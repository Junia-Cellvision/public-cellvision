"""HTTP layer: a long-lived uvicorn server hosting one logic at a time.

Each workflow step calls `serve_blocking(...)` which swaps the active
logic in place on the shared session and blocks until it completes.
The browser stays on the same URL across all steps; it polls
`/api/state` between steps and re-renders when the `generation`
counter advances.

The logic object is opaque to this module. It must implement::

    snapshot() -> dict     # JSON-serializable view of the current state
    is_done   : bool       # True once the step is complete
    wait()                 # blocks until is_done

…and one method per event name dispatched by the front-end. Methods
starting with "_" are not exposed.
"""

from __future__ import annotations

import atexit
import os
import shutil
import socket
import tempfile
import threading
import time
from pathlib import Path
from typing import Any, Protocol

import uvicorn
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

DEFAULT_PORT = "8888"

_STATIC_DIR = Path(__file__).parent / "app/dist"


class _GuiLogic(Protocol):
    is_done: bool
    def snapshot(self) -> dict[str, Any]: ...
    def wait(self) -> None: ...


class RestartRequested(Exception):
    """Raised by `WebGuiSession.run` when the user clicked Restart in the UI.

    The orchestrator (cellvision.py) catches this at the top level and
    re-runs the workflow from scratch.
    """


class ZipProvider(Protocol):
    def build_zip(self, dest: Path) -> Path: ...
    @property
    def filename(self) -> str: ...


class _EventIn(BaseModel):
    name: str
    payload: dict[str, Any] = {}



def _local_url(port: int) -> str:
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        # Connexion fictive → permet de récupérer l’IP réseau réelle
        s.connect(("8.8.8.8", 80))
        host = s.getsockname()[0]
    except Exception:
        host = "localhost"
    finally:
        s.close()
    return f"http://{host}:{port}/"


class WebGuiSession:
    """A long-lived uvicorn server hosting one GUI logic at a time.

    Multiple workflow steps reuse the same server (and therefore the
    same browser URL): each call to `run()` swaps the active logic and
    blocks until that step completes. The frontend polls between steps
    and follows along automatically.
    """

    def __init__(self, host: str = "0.0.0.0", port: int|None = None):
        self.host = host
        self.port = int(port or os.environ['PORT'] or DEFAULT_PORT)

        self._lock = threading.RLock()
        self._logic: _GuiLogic | None = None
        self._title: str = ""
        self._image_path: Path | None = None
        self._extra_files: dict[str, Path] = {}
        self._generation: int = 0
        self._zip_provider: "ZipProvider | None" = None
        self._restart_event = threading.Event()

        # Persistent scratch dir for uploads/zips; lives for the session lifetime.
        self._scratch_dir = Path(tempfile.mkdtemp(prefix="cellvision_web_"))

        self._server: uvicorn.Server | None = None
        self._thread: threading.Thread | None = None
        self._closed: bool = False

    @property
    def scratch_dir(self) -> Path:
        return self._scratch_dir

    def set_image_path(self, path: str | Path) -> None:
        """Switch the image served by `/api/image` (e.g. after upload)."""
        with self._lock:
            self._image_path = Path(path)

    def set_zip_provider(self, provider: "ZipProvider | None") -> None:
        """Configure the source for `GET /api/result.zip`."""
        with self._lock:
            self._zip_provider = provider

    # ── lifecycle ─────────────────────────────────────────────────────────────

    def start(self) -> None:
        if self._server is not None:
            return
        config = uvicorn.Config(
            self._build_app(),
            host=self.host,
            port=self.port,
            log_level="warning",
        )
        self._server = uvicorn.Server(config)
        self._thread = threading.Thread(target=self._server.run, daemon=True)
        self._thread.start()

        deadline = time.monotonic() + 10.0
        while not self._server.started:
            if not self._thread.is_alive():
                raise RuntimeError("uvicorn failed to start")
            if time.monotonic() > deadline:
                raise RuntimeError("uvicorn did not become ready in time")
            time.sleep(0.05)

        print(
            f"\nInterface web Cellvision prête : \n"
            f"Local : http://localhost:{self.port}/\n"
            f"Réseau : {_local_url(self.port)}\n"
            f"Gardez cette URL ouverte pendant tout le workflow.\n",
            flush=True,
        )
        atexit.register(self.close)

    def run(
        self,
        logic: _GuiLogic,
        image_path: str | Path | None,
        title: str,
        extra_files: dict[str, str | Path] | None = None,
    ) -> None:
        if self._server is None:
            self.start()
        # Drop any stale restart signal from a prior step so it can't
        # immediately abort this one.
        self._restart_event.clear()
        with self._lock:
            self._logic = logic
            self._image_path = Path(image_path) if image_path is not None else None
            self._title = title
            self._extra_files = {k: Path(v) for k, v in (extra_files or {}).items()}
            self._generation += 1
        print(f"  → étape : {title}", flush=True)
        # Wait until the step finishes OR the user asked to restart
        # the whole workflow. Polling at 50 ms keeps UI latency
        # imperceptible without busy-spinning.
        while not logic.is_done:
            if self._restart_event.wait(timeout=0.05):
                self._restart_event.clear()
                raise RestartRequested()

    def request_restart(self) -> None:
        """Signal the active step to abort with `RestartRequested`."""
        self._restart_event.set()

    def close(self) -> None:
        if self._closed or self._server is None:
            return
        self._closed = True
        with self._lock:
            self._logic = None
            self._image_path = None
            self._extra_files = {}
            self._generation += 1
        self._server.should_exit = True
        if self._thread is not None:
            self._thread.join(timeout=5)
        self._server = None
        self._thread = None
        shutil.rmtree(self._scratch_dir, ignore_errors=True)

    # ── HTTP routes ───────────────────────────────────────────────────────────

    def _build_app(self) -> FastAPI:
        app = FastAPI(title="Cellvision GUI")

        @app.get("/")
        def index() -> FileResponse:
            return FileResponse(_STATIC_DIR / "index.html")

        @app.get("/api/state")
        def state() -> JSONResponse:
            with self._lock:
                if self._logic is None:
                    return JSONResponse({
                        "title": "",
                        "generation": self._generation,
                        "kind": "idle",
                        "phase": "idle",
                    })
                return JSONResponse({
                    "title": self._title,
                    "generation": self._generation,
                    **self._logic.snapshot(),
                })

        @app.get("/api/image")
        def image() -> FileResponse:
            with self._lock:
                p = self._image_path
            if p is None:
                raise HTTPException(404, "no active image")
            return FileResponse(p)

        @app.get("/api/asset/{name}")
        def asset(name: str) -> FileResponse:
            with self._lock:
                p = self._extra_files.get(name)
            if p is None:
                raise HTTPException(404, f"unknown asset {name!r}")
            return FileResponse(p)

        @app.post("/api/upload")
        async def upload(file: UploadFile = File(...)) -> JSONResponse:
            with self._lock:
                logic = self._logic
                if logic is None or not hasattr(logic, "set_uploaded_path"):
                    raise HTTPException(409, "no active upload step")
            suffix = Path(file.filename or "upload.bin").suffix or ".bin"
            dest = self._scratch_dir / f"upload_{int(time.time() * 1000)}{suffix}"
            with dest.open("wb") as out:
                while chunk := await file.read(1 << 20):
                    out.write(chunk)
            logic.set_uploaded_path(dest, file.filename or dest.name)
            with self._lock:
                self._image_path = dest
            return JSONResponse({
                "title": self._title,
                "generation": self._generation,
                **logic.snapshot(),
            })

        @app.get("/api/result.zip")
        def result_zip() -> FileResponse:
            with self._lock:
                provider = self._zip_provider
            if provider is None:
                raise HTTPException(404, "no zip available")
            out = provider.build_zip(self._scratch_dir)
            return FileResponse(
                out,
                media_type="application/zip",
                filename=provider.filename,
            )

        @app.post("/api/restart")
        def restart() -> JSONResponse:
            self.request_restart()
            return JSONResponse({"ok": True})

        @app.post("/api/event")
        def event(ev: _EventIn) -> JSONResponse:
            if ev.name.startswith("_"):
                raise HTTPException(404, f"unknown event {ev.name!r}")
            with self._lock:
                logic = self._logic
                if logic is None:
                    raise HTTPException(409, "no active step")
                handler = getattr(logic, ev.name, None)
                if not callable(handler):
                    raise HTTPException(404, f"unknown event {ev.name!r}")
                try:
                    handler(**ev.payload)
                except TypeError as e:
                    raise HTTPException(400, str(e))
                return JSONResponse({
                    "title": self._title,
                    "generation": self._generation,
                    **logic.snapshot(),
                })

        app.mount("/", StaticFiles(directory=_STATIC_DIR), name="static")
        return app


# ── module-level singleton ────────────────────────────────────────────────────

_singleton_lock = threading.Lock()
_singleton: WebGuiSession | None = None


def serve_blocking(
    logic: _GuiLogic,
    image_path: str | Path | None,
    title: str,
    *,
    extra_files: dict[str, str | Path] | None = None,
) -> None:
    """Host `logic` on the shared session and block until it completes."""
    session = get_session()
    session.run(logic, image_path=image_path, title=title, extra_files=extra_files)


def get_session() -> WebGuiSession:
    """Return the shared session, starting it on first call."""
    global _singleton
    with _singleton_lock:
        if _singleton is None:
            _singleton = WebGuiSession()
            _singleton.start()
        return _singleton
