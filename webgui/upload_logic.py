"""Pure state machine for the initial image-upload step.

The web layer (`webgui.server`) hosts an instance until a file is
uploaded via `POST /api/upload`. The server side saves the upload to
disk and calls `set_uploaded_path(...)` to fulfil the step.
"""

from __future__ import annotations

import threading
from pathlib import Path
from typing import Any


class UploadLogic:
    """Wait for the user to upload one image."""

    def __init__(self, accept: str = "image/*"):
        self._accept = accept
        self._uploaded_path: Path | None = None
        self._uploaded_name: str | None = None
        self._error: str | None = None
        self._phase = "wait"
        self._done = threading.Event()

    @property
    def is_done(self) -> bool:
        return self._done.is_set()

    def wait(self) -> None:
        self._done.wait()

    @property
    def result(self) -> Path:
        if self._uploaded_path is None:
            raise RuntimeError("upload was never completed")
        return self._uploaded_path

    # ── server-side hooks ────────────────────────────────────────────────────

    def set_uploaded_path(self, path: Path, original_name: str) -> None:
        """Called by the HTTP layer once it has saved the file to disk."""
        self._uploaded_path = Path(path)
        self._uploaded_name = original_name
        self._phase = "done"
        self._done.set()

    def set_error(self, message: str) -> None:
        self._error = message

    # ── snapshot ─────────────────────────────────────────────────────────────

    def snapshot(self) -> dict[str, Any]:
        return {
            "kind": "upload",
            "phase": self._phase,
            "accept": self._accept,
            "uploaded_name": self._uploaded_name,
            "error": self._error,
        }
