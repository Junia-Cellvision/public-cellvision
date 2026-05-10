"""Pure state machines for passive display steps.

Each one shows something to the user and waits for a single
"continue" press. No business logic — only presentation state.
"""

from __future__ import annotations

import threading
from pathlib import Path
from typing import Any


class SegmentationDisplayLogic:
    """Show detected electrode boxes (and optional bar polygons) over the image."""

    def __init__(
        self,
        image_size: tuple[int, int],
        detected_boxes: list[tuple[float, float, float, float]],
        bars_poly: list[list[tuple[float, float]]] | None = None,
    ):
        self._image_size = image_size
        self._detected_boxes = list(detected_boxes)
        self._bars_poly = [list(p) for p in (bars_poly or [])]
        self._phase = "show"
        self._done = threading.Event()

    @property
    def is_done(self) -> bool:
        return self._done.is_set()

    def wait(self) -> None:
        self._done.wait()

    # ── events ───────────────────────────────────────────────────────────────

    def confirm(self) -> None:
        if self._phase != "show":
            return
        self._phase = "done"
        self._done.set()

    # ── snapshot ─────────────────────────────────────────────────────────────

    def snapshot(self) -> dict[str, Any]:
        return {
            "kind": "display_segmentation",
            "phase": self._phase,
            "image_size": list(self._image_size),
            "detected_boxes": [list(b) for b in self._detected_boxes],
            "bars_poly": [
                [[float(x), float(y)] for x, y in poly]
                for poly in self._bars_poly
            ],
        }


class CellsDisplayLogic:
    """Show cell-segmentation overlay/metric images and wait for confirm."""

    def __init__(self, image_files: list[tuple[str, Path]]):
        # `image_files`: [(asset_key, path)] — exposed via /api/asset/<key>
        self._items = [{"key": k, "label": Path(p).stem} for k, p in image_files]
        self._phase = "show"
        self._done = threading.Event()

    @property
    def is_done(self) -> bool:
        return self._done.is_set()

    def wait(self) -> None:
        self._done.wait()

    def confirm(self) -> None:
        if self._phase != "show":
            return
        self._phase = "done"
        self._done.set()

    def snapshot(self) -> dict[str, Any]:
        return {
            "kind": "display_cells",
            "phase": self._phase,
            "items": list(self._items),
        }


class WaitingLogic:
    """Show a spinner with a status message while a background task runs."""

    def __init__(self, message: str):
        self._message = message
        self._phase = "wait"
        self._done = threading.Event()

    @property
    def is_done(self) -> bool:
        return self._done.is_set()

    def wait(self) -> None:
        self._done.wait()

    def finish(self) -> None:
        """Called from the background thread when the work is over."""
        self._phase = "done"
        self._done.set()

    def update_message(self, message: str) -> None:
        self._message = message

    def snapshot(self) -> dict[str, Any]:
        return {
            "kind": "waiting",
            "phase": self._phase,
            "message": self._message,
        }


class DownloadLogic:
    """Final step: present a zip download + a 'restart' button."""

    def __init__(
        self,
        summary: dict[str, Any],
        viz_items: list[dict] | None = None,
        unconnected_viz: dict | None = None,
    ):
        self._summary = dict(summary)
        self._viz_items = list(viz_items) if viz_items else []
        self._unconnected_viz = dict(unconnected_viz) if unconnected_viz else None
        self._phase = "show"
        self._restart = False
        self._done = threading.Event()

    @property
    def is_done(self) -> bool:
        return self._done.is_set()

    def wait(self) -> None:
        self._done.wait()

    @property
    def restart_requested(self) -> bool:
        return self._restart

    def finish(self) -> None:
        if self._phase != "show":
            return
        self._phase = "done"
        self._done.set()

    def restart(self) -> None:
        if self._phase != "show":
            return
        self._restart = True
        self._phase = "done"
        self._done.set()

    def snapshot(self) -> dict[str, Any]:
        s: dict[str, Any] = {
            "kind": "download",
            "phase": self._phase,
            "summary": dict(self._summary),
            "zip_url": "/api/result.zip",
            "viz_items": [
                {"label": item["label"], "url": f"/api/asset/{item['key']}"}
                for item in self._viz_items
            ],
        }
        if self._unconnected_viz:
            s["unconnected_viz"] = {
                "label": self._unconnected_viz["label"],
                "url": f"/api/asset/{self._unconnected_viz['key']}",
            }
        return s
