"""Pure state machine for the manual MEA-family selector.

No web/IO concerns — only state and transitions. Hosted by `webgui.server`
or any other transport.
"""

from __future__ import annotations

import threading
from typing import Any


class MeaSelectLogic:
    """Show the user the auto-detected family + alternatives, capture the choice."""

    def __init__(
        self,
        classes: list[str],
        detected_class: str | None,
        confidence: float,
    ):
        self._classes = list(classes)
        self._detected = (
            detected_class if detected_class in self._classes
            else (self._classes[0] if self._classes else None)
        )
        self._confidence = float(confidence)
        self._result: str | None = self._detected
        self._phase = "select"
        self._done = threading.Event()

    @property
    def is_done(self) -> bool:
        return self._done.is_set()

    def wait(self) -> None:
        self._done.wait()

    @property
    def result(self) -> str | None:
        return self._result

    # ── events ────────────────────────────────────────────────────────────────

    def select_class(self, mea_class: str) -> None:
        if self._phase != "select":
            return
        if mea_class not in self._classes:
            return
        self._result = mea_class
        self._phase = "done"
        self._done.set()

    # ── snapshot ──────────────────────────────────────────────────────────────

    def snapshot(self) -> dict[str, Any]:
        return {
            "kind": "mea_select",
            "phase": self._phase,
            "classes": list(self._classes),
            "detected_class": self._detected,
            "confidence": self._confidence,
        }
