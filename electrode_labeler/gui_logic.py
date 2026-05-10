"""Pure state machines for the electrode-labeler interactive GUIs.

No web/IO dependencies — only state and transitions. The web layer
(`webgui.server`) hosts these and exposes them over HTTP; the same
logic could be driven from any other front-end.

Each state machine exposes:
    snapshot()       — JSON-serializable view of the current state
    is_done / wait() — completion signaling
    result           — final result once done
    <event methods>  — mutate state; the web layer dispatches by name
"""

from __future__ import annotations

import threading
from typing import Any

import numpy as np
from numpy.typing import NDArray
from scipy.spatial import cKDTree


class VerificationLogic:
    """Approve / reject the automatic labeling, then place missing electrodes."""

    def __init__(
        self,
        pattern_points: NDArray[np.float64],
        pattern_labels: list[str],
        detected_boxes: list[tuple[float, float, float, float]],
        label_per_electrode_idx: dict[int, str],
        transformed_pattern_points: NDArray[np.float64],
        mea_family: str,
        image_size: tuple[int, int],
    ):
        self._pattern_points = pattern_points
        self._pattern_labels = list(pattern_labels)
        self._detected_boxes = list(detected_boxes)
        self._label_per_box = dict(label_per_electrode_idx)
        self._mea_family = mea_family
        self._image_size = image_size

        labeled = set(self._label_per_box.values())
        self._missing_labels = [lbl for lbl in pattern_labels if lbl not in labeled]
        self._missing_hint = {
            pattern_labels[i]: (
                float(transformed_pattern_points[i][0]),
                float(transformed_pattern_points[i][1]),
            )
            for i in range(len(pattern_labels))
            if pattern_labels[i] not in labeled
        }

        self._phase = "review"
        self._missing_step = 0
        self._manually_placed: dict[str, tuple[float, float]] = {}
        self._approved = False
        self._done = threading.Event()

    @property
    def is_done(self) -> bool:
        return self._done.is_set()

    def wait(self) -> None:
        self._done.wait()

    @property
    def result(self) -> tuple[bool, dict[str, tuple[float, float]]]:
        return self._approved, self._manually_placed

    # ── events ────────────────────────────────────────────────────────────────

    def approve(self) -> None:
        if self._phase != "review":
            return
        self._approved = True
        if self._missing_labels:
            self._phase = "missing"
            self._missing_step = 0
        else:
            self._finish()

    def reject(self) -> None:
        if self._phase != "review":
            return
        self._approved = False
        self._finish()

    def place_at(self, x: float, y: float) -> None:
        if self._phase != "missing":
            return
        lbl = self._missing_labels[self._missing_step]
        self._manually_placed[lbl] = (float(x), float(y))
        self._advance_missing()

    def autoplace(self) -> None:
        if self._phase != "missing":
            return
        lbl = self._missing_labels[self._missing_step]
        self.place_at(*self._missing_hint[lbl])

    def skip(self) -> None:
        if self._phase != "missing":
            return
        self._advance_missing()

    def _advance_missing(self) -> None:
        self._missing_step += 1
        if self._missing_step >= len(self._missing_labels):
            self._finish()

    def _finish(self) -> None:
        self._phase = "done"
        self._done.set()

    # ── snapshot ──────────────────────────────────────────────────────────────

    def snapshot(self) -> dict[str, Any]:
        cur = (
            self._missing_labels[self._missing_step]
            if self._phase == "missing"
            else None
        )
        return {
            "kind": "verification",
            "phase": self._phase,
            "mea_family": self._mea_family,
            "image_size": list(self._image_size),
            "pattern_points": [[float(x), float(y)] for x, y in self._pattern_points],
            "pattern_labels": list(self._pattern_labels),
            "detected_boxes": [list(b) for b in self._detected_boxes],
            "label_per_box": {str(k): v for k, v in self._label_per_box.items()},
            "missing_labels": list(self._missing_labels),
            "missing_hint": {k: list(v) for k, v in self._missing_hint.items()},
            "manually_placed": {k: list(v) for k, v in self._manually_placed.items()},
            "current_missing_label": cur,
            "missing_step": self._missing_step,
        }


class SelectionLogic:
    """Manual labeling: pick landmarks → verify → place missing electrodes."""

    def __init__(
        self,
        pattern_points: NDArray[np.float64],
        pattern_labels: list[str],
        detected_boxes: list[tuple[float, float, float, float]],
        mea_family: str,
        image_size: tuple[int, int],
        lm_indices: list[int],
        estimate_similarity_transform_fn,
        apply_transform_fn,
    ):
        self._pattern_points = pattern_points
        self._pattern_labels = list(pattern_labels)
        self._detected_boxes = list(detected_boxes)
        self._box_centers = [
            ((x1 + x2) / 2.0, (y1 + y2) / 2.0)
            for x1, y1, x2, y2 in detected_boxes
        ]
        self._mea_family = mea_family
        self._image_size = image_size
        self._lm_indices = list(lm_indices)
        self._estimate = estimate_similarity_transform_fn
        self._apply = apply_transform_fn

        self._phase = "select"
        self._step = 0
        self._lm_mode = False
        self._selections: list[dict] = []
        self._box_labels: dict[int, str] = {}
        self._transformed: NDArray[np.float64] | None = None

        self._missing_labels: list[str] = []
        self._missing_step = 0
        self._manually_placed: dict[str, tuple[float, float]] = {}
        self._missing_hint: dict[str, tuple[float, float]] = {}

        self._cancelled = False
        self._done = threading.Event()

    @property
    def is_done(self) -> bool:
        return self._done.is_set()

    @property
    def cancelled(self) -> bool:
        return self._cancelled

    def wait(self) -> None:
        self._done.wait()

    @property
    def result(self):
        if self._cancelled:
            return None
        image_pts = [(s["pos"][0], s["pos"][1]) for s in self._selections]
        mea_pts = [
            self._pattern_points[self._lm_indices[i]].tolist()
            for i in range(len(self._lm_indices))
        ]
        return image_pts, mea_pts, self._manually_placed

    # ── events ────────────────────────────────────────────────────────────────

    def select_box(self, box_idx: int) -> None:
        if self._phase != "select" or self._lm_mode:
            return
        if self._step >= len(self._lm_indices):
            return
        box_idx = int(box_idx)
        if any(s.get("box_idx") == box_idx for s in self._selections):
            return
        self._selections.append({
            "pos": self._box_centers[box_idx],
            "type": "box",
            "box_idx": box_idx,
        })
        self._step += 1

    def select_image_point(self, x: float, y: float) -> None:
        if self._phase == "missing":
            self._place_missing(float(x), float(y))
            return
        if self._phase != "select" or self._lm_mode:
            return
        if self._step >= len(self._lm_indices):
            return
        self._selections.append({
            "pos": (float(x), float(y)),
            "type": "manual",
            "box_idx": None,
        })
        self._step += 1

    def toggle_lm_mode(self) -> None:
        if self._phase != "select" or self._step >= len(self._lm_indices):
            return
        self._lm_mode = not self._lm_mode

    def change_landmark(self, pattern_idx: int) -> None:
        if not self._lm_mode or self._phase != "select":
            return
        used = {
            self._lm_indices[i]
            for i in range(len(self._lm_indices))
            if i != self._step
        }
        if pattern_idx in used:
            return
        self._lm_indices[self._step] = int(pattern_idx)
        self._lm_mode = False

    def undo(self) -> None:
        if self._phase != "select" or not self._selections:
            return
        self._selections.pop()
        self._step -= 1
        self._lm_mode = False

    def confirm(self) -> None:
        if self._phase == "select":
            if self._step < len(self._lm_indices):
                return
            img_pts = np.array([(s["pos"][0], s["pos"][1]) for s in self._selections])
            mea_pts = np.array([self._pattern_points[i] for i in self._lm_indices])
            s, R, t = self._estimate(mea_pts, img_pts)
            transformed = self._apply(self._pattern_points, s, R, t)
            det_pts = np.array(self._box_centers)
            _, det_idx = cKDTree(det_pts).query(transformed)
            self._box_labels = {
                int(det_idx[i]): self._pattern_labels[i]
                for i in range(len(self._pattern_labels))
            }
            self._transformed = transformed
            self._phase = "verify"
            return

        if self._phase == "verify":
            labeled = set(self._box_labels.values())
            missing = [lbl for lbl in self._pattern_labels if lbl not in labeled]
            if not missing:
                self._finish()
                return
            assert self._transformed is not None
            self._missing_labels = missing
            self._missing_step = 0
            self._manually_placed = {}
            self._missing_hint = {
                self._pattern_labels[i]: (
                    float(self._transformed[i][0]),
                    float(self._transformed[i][1]),
                )
                for i in range(len(self._pattern_labels))
                if self._pattern_labels[i] in set(missing)
            }
            self._phase = "missing"

    def back(self) -> None:
        if self._phase != "verify":
            return
        self._phase = "select"
        self._box_labels = {}

    def cancel(self) -> None:
        self._cancelled = True
        self._finish()

    def autoplace(self) -> None:
        if self._phase != "missing":
            return
        lbl = self._missing_labels[self._missing_step]
        self._place_missing(*self._missing_hint[lbl])

    def skip(self) -> None:
        if self._phase != "missing":
            return
        self._advance_missing()

    def _place_missing(self, x: float, y: float) -> None:
        lbl = self._missing_labels[self._missing_step]
        self._manually_placed[lbl] = (float(x), float(y))
        self._advance_missing()

    def _advance_missing(self) -> None:
        self._missing_step += 1
        if self._missing_step >= len(self._missing_labels):
            self._finish()

    def _finish(self) -> None:
        self._phase = "done"
        self._done.set()

    # ── snapshot ──────────────────────────────────────────────────────────────

    def snapshot(self) -> dict[str, Any]:
        cur_missing = (
            self._missing_labels[self._missing_step]
            if self._phase == "missing"
            else None
        )
        return {
            "kind": "selection",
            "phase": self._phase,
            "mea_family": self._mea_family,
            "image_size": list(self._image_size),
            "pattern_points": [[float(x), float(y)] for x, y in self._pattern_points],
            "pattern_labels": list(self._pattern_labels),
            "detected_boxes": [list(b) for b in self._detected_boxes],
            "box_centers": [list(c) for c in self._box_centers],
            "lm_indices": list(self._lm_indices),
            "n_landmarks": len(self._lm_indices),
            "step": self._step,
            "lm_mode": self._lm_mode,
            "selections": [
                {
                    "type": s["type"],
                    "pos": [float(s["pos"][0]), float(s["pos"][1])],
                    "box_idx": s["box_idx"],
                }
                for s in self._selections
            ],
            "box_labels": {str(k): v for k, v in self._box_labels.items()},
            "missing_labels": list(self._missing_labels),
            "missing_step": self._missing_step,
            "manually_placed": {
                k: list(v) for k, v in self._manually_placed.items()
            },
            "missing_hint": {k: list(v) for k, v in self._missing_hint.items()},
            "current_missing_label": cur_missing,
        }
