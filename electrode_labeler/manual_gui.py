"""Public GUI entry-points for the electrode labeler.

Thin wrappers around `gui_logic` (state machines) and `webgui.server`
(HTTP transport). The function signatures match the historical tkinter
versions so callers (`helper.py`, `manual.py`) stay unchanged.
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

import numpy as np
from numpy.typing import NDArray
from PIL import Image

from electrode_labeler.gui_logic import SelectionLogic, VerificationLogic
from webgui import serve_blocking


def _image_size(image_path: str | Path) -> tuple[int, int]:
    with Image.open(image_path) as im:
        return im.size


def run_verification_gui(
    image_path: str | Path,
    pattern_points: NDArray[np.float64],
    pattern_labels: list[str],
    detected_boxes: list[tuple[float, float, float, float]],
    label_per_electrode_idx: dict[int, str],
    transformed_pattern_points: NDArray[np.float64],
    mea_family: str = "",
) -> tuple[bool, dict[str, tuple[float, float]]]:
    logic = VerificationLogic(
        pattern_points=pattern_points,
        pattern_labels=pattern_labels,
        detected_boxes=detected_boxes,
        label_per_electrode_idx=label_per_electrode_idx,
        transformed_pattern_points=transformed_pattern_points,
        mea_family=mea_family,
        image_size=_image_size(image_path),
    )
    serve_blocking(
        logic,
        image_path=image_path,
        title=f"Vérification du labelling automatique — {mea_family}",
    )
    return logic.result


def run_selection_gui(
    image_path: str | Path,
    pattern_points: NDArray[np.float64],
    pattern_labels: list[str],
    detected_boxes: list[tuple[float, float, float, float]],
    mea_family: str,
    select_landmark_electrodes: Callable[
        [NDArray[np.float64], list[tuple[float, float, float, float]]],
        list[int],
    ],
    estimate_similarity_transform_fn: Callable[
        [NDArray[np.float64], NDArray[np.float64]],
        tuple[float, NDArray[np.float64], NDArray[np.float64]],
    ],
    apply_transform_fn: Callable[
        [NDArray[np.float64], float, NDArray[np.float64], NDArray[np.float64]],
        NDArray[np.float64],
    ],
) -> tuple[list[tuple[float, float]], list[list[float]], dict[str, tuple[float, float]]]:
    lm_indices = select_landmark_electrodes(pattern_points, detected_boxes)
    logic = SelectionLogic(
        pattern_points=pattern_points,
        pattern_labels=pattern_labels,
        detected_boxes=detected_boxes,
        mea_family=mea_family,
        image_size=_image_size(image_path),
        lm_indices=lm_indices,
        estimate_similarity_transform_fn=estimate_similarity_transform_fn,
        apply_transform_fn=apply_transform_fn,
    )
    serve_blocking(
        logic,
        image_path=image_path,
        title=f"Labelling manuel des électrodes — {mea_family}",
    )

    if logic.cancelled or logic.result is None:
        raise RuntimeError("Manual labeling was cancelled.")
    return logic.result
