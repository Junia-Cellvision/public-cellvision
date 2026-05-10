from pathlib import Path

import numpy as np
from numpy.typing import NDArray

from electrode_labeler._core import get_pattern_data, mea_from_family
from electrode_labeler.automatic import AutomaticElectrodeLabeler
from electrode_labeler.manual import ManualElectrodeLabeler
from electrode_labeler.manual_gui import run_verification_gui


def labelize_electrodes(
    image_path: str | Path,
    detected_boxes: list[tuple[float, float, float, float]],
    mea_family: str,
    auto: bool = False
) -> tuple[dict[int, str], NDArray[np.float64], dict[str, tuple[float, float]]]:
    """Run automatic labeling, let the user verify, and fall back to manual if rejected.

    Returns (label_per_electrode_idx, transformed_pattern_points, manually_placed).
    """
    label_per_electrode_idx, transformed_pattern_points = AutomaticElectrodeLabeler(
        image_path
    ).labelize(detected_boxes, mea_family)

    manually_placed = {}
    
    if not auto:
        pattern_points, pattern_labels = get_pattern_data(mea_from_family(mea_family))
        pattern_points[:, 0] = -pattern_points[:, 0]

        approved, manually_placed = run_verification_gui(
            image_path=image_path,
            pattern_points=pattern_points,
            pattern_labels=pattern_labels,
            detected_boxes=detected_boxes,
            label_per_electrode_idx=label_per_electrode_idx,
            transformed_pattern_points=transformed_pattern_points,
            mea_family=mea_family,
        )

        if not approved:
            label_per_electrode_idx, transformed_pattern_points, manually_placed = (
                ManualElectrodeLabeler(image_path).labelize(detected_boxes, mea_family)
            )

    return label_per_electrode_idx, transformed_pattern_points, manually_placed
