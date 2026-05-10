import numpy as np
from pathlib import Path

from electrode_labeler._core import (
    boxes_to_points,
    mea_from_family,
    get_pattern_data,
    match_electrodes,
)


class AutomaticElectrodeLabeler:

    def __init__(self, image_path: str | Path):
        self.image_path = image_path

    def labelize(
            self,
            detected_boxes: list[tuple[float, float, float, float]],
            mea_family: str,
        ):
        """Return : {detected_electrod_idx: electrod_id}"""
        detected_points = boxes_to_points(detected_boxes)

        mea = mea_from_family(mea_family)
        pattern_points, pattern_labels = get_pattern_data(mea)
        pattern_points[:, 0] = -pattern_points[:, 0]

        detected_boxes_count = len(detected_points)
        pattern_electrods_count = len(pattern_points)

        print(f"Nombre d'électrodes détectées {detected_boxes_count}")
        print(f"Nombre d'électrodes du pattern {pattern_electrods_count}")

        if detected_boxes_count == 0:
            print("Aucune électrode détectée — étiquetage automatique ignoré.")
            return {}, pattern_points.copy()

        label_per_electrode_idx, transformed_pattern_points = match_electrodes(
            pattern_points,
            pattern_labels,
            detected_points,
            threshold=1000.0
        )

        print("Association (index_box_detectée -> id_pattern) :")
        print(label_per_electrode_idx)

        print(f"Étendue du pattern : {np.max(pattern_points, axis=0) - np.min(pattern_points, axis=0)}")
        print(f"Étendue des détections : {np.max(detected_points, axis=0) - np.min(detected_points, axis=0)}")

        return label_per_electrode_idx, transformed_pattern_points
