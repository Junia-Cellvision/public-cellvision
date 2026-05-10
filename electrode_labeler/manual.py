import numpy as np
from numpy.typing import NDArray
from pathlib import Path

from electrode_labeler._core import (
    boxes_to_points,
    mea_from_family,
    get_pattern_data,
    estimate_similarity_transform,
    apply_transform,
)
from electrode_labeler.manual_gui import run_selection_gui
from scipy.spatial import cKDTree


class ManualElectrodeLabeler:
    """
    Interactive labeler: the computer highlights 3 landmark electrodes on the MEA
    schema (right panel); the user clicks matching detected boxes on the microscope
    image (left panel).  Image rotation is supported.
    """

    def __init__(self, image_path: str | Path):
        self.image_path = image_path

    def labelize(
            self,
            detected_boxes: list[tuple[float, float, float, float]],
            mea_family: str,
        ):
        """Return : (label_per_electrode_idx, transformed_pattern_points)"""
        detected_points = boxes_to_points(detected_boxes)

        mea = mea_from_family(mea_family)
        pattern_points, pattern_labels = get_pattern_data(mea)
        pattern_points[:, 0] = -pattern_points[:, 0]

        image_pts, mea_pts, manually_placed = self._run_selection_gui(
            self.image_path, pattern_points, pattern_labels, detected_boxes, mea_family
        )

        s, R, t = estimate_similarity_transform(np.array(mea_pts), np.array(image_pts))
        transformed_pattern_points = apply_transform(pattern_points, s, R, t)

        tree = cKDTree(detected_points)
        distances, detected_indices = tree.query(transformed_pattern_points)

        label_per_electrode_idx: dict[int, str] = {
            int(detected_indices[i]): pattern_labels[i]
            for i in range(len(pattern_labels))
        }

        angle_deg = float(np.degrees(np.arctan2(R[1, 0], R[0, 0])))
        mean_err  = float(np.mean(distances))
        max_err   = float(np.max(distances))
        n_labeled = len(label_per_electrode_idx)
        n_detected = len(detected_points)
        sample = dict(list(label_per_electrode_idx.items())[:6])

        print(f"\n{'='*50}")
        print(f"  Labeling quality — {mea_family}")
        print(f"{'='*50}")
        print(f"  Scale factor    : {s:.4f} px/unit")
        print(f"  Rotation        : {angle_deg:.2f}°")
        print(f"  Mean align. err : {mean_err:.1f} px")
        print(f"  Max  align. err : {max_err:.1f} px")
        print(f"  Labeled         : {n_labeled} / {n_detected} detected electrodes")
        print(f"  Sample labels   : {sample}")
        print(f"{'='*50}\n")

        return label_per_electrode_idx, transformed_pattern_points, manually_placed

    def _select_landmark_electrodes(
        self,
        pattern_points: NDArray[np.float64],
        detected_boxes: list[tuple[float, float, float, float]],
    ) -> list[int]:
        """
        Pick n landmark electrodes evenly spaced in angle around the pattern centroid.

        n scales with detection coverage (3–5). When coverage is partial, candidates
        are restricted to the inner ~65 % of the pattern radius so selected landmarks
        are more likely to be visible in the microscope image.  A random angular
        phase offset ensures the selection varies across sessions.
        """
        n_detected = len(detected_boxes)
        n_pattern = len(pattern_points)
        coverage = min(1.0, n_detected / max(n_pattern, 1))

        n_pts = min(5, 3 + max(0, int((coverage - 0.60) / 0.30)))

        pts = pattern_points.copy()
        centroid = pts.mean(axis=0)
        vectors = pts - centroid
        angles = np.arctan2(vectors[:, 1], vectors[:, 0])
        dists = np.linalg.norm(vectors, axis=1)
        max_d = float(dists.max()) if dists.max() > 0 else 1.0

        candidate_mask = np.ones(len(pts), dtype=bool)
        if coverage < 0.75:
            candidate_mask[dists > max_d * 0.65] = False
            if candidate_mask.sum() < n_pts + 3:
                candidate_mask[:] = True

        angle_offset = np.random.uniform(0, 2 * np.pi / n_pts)

        selected: list[int] = []
        for i in range(n_pts):
            target_a = angle_offset + i * (2 * np.pi / n_pts)
            ang_diff = np.abs(np.angle(np.exp(1j * (angles - target_a))))
            scores = ang_diff - 0.3 * (dists / max_d)
            scores[~candidate_mask] = np.inf
            for s in selected:
                scores[s] = np.inf
            selected.append(int(np.argmin(scores)))

        return selected

    def _run_selection_gui(
            self,
            image_path: str | Path,
            pattern_points: NDArray[np.float64],
            pattern_labels: list[str],
            detected_boxes: list[tuple[float, float, float, float]],
            mea_family: str = "",
        ):
        return run_selection_gui(
            image_path=image_path,
            pattern_points=pattern_points,
            pattern_labels=pattern_labels,
            detected_boxes=detected_boxes,
            mea_family=mea_family,
            select_landmark_electrodes=self._select_landmark_electrodes,
            estimate_similarity_transform_fn=estimate_similarity_transform,
            apply_transform_fn=apply_transform,
        )
