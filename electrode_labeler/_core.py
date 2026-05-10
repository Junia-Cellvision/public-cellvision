from dataclasses import dataclass
import numpy as np
from numpy.typing import NDArray
from scipy.spatial import cKDTree
import random

from electrode_labeler.Hardware import MEAs


def estimate_similarity_transform(src: NDArray[np.float64], dst: NDArray[np.float64]):
    """Calcule la transformation de similarité optimale au sens des moindres carrés.
    Modèle : dst = s * R * src + t
    """
    centroid_src = np.mean(src, axis=0)
    centroid_dst = np.mean(dst, axis=0)

    src_c: NDArray[np.float64] = src - centroid_src
    dst_c: NDArray[np.float64] = dst - centroid_dst

    H = src_c.T @ dst_c

    U, S, Vt = np.linalg.svd(H)
    R = Vt.T @ U.T

    if np.linalg.det(R) < 0:
        Vt[1, :] *= -1
        R = Vt.T @ U.T

    var_src = np.sum(src_c ** 2)
    s = np.sum(S) / var_src

    t = centroid_dst - s * R @ centroid_src

    return s, R, t


def apply_transform(points, s, R, t):
    return (s * (R @ points.T)).T + t


def ransac_similarity(
        pattern_points: NDArray[np.float64],
        detected_points: NDArray[np.float64],
        n_iter=10000,
        threshold=10.0
    ):
    scale_pattern = np.linalg.norm(pattern_points - np.mean(pattern_points, axis=0)) / np.sqrt(len(pattern_points))
    scale_detected = np.linalg.norm(detected_points - np.mean(detected_points, axis=0)) / np.sqrt(len(detected_points))

    expected_s = scale_detected / scale_pattern
    print(f"Ratio d'échelle attendu : {expected_s:.4f}")

    best_inliers = []
    best_model = None
    tree = cKDTree(detected_points)

    for _ in range(n_iter):
        idx_p = random.sample(range(len(pattern_points)), 2)
        idx_d = random.sample(range(len(detected_points)), 2)

        pattern_points_sample = pattern_points[idx_p]
        detected_points_sample = detected_points[idx_d]

        try:
            s, R, t = estimate_similarity_transform(pattern_points_sample, detected_points_sample)

            if not (0.1 * expected_s < s < 10 * expected_s):
                continue

            transformed = apply_transform(pattern_points, s, R, t)
            distances, _ = tree.query(transformed)
            inliers = np.where(distances < threshold)[0]

            if len(inliers) > len(best_inliers):
                best_inliers = inliers
                best_model = (s, R, t)

                if len(inliers) > len(pattern_points) * 0.99:
                    break
        except:
            continue

    if best_model is None:
        return None

    s, R, t = best_model
    transformed = apply_transform(pattern_points, s, R, t)
    distances, indices = tree.query(transformed)
    inliers = np.where(tree.query(transformed)[0] < threshold)[0]

    cost = np.sum(distances) / len(distances)

    if len(inliers) >= 2:
        s, R, t = estimate_similarity_transform(pattern_points[inliers], detected_points[indices[inliers]])

    return s, R, t, cost


def match_electrodes(
        pattern_points: NDArray[np.float64],
        pattern_labels,
        detected_points: NDArray[np.float64],
        threshold=10.0
    ):
    """Return : {detected_electrod_idx: electrod_id}"""
    model = ransac_similarity(pattern_points, detected_points)

    if model is None:
        raise RuntimeError("Transformation impossible à estimer.")

    s, R, t, cost = model
    transformed_pattern_points = apply_transform(pattern_points, s, R, t)

    tree = cKDTree(detected_points)
    distances, detected_indices = tree.query(transformed_pattern_points)

    label_per_electrode_idx: dict[int, str] = {}
    for i, (dist, detected_electrod_idx) in enumerate(zip(distances, detected_indices)):
        label_per_electrode_idx[int(detected_electrod_idx)] = pattern_labels[i]

    return label_per_electrode_idx, transformed_pattern_points


SUPPORTED_FAMILIES: dict[str, MEAs.MEA] = {
    "MEA_60MEA_200": MEAs.MEA_60MEA_200_30,
    "MEA_60MEA_500": MEAs.MEA_60MEA_500_30,
    "MEA_60HexaMEA_40_10": MEAs.MEA_60HexaMEA_40_10,
}


def mea_from_family(mea_family: str):
    try:
        return SUPPORTED_FAMILIES[mea_family]
    except KeyError:
        raise ValueError(f"Pattern '{mea_family}' is not in the list of supported families")


def get_pattern_data(mea: MEAs.MEA) -> tuple[NDArray[np.float64], list[str]]:
    points = []
    labels = []

    for electrode in mea:
        if "REF" in electrode.label:
            continue
        labels.append(electrode.label)
        points.append([float(electrode.position.x), float(electrode.position.y)])

    return np.array(points), labels


def boxes_to_points(
        boxes: list[tuple[float, float, float, float]]
    ) -> NDArray[np.float64]:
    """boxes: list of [x1, y1, x2, y2] — returns Nx2 array of box centres."""
    points = []
    for box in boxes:
        x1, y1, x2, y2 = box
        points.append([(x1 + x2) / 2.0, (y1 + y2) / 2.0])
    return np.array(points)


@dataclass
class Result:
    label_per_electrode_idx: dict[int, str]
    transformed_pattern_points: NDArray[np.float64]
