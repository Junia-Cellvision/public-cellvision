"""Cellular pipeline helpers: run the segmenter, load metrics, enrich electrodes."""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
from PIL import Image

from cell_segmenter import CellSegmenter


def run_cell_segmenter(image_path: str, cell_dir: Path) -> dict:
    cell_dir.mkdir(parents=True, exist_ok=True)
    seg = CellSegmenter(image_path, str(cell_dir))
    seg.segment()
    seg.Mix(show=False)
    cpm = seg.Metrics(show=False)
    seg.clearAndCopy()
    return cpm


def count_cells(cell_dir: Path) -> int:
    """Sum `n_components` across the metrics JSONs written by CellSegmenter."""
    total = 0
    for p in cell_dir.glob("metrics.json"):
        try:
            data = json.loads(p.read_text())
        except (json.JSONDecodeError, OSError):
            continue
        total += int(data.get("n_components", 0))
    return total


def load_cell_metrics(cell_dir: Path) -> list[dict]:
    """Return the components list from cell_segmenter/metrics.json, or []."""
    p = cell_dir / "metrics.json"
    if not p.exists():
        return []
    try:
        data = json.loads(p.read_text())
    except (json.JSONDecodeError, OSError):
        return []
    return data.get("components", [])


def build_label_image(cells: list[dict], masks_dir: Path) -> np.ndarray | None:
    """Combine per-cell PNG masks into a single label image (pixel = component_id)."""
    if not cells or not masks_dir.exists():
        return None
    sample_path = masks_dir / f"cell_{cells[0]['component_id']:04d}.png"
    if not sample_path.exists():
        return None
    h, w = np.array(Image.open(sample_path).convert("L")).shape
    label_img = np.zeros((h, w), dtype=np.int32)
    for cell in cells:
        cid = cell["component_id"]
        mp = masks_dir / f"cell_{cid:04d}.png"
        if mp.exists():
            label_img[np.array(Image.open(mp).convert("L")) > 0] = cid
    return label_img


def enrich_electrodes(
    electrodes: list[dict],
    cells: list[dict],
    label_img: np.ndarray | None,
) -> list[dict]:
    """Add cell_ids (overlap) and cell_distances to each electrode entry."""
    h, w = (label_img.shape if label_img is not None else (0, 0))
    enriched = []
    for entry in electrodes:
        entry = dict(entry)
        if entry.get("manually_placed"):
            cx, cy = entry["center"]
        else:
            x1, y1, x2, y2 = entry["end_box"]
            cx, cy = (x1 + x2) / 2, (y1 + y2) / 2

        # Which cell (if any) covers the electrode tip pixel?
        cell_ids: list[int] = []
        if label_img is not None:
            px, py = int(round(cx)), int(round(cy))
            if 0 <= py < h and 0 <= px < w:
                cid = int(label_img[py, px])
                if cid > 0:
                    cell_ids = [cid]

        # Distance from electrode tip to every cell centroid
        cell_distances = [
            {
                "cell_id": cell["component_id"],
                "distance_px": round(
                    math.hypot(cx - cell["centroid_x"], cy - cell["centroid_y"]), 1
                ),
            }
            for cell in cells
        ]

        entry["cell_ids"] = cell_ids
        entry["cell_distances"] = cell_distances
        enriched.append(entry)
    return enriched
