"""Per-electrode result visualization (one PNG per electrode + one for unconnected cells)."""

from __future__ import annotations

import re
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw


_VIZ_MAX_W = 1024
_VIZ_DIM   = 0.25   # brightness factor for background cells
_VIZ_ALPHA = 0.65   # highlight opacity
_COLOR_CONNECTED   = np.array([0.25, 1.00, 0.35], dtype=np.float32)  # bright green
_COLOR_UNCONNECTED = np.array([1.00, 0.15, 0.15], dtype=np.float32)  # red
_ELECTRODE_OUTLINE = "#ffd400"  # bright yellow — distinct from green/red


def generate_per_electrode_viz(
    image_path: str,
    cells: list[dict],
    electrodes: list[dict],
    label_img: np.ndarray | None,
    output_dir: Path,
) -> tuple[list[dict], dict]:
    """
    Generate one PNG per electrode (connected cells highlighted green, with
    the electrode itself outlined) + one PNG for unconnected cells (red).

    Returns (per_electrode_assets, unconnected_asset).
    Each asset is {"key": str, "label": str, "path": Path}.
    """
    img_pil = Image.open(image_path).convert("RGB")
    scale = 1.0
    if img_pil.width > _VIZ_MAX_W:
        scale = _VIZ_MAX_W / img_pil.width
        img_pil = img_pil.resize(
            (_VIZ_MAX_W, int(img_pil.height * scale)), Image.LANCZOS
        )
        if label_img is not None:
            lbl_pil = Image.fromarray(label_img.astype(np.int32), mode="I")
            label_img = np.array(
                lbl_pil.resize((_VIZ_MAX_W, int(lbl_pil.height * scale)), Image.NEAREST)
            )

    base = np.array(img_pil).astype(np.float32) / 255.0
    H, W = base.shape[:2]

    # Dim all cells in the base image so highlights stand out clearly
    if label_img is not None:
        all_cells = label_img > 0
        base[all_cells] *= _VIZ_DIM

    cell_to_electrode: dict[int, str] = {}
    for elec in electrodes:
        for cid in elec.get("cell_ids", []):
            cell_to_electrode[cid] = elec["label"]

    all_cell_ids = {c["component_id"] for c in cells}
    unconnected_ids = all_cell_ids - set(cell_to_electrode)

    def _save(
        highlight_ids: set[int],
        color: np.ndarray,
        key: str,
        label: str,
        fname: str,
        elec: dict | None = None,
    ) -> dict:
        out = base.copy()
        if label_img is not None and highlight_ids:
            mask = np.isin(label_img, list(highlight_ids))
            out[mask] = (1 - _VIZ_ALPHA) * out[mask] + _VIZ_ALPHA * color
        img = Image.fromarray((np.clip(out, 0, 1) * 255).astype(np.uint8))

        if elec is not None:
            draw = ImageDraw.Draw(img)
            stroke = max(2, int(round(min(W, H) * 0.004)))
            if elec.get("manually_placed"):
                cx, cy = elec["center"]
                cx, cy = cx * scale, cy * scale
                r = max(8, int(round(min(W, H) * 0.012)))
                draw.ellipse(
                    (cx - r, cy - r, cx + r, cy + r),
                    outline=_ELECTRODE_OUTLINE, width=stroke,
                )
            else:
                x1, y1, x2, y2 = elec["end_box"]
                draw.rectangle(
                    (x1 * scale, y1 * scale, x2 * scale, y2 * scale),
                    outline=_ELECTRODE_OUTLINE, width=stroke,
                )

        path = output_dir / fname
        img.save(path, optimize=True)
        return {"key": key, "label": label, "path": path}

    def _natural_key(e: dict):
        parts = re.split(r"(\d+)", e["label"])
        return [int(p) if p.isdigit() else p for p in parts]

    electrode_assets = []
    for elec in sorted(electrodes, key=_natural_key):
        lbl = elec["label"]
        safe = lbl.replace(" ", "_").replace("/", "-")
        n = len(elec.get("cell_ids", []))
        display = f"Électrode {lbl} — {n} cellule{'s' if n != 1 else ''}"
        electrode_assets.append(_save(
            set(elec.get("cell_ids", [])), _COLOR_CONNECTED,
            f"viz_elec_{safe}", display, f"viz_electrode_{safe}.png",
            elec=elec,
        ))

    n_unc = len(unconnected_ids)
    unconnected_asset = _save(
        unconnected_ids, _COLOR_UNCONNECTED,
        "viz_unconnected",
        f"Cellules non connectées — {n_unc}",
        "viz_unconnected.png",
    )
    return electrode_assets, unconnected_asset
