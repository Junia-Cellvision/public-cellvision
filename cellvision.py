# Headless backend must be set before any matplotlib import (cell_segmenter
# imports pyplot at module level), otherwise plt.show() can hang on a Jetson.
import matplotlib
matplotlib.use("Agg")

import json
import os
import threading
from pathlib import Path

from PIL import Image

from electrode_labeler.helper import labelize_electrodes
from electrode_segmenter import ElectrodeSegmenter
from mea_classifier import ManualMEASelector, MEAClassifier
from results_merger.cells import (
    build_label_image,
    count_cells,
    enrich_electrodes,
    load_cell_metrics,
    run_cell_segmenter,
)
from results_merger.output import NumpyEncoder, OutputZipProvider, resolve_image
from results_merger.viz import generate_per_electrode_viz
from webgui import get_session, serve_blocking
from webgui.display_logic import (
    CellsDisplayLogic,
    DownloadLogic,
    SegmentationDisplayLogic,
    WaitingLogic,
)
from webgui.server import DEFAULT_PORT, RestartRequested

OUTPUT_PATH = Path("./output")


def workflow(image_path: str | None, auto: bool):
    # 0. Charger l'image (upload web ou CLI)
    image_path = resolve_image(image_path, auto)
    image = Image.open(image_path)

    # 1. Segmenter les électrodes
    electrode_ends_box, electrode_bars_poly = ElectrodeSegmenter(
        model_path="./data/models/best.pt"
    ).segment(image)
    print(f"Électrodes trouvés : {len(electrode_ends_box)}")
    if not auto:
        seg_view = SegmentationDisplayLogic(
            image_size=image.size,
            detected_boxes=electrode_ends_box,
            bars_poly=electrode_bars_poly,
        )
        serve_blocking(
            seg_view, image_path=image_path, title="Détourage des électrodes",
        )

    # 2. Trouver la famille de MEA
    mea_found, mea_confidence = MEAClassifier().classify(image)
    print(f"MEA Trouvé : {mea_found}, confiance {mea_confidence:.3%}")
    if not auto:
        mea_found = ManualMEASelector().select(image, mea_found, mea_confidence)
        print(f"MEA sélectionné manuellement : {mea_found}")

    # 3. Labelliser les électrodes selon le pattern
    label_per_electrode_idx, transformed_pattern_points, manually_placed = labelize_electrodes(
        image_path,
        electrode_ends_box,
        mea_found,
        auto=auto,
    )

    # 4. Segmenter les cellules — calcul en arrière-plan + spinner web
    cell_dir = OUTPUT_PATH / "cell_segmenter"
    cell_state: dict[str, object] = {}
    if not auto:
        waiting = WaitingLogic("Segmentation cellulaire en cours…")

        def _bg():
            try:
                cell_state["cpm"] = run_cell_segmenter(image_path, cell_dir)
            except Exception as exc:  # surface failure on the spinner
                cell_state["error"] = exc
            finally:
                waiting.finish()

        threading.Thread(target=_bg, daemon=True).start()
        serve_blocking(
            waiting, image_path=image_path, title="Segmentation cellulaire",
        )
        if "error" in cell_state:
            raise cell_state["error"]
    else:
        cell_state["cpm"] = run_cell_segmenter(image_path, cell_dir)
    n_cells = count_cells(cell_dir)

    # 5. Affichage des cellules segmentées
    if not auto:
        gen = sorted(cell_dir.glob("overlay.png")) + sorted(cell_dir.glob("metrics.png"))
        extra = {f"cells-{p.name}": p for p in gen}
        cells_view = CellsDisplayLogic(list(extra.items()))
        serve_blocking(
            cells_view,
            image_path=image_path,
            title="Cellules segmentées",
            extra_files=extra,
        )

    # 6. Résumer avant d'écrire le résultat final
    labeled_electrodes = [
        {"index": idx, "label": label_per_electrode_idx[idx]}
        for idx in sorted(label_per_electrode_idx)
    ]
    print(f"\n{'=' * 50}")
    print("  Workflow summary")
    print(f"{'=' * 50}")
    print(f"  Image           : {image_path}")
    print(
        f"  MEA family      : {mea_found}  (classifier confidence {mea_confidence:.4f})"
    )
    print(f"  Electrodes det. : {len(electrode_ends_box)}")
    print(f"  Electrodes lab. : {len(label_per_electrode_idx)}")
    print(f"  Manually placed : {len(manually_placed)}")
    print(f"  Cells           : {n_cells}")
    print(f"  Labeled list    :")
    for entry in labeled_electrodes:
        print(f"    box #{entry['index']:>2}  →  {entry['label']}")
    if manually_placed:
        print(f"  Manually placed :")
        for lbl, pos in manually_placed.items():
            print(f"    {lbl}  →  ({pos[0]:.1f}, {pos[1]:.1f})")
    print(f"{'=' * 50}\n")

    # 7. Sauvegarder le résultat
    cells = load_cell_metrics(cell_dir)
    label_img = build_label_image(cells, cell_dir / "masks")

    electrodes = enrich_electrodes(
        [
            {
                "end_box": end_box,
                "label": label_per_electrode_idx[index],
                "manually_placed": False,
            }
            for index, end_box in enumerate(electrode_ends_box)
            if index in label_per_electrode_idx
        ]
        + [
            {
                "end_box": None,
                "center": list(pos),
                "label": lbl,
                "manually_placed": True,
            }
            for lbl, pos in manually_placed.items()
        ],
        cells,
        label_img,
    )

    results = {
        "mea": {"family": mea_found},
        "cells_count": n_cells,
        "cells": cells,
        "electrodes": electrodes,
    }
    OUTPUT_PATH.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_PATH / "results.json", "w") as f:
        json.dump(results, f, cls=NumpyEncoder)

    electrode_assets, unconnected_asset = generate_per_electrode_viz(
        image_path, cells, electrodes, label_img, OUTPUT_PATH
    )

    # 8. Page de résultats + téléchargement zip
    if not auto:
        session = get_session()
        session.set_zip_provider(OutputZipProvider(OUTPUT_PATH))
        try:
            viz_items = [{"key": a["key"], "label": a["label"]} for a in electrode_assets]
            unconnected_viz = {
                "key": unconnected_asset["key"],
                "label": unconnected_asset["label"],
            }
            download = DownloadLogic(
                {
                    "Famille MEA": mea_found,
                    "Confiance": f"{mea_confidence:.3%}",
                    "Électrodes détectées": len(electrode_ends_box),
                    "Électrodes étiquetées": len(label_per_electrode_idx),
                    "Placées manuellement": len(manually_placed),
                    "Cellules": n_cells,
                },
                viz_items=viz_items,
                unconnected_viz=unconnected_viz,
            )
            extra_files = {a["key"]: a["path"] for a in electrode_assets}
            extra_files[unconnected_asset["key"]] = unconnected_asset["path"]
            serve_blocking(
                download, image_path=None, title="Télécharger le résultat",
                extra_files=extra_files,
            )
            if download.restart_requested:
                # Treat the in-page "Recommencer" button the same way
                # as the header one: bubble up to the orchestrator.
                raise RestartRequested()
        finally:
            session.set_zip_provider(None)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--image",
        default=None,
        type=str,
        help="The image file path (omit to upload via the web GUI)",
    )
    parser.add_argument("--auto", action="store_true", help="Perform all in auto mode")
    parser.add_argument("--port", help="Port to access web GUI")
    args = parser.parse_args()

    OUTPUT_PATH.mkdir(exist_ok=True)
    os.environ["PORT"] = args.port or DEFAULT_PORT

    # Loop so the "Recommencer" button in the web GUI restarts the
    # whole workflow without having to relaunch the process. The CLI
    # image (if any) is only used on the very first run; subsequent
    # restarts always go through the upload page.
    image_arg = args.image
    while True:
        try:
            workflow(image_path=image_arg, auto=args.auto)
            break
        except RestartRequested:
            print("\n→ Restart demandé. Reprise du workflow…\n", flush=True)
            image_arg = None
