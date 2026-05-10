"""Public GUI entry-point for manual MEA-family selection.

Thin wrapper over `gui_logic.MeaSelectLogic` (state machine) and
`webgui.server` (HTTP transport). Renders the MEA-pattern thumbnails
once via matplotlib and exposes them as static assets to the SPA.
"""

from __future__ import annotations

import tempfile
from pathlib import Path

from PIL import Image
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure

from electrode_labeler.labeler import SUPPORTED_FAMILIES, mea_from_family
from mea_classifier.gui_logic import MeaSelectLogic
from webgui import serve_blocking


KNOWN_CLASSES = list(SUPPORTED_FAMILIES.keys())

_THUMB_INCHES = (3.0, 3.0)
_THUMB_DPI = 120


def _render_mea_thumbnail(mea_class: str, out_path: Path) -> None:
    fig = Figure(figsize=_THUMB_INCHES, dpi=_THUMB_DPI, facecolor="white")
    ax = fig.add_subplot(111, facecolor="white")
    try:
        mea_from_family(mea_class).draw(ax, text=False)
        ax.set_title("")
        ax.set_xlabel("")
        ax.set_ylabel("")
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_visible(False)
    except Exception as exc:
        ax.clear()
        ax.text(0.5, 0.5, f"(error: {exc})", ha="center", va="center")
        ax.set_axis_off()
    fig.tight_layout(pad=0.2)
    FigureCanvasAgg(fig).print_png(out_path)


class ManualMEASelector:
    """Web-based manual selection of the MEA family."""

    def select(
        self,
        input_image: Image.Image,
        detected_class: str | None,
        confidence: float,
    ) -> str:
        with tempfile.TemporaryDirectory(prefix="cellvision_mea_") as tmpdir:
            tmp = Path(tmpdir)

            input_path = tmp / "input.jpg"
            input_image.convert("RGB").save(input_path, "JPEG", quality=85)

            extra_files: dict[str, Path] = {}
            for cls in KNOWN_CLASSES:
                p = tmp / f"thumb_{cls}.png"
                _render_mea_thumbnail(cls, p)
                extra_files[cls] = p

            logic = MeaSelectLogic(KNOWN_CLASSES, detected_class, confidence)
            serve_blocking(
                logic,
                image_path=input_path,
                title="Sélection de la famille MEA",
                extra_files=extra_files,
            )

        result = logic.result
        if result is None:
            raise RuntimeError("MEA family selection was cancelled.")
        return result
