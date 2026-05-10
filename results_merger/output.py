"""Output helpers: JSON encoder, zip provider, image-upload bootstrap."""

from __future__ import annotations

import json
import zipfile
from pathlib import Path

import numpy as np

from webgui import serve_blocking
from webgui.upload_logic import UploadLogic


class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        return super().default(obj)


class OutputZipProvider:
    """Build a zip of the whole `output/` directory on demand."""

    filename = "cellvision_results.zip"

    def __init__(self, source: Path):
        self._source = source

    def build_zip(self, dest: Path) -> Path:
        out = dest / "cellvision_results.zip"
        if out.exists():
            out.unlink()
        with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED) as zf:
            for p in self._source.rglob("*"):
                if p.is_file():
                    zf.write(p, p.relative_to(self._source))
        return out


def resolve_image(image_path: str | None, auto: bool) -> str:
    if image_path is not None:
        return image_path
    if auto:
        raise SystemExit("--auto requires an image path on the CLI")
    upload = UploadLogic()
    serve_blocking(upload, image_path=None, title="Importer l'image du microscope")
    return str(upload.result)
