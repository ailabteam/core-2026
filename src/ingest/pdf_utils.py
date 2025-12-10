from __future__ import annotations

from pathlib import Path
from typing import List

from pdf2image import convert_from_path
from PIL import Image


def load_pages(input_path: Path, dpi: int = 300) -> List[Image.Image]:
    """
    Load a PDF or image file and return list of PIL Images (pages).
    """
    suffix = input_path.suffix.lower()
    if suffix == ".pdf":
        return convert_from_path(str(input_path), dpi=dpi)
    if suffix in {".png", ".jpg", ".jpeg", ".tif", ".tiff"}:
        return [Image.open(input_path)]
    raise ValueError(f"Unsupported file extension: {suffix}")

