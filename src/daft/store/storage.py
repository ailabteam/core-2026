from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List

from PIL import Image


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def save_page_image(img: Image.Image, out_dir: Path, page_index: int) -> Path:
    ensure_dir(out_dir)
    out_path = out_dir / f"page_{page_index:04d}.png"
    img.save(out_path, format="PNG")
    return out_path


def save_json(data: Dict[str, Any], out_file: Path) -> None:
    ensure_dir(out_file.parent)
    with out_file.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def save_text(lines: List[str], out_file: Path) -> None:
    ensure_dir(out_file.parent)
    with out_file.open("w", encoding="utf-8") as f:
        f.write("\n".join(lines))

