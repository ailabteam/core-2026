from __future__ import annotations

from typing import Any, Dict, List


def normalize_bbox(bbox: List[float], width: int, height: int) -> List[float]:
    if not bbox or width == 0 or height == 0:
        return bbox
    x0, y0, x1, y1 = bbox
    return [
        max(0.0, min(1.0, x0 / width)),
        max(0.0, min(1.0, y0 / height)),
        max(0.0, min(1.0, x1 / width)),
        max(0.0, min(1.0, y1 / height)),
    ]


def normalize_page(page: Dict[str, Any]) -> Dict[str, Any]:
    width = page.get("width", 1)
    height = page.get("height", 1)
    for block in page.get("blocks", []):
        if "bbox" in block:
            block["bbox_norm"] = normalize_bbox(block["bbox"], width, height)
        for line in block.get("lines", []):
            if "bbox" in line:
                line["bbox_norm"] = normalize_bbox(line["bbox"], width, height)
        if "table" in block and block["table"]:
            for row in block["table"].get("rows", []):
                for cell in row:
                    if "bbox" in cell:
                        cell["bbox_norm"] = normalize_bbox(cell["bbox"], width, height)
    return page


def flatten_text(pages: List[Dict[str, Any]]) -> str:
    parts: List[str] = []
    for page in pages:
        if "full_text" in page and page["full_text"]:
            parts.append(page["full_text"])
            continue
        for block in page.get("blocks", []):
            for line in block.get("lines", []):
                if line.get("text"):
                    parts.append(line["text"])
    return "\n".join(parts)

