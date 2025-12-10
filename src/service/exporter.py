from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List


def blocks_to_markdown(blocks: List[Dict[str, Any]]) -> str:
    lines: List[str] = []
    for block in blocks:
        if block.get("type") == "table" and block.get("table"):
            for row in block["table"].get("rows", []):
                row_text = " | ".join(cell.get("text", "") for cell in row)
                lines.append(row_text)
        else:
            for line in block.get("lines", []):
                if line.get("text"):
                    lines.append(line["text"])
    return "\n".join(lines)


def export_markdown(pages: List[Dict[str, Any]], out_file: Path) -> None:
    chunks: List[str] = []
    for page in pages:
        blocks = page.get("blocks", [])
        chunks.append(f"# Page {page.get('page', 0)}")
        chunks.append(blocks_to_markdown(blocks))
    out_file.parent.mkdir(parents=True, exist_ok=True)
    out_file.write_text("\n\n".join(chunks), encoding="utf-8")

