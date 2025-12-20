from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Any, Dict, List

from rich.console import Console
from rich.table import Table
from tqdm import tqdm

from src.config import get_settings
from src.ingest.pdf_utils import load_pages
from src.ocr.gemini_client import GeminiClient
from src.postprocess.layout import flatten_text, normalize_page
from src.preprocess.image_ops import preprocess_pil
from src.service.doc_semantic import analyze_document, render_debug_html
from src.service.docx_exporter import export_to_docx
from src.service.exporter import export_markdown
from src.store.storage import save_json, save_page_image, save_text

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)

logger = logging.getLogger(__name__)
console = Console()


def run_pipeline(input_path: Path, output_dir: Path) -> int:
    """
    Process a single file through OCR pipeline.
    Returns: total number of API requests made.
    """
    settings = get_settings()
    client = GeminiClient()

    console.print(f"[bold]Processing:[/] {input_path}")
    pages = load_pages(input_path, dpi=settings.pdf_dpi)

    processed_pages = []
    text_lines: List[str] = []

    page_images_dir = output_dir / "pages"
    ocr_json_path = output_dir / "ocr.json"
    text_path = output_dir / "text.txt"

    total_pages = len(pages)
    for page_idx, pil_img in enumerate(tqdm(pages, desc="Pages")):
        processed_pil, png_bytes = preprocess_pil(
            pil_img,
            max_dim=settings.max_image_dim,
            enable_deskew=settings.enable_deskew,
            deskew_threshold=settings.deskew_threshold
        )
        save_page_image(processed_pil, page_images_dir, page_idx)

        requests_before = client.get_request_count()
        ocr_result = client.ocr_page(page_idx, png_bytes, processed_pil.size)
        requests_after = client.get_request_count()
        requests_for_page = requests_after - requests_before
        
        normalized = normalize_page(ocr_result)
        processed_pages.append(normalized)
        
        if requests_for_page > 1:
            logger.info(f"Page {page_idx + 1}/{total_pages} required {requests_for_page} API requests")

    full_text = flatten_text(processed_pages)
    text_lines.extend(full_text.splitlines())

    save_json({"pages": processed_pages, "full_text": full_text}, ocr_json_path)
    save_text(text_lines, text_path)
    export_markdown(processed_pages, output_dir / "layout.md")

    # Export Word document if enabled
    docx_path = output_dir / "document.docx"
    if settings.export_word:
        logger.info("Analyzing semantic structure for Word export...")
        semantic_doc = analyze_document({"pages": processed_pages})

        # Optional debug HTML
        debug_path = output_dir / "debug.html"
        render_debug_html(semantic_doc, debug_path)

        # Adapt semantic blocks to exporter schema
        def adapt_block(blk: Dict[str, Any]) -> Dict[str, Any]:
            btype = blk.get("type", "paragraph")
            if btype == "heading":
                etype = "heading"
            elif btype == "table":
                etype = "table"
            elif btype == "signature":
                etype = "paragraph"  # Signature as paragraph with right alignment
            else:
                etype = "paragraph"
            return {
                "element_type": etype,
                "heading_level": blk.get("heading_level"),
                "alignment": blk.get("alignment", "left"),
                "text": blk.get("text", ""),
                "lines": blk.get("lines", []),
                "bbox": blk.get("bbox", (0, 0, 0, 0)),  # Already in mm
                "bold": blk.get("bold", False),  # Pass bold flag
                "font_size_ratio": blk.get("font_size_ratio", 1.0),
            }

        adapted_pages = []
        for page in semantic_doc.get("pages", []):
            adapted_blocks = [adapt_block(b) for b in page.get("blocks", [])]
            adapted_pages.append(
                {
                    "page": page.get("page", 0),
                    "width": page.get("width", 1.0),
                    "height": page.get("height", 1.0),
                    "blocks": adapted_blocks,
                }
            )

        logger.info("Exporting Word document...")
        style_map = {
            "body": settings.word_font_size_base,
            "list": settings.word_font_size_base,
            "heading1": settings.word_font_size_heading1,
            "heading2": settings.word_font_size_heading2,
            "heading3": settings.word_font_size_heading3,
            "org": settings.word_font_size_org,
        }
        export_to_docx(
            adapted_pages,
            docx_path,
            base_font_size=settings.word_font_size_base,
            detect_columns=settings.word_detect_columns,
            merge_pages=settings.word_merge_pages,
            font_name_body=settings.word_font_body,
            font_name_heading=settings.word_font_heading,
            font_color=settings.word_font_color,
            style_map=style_map,
        )
        console.print(f"[green]Word document exported:[/] {docx_path}")

    total_requests = client.get_request_count()
    total_pages = len(processed_pages)
    avg_requests_per_page = total_requests / total_pages if total_pages > 0 else 0

    console.print(f"\n[bold green]✓ Processing completed[/]")
    console.print(f"[cyan]Total pages:[/] {total_pages}")
    console.print(f"[cyan]Total API requests:[/] {total_requests}")
    console.print(f"[cyan]Average requests per page:[/] {avg_requests_per_page:.2f}")
    
    if total_requests > total_pages:
        console.print(
            f"[yellow]Note:[/] {total_requests - total_pages} retry request(s) were needed "
            f"({((total_requests - total_pages) / total_requests * 100):.1f}% of total)"
        )

    table = Table(title="Outputs")
    table.add_column("Type")
    table.add_column("Path")
    table.add_row("OCR JSON", str(ocr_json_path))
    table.add_row("Text", str(text_path))
    table.add_row("Markdown", str(output_dir / "layout.md"))
    if settings.export_word:
        table.add_row("Word Document", str(docx_path))
    table.add_row("Page Images", str(page_images_dir))
    console.print(table)
    
    return total_requests


def run_batch(input_path: Path, output_root: Path) -> None:
    settings = get_settings()
    allowed = set(settings.allowed_exts)
    files: List[Path] = []
    if input_path.is_file():
        files = [input_path]
    else:
        files = [
            p
            for p in sorted(input_path.rglob("*"))
            if p.is_file() and p.suffix.lower() in allowed
        ]
    if not files:
        raise FileNotFoundError(f"No PDF/image found in {input_path}")

    total_files = len(files)
    total_all_requests = 0
    total_all_pages = 0
    
    for file_idx, file in enumerate(files, 1):
        console.print(f"\n[bold cyan]Processing file {file_idx}/{total_files}:[/] {file.name}")
        out_dir = (output_root / file.stem).resolve()
        out_dir.mkdir(parents=True, exist_ok=True)
        
        requests = run_pipeline(file, out_dir)
        total_all_requests += requests
        
        # Count pages from the file
        pages = load_pages(file, dpi=settings.pdf_dpi)
        total_all_pages += len(pages)
    
    if total_files > 1:
        avg_requests_per_page = total_all_requests / total_all_pages if total_all_pages > 0 else 0
        console.print(f"\n[bold green]Batch processing completed[/]")
        console.print(f"[cyan]Total files:[/] {total_files}")
        console.print(f"[cyan]Total pages:[/] {total_all_pages}")
        console.print(f"[cyan]Total API requests:[/] {total_all_requests}")
        console.print(f"[cyan]Average requests per page:[/] {avg_requests_per_page:.2f}")
        
        if total_all_requests > total_all_pages:
            console.print(
                f"[yellow]Note:[/] {total_all_requests - total_all_pages} retry request(s) were needed "
                f"({((total_all_requests - total_all_pages) / total_all_requests * 100):.1f}% of total)"
            )


def main() -> None:
    parser = argparse.ArgumentParser(description="Batch OCR with Gemini Vision.")
    parser.add_argument("input", type=str, help="Path to PDF or image")
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output directory (default: settings.output_dir/<file_stem>)",
    )
    parser.add_argument(
        "--batch",
        action="store_true",
        help="If set, treat input as directory and process all supported files",
    )
    args = parser.parse_args()

    settings = get_settings()
    input_path = Path(args.input).expanduser().resolve()
    if not input_path.exists():
        raise FileNotFoundError(input_path)

    out_root = Path(args.output) if args.output else Path(settings.output_dir)
    if args.batch or input_path.is_dir():
        out_root.mkdir(parents=True, exist_ok=True)
        run_batch(input_path, out_root)
    else:
        out_dir = (out_root / input_path.stem).resolve()
        out_dir.mkdir(parents=True, exist_ok=True)
        run_pipeline(input_path, out_dir)


if __name__ == "__main__":
    main()

