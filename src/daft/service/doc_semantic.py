from __future__ import annotations

import math
import statistics
from pathlib import Path
from typing import Any, Dict, List, Tuple

#
# Document Semantic Understanding Layer
# Converts raw OCR (Google Document AI or similar) into normalized semantic blocks.
# Optimized for Vietnamese administrative documents with A4 normalization.
#

# A4 dimensions in mm
A4_WIDTH_MM = 210.0
A4_HEIGHT_MM = 297.0
A4_PORTRAIT_RATIO = A4_WIDTH_MM / A4_HEIGHT_MM  # ~0.707
A4_LANDSCAPE_RATIO = A4_HEIGHT_MM / A4_WIDTH_MM  # ~1.414

# DPI to mm conversion (assuming 300 DPI)
DPI_TO_MM = 25.4 / 300.0


# ---------- Geometry helpers ----------


def _bbox_from_poly(poly: Dict[str, Any]) -> Tuple[float, float, float, float]:
    """
    Convert polygon (list of {x,y} or list of tuples) to (x0,y0,x1,y1).
    Assumes coordinates already normalized 0-1; if >1, treated as absolute pixels.
    """
    if not poly:
        return 0.0, 0.0, 0.0, 0.0
    xs, ys = [], []
    for pt in poly:
        x = pt.get("x") if isinstance(pt, dict) else pt[0]
        y = pt.get("y") if isinstance(pt, dict) else pt[1]
        xs.append(float(x))
        ys.append(float(y))
    return min(xs), min(ys), max(xs), max(ys)


def _area(bbox: Tuple[float, float, float, float]) -> float:
    x0, y0, x1, y1 = bbox
    return max(0.0, x1 - x0) * max(0.0, y1 - y0)


# ---------- A4 Normalization & Orientation Detection ----------


def detect_orientation(page_width: float, page_height: float) -> Tuple[str, float, float]:
    """
    Detect page orientation (portrait/landscape) and return A4 dimensions.
    Returns: (orientation, a4_width_mm, a4_height_mm)
    """
    if page_width == 0 or page_height == 0:
        return "portrait", A4_WIDTH_MM, A4_HEIGHT_MM

    aspect_ratio = page_width / page_height

    # Compare with A4 ratios
    dist_portrait = abs(aspect_ratio - A4_PORTRAIT_RATIO)
    dist_landscape = abs(aspect_ratio - A4_LANDSCAPE_RATIO)

    if dist_portrait < dist_landscape:
        return "portrait", A4_WIDTH_MM, A4_HEIGHT_MM
    else:
        return "landscape", A4_HEIGHT_MM, A4_WIDTH_MM


def normalize_to_a4(
    bbox: Tuple[float, float, float, float],
    page_width: float,
    page_height: float,
    orientation: str,
    a4_width_mm: float,
    a4_height_mm: float,
) -> Tuple[float, float, float, float]:
    """
    Normalize bbox from pixel/normalized coordinates to A4 mm coordinates.
    """
    x0, y0, x1, y1 = bbox

    # If coordinates are > 1, assume pixels; convert to normalized first
    if x1 > 1.0 or y1 > 1.0:
        x0_norm = x0 / page_width
        y0_norm = y0 / page_height
        x1_norm = x1 / page_width
        y1_norm = y1 / page_height
    else:
        x0_norm, y0_norm, x1_norm, y1_norm = x0, y0, x1, y1

    # Convert to mm on A4
    x0_mm = x0_norm * a4_width_mm
    y0_mm = y0_norm * a4_height_mm
    x1_mm = x1_norm * a4_width_mm
    y1_mm = y1_norm * a4_height_mm

    return (x0_mm, y0_mm, x1_mm, y1_mm)


# ---------- Extraction (using exact OCR bbox) ----------


def extract_words(page: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    Extract words/lines from OCR JSON, using exact bbox from OCR (not inferred).
    Since Gemini OCR returns line-level bbox, we treat each line as a "word" unit.
    Expected format: {text, bbox: (x0,y0,x1,y1)} in pixels.
    """
    words: List[Dict[str, Any]] = []

    # Preferred: tokens field (Document AI format)
    if "tokens" in page:
        for tok in page["tokens"]:
            text = tok.get("text") or tok.get("content") or ""
            if not text:
                continue
            poly = tok.get("layout", {}).get("boundingPoly") or tok.get("boundingBox")
            bbox = _bbox_from_poly(poly.get("vertices") if isinstance(poly, dict) else poly)
            words.append({"text": text, "bbox": bbox})

    # Gemini OCR format: blocks -> lines -> text with bbox
    if not words and "blocks" in page:
        for block in page.get("blocks", []):
            if block.get("type") != "text":
                continue
            for line in block.get("lines", []):
                line_text = line.get("text", "")
                if not line_text or not line_text.strip():
                    continue
                
                # Get line bbox - this is the exact OCR bbox
                line_bbox = line.get("bbox")
                if isinstance(line_bbox, list) and len(line_bbox) == 4:
                    lbbox = tuple(float(x) for x in line_bbox)
                elif isinstance(line_bbox, dict):
                    # Handle dict format like {"x0": ..., "y0": ..., "x1": ..., "y1": ...}
                    lbbox = (
                        float(line_bbox.get("x0", 0)),
                        float(line_bbox.get("y0", 0)),
                        float(line_bbox.get("x1", 0)),
                        float(line_bbox.get("y1", 0)),
                    )
                else:
                    # Fallback: try to parse from boundingBox or other fields
                    bbox_data = line.get("boundingBox") or line.get("bounding_poly")
                    if bbox_data:
                        lbbox = _bbox_from_poly(bbox_data)
                    else:
                        # Last resort: use block bbox
                        block_bbox = block.get("bbox", [0, 0, 0, 0])
                        if isinstance(block_bbox, list) and len(block_bbox) == 4:
                            lbbox = tuple(float(x) for x in block_bbox)
                        else:
                            continue  # Skip if no bbox available

                # Use line bbox directly - this preserves exact OCR positioning
                words.append({"text": line_text.strip(), "bbox": lbbox})

    return words


# ---------- Statistics (in mm) ----------


def compute_stats_mm(
    words: List[Dict[str, Any]],
    page_width: float,
    page_height: float,
    orientation: str,
    a4_width_mm: float,
    a4_height_mm: float,
) -> Dict[str, float]:
    """
    Compute statistics in mm after A4 normalization.
    """
    heights_mm = []
    gaps_mm = []
    xs_mm = []
    ys_mm = []

    words_sorted = sorted(words, key=lambda w: (w["bbox"][1], w["bbox"][0]))

    for i, w in enumerate(words_sorted):
        x0, y0, x1, y1 = w["bbox"]
        bbox_mm = normalize_to_a4((x0, y0, x1, y1), page_width, page_height, orientation, a4_width_mm, a4_height_mm)
        x0_mm, y0_mm, x1_mm, y1_mm = bbox_mm
        height_mm = max(0.0, y1_mm - y0_mm)
        heights_mm.append(height_mm)
        xs_mm.append(x0_mm)
        ys_mm.append(y0_mm)

        # Compute gaps
        if i > 0:
            prev_w = words_sorted[i - 1]
            px0, py0, px1, py1 = prev_w["bbox"]
            prev_bbox_mm = normalize_to_a4(
                (px0, py0, px1, py1), page_width, page_height, orientation, a4_width_mm, a4_height_mm
            )
            _, prev_y1_mm, _, _ = prev_bbox_mm
            gap_mm = y0_mm - prev_y1_mm
            if gap_mm > 0:
                gaps_mm.append(gap_mm)

    median_h_mm = statistics.median(heights_mm) if heights_mm else 3.0  # Default ~3mm
    avg_gap_mm = statistics.mean(gaps_mm) if gaps_mm else median_h_mm * 1.2
    median_gap_mm = statistics.median(gaps_mm) if gaps_mm else median_h_mm * 1.2

    # Margins (left, right, top, bottom)
    margin_left_mm = min(xs_mm) if xs_mm else 20.0  # Default 20mm left margin
    margin_right_mm = a4_width_mm - max(xs_mm) if xs_mm else 20.0
    margin_top_mm = min(ys_mm) if ys_mm else 20.0
    margin_bottom_mm = a4_height_mm - max(ys_mm) if ys_mm else 20.0

    return {
        "median_word_height_mm": median_h_mm,
        "avg_gap_mm": avg_gap_mm,
        "median_gap_mm": median_gap_mm,
        "margin_left_mm": margin_left_mm,
        "margin_right_mm": margin_right_mm,
        "margin_top_mm": margin_top_mm,
        "margin_bottom_mm": margin_bottom_mm,
    }


# ---------- Grouping (using mm coordinates) ----------


def group_words_to_lines_mm(
    words: List[Dict[str, Any]],
    page_width: float,
    page_height: float,
    orientation: str,
    a4_width_mm: float,
    a4_height_mm: float,
    median_h_mm: float,
) -> List[Dict[str, Any]]:
    """
    Group words into lines using mm-based proximity.
    """
    lines: List[Dict[str, Any]] = []
    if not words:
        return lines

    # Normalize all words to mm first
    words_mm = []
    for w in words:
        bbox_mm = normalize_to_a4(w["bbox"], page_width, page_height, orientation, a4_width_mm, a4_height_mm)
        words_mm.append({"text": w["text"], "bbox": bbox_mm, "bbox_orig": w["bbox"]})

    words_sorted = sorted(words_mm, key=lambda w: (w["bbox"][1], w["bbox"][0]))
    threshold_mm = 0.5 * median_h_mm

    current_line: List[Dict[str, Any]] = []
    current_y_mm = None

    for w in words_sorted:
        x0, y0, x1, y1 = w["bbox"]
        y_center_mm = (y0 + y1) / 2
        if current_y_mm is None:
            current_line = [w]
            current_y_mm = y_center_mm
            continue
        if abs(y_center_mm - current_y_mm) <= threshold_mm:
            current_line.append(w)
            current_y_mm = (current_y_mm * (len(current_line) - 1) + y_center_mm) / len(current_line)
        else:
            lines.append(_finalize_line_mm(current_line))
            current_line = [w]
            current_y_mm = y_center_mm
    if current_line:
        lines.append(_finalize_line_mm(current_line))
    return lines


def _finalize_line_mm(words: List[Dict[str, Any]]) -> Dict[str, Any]:
    words_sorted = sorted(words, key=lambda w: w["bbox"][0])
    text = " ".join(w["text"] for w in words_sorted)
    x0 = min(w["bbox"][0] for w in words_sorted)
    y0 = min(w["bbox"][1] for w in words_sorted)
    x1 = max(w["bbox"][2] for w in words_sorted)
    y1 = max(w["bbox"][3] for w in words_sorted)
    return {"text": text, "words": words_sorted, "bbox": (x0, y0, x1, y1), "bbox_orig": words_sorted[0]["bbox_orig"]}


def group_lines_to_blocks_mm(
    lines: List[Dict[str, Any]], avg_gap_mm: float, a4_width_mm: float
) -> List[Dict[str, Any]]:
    """
    Group lines into blocks using mm-based gaps.
    """
    blocks: List[Dict[str, Any]] = []
    if not lines:
        return blocks
    lines_sorted = sorted(lines, key=lambda l: l["bbox"][1])
    current_block: List[Dict[str, Any]] = [lines_sorted[0]]

    for prev, curr in zip(lines_sorted, lines_sorted[1:]):
        prev_y1 = prev["bbox"][3]
        curr_y0 = curr["bbox"][1]
        gap_mm = curr_y0 - prev_y1

        # Paragraph break if gap > 1.3 * avg_gap OR indent change > 5% page width
        prev_x0 = prev["bbox"][0]
        curr_x0 = curr["bbox"][0]
        indent_change_mm = abs(curr_x0 - prev_x0)

        if gap_mm > 1.3 * avg_gap_mm or indent_change_mm > 0.05 * a4_width_mm:
            blocks.append(_finalize_block_mm(current_block))
            current_block = [curr]
        else:
            current_block.append(curr)
    if current_block:
        blocks.append(_finalize_block_mm(current_block))
    return blocks


def _finalize_block_mm(lines: List[Dict[str, Any]]) -> Dict[str, Any]:
    text = "\n".join(l["text"] for l in lines)
    x0 = min(l["bbox"][0] for l in lines)
    y0 = min(l["bbox"][1] for l in lines)
    x1 = max(l["bbox"][2] for l in lines)
    y1 = max(l["bbox"][3] for l in lines)
    return {"text": text, "lines": lines, "bbox": (x0, y0, x1, y1)}


# ---------- Features / inference (Vietnamese administrative rules) ----------


HEADING_KEYWORDS = {
    "CỘNG HÒA",
    "CỘNG HOÀ",
    "ĐƠN",
    "QUYẾT ĐỊNH",
    "THÔNG BÁO",
    "CÔNG VĂN",
    "CHỈ THỊ",
    "NGHỊ QUYẾT",
}

SIGNATURE_KEYWORDS = {"NGƯỜI LÀM ĐƠN", "KÝ TÊN", "THỦ TRƯỞNG", "KÝ", "KÝ TÊN", "CHỦ TỊCH", "GIÁM ĐỐC"}

# Vietnamese administrative document keywords
QUOC_HIEU_KEYWORDS = {"CỘNG HÒA", "CỘNG HOÀ", "XÃ HỘI", "CHỦ NGHĨA", "VIỆT NAM"}
TIÊU_NGỮ_KEYWORDS = {"ĐỘC LẬP", "TỰ DO", "HẠNH PHÚC"}
CO_QUAN_KEYWORDS = {"UBND", "SỞ", "BỘ", "PHÒNG", "BAN", "TRƯỜNG", "TRUNG TÂM"}


def alignment_from_bbox_mm(bbox: Tuple[float, float, float, float], a4_width_mm: float, margin_left_mm: float) -> str:
    """
    Determine alignment using mm coordinates and margins.
    """
    x0, _, x1, _ = bbox
    center_x = (x0 + x1) / 2
    width = x1 - x0
    page_center = a4_width_mm / 2

    # Center: within 10mm of page center
    if abs(center_x - page_center) < 10.0:
        return "center"

    # Justify: spans most of page width (80%+)
    if width > 0.8 * a4_width_mm:
        return "justify"

    # Left: starts near left margin
    if abs(x0 - margin_left_mm) < 5.0:
        return "left"

    # Right: ends near right margin
    margin_right_mm = a4_width_mm - margin_left_mm
    if abs(x1 - margin_right_mm) < 5.0:
        return "right"

    return "left"


def is_all_caps(text: str) -> bool:
    letters = [c for c in text if c.isalpha()]
    if not letters:
        return False
    return all(c.isupper() for c in letters)


def heading_score_vn(
    block: Dict[str, Any],
    page_height_mm: float,
    median_h_mm: float,
    a4_width_mm: float,
    margin_left_mm: float,
    margin_top_mm: float,
) -> Tuple[bool, int]:
    """
    Heading detection optimized for Vietnamese administrative documents.
    Returns: (is_heading, heading_level 1-6)
    """
    text = block.get("text", "").strip()
    if not text:
        return False, 0

    x0, y0, x1, y1 = block["bbox"]
    block_height_mm = y1 - y0
    font_size_ratio = block_height_mm / median_h_mm if median_h_mm > 0 else 1.0

    conds = 0

    # Condition 1: Font size ratio >= 1.25
    if font_size_ratio >= 1.25:
        conds += 1

    # Condition 2: ALL CAPS
    if is_all_caps(text):
        conds += 1

    # Condition 3: Center alignment
    alignment = alignment_from_bbox_mm(block["bbox"], a4_width_mm, margin_left_mm)
    if alignment == "center":
        conds += 1

    # Condition 4: Top 30% of page (in mm)
    if y0 < margin_top_mm + 0.3 * (page_height_mm - margin_top_mm):
        conds += 1

    # Condition 5: Contains heading keywords
    text_upper = text.upper()
    if any(k in text_upper for k in HEADING_KEYWORDS):
        conds += 1

    # Need at least 2 conditions
    is_heading = conds >= 2

    # Determine heading level
    if font_size_ratio >= 1.6:
        level = 1
    elif font_size_ratio >= 1.35:
        level = 2
    elif font_size_ratio >= 1.25:
        level = 3
    else:
        level = 3  # Default

    return is_heading, level


def bold_inference_vn(
    block: Dict[str, Any],
    page_height_mm: float,
    median_h_mm: float,
    a4_width_mm: float,
    margin_left_mm: float,
    margin_top_mm: float,
) -> bool:
    """
    Bold inference according to Vietnamese administrative document rules:
    - Quốc hiệu/tiêu ngữ: ALL CAPS, top 30%, center → bold
    - Tên cơ quan: ALL CAPS, top 30%, left → bold
    - Heading: font_size_ratio >= 1.25 → bold
    - Nội dung chính: normal
    """
    text = block.get("text", "").strip()
    if not text:
        return False

    x0, y0, x1, y1 = block["bbox"]
    block_height_mm = y1 - y0
    font_size_ratio = block_height_mm / median_h_mm if median_h_mm > 0 else 1.0
    text_upper = text.upper()

    # Rule 1: Quốc hiệu/tiêu ngữ (ALL CAPS, top 30%, center)
    is_top_30 = y0 < margin_top_mm + 0.3 * (page_height_mm - margin_top_mm)
    alignment = alignment_from_bbox_mm(block["bbox"], a4_width_mm, margin_left_mm)
    is_quoc_hieu = any(k in text_upper for k in QUOC_HIEU_KEYWORDS) or any(k in text_upper for k in TIÊU_NGỮ_KEYWORDS)
    if is_all_caps(text) and is_top_30 and alignment == "center" and is_quoc_hieu:
        return True

    # Rule 2: Tên cơ quan (ALL CAPS, top 30%, left)
    is_co_quan = any(k in text_upper for k in CO_QUAN_KEYWORDS)
    if is_all_caps(text) and is_top_30 and alignment == "left" and is_co_quan:
        return True

    # Rule 3: Heading (font_size_ratio >= 1.25)
    if font_size_ratio >= 1.25:
        return True

    # Rule 4: Short ALL CAPS text (< 12 words) in top half
    if is_all_caps(text) and len(text.split()) < 12 and y0 < page_height_mm / 2:
        return True

    return False


def signature_detection_mm(
    block: Dict[str, Any], page_height_mm: float, a4_width_mm: float, margin_left_mm: float
) -> bool:
    """
    Detect signature block: right-aligned, bottom 25%, contains keywords.
    """
    text = block.get("text", "").upper()
    x0, y0, x1, y1 = block["bbox"]
    alignment = alignment_from_bbox_mm(block["bbox"], a4_width_mm, margin_left_mm)

    is_bottom_25 = y0 > 0.75 * page_height_mm
    is_right = alignment == "right"
    has_keyword = any(k in text for k in SIGNATURE_KEYWORDS)

    return is_bottom_25 and (is_right or has_keyword)


# ---------- Main processing ----------


def analyze_page(page: Dict[str, Any]) -> Dict[str, Any]:
    """
    Analyze page with A4 normalization and Vietnamese administrative rules.
    """
    page_width = page.get("width", 1.0)
    page_height = page.get("height", 1.0)

    # Detect orientation and A4 dimensions
    orientation, a4_width_mm, a4_height_mm = detect_orientation(page_width, page_height)

    words = extract_words(page)
    if not words:
        return {
            "page": page.get("page", 0),
            "blocks": [],
            "width": page_width,
            "height": page_height,
            "orientation": orientation,
            "a4_width_mm": a4_width_mm,
            "a4_height_mm": a4_height_mm,
        }

    # Compute statistics in mm
    stats_mm = compute_stats_mm(words, page_width, page_height, orientation, a4_width_mm, a4_height_mm)
    median_h_mm = stats_mm["median_word_height_mm"]
    avg_gap_mm = stats_mm["avg_gap_mm"]
    margin_left_mm = stats_mm["margin_left_mm"]
    margin_top_mm = stats_mm["margin_top_mm"]

    # Group words -> lines -> blocks (using mm)
    lines = group_words_to_lines_mm(words, page_width, page_height, orientation, a4_width_mm, a4_height_mm, median_h_mm)
    blocks = group_lines_to_blocks_mm(lines, avg_gap_mm, a4_width_mm)

    semantic_blocks = []
    for blk in blocks:
        text = blk.get("text", "").strip()
        if not text:
            continue

        x0, y0, x1, y1 = blk["bbox"]
        block_height_mm = y1 - y0
        font_size_ratio = block_height_mm / median_h_mm if median_h_mm > 0 else 1.0

        # Classify block
        is_heading, heading_level = heading_score_vn(
            blk, a4_height_mm, median_h_mm, a4_width_mm, margin_left_mm, margin_top_mm
        )
        is_bold = bold_inference_vn(blk, a4_height_mm, median_h_mm, a4_width_mm, margin_left_mm, margin_top_mm)
        alignment = alignment_from_bbox_mm(blk["bbox"], a4_width_mm, margin_left_mm)
        is_signature = signature_detection_mm(blk, a4_height_mm, a4_width_mm, margin_left_mm)

        block_type = "heading" if is_heading else "paragraph"
        if is_signature:
            block_type = "signature"

        semantic_blocks.append(
            {
                "type": block_type,
                "heading_level": heading_level if is_heading else None,
                "bold": is_bold,
                "text": text,
                "lines": blk["lines"],
                "bbox": blk["bbox"],  # Already in mm
                "bbox_orig": blk["lines"][0]["bbox_orig"] if blk["lines"] else (0, 0, 0, 0),  # Original pixel coords
                "font_size_ratio": font_size_ratio,
                "alignment": alignment,
            }
        )

    return {
        "page": page.get("page", 0),
        "width": page_width,
        "height": page_height,
        "orientation": orientation,
        "a4_width_mm": a4_width_mm,
        "a4_height_mm": a4_height_mm,
        "blocks": semantic_blocks,
        "stats": {
            "median_word_height_mm": median_h_mm,
            "avg_gap_mm": avg_gap_mm,
            "margin_left_mm": margin_left_mm,
            "margin_top_mm": margin_top_mm,
        },
    }


def analyze_document(doc: Dict[str, Any]) -> Dict[str, Any]:
    pages = doc.get("pages", [])
    analyzed = [analyze_page(p) for p in pages]
    return {"pages": analyzed}


# ---------- Debug HTML ----------


def render_debug_html(semantic_doc: Dict[str, Any], out_path: Path) -> None:
    """
    Render HTML with block overlays showing A4-normalized positions and inferred types.
    """
    colors = {
        "heading": "#ff7f0e",
        "paragraph": "#1f77b4",
        "signature": "#2ca02c",
        "table": "#9467bd",
        "list": "#8c564b",
    }

    html_parts = [
        "<html><head><style>",
        "body { font-family: Arial, sans-serif; margin: 20px; }",
        ".page { position: relative; width:800px; border:2px solid #333; margin:20px 0; background:#f9f9f9; }",
        ".block { position:absolute; border:2px solid rgba(0,0,0,0.5); padding:2px; font-size:9px; background:rgba(255,255,255,0.8); }",
        ".stats { margin:10px 0; padding:10px; background:#eee; }",
        "</style></head><body>",
        "<h1>Document Semantic Analysis Debug</h1>",
    ]

    for page in semantic_doc.get("pages", []):
        pw, ph = page.get("width", 1), page.get("height", 1)
        orientation = page.get("orientation", "portrait")
        a4_w_mm = page.get("a4_width_mm", A4_WIDTH_MM)
        a4_h_mm = page.get("a4_height_mm", A4_HEIGHT_MM)
        stats = page.get("stats", {})

        html_parts.append(f'<div class="stats">')
        html_parts.append(f"Page {page.get('page', 0)}: {orientation.upper()}")
        html_parts.append(f" | A4: {a4_w_mm:.1f}mm × {a4_h_mm:.1f}mm")
        html_parts.append(f" | Median height: {stats.get('median_word_height_mm', 0):.2f}mm")
        html_parts.append(f" | Avg gap: {stats.get('avg_gap_mm', 0):.2f}mm")
        html_parts.append(f" | Margins: L={stats.get('margin_left_mm', 0):.1f}mm, T={stats.get('margin_top_mm', 0):.1f}mm")
        html_parts.append("</div>")

        # Scale A4 to 800px width
        scale = 800.0 / a4_w_mm if a4_w_mm > 0 else 1.0
        scaled_height = a4_h_mm * scale

        html_parts.append(f'<div class="page" style="height:{scaled_height}px;">')
        for blk in page.get("blocks", []):
            x0, y0, x1, y1 = blk["bbox"]  # Already in mm
            left = x0 * scale
            top = y0 * scale
            width = (x1 - x0) * scale
            height = (y1 - y0) * scale
            color = colors.get(blk["type"], "#333")
            label = (
                f"{blk['type']} | fsr={blk['font_size_ratio']:.2f} | "
                f"align={blk['alignment']} | bold={blk.get('bold', False)}"
            )
            html_parts.append(
                f'<div class="block" style="left:{left}px;top:{top}px;width:{width}px;height:{height}px;'
                f'border-color:{color};">{label}</div>'
            )
        html_parts.append("</div>")
    html_parts.append("</body></html>")
    out_path.write_text("\n".join(html_parts), encoding="utf-8")
