from __future__ import annotations

import re
from typing import Any, Dict, List, Tuple

# Block types
BLOCK_TYPE_TEXT = "text"
BLOCK_TYPE_TABLE = "table"

# Detected element types
ELEMENT_HEADING = "heading"
ELEMENT_PARAGRAPH = "paragraph"
ELEMENT_LIST = "list"
ELEMENT_TABLE = "table"


def get_block_text(block: Dict[str, Any]) -> str:
    """Extract text from a block."""
    if block.get("type") == BLOCK_TYPE_TABLE:
        return ""
    lines = block.get("lines", [])
    return " ".join(line.get("text", "") for line in lines if line.get("text"))


def get_block_bbox(block: Dict[str, Any]) -> Tuple[float, float, float, float]:
    """Get block bounding box as (x0, y0, x1, y1)."""
    bbox = block.get("bbox", [])
    if len(bbox) >= 4:
        return tuple(bbox[:4])
    return (0.0, 0.0, 0.0, 0.0)


def get_block_height(block: Dict[str, Any]) -> float:
    """Get block height from bbox."""
    x0, y0, x1, y1 = get_block_bbox(block)
    return abs(y1 - y0)


def get_block_width(block: Dict[str, Any]) -> float:
    """Get block width from bbox."""
    x0, y0, x1, y1 = get_block_bbox(block)
    return abs(x1 - x0)


def detect_heading(
    block: Dict[str, Any],
    previous_block: Dict[str, Any] | None,
    page_top: float = 0.0,
    avg_block_height: float = 1.0,
) -> Tuple[bool, int]:
    """
    Detect if block is a heading.
    Returns: (is_heading, heading_level 1-6)
    """
    if block.get("type") == BLOCK_TYPE_TABLE:
        return False, 0

    text = get_block_text(block).strip()
    if not text:
        return False, 0

    bbox = get_block_bbox(block)
    y0 = bbox[1]
    height = get_block_height(block)
    width = get_block_width(block)

    # Heuristics for heading detection
    is_heading = False
    level = 1

    # 1. Position: near top of page
    if y0 < 0.15:  # Within 15% from top
        is_heading = True
        level = 1 if y0 < 0.05 else 2

    # 2. Font size: larger than average (relative height)
    if height > avg_block_height * 1.5:
        is_heading = True
        if height > avg_block_height * 2.5:
            level = 1
        elif height > avg_block_height * 2.0:
            level = 2
        else:
            level = 3

    # 3. Text characteristics: short, uppercase, or bold-like
    text_upper_ratio = sum(1 for c in text if c.isupper()) / len(text) if text else 0
    if len(text) < 100 and (text_upper_ratio > 0.7 or text.isupper()):
        is_heading = True
        if level > 2:
            level = 2

    # 4. Spacing: large gap before this block
    if previous_block:
        prev_bbox = get_block_bbox(previous_block)
        gap = y0 - prev_bbox[3]  # Current y0 - previous y1
        if gap > avg_block_height * 0.5:
            is_heading = True
            if level > 3:
                level = 3

    # 5. Width: full width or centered (typical for headings)
    page_width = 1.0  # Normalized
    if width > page_width * 0.8:  # Wide block
        is_heading = True

    return is_heading, min(6, max(1, level))


def detect_list_item(text: str) -> Tuple[bool, str, int]:
    """
    Detect if text is a list item.
    Returns: (is_list, list_type, indent_level)
    list_type: 'bullet', 'numbered', 'lettered'
    """
    text = text.strip()
    if not text:
        return False, "", 0

    # Bullet patterns
    bullet_patterns = [
        r"^[•·▪▫◦‣⁃]\s+",  # Bullet characters
        r"^[-–—]\s+",  # Dashes
        r"^\*\s+",  # Asterisk
    ]

    # Numbered patterns
    numbered_patterns = [
        r"^(\d+)[.)]\s+",  # 1. or 1)
        r"^\((\d+)\)\s+",  # (1)
    ]

    # Lettered patterns
    lettered_patterns = [
        r"^([a-z])[.)]\s+",  # a. or a)
        r"^\(([a-z])\)\s+",  # (a)
        r"^([A-Z])[.)]\s+",  # A. or A)
    ]

    # Check bullet
    for pattern in bullet_patterns:
        if re.match(pattern, text):
            indent = len(re.match(pattern, text).group(0))
            return True, "bullet", indent

    # Check numbered
    for pattern in numbered_patterns:
        match = re.match(pattern, text)
        if match:
            indent = len(match.group(0))
            return True, "numbered", indent

    # Check lettered
    for pattern in lettered_patterns:
        match = re.match(pattern, text)
        if match:
            indent = len(match.group(0))
            return True, "lettered", indent

    return False, "", 0


def detect_alignment(block: Dict[str, Any], page_width: float = 1.0) -> str:
    """
    Detect text alignment from bbox position.
    Returns: 'left', 'center', 'right', 'justify'
    """
    bbox = get_block_bbox(block)
    x0, y0, x1, y1 = bbox
    block_width = x1 - x0
    block_center_x = (x0 + x1) / 2
    page_center_x = page_width / 2

    # Check if text spans full width (justify)
    if block_width > page_width * 0.9:
        # Check if lines are justified (would need line-level analysis)
        # For now, assume left if starts from left
        if x0 < 0.1:
            return "justify"
        return "left"

    # Check alignment based on position
    if x0 < 0.1:  # Starts from left
        return "left"
    elif abs(block_center_x - page_center_x) < 0.1:  # Centered
        return "center"
    elif x1 > page_width * 0.9:  # Ends at right
        return "right"
    else:
        return "left"  # Default


def detect_columns(
    blocks: List[Dict[str, Any]], page_width: float = 1.0, tolerance: float = 0.05
) -> List[List[Dict[str, Any]]]:
    """
    Detect multi-column layout by grouping blocks horizontally.
    Returns: List of column groups, each group is a list of blocks.
    """
    if not blocks:
        return []

    # Group blocks by similar y-position (same row)
    rows: List[List[Tuple[float, Dict[str, Any]]]] = []
    for block in blocks:
        if block.get("type") == BLOCK_TYPE_TABLE:
            # Tables usually span full width, skip column detection
            continue
        bbox = get_block_bbox(block)
        y_center = (bbox[1] + bbox[3]) / 2

        # Find existing row with similar y
        found = False
        for row in rows:
            if row and abs(row[0][0] - y_center) < tolerance:
                row.append((y_center, block))
                found = True
                break

        if not found:
            rows.append([(y_center, block)])

    # Sort rows by y-position
    rows.sort(key=lambda r: r[0][0] if r else 0)

    # Analyze x-positions to detect columns
    column_groups: List[List[Dict[str, Any]]] = []
    column_boundaries: List[float] = []

    for row in rows:
        if not row:
            continue

        # Get x-positions of blocks in this row
        row_blocks = [b for _, b in row]
        row_blocks.sort(key=lambda b: get_block_bbox(b)[0])  # Sort by x0

        if not column_boundaries:
            # First row: detect column boundaries
            for block in row_blocks:
                bbox = get_block_bbox(block)
                x0, x1 = bbox[0], bbox[2]
                # Check if this block fits in existing column
                found_col = False
                for i, (col_x0, col_x1) in enumerate(
                    zip(column_boundaries[::2], column_boundaries[1::2])
                ):
                    if abs(x0 - col_x0) < tolerance and abs(x1 - col_x1) < tolerance:
                        found_col = True
                        break
                if not found_col:
                    # New column detected
                    if len(column_boundaries) == 0:
                        column_boundaries = [x0, x1]
                    else:
                        column_boundaries.extend([x0, x1])

            # Initialize column groups
            column_groups = [[] for _ in range(len(column_boundaries) // 2)]

        # Assign blocks to columns
        for block in row_blocks:
            bbox = get_block_bbox(block)
            x_center = (bbox[0] + bbox[2]) / 2

            # Find which column this block belongs to
            assigned = False
            for i in range(len(column_boundaries) // 2):
                col_x0 = column_boundaries[i * 2]
                col_x1 = column_boundaries[i * 2 + 1]
                if col_x0 - tolerance <= x_center <= col_x1 + tolerance:
                    column_groups[i].append(block)
                    assigned = True
                    break

            if not assigned:
                # Default to first column
                if column_groups:
                    column_groups[0].append(block)

    # If no columns detected, return all blocks as single column
    if not column_groups or all(not col for col in column_groups):
        return [[b for b in blocks if b.get("type") != BLOCK_TYPE_TABLE]]

    return column_groups


def analyze_page_structure(
    page: Dict[str, Any],
    enable_column_detection: bool = True,
) -> Dict[str, Any]:
    """
    Analyze page structure: detect headings, lists, columns, alignment.
    Returns: Enhanced page dict with detected structure.
    """
    blocks = page.get("blocks", [])
    page_width = page.get("width", 1.0)
    page_height = page.get("height", 1.0)

    # Calculate average block height for heading detection
    text_blocks = [b for b in blocks if b.get("type") == BLOCK_TYPE_TEXT]
    if text_blocks:
        avg_height = sum(get_block_height(b) for b in text_blocks) / len(text_blocks)
    else:
        avg_height = page_height * 0.05  # Default estimate

    # Analyze each block
    analyzed_blocks: List[Dict[str, Any]] = []
    previous_block = None

    for block in blocks:
        block_type = block.get("type", BLOCK_TYPE_TEXT)
        analyzed_block = block.copy()

        if block_type == BLOCK_TYPE_TABLE:
            analyzed_block["element_type"] = ELEMENT_TABLE
        else:
            text = get_block_text(block)

            # Detect heading
            is_heading, heading_level = detect_heading(
                block, previous_block, page_top=0.0, avg_block_height=avg_height
            )

            if is_heading:
                analyzed_block["element_type"] = ELEMENT_HEADING
                analyzed_block["heading_level"] = heading_level
            else:
                # Detect list
                is_list, list_type, indent = detect_list_item(text)
                if is_list:
                    analyzed_block["element_type"] = ELEMENT_LIST
                    analyzed_block["list_type"] = list_type
                    analyzed_block["list_indent"] = indent
                else:
                    analyzed_block["element_type"] = ELEMENT_PARAGRAPH

            # Detect alignment
            analyzed_block["alignment"] = detect_alignment(block, page_width)

        analyzed_blocks.append(analyzed_block)
        previous_block = block

    result = page.copy()
    result["blocks"] = analyzed_blocks

    # Detect columns if enabled
    if enable_column_detection and len(text_blocks) > 1:
        column_groups = detect_columns(analyzed_blocks, page_width)
        if len(column_groups) > 1:
            result["columns"] = column_groups
            result["has_columns"] = True
        else:
            result["has_columns"] = False
    else:
        result["has_columns"] = False

    return result

