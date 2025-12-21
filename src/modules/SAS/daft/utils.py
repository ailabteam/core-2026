"""
Utility functions cho seal và signature detection
"""
import re
import json
from typing import Optional, List
from PIL import Image, ImageDraw
from .models import BoundingBox, DetectionResult


def resize_image_if_needed(
    image: Image.Image,
    max_image_size: Optional[int] = None,
    low_memory_mode: bool = False,
    num_gpus: int = 1
) -> Image.Image:
    """
    Resize ảnh nếu quá lớn để tiết kiệm memory
    
    Args:
        image: PIL Image
        max_image_size: Kích thước tối đa (resize nếu lớn hơn)
        low_memory_mode: Nếu True, tự động resize khi ảnh quá lớn
        num_gpus: Số lượng GPU (để điều chỉnh threshold)
            
    Returns:
        PIL Image (có thể đã được resize)
    """
    width, height = image.size
    
    # Nếu có max_image_size, resize theo chiều dài nhất
    if max_image_size is not None:
        max_dim = max(width, height)
        if max_dim > max_image_size:
            scale = max_image_size / max_dim
            new_width = int(width * scale)
            new_height = int(height * scale)
            image = image.resize((new_width, new_height), Image.Resampling.LANCZOS)
            print(f"📐 Resized image from {width}x{height} to {new_width}x{new_height}")
            return image
    
    # Low memory mode: tự động resize nếu ảnh quá lớn
    if low_memory_mode:
        max_dim = max(width, height)
        threshold = 1536 if num_gpus > 1 else 1024
        target_size = 1536 if num_gpus > 1 else 1024
        
        if max_dim > threshold:
            scale = target_size / max_dim
            new_width = int(width * scale)
            new_height = int(height * scale)
            image = image.resize((new_width, new_height), Image.Resampling.LANCZOS)
            print(f"📐 [Low memory mode] Resized image from {width}x{height} to {new_width}x{new_height}")
    
    return image


def create_detection_prompt(language: str = "vietnamese") -> str:
    """
    Tạo prompt để nhận diện con dấu và chữ ký
    
    Args:
        language: Ngôn ngữ của prompt (vietnamese/english)
    """
    if language == "vietnamese":
        prompt = """Bạn là chuyên gia phân tích tài liệu tiếng Việt. Nhiệm vụ của bạn là nhận diện và định vị CON DẤU (seal/stamp) và CHỮ KÝ (signature) trong hình ảnh tài liệu.

CON DẤU thường có đặc điểm:
- Hình tròn hoặc hình chữ nhật
- Màu đỏ hoặc màu khác nổi bật
- Có văn bản bên trong (tên cơ quan, tổ chức)
- Thường ở góc trên bên phải hoặc gần chữ ký

CHỮ KÝ thường có đặc điểm:
- Nét viết tay, đường nét cong
- Màu đen hoặc xanh
- Thường ở góc dưới bên phải
- Có thể kèm theo tên người ký

Hãy phân tích hình ảnh và trả về CHỈ JSON, không có markdown, không có giải thích.

QUAN TRỌNG - ĐỌC KỸ:
1. bbox phải là mảng 4 số nguyên thực tế [x1, y1, x2, y2] với:
   - x1, y1: tọa độ góc trên bên trái (pixel) - PHẢI LÀ SỐ NGUYÊN THỰC TẾ
   - x2, y2: tọa độ góc dưới bên phải (pixel) - PHẢI LÀ SỐ NGUYÊN THỰC TẾ
   - KHÔNG được dùng placeholder như "number x0", "x0 of seal", "double", "str(...)"
   - KHÔNG được copy nguyên văn từ schema, phải đo tọa độ thực tế từ ảnh

2. confidence phải là số thực từ 0.0 đến 1.0, KHÔNG phải chữ "double"

3. description phải là chuỗi mô tả thực tế, KHÔNG phải "str (...)" hay "Name of ..."

4. width và height phải là kích thước thực tế của ảnh (pixel)

Schema JSON bắt buộc (CHỈ LÀ VÍ DỤ - BẠN PHẢI ĐIỀN SỐ THỰC TẾ):
{
  "page": 0,
  "width": 1200,
  "height": 1600,
  "seals": [
    {
      "type": "seal",
      "bbox": [100, 50, 300, 200],
      "confidence": 0.95,
      "description": "Con dấu UBND"
    }
  ],
  "signatures": [
    {
      "type": "signature",
      "bbox": [800, 1400, 1100, 1550],
      "confidence": 0.90,
      "description": "Chữ ký người đại diện"
    }
  ]
}

Ví dụ: Nếu con dấu ở vị trí từ pixel (100, 50) đến (300, 200), thì bbox = [100, 50, 300, 200] - ĐÂY LÀ SỐ THỰC TẾ, KHÔNG PHẢI PLACEHOLDER.

Trả về CHỈ JSON object với các số thực tế từ ảnh, không có gì khác."""
    else:
        prompt = """You are an expert in Vietnamese document analysis. Your task is to identify and locate SEALS (seal/stamp) and SIGNATURES in document images.

SEALS typically have these characteristics:
- Circular or rectangular shape
- Red or other prominent colors
- Text inside (organization name, agency name)
- Usually in top-right corner or near signature

SIGNATURES typically have these characteristics:
- Handwritten strokes, curved lines
- Black or blue color
- Usually in bottom-right corner
- May be accompanied by signer's name

Analyze the image and return ONLY JSON, no markdown, no explanations.

IMPORTANT - READ CAREFULLY:
1. bbox must be an array of 4 actual integers [x1, y1, x2, y2] where:
   - x1, y1: top-left corner coordinates (pixels) - MUST BE ACTUAL NUMBERS
   - x2, y2: bottom-right corner coordinates (pixels) - MUST BE ACTUAL NUMBERS
   - DO NOT use placeholders like "number x0", "x0 of seal", "double", "str(...)"
   - DO NOT copy schema literally, you must measure actual coordinates from the image

2. confidence must be a real number from 0.0 to 1.0, NOT the word "double"

3. description must be an actual description string, NOT "str (...)" or "Name of ..."

4. width and height must be the actual image dimensions (pixels)

Required JSON schema (THIS IS AN EXAMPLE - YOU MUST FILL IN ACTUAL NUMBERS):
{
  "page": 0,
  "width": 1200,
  "height": 1600,
  "seals": [
    {
      "type": "seal",
      "bbox": [100, 50, 300, 200],
      "confidence": 0.95,
      "description": "Seal description"
    }
  ],
  "signatures": [
    {
      "type": "signature",
      "bbox": [800, 1400, 1100, 1550],
      "confidence": 0.90,
      "description": "Signature description"
    }
  ]
}

Example: If a seal is located from pixel (100, 50) to (300, 200), then bbox = [100, 50, 300, 200] - THESE ARE ACTUAL NUMBERS, NOT PLACEHOLDERS.

Return ONLY the JSON object with actual numbers from the image, nothing else."""
    
    return prompt


def clean_response(response: str) -> str:
    """
    Clean response từ model để extract JSON:
    1. Loại bỏ markdown code blocks (```json ... ```)
    2. Loại bỏ text giải thích trước/sau JSON
    3. Extract chỉ phần JSON
    
    Args:
        response: Raw response text từ model
        
    Returns:
        Cleaned JSON string
    """
    cleaned = response.strip()
    
    # Loại bỏ markdown code blocks
    # Pattern: ```json ... ``` hoặc ``` ... ```
    markdown_patterns = [
        r'```json\s*\n?(.*?)\n?```',
        r'```\s*\n?(.*?)\n?```',
    ]
    
    for pattern in markdown_patterns:
        match = re.search(pattern, cleaned, re.DOTALL | re.IGNORECASE)
        if match:
            cleaned = match.group(1).strip()
            break
    
    # Tìm JSON object trong text (có thể có text trước/sau)
    # Tìm từ { đầu tiên đến } cuối cùng hợp lệ
    json_start = cleaned.find('{')
    if json_start != -1:
        # Tìm } cuối cùng hợp lệ bằng cách đếm số lượng { và }
        brace_count = 0
        json_end = -1
        for i in range(json_start, len(cleaned)):
            if cleaned[i] == '{':
                brace_count += 1
            elif cleaned[i] == '}':
                brace_count -= 1
                if brace_count == 0:
                    json_end = i + 1
                    break
        
        if json_end != -1:
            cleaned = cleaned[json_start:json_end]
    
    return cleaned


def validate_and_normalize_bbox(
    bbox: BoundingBox,
    img_width: int,
    img_height: int,
    min_area: int = 100
) -> Optional[BoundingBox]:
    """
    Validate và normalize bbox coordinates
    
    Args:
        bbox: BoundingBox object
        img_width: Chiều rộng ảnh
        img_height: Chiều cao ảnh
        min_area: Diện tích tối thiểu (pixel^2) để giữ lại bbox
        
    Returns:
        BoundingBox đã được validate và normalize, hoặc None nếu invalid
    """
    if img_width <= 0 or img_height <= 0:
        return bbox  # Không validate nếu không có kích thước ảnh
    
    # Normalize coordinates về phạm vi hợp lệ
    x1 = max(0, min(bbox.x1, img_width - 1))
    y1 = max(0, min(bbox.y1, img_height - 1))
    x2 = max(x1 + 1, min(bbox.x2, img_width))
    y2 = max(y1 + 1, min(bbox.y2, img_height))
    
    # Validate coordinates
    if x2 <= x1 or y2 <= y1:
        return None
    
    # Check minimum area
    area = (x2 - x1) * (y2 - y1)
    if area < min_area:
        return None
    
    # Check aspect ratio (quá dài hoặc quá rộng có thể là lỗi)
    width = x2 - x1
    height = y2 - y1
    aspect_ratio = max(width / height, height / width) if height > 0 else float('inf')
    if aspect_ratio > 20:  # Quá dài hoặc quá rộng
        return None
    
    # Update bbox với normalized coordinates
    bbox.x1 = x1
    bbox.y1 = y1
    bbox.x2 = x2
    bbox.y2 = y2
    
    return bbox


def parse_response(response: str, img_width: int = 0, img_height: int = 0) -> DetectionResult:
    """
    Parse response từ model thành DetectionResult
    
    Args:
        response: Response text từ model
        img_width: Chiều rộng ảnh (nếu không có trong JSON)
        img_height: Chiều cao ảnh (nếu không có trong JSON)
        
    Returns:
        DetectionResult object chứa các bounding boxes
    """
    result = DetectionResult()
    
    # Clean response trước khi parse
    cleaned_response = clean_response(response)
    
    # Tìm JSON trong cleaned response
    json_match = re.search(r'\{[^{}]*(?:"seals"|"signatures")[^{}]*\}', cleaned_response, re.DOTALL)
    if not json_match:
        json_match = re.search(r'\{.*\}', cleaned_response, re.DOTALL)
    
    if json_match:
        try:
            json_str = json_match.group(0)
            data = json.loads(json_str)
            
            # Parse page, width, height
            result.page = data.get("page", 0)
            result.width = data.get("width", img_width)
            result.height = data.get("height", img_height)
            
            # Parse seals
            if "seals" in data and isinstance(data["seals"], list):
                for seal in data["seals"]:
                    # Skip nếu có placeholder values
                    if isinstance(seal, dict):
                        # Check for placeholder values in description
                        description = str(seal.get("description", ""))
                        if any(placeholder in description.lower() for placeholder in ["str (", "name of", "placeholder"]):
                            print(f"⚠️  Skipping seal with placeholder description: {description}")
                            continue
                        
                        # Check for placeholder values in confidence
                        confidence = seal.get("confidence", 1.0)
                        if isinstance(confidence, str) and confidence.lower() in ["double", "float", "number"]:
                            print(f"⚠️  Skipping seal with placeholder confidence: {confidence}")
                            continue
                    
                    if "bbox" in seal and isinstance(seal["bbox"], list) and len(seal["bbox"]) == 4:
                        bbox_arr = seal["bbox"]
                        try:
                            # Check for placeholder values in bbox
                            bbox_str = str(bbox_arr).lower()
                            if any(placeholder in bbox_str for placeholder in ["x0", "y0", "x1", "y1", "number", "of seal", "of signature"]):
                                print(f"⚠️  Skipping seal with placeholder bbox: {bbox_arr}")
                                continue
                            
                            coords = [int(float(x)) for x in bbox_arr]
                            if coords[2] > coords[0] and coords[3] > coords[1]:
                                bbox = BoundingBox(
                                    x1=coords[0],
                                    y1=coords[1],
                                    x2=coords[2],
                                    y2=coords[3],
                                    label="seal",
                                    confidence=float(confidence) if not isinstance(confidence, str) else 1.0,
                                    description=description if not any(ph in description.lower() for ph in ["str (", "name of"]) else ""
                                )
                                # Validate và normalize bbox
                                validated_bbox = validate_and_normalize_bbox(bbox, result.width or img_width, result.height or img_height)
                                if validated_bbox:
                                    result.add_seal(validated_bbox)
                                else:
                                    print(f"⚠️  Skipping invalid seal bbox (failed validation): {bbox_arr}")
                            else:
                                print(f"⚠️  Skipping invalid seal bbox (invalid coordinates): {bbox_arr}")
                        except (ValueError, TypeError) as e:
                            print(f"⚠️  Skipping seal with placeholder/invalid bbox: {bbox_arr} (Error: {e})")
                    elif all(k in seal for k in ["x1", "y1", "x2", "y2"]):
                        try:
                            bbox = BoundingBox(
                                x1=int(seal["x1"]),
                                y1=int(seal["y1"]),
                                x2=int(seal["x2"]),
                                y2=int(seal["y2"]),
                                label="seal",
                                confidence=float(seal.get("confidence", 1.0)),
                                description=seal.get("description", "")
                            )
                            result.add_seal(bbox)
                        except (ValueError, TypeError) as e:
                            print(f"⚠️  Skipping seal with invalid coordinates: {e}")
            
            # Parse signatures
            if "signatures" in data and isinstance(data["signatures"], list):
                for sig in data["signatures"]:
                    # Skip nếu có placeholder values
                    if isinstance(sig, dict):
                        # Check for placeholder values in description
                        description = str(sig.get("description", ""))
                        if any(placeholder in description.lower() for placeholder in ["str (", "name of", "placeholder"]):
                            print(f"⚠️  Skipping signature with placeholder description: {description}")
                            continue
                        
                        # Check for placeholder values in confidence
                        confidence = sig.get("confidence", 1.0)
                        if isinstance(confidence, str) and confidence.lower() in ["double", "float", "number"]:
                            print(f"⚠️  Skipping signature with placeholder confidence: {confidence}")
                            continue
                    
                    if "bbox" in sig and isinstance(sig["bbox"], list) and len(sig["bbox"]) == 4:
                        bbox_arr = sig["bbox"]
                        try:
                            # Check for placeholder values in bbox
                            bbox_str = str(bbox_arr).lower()
                            if any(placeholder in bbox_str for placeholder in ["x0", "y0", "x1", "y1", "number", "of seal", "of signature"]):
                                print(f"⚠️  Skipping signature with placeholder bbox: {bbox_arr}")
                                continue
                            
                            coords = [int(float(x)) for x in bbox_arr]
                            if coords[2] > coords[0] and coords[3] > coords[1]:
                                bbox = BoundingBox(
                                    x1=coords[0],
                                    y1=coords[1],
                                    x2=coords[2],
                                    y2=coords[3],
                                    label="signature",
                                    confidence=float(confidence) if not isinstance(confidence, str) else 1.0,
                                    description=description if not any(ph in description.lower() for ph in ["str (", "name of"]) else ""
                                )
                                # Validate và normalize bbox
                                validated_bbox = validate_and_normalize_bbox(bbox, result.width or img_width, result.height or img_height)
                                if validated_bbox:
                                    result.add_signature(validated_bbox)
                                else:
                                    print(f"⚠️  Skipping invalid signature bbox (failed validation): {bbox_arr}")
                            else:
                                print(f"⚠️  Skipping invalid signature bbox (invalid coordinates): {bbox_arr}")
                        except (ValueError, TypeError) as e:
                            print(f"⚠️  Skipping signature with placeholder/invalid bbox: {bbox_arr} (Error: {e})")
                    elif all(k in sig for k in ["x1", "y1", "x2", "y2"]):
                        try:
                            bbox = BoundingBox(
                                x1=int(sig["x1"]),
                                y1=int(sig["y1"]),
                                x2=int(sig["x2"]),
                                y2=int(sig["y2"]),
                                label="signature",
                                confidence=float(sig.get("confidence", 1.0)),
                                description=sig.get("description", "")
                            )
                            result.add_signature(bbox)
                        except (ValueError, TypeError) as e:
                            print(f"⚠️  Skipping signature with invalid coordinates: {e}")
        except json.JSONDecodeError as e:
            print(f"⚠️  Failed to parse JSON: {e}")
            print(f"Cleaned response: {cleaned_response[:500]}")
            print(f"Original response: {response[:500]}")
    else:
        print(f"⚠️  No JSON found in response")
        print(f"Cleaned response: {cleaned_response[:500]}")
        print(f"Original response: {response[:500]}")
    
    return result


def draw_boxes(
    image: Image.Image,
    result: DetectionResult,
    seal_color: str = "red",
    signature_color: str = "blue",
    line_width: int = 3,
) -> Image.Image:
    """
    Vẽ bounding boxes lên ảnh
    
    Args:
        image: PIL Image
        result: DetectionResult
        seal_color: Màu để vẽ con dấu
        signature_color: Màu để vẽ chữ ký
        line_width: Độ dày đường viền
        
    Returns:
        PIL Image đã được vẽ bounding boxes
    """
    draw = ImageDraw.Draw(image)
    
    # Vẽ seals (màu đỏ)
    for seal in result.seals:
        x1, y1, x2, y2 = seal.get_coords()
        draw.rectangle(
            [x1, y1, x2, y2],
            outline=seal_color,
            width=line_width
        )
        draw.text((x1, y1 - 20), "SEAL", fill=seal_color)
    
    # Vẽ signatures (màu xanh)
    for sig in result.signatures:
        x1, y1, x2, y2 = sig.get_coords()
        draw.rectangle(
            [x1, y1, x2, y2],
            outline=signature_color,
            width=line_width
        )
        draw.text((x1, y1 - 20), "SIGNATURE", fill=signature_color)
    
    return image
