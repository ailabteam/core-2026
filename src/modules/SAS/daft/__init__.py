"""
Seal & Signature Detection Module
Sử dụng DeepSeek-VL2 để nhận diện con dấu và chữ ký trong tài liệu
"""

from .models import BoundingBox, DetectionResult
from .detector import SealSignatureDetector
from .model_manager import ModelManager
from .utils import (
    resize_image_if_needed,
    create_detection_prompt,
    parse_response,
    draw_boxes
)

__all__ = [
    "BoundingBox",
    "DetectionResult",
    "SealSignatureDetector",
    "ModelManager",
    "resize_image_if_needed",
    "create_detection_prompt",
    "parse_response",
    "draw_boxes",
]
