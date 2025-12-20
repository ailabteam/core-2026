from __future__ import annotations

import io
import math
from typing import Tuple

import cv2
import numpy as np
from PIL import Image


def _resize_long_edge(img: np.ndarray, max_dim: int) -> np.ndarray:
    """
    Resize image keeping aspect ratio, only if larger than max_dim.
    """
    h, w = img.shape[:2]
    scale = min(1.0, float(max_dim) / max(h, w))
    if scale < 1.0:
        new_size = (int(w * scale), int(h * scale))
        img = cv2.resize(img, new_size, interpolation=cv2.INTER_AREA)
    return img


def _rotate_image_with_padding(img: np.ndarray, angle: float) -> np.ndarray:
    """
    Rotate image without cropping. Calculates canvas size to fit rotated image.
    """
    if abs(angle) < 0.1:  # Skip if angle too small
        return img
    
    h, w = img.shape[:2]
    # Convert angle to radians
    angle_rad = math.radians(angle)
    
    # Calculate new canvas size to fit rotated image
    cos_a = abs(math.cos(angle_rad))
    sin_a = abs(math.sin(angle_rad))
    new_w = int(h * sin_a + w * cos_a)
    new_h = int(h * cos_a + w * sin_a)
    
    # Center of original and new image
    center_x, center_y = w / 2, h / 2
    new_center_x, new_center_y = new_w / 2, new_h / 2
    
    # Create rotation matrix
    rot_mat = cv2.getRotationMatrix2D((center_x, center_y), angle, 1.0)
    # Adjust translation to center in new canvas
    rot_mat[0, 2] += new_center_x - center_x
    rot_mat[1, 2] += new_center_y - center_y
    
    # Rotate with padding (BORDER_CONSTANT with white background)
    rotated = cv2.warpAffine(
        img,
        rot_mat,
        (new_w, new_h),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(255, 255, 255) if len(img.shape) == 3 else 255
    )
    
    return rotated


def preprocess_pil(
    pil_img: Image.Image,
    max_dim: int = 2000,
    enable_deskew: bool = True,
    deskew_threshold: float = 2.0
) -> Tuple[Image.Image, bytes]:
    """
    Preprocess image: resize, optional deskew (with padding), denoise, contrast.
    
    Args:
        pil_img: Input PIL image
        max_dim: Maximum dimension (long edge) after resize
        enable_deskew: Whether to apply deskew correction
        deskew_threshold: Minimum angle (degrees) to apply deskew
    
    Returns:
        Tuple of (processed PIL image, PNG bytes)
    """
    pil_rgb = pil_img.convert("RGB")
    np_img = np.array(pil_rgb)
    
    # Resize if needed (preserves aspect ratio)
    np_img = _resize_long_edge(np_img, max_dim)
    
    # Optional deskew with padding to prevent cropping
    if enable_deskew:
        gray = cv2.cvtColor(np_img, cv2.COLOR_RGB2GRAY)
        denoised = cv2.fastNlMeansDenoising(gray, h=10)
        
        # Detect skew angle
        _, thresh = cv2.threshold(denoised, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        coords = np.column_stack(np.where(thresh > 0))
        
        angle = 0.0
        if coords.size:
            rect = cv2.minAreaRect(coords)
            detected_angle = rect[-1]
            if detected_angle < -45:
                angle = -(90 + detected_angle)
            else:
                angle = -detected_angle
        
        # Only apply if angle is significant
        if abs(angle) >= deskew_threshold:
            np_img = _rotate_image_with_padding(np_img, angle)
        # else: angle too small, skip rotation
    
    # Boost contrast slightly
    lab = cv2.cvtColor(np_img, cv2.COLOR_RGB2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    l = clahe.apply(l)
    lab = cv2.merge((l, a, b))
    enhanced = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)
    
    final_pil = Image.fromarray(enhanced)
    buf = io.BytesIO()
    final_pil.save(buf, format="PNG")
    return final_pil, buf.getvalue()

