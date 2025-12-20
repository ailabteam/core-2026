"""
Utility functions for Document Image Enhancement Pipeline
"""

import cv2
import numpy as np
from PIL import Image
from pdf2image import convert_from_path
from typing import Tuple, Optional, List
import os

from .config import *


def load_image(path: str) -> np.ndarray:
    """
    Load image from file path
    
    Args:
        path: Path to image file
        
    Returns:
        Image as numpy array (BGR format)
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"Image not found: {path}")
    
    image = cv2.imread(path)
    if image is None:
        raise ValueError(f"Failed to load image: {path}")
    
    return image


def load_pdf_page(pdf_path: str, page_num: int = 0, dpi: int = 300) -> np.ndarray:
    """
    Extract a page from PDF as image
    
    Args:
        pdf_path: Path to PDF file
        page_num: Page number (0-indexed)
        dpi: DPI for rendering
        
    Returns:
        Image as numpy array (BGR format)
    """
    if not os.path.exists(pdf_path):
        raise FileNotFoundError(f"PDF not found: {pdf_path}")
    
    # Convert PDF page to image
    images = convert_from_path(pdf_path, dpi=dpi, first_page=page_num+1, last_page=page_num+1)
    
    if not images:
        raise ValueError(f"Failed to extract page {page_num} from PDF")
    
    # Convert PIL Image to numpy array (RGB)
    pil_image = images[0]
    image_rgb = np.array(pil_image)
    
    # Convert RGB to BGR for OpenCV
    image_bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
    
    return image_bgr


def resize_with_aspect_ratio(image: np.ndarray, 
                             target_width: Optional[int] = None,
                             target_height: Optional[int] = None,
                             max_dimension: Optional[int] = None) -> np.ndarray:
    """
    Resize image maintaining aspect ratio
    
    Args:
        image: Input image
        target_width: Target width (optional)
        target_height: Target height (optional)
        max_dimension: Maximum dimension for either width or height (optional)
        
    Returns:
        Resized image
    """
    h, w = image.shape[:2]
    
    if max_dimension is not None:
        # Scale based on maximum dimension
        if max(h, w) > max_dimension:
            if h > w:
                scale = max_dimension / h
            else:
                scale = max_dimension / w
            new_w = int(w * scale)
            new_h = int(h * scale)
        else:
            return image
    elif target_width is not None:
        # Scale based on width
        scale = target_width / w
        new_w = target_width
        new_h = int(h * scale)
    elif target_height is not None:
        # Scale based on height
        scale = target_height / h
        new_h = target_height
        new_w = int(w * scale)
    else:
        return image
    
    resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)
    return resized


def compute_skew_angle(image: np.ndarray, method: str = 'hough') -> float:
    """
    Calculate skew angle of document
    
    Args:
        image: Input image (grayscale)
        method: Method to use ('hough' or 'projection')
        
    Returns:
        Rotation angle in degrees
    """
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image.copy()
    
    if method == 'hough':
        return _compute_skew_hough(gray)
    elif method == 'projection':
        return _compute_skew_projection(gray)
    else:
        raise ValueError(f"Unknown method: {method}")


def _compute_skew_hough(gray: np.ndarray) -> float:
    """Compute skew using Hough Line Transform"""
    # Edge detection
    edges = cv2.Canny(gray, CANNY_THRESHOLD1, CANNY_THRESHOLD2, apertureSize=CANNY_APERTURE)
    
    # Detect lines
    lines = cv2.HoughLinesP(
        edges,
        rho=HOUGH_RHO,
        theta=np.pi / 180 * HOUGH_THETA,
        threshold=HOUGH_THRESHOLD,
        minLineLength=HOUGH_MIN_LINE_LENGTH,
        maxLineGap=HOUGH_MAX_LINE_GAP
    )
    
    if lines is None or len(lines) == 0:
        return 0.0
    
    # Calculate angles
    angles = []
    for line in lines:
        x1, y1, x2, y2 = line[0]
        angle = np.degrees(np.arctan2(y2 - y1, x2 - x1))
        # Normalize to [-45, 45] range
        if angle < -45:
            angle += 90
        elif angle > 45:
            angle -= 90
        angles.append(angle)
    
    # Use median to be robust to outliers
    if angles:
        median_angle = np.median(angles)
        return median_angle
    
    return 0.0


def _compute_skew_projection(gray: np.ndarray) -> float:
    """Compute skew using projection profile (fallback method)"""
    # This is a simpler fallback method
    # Use moments to find the orientation
    coords = np.column_stack(np.where(gray > 0))
    if len(coords) < 10:
        return 0.0
    
    # Calculate angle using PCA-like approach
    angle = cv2.minAreaRect(coords)[-1]
    
    # Normalize angle
    if angle < -45:
        angle = 90 + angle
    
    return -angle


def find_document_contour(image: np.ndarray) -> Optional[np.ndarray]:
    """
    Find the largest rectangular contour (document boundary)
    
    Args:
        image: Input image
        
    Returns:
        4-point contour or None if not found
    """
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image.copy()
    
    # Apply Gaussian blur
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    
    # Edge detection
    edges = cv2.Canny(blurred, CANNY_THRESHOLD1, CANNY_THRESHOLD2)
    
    # Dilate edges to close gaps
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    dilated = cv2.dilate(edges, kernel, iterations=2)
    
    # Find contours
    contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if not contours:
        return None
    
    # Sort by area and get the largest
    contours = sorted(contours, key=cv2.contourArea, reverse=True)
    
    for contour in contours[:5]:  # Check top 5 largest contours
        # Skip if too small
        if cv2.contourArea(contour) < MIN_CONTOUR_AREA:
            continue
        
        # Approximate contour to polygon
        epsilon = CONTOUR_EPSILON_FACTOR * cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, epsilon, True)
        
        # Check if it's a quadrilateral
        if len(approx) == 4:
            return approx.reshape(4, 2)
    
    return None


def order_points(pts: np.ndarray) -> np.ndarray:
    """
    Order points in clockwise order: top-left, top-right, bottom-right, bottom-left
    
    Args:
        pts: 4 corner points
        
    Returns:
        Ordered points
    """
    # Sort by y-coordinate
    pts_sorted = pts[np.argsort(pts[:, 1])]
    
    # Top two points
    top_pts = pts_sorted[:2]
    top_pts = top_pts[np.argsort(top_pts[:, 0])]  # Sort by x
    tl, tr = top_pts
    
    # Bottom two points
    bottom_pts = pts_sorted[2:]
    bottom_pts = bottom_pts[np.argsort(bottom_pts[:, 0])]  # Sort by x
    bl, br = bottom_pts
    
    return np.array([tl, tr, br, bl], dtype=np.float32)


def apply_clahe(image: np.ndarray, 
                clip_limit: float = CLAHE_CLIP_LIMIT,
                tile_grid_size: Tuple[int, int] = CLAHE_TILE_GRID_SIZE) -> np.ndarray:
    """
    Apply CLAHE (Contrast Limited Adaptive Histogram Equalization)
    
    Args:
        image: Input grayscale image
        clip_limit: Threshold for contrast limiting
        tile_grid_size: Size of grid for histogram equalization
        
    Returns:
        Enhanced image
    """
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image.copy()
    
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
    enhanced = clahe.apply(gray)
    
    return enhanced


def calculate_image_quality_score(image: np.ndarray) -> dict:
    """
    Calculate various quality metrics for the image
    
    Args:
        image: Input image
        
    Returns:
        Dictionary with quality metrics
    """
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image.copy()
    
    # Sharpness (variance of Laplacian)
    laplacian = cv2.Laplacian(gray, cv2.CV_64F)
    sharpness = laplacian.var()
    
    # Contrast (standard deviation)
    contrast = gray.std()
    
    # Brightness (mean)
    brightness = gray.mean()
    
    # Normalized metrics
    return {
        'sharpness': float(sharpness),
        'contrast': float(contrast),
        'brightness': float(brightness),
        'overall_score': float(sharpness / 1000 + contrast / 100)  # Weighted combination
    }


def apply_unsharp_mask(image: np.ndarray, 
                       kernel_size: Tuple[int, int] = (5, 5),
                       sigma: float = 1.0,
                       amount: float = 1.0,
                       threshold: int = 0) -> np.ndarray:
    """
    Apply unsharp masking for image sharpening
    
    Args:
        image: Input image
        kernel_size: Gaussian kernel size
        sigma: Gaussian kernel standard deviation
        amount: Strength of sharpening
        threshold: Threshold for the difference
        
    Returns:
        Sharpened image
    """
    # Blur the image
    blurred = cv2.GaussianBlur(image, kernel_size, sigma)
    
    # Calculate the sharpened image
    sharpened = cv2.addWeighted(image, 1.0 + amount, blurred, -amount, 0)
    
    # Apply threshold if specified
    if threshold > 0:
        low_contrast_mask = np.abs(image - blurred) < threshold
        sharpened = np.where(low_contrast_mask, image, sharpened)
    
    return sharpened


def ensure_grayscale(image: np.ndarray) -> np.ndarray:
    """
    Ensure image is grayscale
    
    Args:
        image: Input image
        
    Returns:
        Grayscale image
    """
    if len(image.shape) == 3:
        return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return image


def rotate_image(image: np.ndarray, angle: float, background_color: Tuple[int, int, int] = (255, 255, 255)) -> np.ndarray:
    """
    Rotate image by specified angle
    
    Args:
        image: Input image
        angle: Rotation angle in degrees (positive = counter-clockwise)
        background_color: Color for the background
        
    Returns:
        Rotated image
    """
    h, w = image.shape[:2]
    center = (w // 2, h // 2)
    
    # Get rotation matrix
    rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
    
    # Calculate new dimensions
    cos = np.abs(rotation_matrix[0, 0])
    sin = np.abs(rotation_matrix[0, 1])
    new_w = int((h * sin) + (w * cos))
    new_h = int((h * cos) + (w * sin))
    
    # Adjust rotation matrix for new dimensions
    rotation_matrix[0, 2] += (new_w / 2) - center[0]
    rotation_matrix[1, 2] += (new_h / 2) - center[1]
    
    # Perform rotation
    rotated = cv2.warpAffine(
        image,
        rotation_matrix,
        (new_w, new_h),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=background_color
    )
    
    return rotated


def perspective_transform(image: np.ndarray, src_points: np.ndarray, target_size: Optional[Tuple[int, int]] = None) -> np.ndarray:
    """
    Apply perspective transformation to straighten document
    
    Args:
        image: Input image
        src_points: 4 corner points of document (ordered)
        target_size: Target size (width, height), auto-calculated if None
        
    Returns:
        Transformed image
    """
    # Order points
    src_points = order_points(src_points)
    
    # Calculate target size if not provided
    if target_size is None:
        # Calculate width
        width_top = np.linalg.norm(src_points[0] - src_points[1])
        width_bottom = np.linalg.norm(src_points[3] - src_points[2])
        max_width = int(max(width_top, width_bottom))
        
        # Calculate height
        height_left = np.linalg.norm(src_points[0] - src_points[3])
        height_right = np.linalg.norm(src_points[1] - src_points[2])
        max_height = int(max(height_left, height_right))
        
        target_size = (max_width, max_height)
    
    # Define destination points
    dst_points = np.array([
        [0, 0],
        [target_size[0] - 1, 0],
        [target_size[0] - 1, target_size[1] - 1],
        [0, target_size[1] - 1]
    ], dtype=np.float32)
    
    # Get perspective transform matrix
    matrix = cv2.getPerspectiveTransform(src_points, dst_points)
    
    # Apply transformation
    warped = cv2.warpPerspective(image, matrix, target_size)
    
    return warped
