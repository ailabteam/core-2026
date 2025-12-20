"""
Document Image Enhancement Pipeline

Pipeline chuẩn xử lý ảnh tài liệu (Document Image Enhancement)
Bao gồm 5 bước chính:
1. Chuẩn hoá đầu vào (format, DPI, màu)
2. Căn chỉnh & sửa méo (deskew / dewarp)
3. Crop vùng tài liệu
4. Khử nhiễu & tăng chất lượng
5. Chuẩn hoá cho OCR / AI
"""

import cv2
import numpy as np
import time
from typing import Tuple, Optional, Union, Dict, Any
import os

from .config import *
from .utils import (
    load_image,
    load_pdf_page,
    resize_with_aspect_ratio,
    compute_skew_angle,
    find_document_contour,
    order_points,
    apply_clahe,
    calculate_image_quality_score,
    apply_unsharp_mask,
    ensure_grayscale,
    rotate_image,
    perspective_transform
)


class DocumentEnhancer:
    """
    Document Image Enhancement Pipeline
    
    Processes document images through a 5-stage pipeline to prepare them
    for OCR and AI analysis.
    
    Attributes:
        target_dpi: Target DPI for output images
        grayscale: Whether to convert to grayscale
        enable_deskew: Enable automatic skew correction
        enable_crop: Enable automatic document cropping
        enable_enhance: Enable denoising and enhancement
        enable_ocr_prep: Enable OCR preparation stage
    """
    
    def __init__(
        self,
        target_dpi: int = TARGET_DPI,
        grayscale: bool = DEFAULT_GRAYSCALE,
        enable_deskew: bool = True,
        enable_crop: bool = True,
        enable_enhance: bool = True,
        enable_ocr_prep: bool = True
    ):
        """
        Initialize DocumentEnhancer
        
        Args:
            target_dpi: Target DPI for processing (default: 300)
            grayscale: Convert to grayscale (default: True)
            enable_deskew: Enable skew correction (default: True)
            enable_crop: Enable document cropping (default: True)
            enable_enhance: Enable enhancement filters (default: True)
            enable_ocr_prep: Enable OCR preparation (default: True)
        """
        self.target_dpi = target_dpi
        self.grayscale = grayscale
        self.enable_deskew = enable_deskew
        self.enable_crop = enable_crop
        self.enable_enhance = enable_enhance
        self.enable_ocr_prep = enable_ocr_prep
        
    def process(
        self,
        input_source: Union[str, np.ndarray],
        input_type: str = 'auto'
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Main pipeline entry point
        
        Args:
            input_source: Path to image/PDF or numpy array
            input_type: 'image', 'pdf', or 'auto' (auto-detect from extension)
            
        Returns:
            Tuple of (enhanced_image, metadata)
        """
        start_time = time.time()
        
        # Initialize metadata
        metadata = {
            'input_type': input_type,
            'processing_steps': [],
            'original_size': None,
            'rotation_angle': 0.0,
            'crop_coordinates': None,
            'enhancement_params': {},
            'quality_metrics': {},
            'final_size': None,
            'processing_time_ms': 0.0
        }
        
        # Load image
        if isinstance(input_source, str):
            # Auto-detect type from extension
            if input_type == 'auto':
                ext = os.path.splitext(input_source)[1].lower()
                if ext == '.pdf':
                    input_type = 'pdf'
                else:
                    input_type = 'image'
            
            metadata['input_type'] = input_type
            
            if input_type == 'pdf':
                image = load_pdf_page(input_source, page_num=0, dpi=self.target_dpi)
            else:
                image = load_image(input_source)
        else:
            image = input_source.copy()
            metadata['input_type'] = 'array'
        
        metadata['original_size'] = (image.shape[1], image.shape[0])
        
        # Stage 1: Normalize input
        image = self.normalize_input(image)
        metadata['processing_steps'].append('normalize')
        
        # Stage 2: Align image (deskew)
        if self.enable_deskew:
            image, rotation_angle = self.align_image(image)
            metadata['rotation_angle'] = rotation_angle
            metadata['processing_steps'].append('deskew')
        
        # Stage 3: Crop document
        if self.enable_crop:
            result = self.crop_document(image)
            if result is not None:
                image, crop_coords = result
                metadata['crop_coordinates'] = crop_coords.tolist() if crop_coords is not None else None
                metadata['processing_steps'].append('crop')
        
        # Stage 4: Denoise and enhance
        if self.enable_enhance:
            image, enhance_params = self.denoise_enhance(image)
            metadata['enhancement_params'] = enhance_params
            metadata['processing_steps'].append('enhance')
        
        # Stage 5: Prepare for OCR
        if self.enable_ocr_prep:
            image = self.prepare_for_ocr(image)
            metadata['processing_steps'].append('ocr_prep')
        
        # Calculate final quality metrics
        metadata['quality_metrics'] = calculate_image_quality_score(image)
        metadata['final_size'] = (image.shape[1], image.shape[0])
        
        # Calculate processing time
        processing_time = (time.time() - start_time) * 1000
        metadata['processing_time_ms'] = round(processing_time, 2)
        
        return image, metadata
    
    def normalize_input(self, image: np.ndarray) -> np.ndarray:
        """
        Stage 1: Format, DPI, and color normalization
        
        Converts image to standard format:
        - Grayscale conversion (if enabled)
        - Resize to target dimensions maintaining aspect ratio
        
        Args:
            image: Input image
            
        Returns:
            Normalized image
        """
        # Convert to grayscale if requested
        if self.grayscale and len(image.shape) == 3:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Resize to target height while maintaining aspect ratio
        if image.shape[0] != TARGET_HEIGHT:
            image = resize_with_aspect_ratio(image, target_height=TARGET_HEIGHT)
        
        return image
    
    def align_image(self, image: np.ndarray) -> Tuple[np.ndarray, float]:
        """
        Stage 2: Deskew using Hough Transform
        
        Detects and corrects image rotation using line detection.
        
        Args:
            image: Input image
            
        Returns:
            Tuple of (aligned_image, rotation_angle)
        """
        # Compute skew angle
        angle = compute_skew_angle(image, method='hough')
        
        # Only correct if angle is significant
        if abs(angle) < 0.1:
            return image, 0.0
        
        # Limit correction angle
        if abs(angle) > MAX_SKEW_ANGLE:
            angle = np.sign(angle) * MAX_SKEW_ANGLE
        
        # Rotate image
        if len(image.shape) == 3:
            bg_color = (255, 255, 255)
        else:
            bg_color = (255,)
        
        rotated = rotate_image(image, angle, background_color=bg_color)
        
        return rotated, angle
    
    def crop_document(self, image: np.ndarray) -> Optional[Tuple[np.ndarray, np.ndarray]]:
        """
        Stage 3: Detect and crop document boundaries
        
        Uses edge detection and contour finding to identify document boundaries
        and applies perspective transformation.
        
        Args:
            image: Input image
            
        Returns:
            Tuple of (cropped_image, corner_coordinates) or None if detection fails
        """
        # Find document contour
        contour = find_document_contour(image)
        
        if contour is None:
            # If detection fails, return original image
            return None
        
        # Apply perspective transformation
        cropped = perspective_transform(image, contour)
        
        return cropped, contour
    
    def denoise_enhance(self, image: np.ndarray) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Stage 4: Apply filters and CLAHE for enhancement
        
        Applies multiple filters to reduce noise and enhance quality:
        - Bilateral filter (edge-preserving noise reduction)
        - Median filter (salt-and-pepper noise)
        - CLAHE (contrast enhancement)
        - Unsharp masking (sharpening)
        
        Args:
            image: Input image
            
        Returns:
            Tuple of (enhanced_image, enhancement_parameters)
        """
        enhanced = image.copy()
        params = {}
        
        # Ensure grayscale
        if len(enhanced.shape) == 3:
            enhanced = cv2.cvtColor(enhanced, cv2.COLOR_BGR2GRAY)
        
        # Step 1: Bilateral filter (edge-preserving denoising)
        enhanced = cv2.bilateralFilter(
            enhanced,
            d=BILATERAL_D,
            sigmaColor=BILATERAL_SIGMA_COLOR,
            sigmaSpace=BILATERAL_SIGMA_SPACE
        )
        params['bilateral_d'] = BILATERAL_D
        
        # Step 2: Median filter (remove salt-and-pepper noise)
        enhanced = cv2.medianBlur(enhanced, MEDIAN_KERNEL_SIZE)
        params['median_kernel'] = MEDIAN_KERNEL_SIZE
        
        # Step 3: CLAHE (contrast enhancement)
        enhanced = apply_clahe(
            enhanced,
            clip_limit=CLAHE_CLIP_LIMIT,
            tile_grid_size=CLAHE_TILE_GRID_SIZE
        )
        params['clahe_clip'] = CLAHE_CLIP_LIMIT
        params['clahe_grid'] = CLAHE_TILE_GRID_SIZE
        
        # Step 4: Unsharp masking (sharpening)
        enhanced = apply_unsharp_mask(
            enhanced,
            kernel_size=(5, 5),
            sigma=1.0,
            amount=0.5
        )
        params['unsharp_amount'] = 0.5
        
        return enhanced, params
    
    def prepare_for_ocr(self, image: np.ndarray) -> np.ndarray:
        """
        Stage 5: Adaptive threshold and morphology for OCR
        
        Prepares image for optimal OCR performance:
        - Adaptive thresholding (handles varying lighting)
        - Morphological operations (clean up noise)
        - Ensure white background, black text
        
        Args:
            image: Input image
            
        Returns:
            OCR-ready binary image
        """
        # Ensure grayscale
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()
        
        # Adaptive thresholding
        binary = cv2.adaptiveThreshold(
            gray,
            255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY,
            ADAPTIVE_BLOCK_SIZE,
            ADAPTIVE_C
        )
        
        # Morphological operations to clean up
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, MORPH_KERNEL_SIZE)
        
        # Opening: remove small white noise
        opened = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel, iterations=MORPH_ITERATIONS)
        
        # Closing: fill small black holes
        closed = cv2.morphologyEx(opened, cv2.MORPH_CLOSE, kernel, iterations=MORPH_ITERATIONS)
        
        return closed
    
    def process_batch(
        self,
        input_sources: list,
        input_type: str = 'auto'
    ) -> list:
        """
        Process multiple images in batch
        
        Args:
            input_sources: List of image paths or arrays
            input_type: Type of input ('image', 'pdf', or 'auto')
            
        Returns:
            List of tuples (enhanced_image, metadata)
        """
        results = []
        for source in input_sources:
            try:
                result = self.process(source, input_type)
                results.append(result)
            except Exception as e:
                print(f"Error processing {source}: {str(e)}")
                results.append((None, {'error': str(e)}))
        
        return results
    
    def get_config(self) -> Dict[str, Any]:
        """
        Get current configuration
        
        Returns:
            Configuration dictionary
        """
        return {
            'target_dpi': self.target_dpi,
            'grayscale': self.grayscale,
            'enable_deskew': self.enable_deskew,
            'enable_crop': self.enable_crop,
            'enable_enhance': self.enable_enhance,
            'enable_ocr_prep': self.enable_ocr_prep
        }


def quick_enhance(
    input_source: Union[str, np.ndarray],
    grayscale: bool = True
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """
    Quick enhancement with default settings
    
    Convenience function for simple usage.
    
    Args:
        input_source: Path to image/PDF or numpy array
        grayscale: Convert to grayscale
        
    Returns:
        Tuple of (enhanced_image, metadata)
    """
    enhancer = DocumentEnhancer(grayscale=grayscale)
    return enhancer.process(input_source)


if __name__ == "__main__":
    # Simple CLI for testing
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python app.py <image_path>")
        sys.exit(1)
    
    image_path = sys.argv[1]
    
    # Process image
    enhancer = DocumentEnhancer()
    enhanced, metadata = enhancer.process(image_path)
    
    # Print results
    print("=" * 60)
    print("Document Enhancement Complete")
    print("=" * 60)
    print(f"Input: {image_path}")
    print(f"Original size: {metadata['original_size']}")
    print(f"Final size: {metadata['final_size']}")
    print(f"Rotation angle: {metadata['rotation_angle']:.2f}°")
    print(f"Processing steps: {', '.join(metadata['processing_steps'])}")
    print(f"Processing time: {metadata['processing_time_ms']:.2f} ms")
    print(f"\nQuality metrics:")
    for key, value in metadata['quality_metrics'].items():
        print(f"  {key}: {value:.2f}")
    
    # Save result
    output_path = image_path.rsplit('.', 1)[0] + '_enhanced.png'
    cv2.imwrite(output_path, enhanced)
    print(f"\nSaved to: {output_path}")
