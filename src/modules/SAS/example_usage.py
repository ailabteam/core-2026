"""
Example usage của Seal & Signature Detector

Các ví dụ này cho thấy cách sử dụng detector với model reuse để tránh OOM
"""

import os
import json
import torch
from .detector import SealSignatureDetector
from .models import DetectionResult
from .model_manager import ModelManager


def example_single_image():
    """Ví dụ: Xử lý một ảnh"""
    print("=" * 60)
    print("Example 1: Single Image Detection")
    print("=" * 60)
    
    # Khởi tạo detector - model sẽ được load một lần
    detector = SealSignatureDetector(
        model_path="deepseek-ai/deepseek-vl2-tiny",
        low_memory_mode=True,
        max_image_size=1024,
        use_multi_gpu=True,
    )
    
    # Xử lý ảnh
    result = detector.detect(
        image_path="path/to/image.jpg",
        language="vietnamese",
        return_image=False
    )
    
    print(f"Found {len(result.seals)} seals and {len(result.signatures)} signatures")


def example_batch_processing():
    """Ví dụ: Xử lý nhiều ảnh - model chỉ load một lần"""
    print("=" * 60)
    print("Example 2: Batch Processing (Model Reuse)")
    print("=" * 60)
    
    image_paths = [
        "path/to/image1.jpg",
        "path/to/image2.jpg",
        "path/to/image3.jpg",
    ]
    
    # Khởi tạo detector một lần
    # Model sẽ được load và reuse cho tất cả các ảnh
    detector = SealSignatureDetector(
        model_path="deepseek-ai/deepseek-vl2-tiny",
        low_memory_mode=True,
        max_image_size=1024,
        use_multi_gpu=True,
    )
    
    # Batch processing - model không bị reload
    results = detector.detect_batch(
        image_paths=image_paths,
        language="vietnamese",
        return_images=False
    )
    
    for result, image_path in zip(results, image_paths):
        print(f"{os.path.basename(image_path)}: {len(result.seals)} seals, {len(result.signatures)} signatures")


def example_reuse_across_cells():
    """
    Ví dụ: Sử dụng trong notebook với nhiều cells
    
    Model sẽ được reuse giữa các cells nếu cùng một instance
    """
    print("=" * 60)
    print("Example 3: Reuse Across Multiple Calls")
    print("=" * 60)
    
    # Cell 1: Khởi tạo detector
    detector = SealSignatureDetector(
        model_path="deepseek-ai/deepseek-vl2-tiny",
        low_memory_mode=True,
        use_multi_gpu=True,
    )
    
    # Cell 2: Xử lý ảnh đầu tiên
    result1 = detector.detect("image1.jpg", language="vietnamese")
    print(f"Image 1: {len(result1.seals)} seals")
    
    # Cell 3: Xử lý ảnh thứ hai - model không bị reload!
    result2 = detector.detect("image2.jpg", language="vietnamese")
    print(f"Image 2: {len(result2.seals)} seals")
    
    # Cell 4: Xử lý ảnh thứ ba - model vẫn được reuse!
    result3 = detector.detect("image3.jpg", language="vietnamese")
    print(f"Image 3: {len(result3.seals)} seals")
    
    print("\n✅ Model was loaded only once and reused for all 3 images!")


def example_explicit_model_manager():
    """Ví dụ: Sử dụng ModelManager trực tiếp để kiểm soát tốt hơn"""
    print("=" * 60)
    print("Example 4: Explicit Model Manager Usage")
    print("=" * 60)
    
    from model_manager import ModelManager
    
    # Load model một lần
    model_manager = ModelManager()
    model_manager.load_model(
        model_path="deepseek-ai/deepseek-vl2-tiny",
        low_memory_mode=True,
        use_multi_gpu=True,
    )
    
    # Sử dụng model cho nhiều detector instances
    detector1 = SealSignatureDetector(model_path="deepseek-ai/deepseek-vl2-tiny")
    detector2 = SealSignatureDetector(model_path="deepseek-ai/deepseek-vl2-tiny")
    
    # Cả hai detector đều sử dụng cùng một model instance!
    result1 = detector1.detect("image1.jpg")
    result2 = detector2.detect("image2.jpg")
    
    print("✅ Both detectors share the same model instance")
    
    # Release memory khi không cần nữa
    model_manager.release_memory()


def example_memory_optimization():
    """Ví dụ: Tối ưu memory cho nhiều ảnh lớn"""
    print("=" * 60)
    print("Example 5: Memory Optimization")
    print("=" * 60)
    
    # Khởi tạo với low_memory_mode và max_image_size nhỏ
    detector = SealSignatureDetector(
        model_path="deepseek-ai/deepseek-vl2-tiny",
        low_memory_mode=True,      # Clear cache sau mỗi step
        max_image_size=1024,       # Resize ảnh lớn xuống 1024px
        use_multi_gpu=True,        # Phân tán trên nhiều GPU
        device_map="balanced_low_0",  # Ưu tiên GPU 1 để GPU 0 có nhiều memory hơn
    )
    
    # Xử lý nhiều ảnh lớn
    large_images = ["large_image1.jpg", "large_image2.jpg", "large_image3.jpg"]
    
    for image_path in large_images:
        # Mỗi lần detect sẽ tự động clear cache nếu low_memory_mode=True
        result = detector.detect(image_path, language="vietnamese")
        print(f"Processed {os.path.basename(image_path)}")
    
    print("✅ All large images processed with memory optimization")


if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("Seal & Signature Detector - Usage Examples")
    print("=" * 60 + "\n")
    
    print("""
Các ví dụ này cho thấy cách sử dụng detector với model reuse:

1. Single Image: Xử lý một ảnh
2. Batch Processing: Xử lý nhiều ảnh, model chỉ load một lần
3. Reuse Across Cells: Sử dụng trong notebook, model được reuse
4. Explicit Model Manager: Kiểm soát model lifecycle thủ công
5. Memory Optimization: Tối ưu memory cho ảnh lớn

Chạy từng function để xem ví dụ cụ thể.
    """)
