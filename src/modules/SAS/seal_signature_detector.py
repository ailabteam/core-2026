# ==============================
# DeepSeek-VL2 Seal & Signature Detector
# ==============================
"""
Module để nhận diện và định vị con dấu (seal/stamp) và chữ ký (signature) 
trong hình ảnh tài liệu sử dụng DeepSeek-VL2.

File này giữ lại để backward compatibility.
Sử dụng từ detector.py và các module khác để tối ưu memory và reuse model.
"""

import os
import json
import torch
from typing import List, Optional

# Import từ các module mới
from .detector import SealSignatureDetector
from .models import BoundingBox, DetectionResult
from .model_manager import ModelManager

# Export để backward compatibility
__all__ = ["SealSignatureDetector", "BoundingBox", "DetectionResult", "ModelManager"]


# ------------------------------
# MAIN / Example Usage
# ------------------------------
def main():
    """
    Example usage - Single image detection
    
    Model sẽ chỉ được load một lần và reuse cho các lần chạy tiếp theo
    """
    # Setup
    model_name = "deepseek-ai/deepseek-vl2-tiny"
    image_path = "/kaggle/input/test-sign/1-5_Opt.jpg"  # Điều chỉnh đường dẫn theo cần thiết
    
    # Kiểm tra file ảnh tồn tại
    if not os.path.exists(image_path):
        print(f"❌ Image not found: {image_path}")
        print("Please update the image_path variable")
        return
    
    # Khởi tạo detector với multi-GPU và low_memory_mode
    # Model sẽ chỉ được load một lần (singleton pattern)
    num_gpus = torch.cuda.device_count() if torch.cuda.is_available() else 0
    max_img_size = 1536 if num_gpus > 1 else 1024
    
    detector = SealSignatureDetector(
        model_path=model_name,
        low_memory_mode=True,
        max_image_size=max_img_size,
        use_multi_gpu=True,
        device_map="auto",
    )
    
    # Nhận diện
    print(f"\n🔍 Detecting seals and signatures in: {image_path}\n")
    result, annotated_image = detector.detect(
        image_path=image_path,
        language="vietnamese",
        return_image=True
    )
    
    # In kết quả
    print(f"\n📊 Detection Results:")
    print(f"  - Seals found: {len(result.seals)}")
    print(f"  - Signatures found: {len(result.signatures)}")
    
    if result.seals:
        print("\n  🏷️  Seals:")
        for i, seal in enumerate(result.seals, 1):
            print(f"    {i}. {seal}")
    
    if result.signatures:
        print("\n  ✍️  Signatures:")
        for i, sig in enumerate(result.signatures, 1):
            print(f"    {i}. {sig}")
    
    # Lưu kết quả
    base_name = os.path.basename(image_path)
    name_without_ext = os.path.splitext(base_name)[0]
    
    # Xác định output directory
    if os.path.exists("/kaggle/working"):
        output_dir = "/kaggle/working"
    else:
        output_dir = os.path.dirname(image_path) if os.path.dirname(image_path) else "."
    
    output_json_path = os.path.join(output_dir, f"{name_without_ext}_detection.json")
    with open(output_json_path, "w", encoding="utf-8") as f:
        json.dump(result.to_json_format(), f, indent=2, ensure_ascii=False)
    print(f"\n✅ Saved JSON results to: {output_json_path}")
    
    # Lưu ảnh đã được vẽ
    ext = os.path.splitext(base_name)[1] or ".jpg"
    output_image_path = os.path.join(output_dir, f"{name_without_ext}_detected{ext}")
    annotated_image.save(output_image_path)
    print(f"✅ Saved annotated image to: {output_image_path}")


def main_batch():
    """
    Example usage - Batch processing nhiều ảnh
    
    Model chỉ được load một lần và reuse cho tất cả các ảnh
    """
    # Setup
    model_name = "deepseek-ai/deepseek-vl2-tiny"
    image_paths = [
        "/kaggle/input/test-sign/image1.jpg",
        "/kaggle/input/test-sign/image2.jpg",
        "/kaggle/input/test-sign/image3.jpg",
    ]
    
    # Filter chỉ các file tồn tại
    image_paths = [p for p in image_paths if os.path.exists(p)]
    
    if not image_paths:
        print("❌ No images found")
        return
    
    # Khởi tạo detector một lần
    # Model sẽ được load và reuse cho tất cả các ảnh
    num_gpus = torch.cuda.device_count() if torch.cuda.is_available() else 0
    max_img_size = 1536 if num_gpus > 1 else 1024
    
    print(f"🚀 Initializing detector (model will be loaded once)...")
    detector = SealSignatureDetector(
        model_path=model_name,
        low_memory_mode=True,
        max_image_size=max_img_size,
        use_multi_gpu=True,
        device_map="auto",
    )
    
    print(f"\n📦 Processing {len(image_paths)} images...")
    print("   (Model is reused for all images - no reloading)\n")
    
    # Batch processing
    results = detector.detect_batch(
        image_paths=image_paths,
        language="vietnamese",
        return_images=True
    )
    
    # Lưu kết quả cho mỗi ảnh
    if os.path.exists("/kaggle/working"):
        output_dir = "/kaggle/working"
    else:
        output_dir = "."
    
    for (result, annotated_image), image_path in zip(results, image_paths):
        base_name = os.path.basename(image_path)
        name_without_ext = os.path.splitext(base_name)[0]
        
        # Lưu JSON
        output_json_path = os.path.join(output_dir, f"{name_without_ext}_detection.json")
        with open(output_json_path, "w", encoding="utf-8") as f:
            json.dump(result.to_json_format(), f, indent=2, ensure_ascii=False)
        
        # Lưu ảnh
        if annotated_image:
            ext = os.path.splitext(base_name)[1] or ".jpg"
            output_image_path = os.path.join(output_dir, f"{name_without_ext}_detected{ext}")
            annotated_image.save(output_image_path)
        
        print(f"✅ Processed: {base_name} - {len(result.seals)} seals, {len(result.signatures)} signatures")
    
    print(f"\n✅ All {len(image_paths)} images processed!")
    print("💡 Model was loaded only once and reused for all images")


if __name__ == "__main__":
    # Chạy single image detection
    main()
    
    # Hoặc chạy batch processing
    # main_batch()
