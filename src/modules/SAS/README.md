# Seal & Signature Detector

Module nhận diện và định vị con dấu (seal/stamp) và chữ ký (signature) trong hình ảnh tài liệu sử dụng DeepSeek-VL2.

## 🎯 Tính năng chính

- ✅ **Model Reuse**: Model chỉ được load một lần và reuse cho nhiều requests (tránh OOM)
- ✅ **Multi-GPU Support**: Tự động phân tán model trên nhiều GPU
- ✅ **Memory Optimization**: Low memory mode với cache clearing tự động
- ✅ **Batch Processing**: Xử lý nhiều ảnh với model reuse
- ✅ **Modular Design**: Code được tách thành các module riêng biệt

## 📁 Cấu trúc Module

```
signatures-and-stamps/
├── __init__.py              # Exports chính
├── models.py                 # Data models (BoundingBox, DetectionResult)
├── model_manager.py         # Singleton để quản lý model lifecycle
├── detector.py              # Core detection logic
├── utils.py                 # Utilities (resize, drawing, prompts, parsing)
├── seal_signature_detector.py  # Main file (backward compatibility)
├── example_usage.py          # Ví dụ sử dụng
└── README.md                 # Tài liệu này
```

## 🚀 Cài đặt

```bash
pip install transformers deepseek-vl2 accelerate torch pillow
```

## 💻 Cách sử dụng

### 1. Single Image Detection

```python
from src.modules.signatures_and_stamps import SealSignatureDetector

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
    return_image=True
)

print(f"Found {len(result.seals)} seals and {len(result.signatures)} signatures")
```

### 2. Batch Processing (Model Reuse)

```python
# Model chỉ được load một lần và reuse cho tất cả các ảnh
detector = SealSignatureDetector(
    model_path="deepseek-ai/deepseek-vl2-tiny",
    low_memory_mode=True,
    use_multi_gpu=True,
)

image_paths = ["image1.jpg", "image2.jpg", "image3.jpg"]

# Batch processing - model không bị reload
results = detector.detect_batch(
    image_paths=image_paths,
    language="vietnamese",
    return_images=True
)

for (result, annotated_image), image_path in zip(results, image_paths):
    print(f"{image_path}: {len(result.seals)} seals")
```

### 3. Reuse Across Multiple Calls

```python
# Trong notebook hoặc script với nhiều lần gọi

# Lần đầu: Model được load
detector = SealSignatureDetector(model_path="deepseek-ai/deepseek-vl2-tiny")

# Các lần sau: Model được reuse (không reload)
result1 = detector.detect("image1.jpg")
result2 = detector.detect("image2.jpg")
result3 = detector.detect("image3.jpg")

# ✅ Model chỉ được load một lần!
```

### 4. Explicit Model Manager

```python
from src.modules.signatures_and_stamps import ModelManager, SealSignatureDetector

# Load model một lần thủ công
model_manager = ModelManager()
model_manager.load_model(
    model_path="deepseek-ai/deepseek-vl2-tiny",
    low_memory_mode=True,
    use_multi_gpu=True,
)

# Tạo nhiều detector instances - tất cả đều dùng chung model
detector1 = SealSignatureDetector(model_path="deepseek-ai/deepseek-vl2-tiny")
detector2 = SealSignatureDetector(model_path="deepseek-ai/deepseek-vl2-tiny")

# Release memory khi không cần nữa
model_manager.release_memory()
```

## ⚙️ Parameters

### SealSignatureDetector

- `model_path` (str): Đường dẫn đến model (default: "deepseek-ai/deepseek-vl2-tiny")
- `device` (torch.device): Thiết bị để chạy model (None = auto-detect)
- `max_new_tokens` (int): Số token tối đa cho response (default: 1024)
- `max_image_size` (int): Kích thước tối đa của ảnh, resize nếu lớn hơn (None = không resize)
- `low_memory_mode` (bool): Bật chế độ tiết kiệm memory (default: False)
- `use_multi_gpu` (bool): Sử dụng nhiều GPU nếu có (default: True)
- `device_map` (str): Device map strategy ("auto", "balanced", "balanced_low_0")

### detect()

- `image_path` (str): Đường dẫn đến file ảnh
- `language` (str): Ngôn ngữ của prompt ("vietnamese" hoặc "english")
- `return_image` (bool): Trả về PIL Image đã được vẽ bounding boxes

## 🔧 Model Manager (Singleton Pattern)

`ModelManager` sử dụng singleton pattern để đảm bảo model chỉ được load một lần:

```python
from src.modules.signatures_and_stamps import ModelManager

# Lần đầu: Load model
manager1 = ModelManager()
manager1.load_model(model_path="deepseek-ai/deepseek-vl2-tiny")

# Lần sau: Reuse cùng instance
manager2 = ModelManager()
# manager2 là cùng một instance với manager1
# Model không bị reload!
```

## 📊 Output Format

```json
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
```

## 💡 Tips để tránh OOM

1. **Sử dụng `low_memory_mode=True`**: Tự động clear cache và resize ảnh
2. **Set `max_image_size`**: Giới hạn kích thước ảnh (ví dụ: 1024 hoặc 1536)
3. **Reuse model**: Không tạo detector mới mỗi lần, reuse instance cũ
4. **Batch processing**: Sử dụng `detect_batch()` thay vì loop qua từng ảnh
5. **Multi-GPU**: Bật `use_multi_gpu=True` để phân tán model trên nhiều GPU
6. **Device map**: Thử `device_map="balanced_low_0"` để ưu tiên GPU 1

## 🐛 Troubleshooting

### Out of Memory Error

```python
# Giải pháp 1: Giảm max_image_size
detector = SealSignatureDetector(
    max_image_size=768,  # Giảm từ 1024 xuống 768
    low_memory_mode=True,
)

# Giải pháp 2: Restart kernel và load lại model
model_manager = ModelManager()
model_manager.release_memory()  # Giải phóng memory cũ
# Sau đó load lại
```

### Model không được reuse

Đảm bảo bạn đang sử dụng cùng một instance:

```python
# ✅ Đúng: Reuse cùng instance
detector = SealSignatureDetector(...)
result1 = detector.detect("img1.jpg")
result2 = detector.detect("img2.jpg")  # Model được reuse

# ❌ Sai: Tạo instance mới mỗi lần
result1 = SealSignatureDetector(...).detect("img1.jpg")
result2 = SealSignatureDetector(...).detect("img2.jpg")  # Model bị reload!
```

## 📝 Examples

Xem file `example_usage.py` để có thêm ví dụ chi tiết.

## 🔄 Migration từ code cũ

Code cũ vẫn hoạt động (backward compatible):

```python
# Code cũ vẫn hoạt động
from src.modules.signatures_and_stamps.seal_signature_detector import SealSignatureDetector

detector = SealSignatureDetector(...)
```

Nhưng khuyến nghị sử dụng import mới:

```python
# Code mới (khuyến nghị)
from src.modules.signatures_and_stamps import SealSignatureDetector

detector = SealSignatureDetector(...)
```
