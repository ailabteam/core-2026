# DCAD - Document Collection And Digitization

**Pipeline chuẩn xử lý ảnh tài liệu (Document Image Enhancement)**

Module DCAD cung cấp một pipeline hoàn chỉnh để xử lý và tăng cường chất lượng ảnh tài liệu, chuẩn bị cho các bước OCR và phân tích AI tiếp theo.

---

## 📋 Tổng quan

DCAD xử lý ảnh tài liệu qua 5 bước chính:

```
Input (camera/scan/PDF)
   ↓
[1] Chuẩn hoá đầu vào (format, DPI, màu)
   ↓
[2] Căn chỉnh & sửa méo (deskew / dewarp)
   ↓
[3] Crop vùng tài liệu
   ↓
[4] Khử nhiễu & tăng chất lượng
   ↓
[5] Chuẩn hoá cho OCR / AI
   ↓
Output (ảnh sạch + metadata)
```

---

## 🚀 Cài đặt

### Yêu cầu

- Python 3.8+
- pip hoặc conda

### Cài đặt dependencies

```bash
cd src/modules/DCAD

pip install -r requirements.txt
```

### Dependencies chính

- **opencv-python** - Xử lý ảnh core
- **Pillow** - I/O ảnh
- **pdf2image** - Chuyển đổi PDF
- **numpy** - Tính toán số học
- **scikit-image** - Xử lý ảnh nâng cao
- **scipy** - Tính toán khoa học

---

## 💡 Sử dụng cơ bản

### Quick Start

```python
from modules.DCAD.app import DocumentEnhancer

# Khởi tạo
enhancer = DocumentEnhancer()

# Xử lý ảnh
enhanced_image, metadata = enhancer.process('path/to/document.jpg')

# Hoặc dùng quick enhance
from modules.DCAD.app import quick_enhance
enhanced_image, metadata = quick_enhance('path/to/document.jpg')
```

### Xử lý PDF

```python
# Xử lý trang PDF
enhancer = DocumentEnhancer()
enhanced, metadata = enhancer.process('document.pdf', input_type='pdf')
```

### Xử lý batch

```python
# Xử lý nhiều ảnh
image_paths = ['img1.jpg', 'img2.jpg', 'img3.jpg']
results = enhancer.process_batch(image_paths)

for enhanced_image, metadata in results:
    print(f"Processing time: {metadata['processing_time_ms']} ms")
```

### Configuration tùy chỉnh

```python
enhancer = DocumentEnhancer(
    target_dpi=300,          # DPI mục tiêu
    grayscale=True,          # Chuyển sang grayscale
    enable_deskew=True,      # Bật tự động xoay
    enable_crop=True,        # Bật tự động crop
    enable_enhance=True,     # Bật tăng cường
    enable_ocr_prep=True     # Bật chuẩn bị OCR
)

enhanced, metadata = enhancer.process('image.jpg')
```

---

## 📖 API Reference

### Class: `DocumentEnhancer`

Pipeline chính để xử lý ảnh tài liệu.

#### Constructor

```python
DocumentEnhancer(
    target_dpi: int = 300,
    grayscale: bool = True,
    enable_deskew: bool = True,
    enable_crop: bool = True,
    enable_enhance: bool = True,
    enable_ocr_prep: bool = True
)
```

**Parameters:**
- `target_dpi` - DPI mục tiêu cho output (mặc định: 300)
- `grayscale` - Chuyển sang grayscale (mặc định: True)
- `enable_deskew` - Bật tự động căn chỉnh góc (mặc định: True)
- `enable_crop` - Bật tự động crop tài liệu (mặc định: True)
- `enable_enhance` - Bật khử nhiễu và tăng cường (mặc định: True)
- `enable_ocr_prep` - Bật chuẩn bị cho OCR (mặc định: True)

#### Methods

##### `process(input_source, input_type='auto')`

Xử lý một ảnh/PDF thông qua toàn bộ pipeline.

**Parameters:**
- `input_source` (str | np.ndarray) - Đường dẫn hoặc numpy array
- `input_type` (str) - Loại input: 'image', 'pdf', hoặc 'auto'

**Returns:**
- `tuple[np.ndarray, dict]` - (enhanced_image, metadata)

**Example:**
```python
enhanced, metadata = enhancer.process('document.jpg')
```

##### `process_batch(input_sources, input_type='auto')`

Xử lý nhiều ảnh cùng lúc.

**Parameters:**
- `input_sources` (list) - Danh sách đường dẫn hoặc arrays
- `input_type` (str) - Loại input

**Returns:**
- `list[tuple]` - Danh sách (enhanced_image, metadata)

##### `normalize_input(image)`

**Stage 1:** Chuẩn hóa format, DPI, và màu sắc.

##### `align_image(image)`

**Stage 2:** Căn chỉnh và xoay ảnh sử dụng Hough Transform.

**Returns:** `(aligned_image, rotation_angle)`

##### `crop_document(image)`

**Stage 3:** Phát hiện và crop biên tài liệu.

**Returns:** `(cropped_image, corner_coordinates)` hoặc `None`

##### `denoise_enhance(image)`

**Stage 4:** Khử nhiễu và tăng cường chất lượng.

**Returns:** `(enhanced_image, enhancement_params)`

##### `prepare_for_ocr(image)`

**Stage 5:** Chuẩn bị cho OCR với adaptive threshold.

**Returns:** `ocr_ready_image`

##### `get_config()`

Lấy cấu hình hiện tại.

**Returns:** `dict` - Configuration dictionary

---

### Function: `quick_enhance()`

Hàm tiện ích cho xử lý nhanh với cấu hình mặc định.

```python
quick_enhance(input_source, grayscale=True)
```

**Example:**
```python
from modules.DCAD.app import quick_enhance
enhanced, metadata = quick_enhance('document.jpg')
```

---

## 📊 Metadata Structure

Mỗi lần xử lý sẽ trả về metadata chi tiết:

```python
metadata = {
    'input_type': 'image',              # Loại input
    'original_size': (1920, 1080),      # Kích thước gốc (w, h)
    'processing_steps': [               # Các bước đã thực hiện
        'normalize', 
        'deskew', 
        'crop', 
        'enhance', 
        'ocr_prep'
    ],
    'rotation_angle': -2.3,             # Góc xoay (độ)
    'crop_coordinates': [               # Tọa độ crop (nếu có)
        [10, 20], 
        [1910, 25], 
        [1905, 1075], 
        [15, 1070]
    ],
    'enhancement_params': {             # Tham số tăng cường
        'bilateral_d': 9,
        'median_kernel': 3,
        'clahe_clip': 2.0,
        'clahe_grid': (8, 8),
        'unsharp_amount': 0.5
    },
    'quality_metrics': {                # Chỉ số chất lượng
        'sharpness': 1234.56,
        'contrast': 78.9,
        'brightness': 145.2,
        'overall_score': 2.02
    },
    'final_size': (1754, 2480),         # Kích thước cuối (w, h)
    'processing_time_ms': 1234.56       # Thời gian xử lý (ms)
}
```

---

## 🔧 Configuration

### File: `config.py`

Tất cả tham số có thể được điều chỉnh trong `config.py`:

```python
# Target dimensions
TARGET_DPI = 300
TARGET_HEIGHT = 2480
TARGET_WIDTH = 1754

# Filters
MEDIAN_KERNEL_SIZE = 3
BILATERAL_D = 9

# CLAHE
CLAHE_CLIP_LIMIT = 2.0
CLAHE_TILE_GRID_SIZE = (8, 8)

# Edge detection
CANNY_THRESHOLD1 = 50
CANNY_THRESHOLD2 = 150

# Adaptive threshold
ADAPTIVE_BLOCK_SIZE = 11
ADAPTIVE_C = 2

# Hough Transform
HOUGH_THRESHOLD = 100
HOUGH_MIN_LINE_LENGTH = 100
HOUGH_MAX_LINE_GAP = 10
```

### Tuning Tips

**Nếu ảnh quá tối:**
- Tăng `CLAHE_CLIP_LIMIT` (2.0 → 3.0)
- Giảm `ADAPTIVE_C` (2 → 1)

**Nếu nhiễu nhiều:**
- Tăng `MEDIAN_KERNEL_SIZE` (3 → 5)
- Tăng `BILATERAL_D` (9 → 11)

**Nếu không detect được góc xoay:**
- Giảm `HOUGH_THRESHOLD` (100 → 50)
- Tăng `HOUGH_MIN_LINE_LENGTH` (100 → 150)

**Nếu crop không chính xác:**
- Điều chỉnh `CANNY_THRESHOLD1` và `CANNY_THRESHOLD2`
- Tăng `CONTOUR_EPSILON_FACTOR` (0.02 → 0.03)

---

## 🎯 Chi tiết kỹ thuật

### Stage 1: Normalize Input

**Mục đích:** Đưa mọi input về cùng chuẩn

**Kỹ thuật:**
- Convert PDF → Image (300 DPI)
- Chuyển về grayscale
- Resize theo chiều dài chuẩn (2480px ~ A4 @300dpi)
- Maintain aspect ratio

### Stage 2: Deskew

**Mục đích:** Căn chỉnh ảnh bị nghiêng

**Kỹ thuật:** Hough Line Transform
```python
# Detect edges
edges = cv2.Canny(gray, 50, 150)

# Detect lines
lines = cv2.HoughLinesP(edges, rho=1, theta=np.pi/180, 
                        threshold=100, minLineLength=100)

# Calculate median angle
angles = [arctan2(y2-y1, x2-x1) for each line]
rotation_angle = median(angles)

# Rotate image
M = cv2.getRotationMatrix2D(center, angle, 1.0)
rotated = cv2.warpAffine(image, M, (w, h))
```

**Ưu điểm:**
- Robust với nhiễu
- Có thể detect multiple lines
- Sử dụng median để tránh outliers

### Stage 3: Crop Document

**Mục đích:** Loại bỏ nền, tay người, bóng

**Kỹ thuật:** Contour detection + Perspective transform
```python
# Edge detection
edges = cv2.Canny(blurred, 50, 150)

# Find contours
contours = cv2.findContours(edges, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE)

# Get largest quadrilateral
largest = max(contours, key=cv2.contourArea)
approx = cv2.approxPolyDP(largest, epsilon, True)

# Perspective transform
M = cv2.getPerspectiveTransform(src_points, dst_points)
warped = cv2.warpPerspective(image, M, (w, h))
```

### Stage 4: Denoise & Enhance

**Mục đích:** Tăng chất lượng cho OCR

**Pipeline:**
1. **Bilateral Filter** - Khử nhiễu giữ edge
2. **Median Filter** - Loại salt-and-pepper noise
3. **CLAHE** - Tăng contrast cục bộ
4. **Unsharp Masking** - Làm sắc nét

```python
# Bilateral filter
enhanced = cv2.bilateralFilter(image, d=9, sigmaColor=75, sigmaSpace=75)

# Median filter
enhanced = cv2.medianBlur(enhanced, 3)

# CLAHE
clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
enhanced = clahe.apply(enhanced)

# Unsharp mask
blurred = cv2.GaussianBlur(enhanced, (5,5), 1.0)
enhanced = cv2.addWeighted(enhanced, 1.5, blurred, -0.5, 0)
```

**CLAHE là gì?**
- Contrast Limited Adaptive Histogram Equalization
- Tăng contrast cục bộ mà không làm quá sáng vùng đồng nhất
- Rất hiệu quả cho tài liệu có ánh sáng không đều

### Stage 5: OCR Preparation

**Mục đích:** Output lý tưởng cho OCR

**Yêu cầu output:**
- ✅ 300 DPI
- ✅ Background trắng
- ✅ Text đen
- ✅ Không bóng, không nhiễu
- ✅ Binary image

**Kỹ thuật:** Adaptive Threshold + Morphology
```python
# Adaptive threshold
binary = cv2.adaptiveThreshold(
    gray, 255, 
    cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
    cv2.THRESH_BINARY, 11, 2
)

# Morphological operations
kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))

# Remove noise
opened = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)

# Fill holes
closed = cv2.morphologyEx(opened, cv2.MORPH_CLOSE, kernel)
```

**Tại sao dùng Adaptive Threshold?**
- Xử lý được ảnh có ánh sáng không đều
- Tốt hơn global threshold
- Tự động điều chỉnh theo từng vùng

---

## 📝 Examples

### Example 1: Xử lý ảnh chụp từ điện thoại

```python
from modules.DCAD.app import DocumentEnhancer

# Ảnh chụp từ camera thường bị nghiêng, có bóng
enhancer = DocumentEnhancer(
    grayscale=True,
    enable_deskew=True,   # Sửa góc nghiêng
    enable_crop=True,     # Loại bỏ nền
    enable_enhance=True   # Khử bóng và nhiễu
)

enhanced, metadata = enhancer.process('photo_from_phone.jpg')

print(f"Đã xoay: {metadata['rotation_angle']}°")
print(f"Chất lượng: {metadata['quality_metrics']['overall_score']}")
```

### Example 2: Xử lý scan chất lượng cao

```python
# Scan thường đã thẳng và sạch, chỉ cần enhance
enhancer = DocumentEnhancer(
    grayscale=True,
    enable_deskew=False,  # Không cần xoay
    enable_crop=False,    # Không cần crop
    enable_enhance=True,  # Chỉ tăng cường
    enable_ocr_prep=True  # Chuẩn bị OCR
)

enhanced, metadata = enhancer.process('scanned_doc.jpg')
```

### Example 3: Xử lý PDF nhiều trang

```python
from pdf2image import convert_from_path

# Convert all pages
images = convert_from_path('document.pdf', dpi=300)

enhancer = DocumentEnhancer()
enhanced_pages = []

for i, img in enumerate(images):
    # Convert PIL to numpy
    img_array = np.array(img)
    img_bgr = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
    
    # Process
    enhanced, metadata = enhancer.process(img_bgr)
    enhanced_pages.append(enhanced)
    
    print(f"Page {i+1}: {metadata['processing_time_ms']} ms")
```

### Example 4: Batch processing với output

```python
import os
from pathlib import Path

# Setup
input_dir = Path('input_images')
output_dir = Path('output_enhanced')
output_dir.mkdir(exist_ok=True)

# Get all images
image_files = list(input_dir.glob('*.jpg')) + list(input_dir.glob('*.png'))

# Process
enhancer = DocumentEnhancer()

for img_path in image_files:
    enhanced, metadata = enhancer.process(str(img_path))
    
    # Save
    output_path = output_dir / f"{img_path.stem}_enhanced.png"
    cv2.imwrite(str(output_path), enhanced)
    
    print(f"✅ {img_path.name} → {output_path.name}")
```

---

## 🧪 Testing

### Chạy examples

```bash
cd src/modules/DCAD
python example_usage.py
```

Menu sẽ hiện ra với các tùy chọn:
1. Basic Usage
2. PDF Processing
3. Custom Configuration
4. Stage-by-Stage Processing
5. Batch Processing
6. Quick Enhance

### Test với CLI

```bash
python app.py path/to/image.jpg
```

### Test từng stage

```python
from modules.DCAD.app import DocumentEnhancer
import cv2

enhancer = DocumentEnhancer()
img = cv2.imread('test.jpg')

# Test từng stage
stage1 = enhancer.normalize_input(img)
stage2, angle = enhancer.align_image(stage1)
stage3_result = enhancer.crop_document(stage2)
if stage3_result:
    stage3, coords = stage3_result
else:
    stage3 = stage2
stage4, params = enhancer.denoise_enhance(stage3)
stage5 = enhancer.prepare_for_ocr(stage4)

# Visualize
import matplotlib.pyplot as plt
stages = [img, stage1, stage2, stage3, stage4, stage5]
titles = ['Original', 'Normalized', 'Aligned', 'Cropped', 'Enhanced', 'OCR Ready']

fig, axes = plt.subplots(2, 3, figsize=(15, 10))
for ax, stage, title in zip(axes.flat, stages, titles):
    ax.imshow(stage if len(stage.shape) == 3 else stage, cmap='gray')
    ax.set_title(title)
    ax.axis('off')
plt.show()
```

---

## 🔗 Integration

### Tích hợp với OCR pipeline

```python
from modules.DCAD.app import DocumentEnhancer
from modules.daft.ocr.gemini_client import GeminiOCR

# Step 1: Enhance image
enhancer = DocumentEnhancer()
enhanced, metadata = enhancer.process('document.jpg')

# Step 2: OCR
ocr = GeminiOCR()
text = ocr.extract_text(enhanced)

print(f"Processing time: {metadata['processing_time_ms']} ms")
print(f"Extracted text: {text}")
```

### Tích hợp với Signature Detection

```python
from modules.DCAD.app import DocumentEnhancer
from modules.SAS.detector import SignatureDetector

# Step 1: Enhance
enhancer = DocumentEnhancer()
enhanced, metadata = enhancer.process('contract.jpg')

# Step 2: Detect signatures
detector = SignatureDetector()
signatures = detector.detect(enhanced)

print(f"Found {len(signatures)} signatures")
```

### Tích hợp vào Web API

```python
from fastapi import FastAPI, UploadFile
from modules.DCAD.app import DocumentEnhancer
import cv2
import numpy as np

app = FastAPI()
enhancer = DocumentEnhancer()

@app.post("/enhance")
async def enhance_document(file: UploadFile):
    # Read file
    contents = await file.read()
    nparr = np.frombuffer(contents, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    
    # Process
    enhanced, metadata = enhancer.process(img)
    
    # Encode to bytes
    _, buffer = cv2.imencode('.png', enhanced)
    
    return {
        "image": buffer.tobytes(),
        "metadata": metadata
    }
```

---

## ⚡ Performance

### Benchmarks

Thời gian xử lý trung bình (Intel Core i7, 16GB RAM):

| Image Size | Processing Time | Notes |
|------------|----------------|-------|
| 1920x1080 | ~500 ms | Camera photo |
| 2480x3508 (A4@300dpi) | ~800 ms | Scan |
| 3840x2160 (4K) | ~1200 ms | High-res |

### Optimization Tips

**Để tăng tốc:**
1. **Disable các stage không cần:**
   ```python
   enhancer = DocumentEnhancer(
       enable_crop=False,    # Nếu ảnh đã crop sẵn
       enable_deskew=False   # Nếu ảnh đã thẳng
   )
   ```

2. **Giảm resolution trước khi xử lý:**
   ```python
   # Resize xuống trước
   small = cv2.resize(img, None, fx=0.5, fy=0.5)
   enhanced, _ = enhancer.process(small)
   ```

3. **Batch processing thay vì từng ảnh:**
   ```python
   # Nhanh hơn vì reuse enhancer instance
   results = enhancer.process_batch(image_paths)
   ```

---

## 🐛 Troubleshooting

### Vấn đề thường gặp

**1. ImportError: No module named 'cv2'**
```bash
pip install opencv-python
```

**2. pdf2image: PDFPageCountError**
```bash
# MacOS
brew install poppler

# Ubuntu
sudo apt-get install poppler-utils

# Windows: Download poppler binary
```

**3. Ảnh bị xoay ngược**

Điều chỉnh `MAX_SKEW_ANGLE` trong `config.py`:
```python
MAX_SKEW_ANGLE = 10  # Giảm xuống nếu xoay quá nhiều
```

**4. Crop không chính xác**

Điều chỉnh edge detection:
```python
CANNY_THRESHOLD1 = 30  # Thử giảm xuống
CANNY_THRESHOLD2 = 100
```

**5. Ảnh quá tối sau enhance**

Tăng CLAHE:
```python
CLAHE_CLIP_LIMIT = 3.0  # Tăng lên
```

**6. Nhiễu nhiều sau OCR prep**

Tăng median filter:
```python
MEDIAN_KERNEL_SIZE = 5  # Tăng lên (phải là số lẻ)
```

---

## 📄 License

MIT License - Tự do sử dụng cho mục đích cá nhân và thương mại.

---

## 🙋 Support

Nếu gặp vấn đề hoặc có câu hỏi:

1. Kiểm tra [Troubleshooting](#-troubleshooting)
2. Xem [Examples](#-examples)
3. Review [API Reference](#-api-reference)

---

## 🔮 Roadmap

Tính năng sắp tới:

- [ ] Dewarp (sửa méo cho sách)
- [ ] Shadow removal (loại bóng thông minh)
- [ ] Text line detection
- [ ] Multi-language support trong docs
- [ ] GPU acceleration với CUDA
- [ ] Pre-trained models cho crop & deskew
- [ ] Web interface demo

---

**Made with ❤️ for CORE AI HUB**
