# core-2026

Pipeline OCR với Gemini (server/batch) cho PDF và ảnh, giữ cấu trúc bố cục.

## Yêu cầu
- Python 3.10+
- Poppler (để `pdf2image` convert PDF → image, trên macOS có thể `brew install poppler`)
- Thiết lập biến môi trường `GEMINI_API_KEY` (hoặc dùng file `.env`)

## Cài đặt
```bash
pip install -r requirements.txt
```

## Chạy pipeline mẫu
```bash
python -m src.pipeline data-test/1.pdf --output outputs
```
- Kết quả:  
  - `outputs/<file_stem>/ocr.json`: JSON chứa bố cục + text.  
  - `outputs/<file_stem>/text.txt`: Text phẳng.  
  - `outputs/<file_stem>/layout.md`: Markdown giữ trật tự đọc/block/bảng.
  - `outputs/<file_stem>/pages/`: Ảnh sau tiền xử lý mỗi trang.

Chạy batch trên thư mục:
```bash
python -m src.pipeline data-test --batch --output outputs
```

Đánh giá nhanh CER/WER khi có ground-truth:
```bash
python -m src.eval.benchmark --ref refs_dir --hyp outputs/<file_stem>
```

## Cấu trúc chính
- `src/config.py`: cấu hình (API key, model, dpi, kích thước ảnh).  
- `src/ingest/pdf_utils.py`: tách PDF thành ảnh trang.  
- `src/preprocess/image_ops.py`: tiền xử lý ảnh (resize, denoise, deskew nhẹ).  
- `src/ocr/gemini_client.py`: gọi Gemini với prompt JSON cấu trúc.  
- `src/postprocess/layout.py`: chuẩn hóa bbox, ghép text.  
- `src/store/storage.py`: lưu ảnh, JSON, text.  
- `src/pipeline.py`: CLI orchestrator.

## Ghi chú
- Mặc định model `gemini-1.5-pro`; có thể đổi qua biến `GEMINI_MODEL_NAME`.  
- Giới hạn ảnh `OCR_MAX_IMAGE_DIM` (mặc định 1600) để giảm token.  
- `OCR_RETRY_TIMES`, `OCR_RETRY_BACKOFF` để cấu hình retry khi gọi API.