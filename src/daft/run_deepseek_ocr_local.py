import torch
from transformers import AutoModel, AutoTokenizer
import os
from PIL import Image
import sys

def main():
    # 1. Cài đặt đường dẫn và thiết lập môi trường
    model_path = "./models/DeepSeek-OCR"  # Đường dẫn tới thư mục đã clone
    image_path = "./data-test/4_1.jpg"  # Đường dẫn tới ảnh cần OCR
    output_dir = "./outputs"  # Thư mục xuất kết quả (nếu cần)
    
    # Kiểm tra xem thư mục mô hình có tồn tại không
    if not os.path.exists(model_path):
        print(f"Lỗi: Thư mục mô hình '{model_path}' không tồn tại.")
        print("Hãy chắc chắn bạn đã clone mô hình về đúng đường dẫn này.")
        sys.exit(1)
    
    # Kiểm tra ảnh tồn tại
    if not os.path.exists(image_path):
        print(f"Lỗi: Ảnh '{image_path}' không tồn tại.")
        print("Hãy thay thế bằng đường dẫn ảnh của bạn.")
        sys.exit(1)
    
    # 2. Thiết lập GPU (nếu có) - tự động phát hiện
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Sử dụng thiết bị: {device}")
    
    # Nếu dùng GPU, đặt CUDA_VISIBLE_DEVICES
    if device.type == "cuda":
        os.environ["CUDA_VISIBLE_DEVICES"] = "0"
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    else:
        print("Cảnh báo: Không tìm thấy GPU, chạy trên CPU có thể rất chậm.")
    
    # 3. Load tokenizer và mô hình từ local
    print("Đang tải tokenizer...")
    try:
        tokenizer = AutoTokenizer.from_pretrained(
            model_path,
            trust_remote_code=True,
            local_files_only=True
        )
    except Exception as e:
        print(f"Lỗi khi tải tokenizer: {e}")
        sys.exit(1)
    
    print("Đang tải mô hình...")
    model = None
    
    # Thử nhiều cách load mô hình để xử lý lỗi safetensors
    loading_strategies = [
        # Strategy 1: Với flash_attention và safetensors
        {
            "_attn_implementation": "flash_attention_2" if device.type == "cuda" else "eager",
            "use_safetensors": True,
            "description": "với flash_attention và safetensors"
        },
        # Strategy 2: Không flash_attention nhưng vẫn dùng safetensors
        {
            "use_safetensors": True,
            "description": "không flash_attention nhưng vẫn dùng safetensors"
        },
        # Strategy 3: Không dùng safetensors (fallback nếu safetensors bị lỗi)
        {
            "use_safetensors": False,
            "description": "không dùng safetensors (fallback)"
        },
        # Strategy 4: Không safetensors và không flash_attention
        {
            "_attn_implementation": "eager",
            "use_safetensors": False,
            "description": "không safetensors và eager attention"
        }
    ]
    
    for i, strategy in enumerate(loading_strategies, 1):
        try:
            print(f"Thử cách {i}: Tải mô hình {strategy['description']}...")
            load_kwargs = {
                "trust_remote_code": True,
                "local_files_only": True,
            }
            load_kwargs.update({k: v for k, v in strategy.items() if k != "description"})
            
            model = AutoModel.from_pretrained(model_path, **load_kwargs)
            print(f"✓ Tải mô hình thành công với cách {i}!")
            break
        except Exception as e:
            print(f"✗ Cách {i} thất bại: {e}")
            if i == len(loading_strategies):
                print("Tất cả các cách tải mô hình đều thất bại.")
                print("\nGợi ý:")
                print("1. Kiểm tra xem các file mô hình có bị hỏng không")
                print("2. Thử tải lại mô hình từ HuggingFace")
                print("3. Kiểm tra phiên bản safetensors: pip install --upgrade safetensors")
                sys.exit(1)
    
    if model is None:
        print("Không thể tải mô hình với bất kỳ cách nào.")
        sys.exit(1)
    
    # Chuyển mô hình tới thiết bị và chuyển kiểu dữ liệu
    model = model.to(device)
    if device.type == "cuda":
        model = model.to(torch.bfloat16)
    model = model.eval()
    print("Mô hình đã sẵn sàng.")
    
    # 4. Chuẩn bị prompt và ảnh
    prompt = "<image>\nFree OCR."  # Có thể thay đổi prompt tùy nhu cầu
    # Ví dụ: "<image>\n<|grounding|>Convert the document to markdown. "
    
    # 5. Chạy inference
    print(f"Đang xử lý ảnh: {image_path}")
    try:
        # Đảm bảo thư mục output tồn tại
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        
        # Gọi phương thức infer của mô hình
        result = model.infer(
            tokenizer, 
            prompt=prompt, 
            image_file=image_path,  # Sửa: image_file -> image_path
            output_path=output_dir,  # Sửa: output_path -> output_dir
            base_size=1024, 
            image_size=640, 
            crop_mode=True, 
            save_results=True, 
            test_compress=True
        )
        print("OCR thành công!")
        
        # In kết quả nếu có
        if result is not None:
            print("Kết quả trả về:")
            print(result)
            
            # Lưu kết quả vào file text
            output_txt_path = os.path.join(output_dir, "ocr_result.txt")
            with open(output_txt_path, "w", encoding="utf-8") as f:
                f.write(str(result))
            print(f"Đã lưu kết quả vào {output_txt_path}")
        else:
            print("Kết quả đã được lưu vào thư mục output (nếu save_results=True)")
        
    except Exception as e:
        print(f"Lỗi trong quá trình OCR: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()