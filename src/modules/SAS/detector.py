"""
Core detection logic cho seal và signature detection
Sử dụng ModelManager để reuse model instance
"""
import time
import torch
from typing import Optional, List, Tuple
from PIL import Image

from .model_manager import ModelManager
from .models import DetectionResult
from .utils import (
    resize_image_if_needed,
    create_detection_prompt,
    parse_response,
    draw_boxes
)


def timer(func):
    """Timer decorator"""
    def wrapper(*args, **kwargs):
        start = time.perf_counter()
        result = func(*args, **kwargs)
        end = time.perf_counter()
        print(f"[TIMER] {func.__name__} took {end - start:.3f} seconds")
        return result
    return wrapper


class SealSignatureDetector:
    """Class để nhận diện và định vị con dấu và chữ ký trong ảnh"""
    
    def __init__(
        self,
        model_path: str = "deepseek-ai/deepseek-vl2-tiny",
        device: Optional[torch.device] = None,
        max_new_tokens: int = 1024,
        max_image_size: Optional[int] = None,
        low_memory_mode: bool = False,
        use_multi_gpu: bool = True,
        device_map: Optional[str] = None,
    ):
        """
        Khởi tạo detector
        
        Args:
            model_path: Đường dẫn đến model DeepSeek-VL2
            device: Thiết bị để chạy model (cuda/cpu). Nếu None thì tự động phát hiện
            max_new_tokens: Số token tối đa cho response
            max_image_size: Kích thước tối đa của ảnh (resize nếu lớn hơn). None = không resize
            low_memory_mode: Nếu True, sẽ clear cache sau mỗi step và resize ảnh tự động
            use_multi_gpu: Nếu True và có nhiều GPU, sẽ tự động phân tán model trên các GPU
            device_map: Device map strategy ("auto", "balanced", "balanced_low_0", hoặc dict)
        """
        self.model_path = model_path
        self.max_new_tokens = max_new_tokens
        self.max_image_size = max_image_size
        self.low_memory_mode = low_memory_mode
        
        # Lấy ModelManager instance (singleton)
        self.model_manager = ModelManager()
        
        # Load model nếu chưa được load
        if not self.model_manager.is_loaded():
            self.model_manager.load_model(
                model_path=model_path,
                device=device,
                max_image_size=max_image_size,
                low_memory_mode=low_memory_mode,
                use_multi_gpu=use_multi_gpu,
                device_map=device_map,
            )
        
        # Lấy các components từ model manager
        self.model = self.model_manager.get_model()
        self.processor = self.model_manager.get_processor()
        self.tokenizer = self.model_manager.get_tokenizer()
        self.device = self.model_manager.get_device()
        self.num_gpus = self.model_manager.get_num_gpus()
    
    def _clear_cuda_cache(self):
        """Clear CUDA cache"""
        self.model_manager.clear_cache()
    
    @timer
    def detect(
        self,
        image_path: str,
        language: str = "vietnamese",
        return_image: bool = False,
    ) -> DetectionResult:
        """
        Nhận diện con dấu và chữ ký trong ảnh
        
        Args:
            image_path: Đường dẫn đến file ảnh
            language: Ngôn ngữ của prompt (vietnamese/english)
            return_image: Nếu True, trả về thêm PIL Image đã được vẽ bounding boxes
        
        Returns:
            DetectionResult object hoặc tuple (DetectionResult, PIL.Image) nếu return_image=True
        """
        # Clear cache trước khi xử lý (nếu low memory mode)
        if self.low_memory_mode:
            self._clear_cuda_cache()
        
        # Load image
        image = Image.open(image_path).convert("RGB")
        
        # Resize ảnh nếu cần
        image = resize_image_if_needed(
            image,
            max_image_size=self.max_image_size,
            low_memory_mode=self.low_memory_mode,
            num_gpus=self.num_gpus
        )
        img_width, img_height = image.size
        
        # Tạo prompt
        prompt_text = create_detection_prompt(language=language)
        
        # Tạo conversation
        conversation = [
            {
                "role": "<|User|>",
                "content": f"<image>\n{prompt_text}",
                "images": [image],
            },
            {"role": "<|Assistant|>", "content": ""},
        ]
        
        # Process inputs
        model_inputs = self.processor(
            conversations=conversation,
            images=[image],
            force_batchify=True,
            system_prompt="You are an expert in document analysis and object detection."
        )
        
        # Move inputs to device
        dtype = torch.float16 if self.device.type == "cuda" else torch.float32
        
        # Xác định device để move inputs
        input_device = self.device
        if hasattr(self.model, "hf_device_map") and self.model.hf_device_map:
            first_device_value = list(self.model.hf_device_map.values())[0]
            if isinstance(first_device_value, (int, str)):
                if isinstance(first_device_value, int):
                    input_device = torch.device(f"cuda:{first_device_value}")
                else:
                    input_device = torch.device(first_device_value)
            elif isinstance(first_device_value, list) and len(first_device_value) > 0:
                first_dev = first_device_value[0]
                if isinstance(first_dev, int):
                    input_device = torch.device(f"cuda:{first_dev}")
                elif isinstance(first_dev, str):
                    input_device = torch.device(first_dev)
        
        model_inputs["images"] = model_inputs["images"].to(input_device, dtype=dtype)
        model_inputs["images_spatial_crop"] = model_inputs["images_spatial_crop"].to(input_device)
        model_inputs["images_seq_mask"] = model_inputs["images_seq_mask"].to(input_device)
        model_inputs["input_ids"] = model_inputs["input_ids"].to(input_device)
        model_inputs["attention_mask"] = model_inputs["attention_mask"].to(input_device)
        
        # Clear cache trước khi prepare inputs
        if self.low_memory_mode:
            self._clear_cuda_cache()
        
        # Generate response
        try:
            with torch.no_grad():
                inputs_embeds = self.model.prepare_inputs_embeds(**model_inputs)
                
                if self.low_memory_mode:
                    self._clear_cuda_cache()
                
                attention_mask = model_inputs["attention_mask"]
                
                outputs = self.model.language.generate(
                    inputs_embeds=inputs_embeds,
                    attention_mask=attention_mask,
                    max_new_tokens=self.max_new_tokens,
                    pad_token_id=self.tokenizer.eos_token_id,
                    eos_token_id=self.tokenizer.eos_token_id,
                    do_sample=False,
                    use_cache=True,
                )
            
            # Decode response
            response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            # Clear memory sau khi generate
            del inputs_embeds
            del outputs
            del attention_mask
            if self.low_memory_mode:
                self._clear_cuda_cache()
                
        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                print(f"❌ CUDA out of memory error occurred!")
                print(f"💡 Suggestions:")
                print(f"   1. Use low_memory_mode=True when initializing detector")
                print(f"   2. Set max_image_size to a smaller value (e.g., 1024)")
                print(f"   3. Process smaller images or reduce max_new_tokens")
                if self.device.type == "cuda":
                    self._clear_cuda_cache()
                raise
            else:
                raise
        
        print("\n========== MODEL RESPONSE ==========\n")
        print(response)
        print("\n====================================\n")
        
        # Parse response
        result = parse_response(response, img_width=img_width, img_height=img_height)
        
        # Set width/height nếu chưa có
        if result.width == 0:
            result.width = img_width
        if result.height == 0:
            result.height = img_height
        
        # Validate và normalize coordinates
        for bbox in result.get_all():
            bbox.x1 = max(0, min(bbox.x1, img_width - 1))
            bbox.y1 = max(0, min(bbox.y1, img_height - 1))
            bbox.x2 = max(bbox.x1 + 1, min(bbox.x2, img_width))
            bbox.y2 = max(bbox.y1 + 1, min(bbox.y2, img_height))
        
        if return_image:
            annotated_image = draw_boxes(image.copy(), result)
            return result, annotated_image
        
        return result
    
    def detect_batch(
        self,
        image_paths: List[str],
        language: str = "vietnamese",
        return_images: bool = False,
    ) -> List[Tuple[DetectionResult, Optional[Image.Image]]]:
        """
        Nhận diện con dấu và chữ ký cho nhiều ảnh
        
        Args:
            image_paths: Danh sách đường dẫn đến các file ảnh
            language: Ngôn ngữ của prompt
            return_images: Nếu True, trả về thêm các PIL Images đã được vẽ bounding boxes
        
        Returns:
            List of (DetectionResult, Optional[PIL.Image]) tuples
        """
        results = []
        for i, image_path in enumerate(image_paths, 1):
            print(f"\n[{i}/{len(image_paths)}] Processing: {image_path}")
            try:
                if return_images:
                    result, annotated_image = self.detect(
                        image_path=image_path,
                        language=language,
                        return_image=True
                    )
                    results.append((result, annotated_image))
                else:
                    result = self.detect(
                        image_path=image_path,
                        language=language,
                        return_image=False
                    )
                    results.append((result, None))
            except Exception as e:
                print(f"❌ Error processing {image_path}: {e}")
                # Tạo empty result để giữ index
                empty_result = DetectionResult()
                results.append((empty_result, None))
        
        return results
    
    def save_result_image(
        self,
        image_path: str,
        result: DetectionResult,
        output_path: str,
        seal_color: str = "red",
        signature_color: str = "blue",
    ):
        """
        Lưu ảnh đã được vẽ bounding boxes
        
        Args:
            image_path: Đường dẫn đến ảnh gốc
            result: DetectionResult
            output_path: Đường dẫn để lưu ảnh kết quả
            seal_color: Màu để vẽ con dấu
            signature_color: Màu để vẽ chữ ký
        """
        image = Image.open(image_path).convert("RGB")
        annotated_image = draw_boxes(image, result, seal_color, signature_color)
        annotated_image.save(output_path)
        print(f"✅ Saved annotated image to {output_path}")
    
    def release_memory(self):
        """Giải phóng model khỏi GPU"""
        self.model_manager.release_memory()
