"""
Model Manager với Singleton pattern để quản lý model lifecycle
Cho phép reuse model instance thay vì load lại mỗi lần
"""
import torch
from typing import Optional
from transformers import AutoModelForCausalLM
from deepseek_vl2.models import DeepseekVLV2Processor

# Try to import accelerate for multi-GPU support
try:
    from accelerate import init_empty_weights, load_checkpoint_and_dispatch
    ACCELERATE_AVAILABLE = True
except ImportError:
    ACCELERATE_AVAILABLE = False


class ModelManager:
    """
    Singleton class để quản lý model instance
    Đảm bảo model chỉ được load một lần và reuse cho nhiều requests
    """
    _instance = None
    _model = None
    _processor = None
    _tokenizer = None
    _device = None
    _num_gpus = 0
    _is_loaded = False
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(ModelManager, cls).__new__(cls)
        return cls._instance
    
    def __init__(self):
        if not hasattr(self, '_initialized'):
            self._initialized = True
            self._model = None
            self._processor = None
            self._tokenizer = None
            self._device = None
            self._num_gpus = 0
            self._is_loaded = False
    
    def load_model(
        self,
        model_path: str = "deepseek-ai/deepseek-vl2-tiny",
        device: Optional[torch.device] = None,
        max_image_size: Optional[int] = None,
        low_memory_mode: bool = False,
        use_multi_gpu: bool = True,
        device_map: Optional[str] = None,
    ):
        """
        Load model nếu chưa được load
        
        Args:
            model_path: Đường dẫn đến model
            device: Thiết bị để chạy model
            max_image_size: Kích thước tối đa của ảnh
            low_memory_mode: Chế độ tiết kiệm memory
            use_multi_gpu: Sử dụng multi-GPU
            device_map: Device map strategy
        """
        if self._is_loaded:
            print("✅ Model already loaded, reusing existing instance")
            return
        
        print(f"📦 Loading model from {model_path}...")
        
        # Device setup
        if device is None:
            if torch.cuda.is_available():
                self._device = torch.device("cuda")
                torch.backends.cuda.matmul.allow_tf32 = True
            else:
                self._device = torch.device("cpu")
        else:
            self._device = device
        
        self._num_gpus = torch.cuda.device_count() if torch.cuda.is_available() else 0
        
        print(f"🔧 Using device: {self._device}")
        if self._num_gpus > 1:
            print(f"🎯 Found {self._num_gpus} GPUs available")
            if use_multi_gpu:
                print(f"✅ Multi-GPU mode enabled")
        
        # Clear cache trước khi load
        if self._device.type == "cuda":
            print("🧹 Clearing CUDA cache before loading model...")
            self._clear_cuda_cache()
            import os
            if "PYTORCH_CUDA_ALLOC_CONF" not in os.environ:
                os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:512"
                print("⚙️  Set PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512")
        
        # Load processor
        self._processor = DeepseekVLV2Processor.from_pretrained(model_path)
        self._tokenizer = self._processor.tokenizer
        
        # Load model
        dtype = torch.float16 if self._device.type == "cuda" else torch.float32
        
        # Multi-GPU support
        if use_multi_gpu and self._num_gpus > 1 and ACCELERATE_AVAILABLE:
            device_map_strategy = device_map if device_map else "auto"
            print(f"🚀 Loading model with device_map='{device_map_strategy}' across {self._num_gpus} GPUs...")
            
            try:
                # Tính toán max memory
                max_memory = {}
                for i in range(self._num_gpus):
                    gpu_memory_gb = torch.cuda.get_device_properties(i).total_memory / (1024**3)
                    reserve_gb = 3 if low_memory_mode else 2
                    max_memory[i] = f"{int(gpu_memory_gb - reserve_gb)}GiB"
                max_memory["cpu"] = "30GiB"
                
                load_kwargs = {
                    "trust_remote_code": True,
                    "torch_dtype": dtype,
                    "device_map": device_map_strategy,
                    "low_cpu_mem_usage": True,
                    "max_memory": max_memory,
                }
                
                if low_memory_mode:
                    print(f"💾 Using max_memory limits: {max_memory}")
                
                print("📥 Loading model weights...")
                self._model = AutoModelForCausalLM.from_pretrained(
                    model_path,
                    **load_kwargs
                ).eval()
                
                self._clear_cuda_cache()
                print("✅ Model loaded successfully with multi-GPU distribution!")
                
                if hasattr(self._model, "hf_device_map"):
                    print(f"📊 Device map: {self._model.hf_device_map}")
                
            except Exception as e:
                print(f"⚠️  Failed to load with device_map: {type(e).__name__}")
                print(f"   Clearing cache and trying alternative strategies...")
                self._clear_cuda_cache()
                
                # Try alternative strategies
                alternative_strategies = []
                if device_map_strategy != "balanced_low_0":
                    alternative_strategies.append("balanced_low_0")
                if device_map_strategy != "balanced":
                    alternative_strategies.append("balanced")
                
                loaded = False
                for alt_strategy in alternative_strategies:
                    try:
                        print(f"🔄 Retrying with device_map='{alt_strategy}'...")
                        load_kwargs_alt = {
                            "trust_remote_code": True,
                            "torch_dtype": dtype,
                            "device_map": alt_strategy,
                            "low_cpu_mem_usage": True,
                            "max_memory": max_memory,
                        }
                        self._model = AutoModelForCausalLM.from_pretrained(
                            model_path,
                            **load_kwargs_alt
                        ).eval()
                        self._clear_cuda_cache()
                        print(f"✅ Model loaded successfully with device_map='{alt_strategy}'!")
                        if hasattr(self._model, "hf_device_map"):
                            print(f"📊 Device map: {self._model.hf_device_map}")
                        loaded = True
                        break
                    except Exception as e_alt:
                        print(f"   ⚠️  Failed with '{alt_strategy}': {type(e_alt).__name__}")
                        self._clear_cuda_cache()
                        continue
                
                if not loaded:
                    print("🔄 All device_map strategies failed. Loading on single GPU...")
                    self._clear_cuda_cache()
                    self._model = AutoModelForCausalLM.from_pretrained(
                        model_path,
                        trust_remote_code=True,
                        torch_dtype=dtype,
                    ).to(self._device).eval()
                    print("✅ Model loaded successfully on single GPU!")
        else:
            # Single GPU hoặc CPU
            if use_multi_gpu and self._num_gpus > 1:
                print(f"⚠️  Multi-GPU requested but accelerate not available. Using single GPU.")
            
            self._model = AutoModelForCausalLM.from_pretrained(
                model_path,
                trust_remote_code=True,
                torch_dtype=dtype,
            ).to(self._device).eval()
            print("✅ Model loaded successfully!")
        
        self._is_loaded = True
    
    def get_model(self):
        """Lấy model instance"""
        if not self._is_loaded:
            raise RuntimeError("Model not loaded. Call load_model() first.")
        return self._model
    
    def get_processor(self):
        """Lấy processor instance"""
        if not self._is_loaded:
            raise RuntimeError("Model not loaded. Call load_model() first.")
        return self._processor
    
    def get_tokenizer(self):
        """Lấy tokenizer instance"""
        if not self._is_loaded:
            raise RuntimeError("Model not loaded. Call load_model() first.")
        return self._tokenizer
    
    def get_device(self):
        """Lấy device"""
        return self._device
    
    def get_num_gpus(self):
        """Lấy số lượng GPU"""
        return self._num_gpus
    
    def is_loaded(self):
        """Kiểm tra xem model đã được load chưa"""
        return self._is_loaded
    
    def _clear_cuda_cache(self):
        """Clear CUDA cache"""
        if self._device and self._device.type == "cuda":
            if self._num_gpus > 1:
                for i in range(self._num_gpus):
                    with torch.cuda.device(i):
                        torch.cuda.empty_cache()
                        torch.cuda.synchronize()
            else:
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
    
    def release_memory(self):
        """Giải phóng model khỏi GPU"""
        if self._model is not None:
            del self._model
        if self._processor is not None:
            del self._processor
        if self._tokenizer is not None:
            del self._tokenizer
        self._clear_cuda_cache()
        self._is_loaded = False
        print("✅ Memory released")
    
    def clear_cache(self):
        """Clear CUDA cache (giữ lại model)"""
        self._clear_cuda_cache()
