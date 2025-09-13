"""Model loading and caching utilities.

This module provides utilities for loading, caching, and managing ML models
with support for edge computing and memory-efficient loading patterns.
"""

import logging
import asyncio
import hashlib
import pickle
import json
from typing import Dict, Any, Optional, Union, List, Callable, TypeVar, Generic
from pathlib import Path
from abc import ABC, abstractmethod
import weakref
import threading
from datetime import datetime, timedelta
from functools import wraps
import time

from app.core.logging import LoggerMixin
from app.core.config import settings

try:
    import torch
    import torch.nn as nn
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    nn = None

try:
    from transformers import AutoModel, AutoTokenizer
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False

try:
    import onnx
    import onnxruntime as ort
    ONNX_AVAILABLE = True
except ImportError:
    ONNX_AVAILABLE = False

T = TypeVar('T')


class ModelCacheEntry:
    """Cache entry for a loaded model."""
    
    def __init__(self, model: Any, metadata: Dict[str, Any]):
        self.model = model
        self.metadata = metadata
        self.last_access = datetime.now()
        self.access_count = 0
        self.memory_size = self._estimate_memory_size(model)
    
    def _estimate_memory_size(self, model: Any) -> int:
        """Estimate memory usage of model in bytes."""
        try:
            if TORCH_AVAILABLE and isinstance(model, nn.Module):
                # PyTorch model
                param_size = sum(p.numel() * p.element_size() for p in model.parameters())
                buffer_size = sum(b.numel() * b.element_size() for b in model.buffers())
                return param_size + buffer_size
            
            elif hasattr(model, '__sizeof__'):
                return model.__sizeof__()
            
            else:
                # Rough estimate using pickle
                return len(pickle.dumps(model))
                
        except Exception:
            return 1024 * 1024  # Default 1MB estimate
    
    def touch(self):
        """Update access time and count."""
        self.last_access = datetime.now()
        self.access_count += 1


class ModelCache(LoggerMixin):
    """Thread-safe model cache with LRU eviction and memory management."""
    
    def __init__(self, max_memory_mb: int = 2048, max_entries: int = 50):
        super().__init__()
        self._cache: Dict[str, ModelCacheEntry] = {}
        self._lock = threading.RLock()
        self.max_memory_bytes = max_memory_mb * 1024 * 1024
        self.max_entries = max_entries
        self._total_memory = 0
    
    def get(self, key: str) -> Optional[Any]:
        """Get model from cache."""
        with self._lock:
            if key in self._cache:
                entry = self._cache[key]
                entry.touch()
                self.logger.debug(f"Cache hit for model: {key}")
                return entry.model
            
            self.logger.debug(f"Cache miss for model: {key}")
            return None
    
    def put(self, key: str, model: Any, metadata: Dict[str, Any] = None) -> bool:
        """Put model in cache with eviction if needed."""
        if metadata is None:
            metadata = {}
        
        with self._lock:
            entry = ModelCacheEntry(model, metadata)
            
            # Check if we need to evict entries
            if self._needs_eviction(entry.memory_size):
                self._evict_entries(entry.memory_size)
            
            # Add entry if we have space
            if len(self._cache) < self.max_entries and \
               (self._total_memory + entry.memory_size) <= self.max_memory_bytes:
                
                self._cache[key] = entry
                self._total_memory += entry.memory_size
                
                self.logger.debug(
                    f"Cached model {key} (size: {entry.memory_size / 1024 / 1024:.1f}MB, "
                    f"total: {self._total_memory / 1024 / 1024:.1f}MB)"
                )
                return True
            
            self.logger.warning(f"Failed to cache model {key}: insufficient space")
            return False
    
    def _needs_eviction(self, new_size: int) -> bool:
        """Check if eviction is needed for new entry."""
        return (len(self._cache) >= self.max_entries or 
                (self._total_memory + new_size) > self.max_memory_bytes)
    
    def _evict_entries(self, needed_size: int):
        """Evict entries using LRU policy."""
        if not self._cache:
            return
        
        # Sort by last access time (oldest first)
        sorted_entries = sorted(
            self._cache.items(),
            key=lambda x: x[1].last_access
        )
        
        freed_space = 0
        entries_to_remove = []
        
        for key, entry in sorted_entries:
            if (freed_space >= needed_size and 
                len(self._cache) - len(entries_to_remove) < self.max_entries):
                break
            
            entries_to_remove.append(key)
            freed_space += entry.memory_size
        
        # Remove entries
        for key in entries_to_remove:
            entry = self._cache.pop(key)
            self._total_memory -= entry.memory_size
            self.logger.debug(f"Evicted model {key} from cache")
    
    def clear(self):
        """Clear all cached models."""
        with self._lock:
            self._cache.clear()
            self._total_memory = 0
            self.logger.info("Cleared model cache")
    
    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        with self._lock:
            return {
                "entries": len(self._cache),
                "max_entries": self.max_entries,
                "memory_used_mb": self._total_memory / 1024 / 1024,
                "max_memory_mb": self.max_memory_bytes / 1024 / 1024,
                "memory_usage_percent": (self._total_memory / self.max_memory_bytes) * 100,
                "models": {
                    key: {
                        "memory_mb": entry.memory_size / 1024 / 1024,
                        "last_access": entry.last_access.isoformat(),
                        "access_count": entry.access_count
                    }
                    for key, entry in self._cache.items()
                }
            }


class BaseModelLoader(ABC, LoggerMixin):
    """Abstract base class for model loaders."""
    
    def __init__(self, cache: Optional[ModelCache] = None):
        super().__init__()
        self.cache = cache or ModelCache()
    
    @abstractmethod
    async def load_model(self, model_path: Union[str, Path], **kwargs) -> Any:
        """Load model from path."""
        pass
    
    @abstractmethod
    def get_model_info(self, model: Any) -> Dict[str, Any]:
        """Get model information."""
        pass
    
    def _generate_cache_key(self, model_path: Union[str, Path], **kwargs) -> str:
        """Generate cache key for model."""
        path_str = str(model_path)
        kwargs_str = json.dumps(kwargs, sort_keys=True)
        combined = f"{path_str}:{kwargs_str}"
        return hashlib.md5(combined.encode()).hexdigest()


class TorchModelLoader(BaseModelLoader):
    """PyTorch model loader with caching."""
    
    async def load_model(
        self,
        model_path: Union[str, Path],
        device: Optional[str] = None,
        **kwargs
    ) -> Any:
        """Load PyTorch model."""
        
        if not TORCH_AVAILABLE:
            raise RuntimeError("PyTorch not available")
        
        # Generate cache key
        cache_key = self._generate_cache_key(model_path, device=device, **kwargs)
        
        # Check cache first
        cached_model = self.cache.get(cache_key)
        if cached_model is not None:
            return cached_model
        
        try:
            # Determine device
            if device is None:
                device = "cuda" if torch.cuda.is_available() else "cpu"
            
            # Load model in thread pool
            model = await asyncio.get_event_loop().run_in_executor(
                None,
                self._load_torch_model_sync,
                model_path, device, kwargs
            )
            
            # Cache the model
            metadata = {
                "loader_type": "torch",
                "device": device,
                "model_path": str(model_path),
                "load_time": datetime.now().isoformat()
            }
            
            self.cache.put(cache_key, model, metadata)
            
            return model
            
        except Exception as e:
            self.logger.error(f"Failed to load PyTorch model {model_path}: {e}")
            raise
    
    def _load_torch_model_sync(self, model_path: Union[str, Path], device: str, kwargs: Dict) -> Any:
        """Load PyTorch model synchronously."""
        
        path = Path(model_path)
        
        if path.suffix == '.pt' or path.suffix == '.pth':
            # Load state dict or full model
            model = torch.load(path, map_location=device, **kwargs)
        else:
            raise ValueError(f"Unsupported PyTorch model format: {path.suffix}")
        
        if hasattr(model, 'eval'):
            model.eval()
        
        return model
    
    def get_model_info(self, model: Any) -> Dict[str, Any]:
        """Get PyTorch model information."""
        
        if not TORCH_AVAILABLE or not isinstance(model, nn.Module):
            return {"type": "unknown"}
        
        try:
            total_params = sum(p.numel() for p in model.parameters())
            trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
            
            return {
                "type": "pytorch",
                "class_name": model.__class__.__name__,
                "total_parameters": total_params,
                "trainable_parameters": trainable_params,
                "device": str(next(model.parameters()).device),
                "dtype": str(next(model.parameters()).dtype)
            }
            
        except Exception as e:
            self.logger.warning(f"Failed to get model info: {e}")
            return {"type": "pytorch", "error": str(e)}


class TransformersModelLoader(BaseModelLoader):
    """Hugging Face Transformers model loader."""
    
    async def load_model(
        self,
        model_name: str,
        model_type: str = "auto",
        device: Optional[str] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """Load Transformers model and tokenizer."""
        
        if not TRANSFORMERS_AVAILABLE:
            raise RuntimeError("Transformers not available")
        
        # Generate cache key
        cache_key = self._generate_cache_key(
            model_name, 
            model_type=model_type, 
            device=device, 
            **kwargs
        )
        
        # Check cache
        cached_model = self.cache.get(cache_key)
        if cached_model is not None:
            return cached_model
        
        try:
            # Load model and tokenizer in thread pool
            model_dict = await asyncio.get_event_loop().run_in_executor(
                None,
                self._load_transformers_model_sync,
                model_name, model_type, device, kwargs
            )
            
            # Cache the model
            metadata = {
                "loader_type": "transformers",
                "model_name": model_name,
                "model_type": model_type,
                "device": device,
                "load_time": datetime.now().isoformat()
            }
            
            self.cache.put(cache_key, model_dict, metadata)
            
            return model_dict
            
        except Exception as e:
            self.logger.error(f"Failed to load Transformers model {model_name}: {e}")
            raise
    
    def _load_transformers_model_sync(
        self,
        model_name: str,
        model_type: str,
        device: Optional[str],
        kwargs: Dict
    ) -> Dict[str, Any]:
        """Load Transformers model synchronously."""
        
        # Load tokenizer
        tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            cache_dir=settings.MODEL_CACHE_DIR,
            **kwargs
        )
        
        # Load model
        model = AutoModel.from_pretrained(
            model_name,
            cache_dir=settings.MODEL_CACHE_DIR,
            **kwargs
        )
        
        # Move to device if specified
        if device:
            if device == "cuda" and torch.cuda.is_available():
                model = model.cuda()
            elif device == "cpu":
                model = model.cpu()
        
        model.eval()
        
        return {
            "model": model,
            "tokenizer": tokenizer,
            "config": model.config
        }
    
    def get_model_info(self, model_dict: Dict[str, Any]) -> Dict[str, Any]:
        """Get Transformers model information."""
        
        try:
            model = model_dict.get("model")
            tokenizer = model_dict.get("tokenizer")
            config = model_dict.get("config")
            
            info = {
                "type": "transformers",
                "model_type": config.model_type if config else "unknown",
                "vocab_size": len(tokenizer) if tokenizer else 0
            }
            
            if TORCH_AVAILABLE and hasattr(model, "parameters"):
                info.update({
                    "total_parameters": sum(p.numel() for p in model.parameters()),
                    "device": str(next(model.parameters()).device)
                })
            
            return info
            
        except Exception as e:
            return {"type": "transformers", "error": str(e)}


class ONNXModelLoader(BaseModelLoader):
    """ONNX model loader with ONNXRuntime."""
    
    async def load_model(
        self,
        model_path: Union[str, Path],
        providers: Optional[List[str]] = None,
        **kwargs
    ) -> Any:
        """Load ONNX model."""
        
        if not ONNX_AVAILABLE:
            raise RuntimeError("ONNX/ONNXRuntime not available")
        
        # Generate cache key
        cache_key = self._generate_cache_key(
            model_path,
            providers=providers,
            **kwargs
        )
        
        # Check cache
        cached_model = self.cache.get(cache_key)
        if cached_model is not None:
            return cached_model
        
        try:
            # Load model in thread pool
            session = await asyncio.get_event_loop().run_in_executor(
                None,
                self._load_onnx_model_sync,
                model_path, providers, kwargs
            )
            
            # Cache the session
            metadata = {
                "loader_type": "onnx",
                "model_path": str(model_path),
                "providers": providers,
                "load_time": datetime.now().isoformat()
            }
            
            self.cache.put(cache_key, session, metadata)
            
            return session
            
        except Exception as e:
            self.logger.error(f"Failed to load ONNX model {model_path}: {e}")
            raise
    
    def _load_onnx_model_sync(
        self,
        model_path: Union[str, Path],
        providers: Optional[List[str]],
        kwargs: Dict
    ) -> ort.InferenceSession:
        """Load ONNX model synchronously."""
        
        # Set default providers
        if providers is None:
            available_providers = ort.get_available_providers()
            providers = []
            
            # Prefer GPU providers
            if "CUDAExecutionProvider" in available_providers:
                providers.append("CUDAExecutionProvider")
            if "CPUExecutionProvider" in available_providers:
                providers.append("CPUExecutionProvider")
        
        # Create inference session
        session = ort.InferenceSession(
            str(model_path),
            providers=providers,
            **kwargs
        )
        
        return session
    
    def get_model_info(self, session: ort.InferenceSession) -> Dict[str, Any]:
        """Get ONNX model information."""
        
        try:
            return {
                "type": "onnx",
                "providers": session.get_providers(),
                "input_names": [input.name for input in session.get_inputs()],
                "output_names": [output.name for output in session.get_outputs()],
                "input_shapes": {
                    input.name: input.shape for input in session.get_inputs()
                },
                "output_shapes": {
                    output.name: output.shape for output in session.get_outputs()
                }
            }
            
        except Exception as e:
            return {"type": "onnx", "error": str(e)}


class ModelLoader(LoggerMixin):
    """Unified model loader with automatic format detection."""
    
    def __init__(self, cache: Optional[ModelCache] = None):
        super().__init__()
        self.cache = cache or ModelCache()
        
        # Initialize format-specific loaders
        self.torch_loader = TorchModelLoader(self.cache)
        self.transformers_loader = TransformersModelLoader(self.cache)
        self.onnx_loader = ONNXModelLoader(self.cache)
    
    async def load_model(
        self,
        model_path_or_name: Union[str, Path],
        model_format: Optional[str] = None,
        **kwargs
    ) -> Any:
        """Load model with automatic format detection."""
        
        try:
            # Auto-detect format if not specified
            if model_format is None:
                model_format = self._detect_model_format(model_path_or_name)
            
            # Load using appropriate loader
            if model_format == "torch":
                return await self.torch_loader.load_model(model_path_or_name, **kwargs)
            elif model_format == "transformers":
                return await self.transformers_loader.load_model(model_path_or_name, **kwargs)
            elif model_format == "onnx":
                return await self.onnx_loader.load_model(model_path_or_name, **kwargs)
            else:
                raise ValueError(f"Unsupported model format: {model_format}")
                
        except Exception as e:
            self.logger.error(f"Failed to load model {model_path_or_name}: {e}")
            raise
    
    def _detect_model_format(self, model_path_or_name: Union[str, Path]) -> str:
        """Auto-detect model format."""
        
        if isinstance(model_path_or_name, (str, Path)):
            path = Path(model_path_or_name)
            
            # Check file extension
            if path.suffix in ['.pt', '.pth']:
                return "torch"
            elif path.suffix in ['.onnx']:
                return "onnx"
            elif path.is_dir() or '/' in str(path):
                # Could be Transformers model path/name
                return "transformers"
        
        # Default to transformers for model names
        return "transformers"
    
    def get_model_info(self, model: Any, model_format: Optional[str] = None) -> Dict[str, Any]:
        """Get model information."""
        
        if model_format == "torch" or (TORCH_AVAILABLE and isinstance(model, nn.Module)):
            return self.torch_loader.get_model_info(model)
        elif model_format == "transformers" or isinstance(model, dict):
            return self.transformers_loader.get_model_info(model)
        elif model_format == "onnx" or (ONNX_AVAILABLE and isinstance(model, ort.InferenceSession)):
            return self.onnx_loader.get_model_info(model)
        else:
            return {"type": "unknown"}
    
    def get_cache_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        return self.cache.get_stats()
    
    def clear_cache(self):
        """Clear model cache."""
        self.cache.clear()


def cached_model(cache_ttl: int = 3600):
    """Decorator for caching model inference results."""
    
    def decorator(func: Callable) -> Callable:
        cache = {}
        cache_times = {}
        
        @wraps(func)
        async def wrapper(*args, **kwargs):
            # Generate cache key
            key = f"{func.__name__}:{hash((args, tuple(sorted(kwargs.items()))))}"
            
            # Check cache
            if key in cache:
                cache_time = cache_times[key]
                if (datetime.now() - cache_time).seconds < cache_ttl:
                    return cache[key]
                else:
                    # Remove expired entry
                    del cache[key]
                    del cache_times[key]
            
            # Execute function
            result = await func(*args, **kwargs)
            
            # Cache result
            cache[key] = result
            cache_times[key] = datetime.now()
            
            return result
            
        return wrapper
    return decorator
