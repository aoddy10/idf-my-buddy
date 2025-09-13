"""Edge computing utilities for optimized model inference.

This module provides utilities for optimizing ML model inference for edge devices
with support for quantization, pruning, and hardware acceleration.
"""

import logging
import asyncio
import time
from typing import Dict, Any, Optional, List, Tuple, Union, Callable
from pathlib import Path
from abc import ABC, abstractmethod
import multiprocessing
import threading
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
import json

from app.core.logging import LoggerMixin
from app.core.config import settings

try:
    import torch
    import torch.nn as nn
    from torch.quantization import quantize_dynamic, QConfig
    from torch.nn.utils import prune
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    nn = None

try:
    import onnxruntime as ort
    ONNX_AVAILABLE = True
except ImportError:
    ONNX_AVAILABLE = False

try:
    import numpy as np
    NUMPY_AVAILABLE = True
except ImportError:
    NUMPY_AVAILABLE = False


class HardwareDetector(LoggerMixin):
    """Detect available hardware capabilities for edge computing."""
    
    def __init__(self):
        super().__init__()
        self._capabilities = None
    
    def get_capabilities(self) -> Dict[str, Any]:
        """Get hardware capabilities."""
        
        if self._capabilities is None:
            self._capabilities = self._detect_capabilities()
        
        return self._capabilities
    
    def _detect_capabilities(self) -> Dict[str, Any]:
        """Detect hardware capabilities."""
        
        capabilities = {
            "cpu": self._get_cpu_info(),
            "memory": self._get_memory_info(),
            "gpu": self._get_gpu_info(),
            "accelerators": self._get_accelerators()
        }
        
        self.logger.info(f"Detected hardware capabilities: {capabilities}")
        return capabilities
    
    def _get_cpu_info(self) -> Dict[str, Any]:
        """Get CPU information."""
        
        try:
            cpu_count = multiprocessing.cpu_count()
            
            # Try to get more detailed CPU info
            cpu_info = {"cores": cpu_count}
            
            try:
                import psutil
                cpu_freq = psutil.cpu_freq()
                cpu_info.update({
                    "max_freq_mhz": cpu_freq.max if cpu_freq else None,
                    "current_freq_mhz": cpu_freq.current if cpu_freq else None,
                    "usage_percent": psutil.cpu_percent(interval=1)
                })
            except ImportError:
                pass
            
            return cpu_info
            
        except Exception as e:
            self.logger.warning(f"Failed to get CPU info: {e}")
            return {"cores": 1}
    
    def _get_memory_info(self) -> Dict[str, Any]:
        """Get memory information."""
        
        try:
            import psutil
            memory = psutil.virtual_memory()
            
            return {
                "total_gb": memory.total / (1024**3),
                "available_gb": memory.available / (1024**3),
                "usage_percent": memory.percent
            }
            
        except ImportError:
            return {"total_gb": None, "available_gb": None}
        except Exception as e:
            self.logger.warning(f"Failed to get memory info: {e}")
            return {"total_gb": None, "available_gb": None}
    
    def _get_gpu_info(self) -> Dict[str, Any]:
        """Get GPU information."""
        
        gpu_info = {"available": False, "devices": []}
        
        # Check CUDA
        if TORCH_AVAILABLE:
            try:
                if torch.cuda.is_available():
                    gpu_count = torch.cuda.device_count()
                    devices = []
                    
                    for i in range(gpu_count):
                        device_props = torch.cuda.get_device_properties(i)
                        devices.append({
                            "id": i,
                            "name": device_props.name,
                            "memory_gb": device_props.total_memory / (1024**3),
                            "compute_capability": f"{device_props.major}.{device_props.minor}",
                            "multiprocessor_count": device_props.multi_processor_count
                        })
                    
                    gpu_info.update({
                        "available": True,
                        "cuda_version": torch.version.cuda,
                        "devices": devices
                    })
                    
            except Exception as e:
                self.logger.warning(f"Failed to get CUDA info: {e}")
        
        # Check Metal (macOS)
        if TORCH_AVAILABLE and hasattr(torch.backends, 'mps'):
            try:
                if torch.backends.mps.is_available():
                    gpu_info.update({
                        "available": True,
                        "mps_available": True
                    })
            except Exception:
                pass
        
        return gpu_info
    
    def _get_accelerators(self) -> Dict[str, Any]:
        """Get AI accelerator information."""
        
        accelerators = {}
        
        # Check ONNX Runtime providers
        if ONNX_AVAILABLE:
            try:
                available_providers = ort.get_available_providers()
                accelerators["onnx_providers"] = available_providers
            except Exception:
                pass
        
        # Check Intel OpenVINO
        try:
            import openvino
            accelerators["openvino_available"] = True
        except ImportError:
            accelerators["openvino_available"] = False
        
        # Check TensorRT
        try:
            import tensorrt
            accelerators["tensorrt_available"] = True
        except ImportError:
            accelerators["tensorrt_available"] = False
        
        return accelerators
    
    def recommend_device(self, model_size_mb: float = 100) -> str:
        """Recommend optimal device for model inference."""
        
        capabilities = self.get_capabilities()
        
        # Check GPU availability and memory
        gpu_info = capabilities["gpu"]
        if gpu_info["available"]:
            gpu_devices = gpu_info.get("devices", [])
            for device in gpu_devices:
                # Need at least 2x model size for inference
                if device["memory_gb"] * 1024 >= model_size_mb * 2:
                    return f"cuda:{device['id']}"
            
            # Check MPS (Metal Performance Shaders)
            if gpu_info.get("mps_available"):
                memory_info = capabilities["memory"]
                if memory_info["available_gb"] and memory_info["available_gb"] * 1024 >= model_size_mb * 2:
                    return "mps"
        
        return "cpu"


class ModelOptimizer(LoggerMixin):
    """Model optimization for edge deployment."""
    
    def __init__(self, hardware_detector: Optional[HardwareDetector] = None):
        super().__init__()
        self.hardware_detector = hardware_detector or HardwareDetector()
    
    async def optimize_model(
        self,
        model: Any,
        optimization_level: str = "balanced",
        target_device: Optional[str] = None
    ) -> Any:
        """Optimize model for edge deployment."""
        
        if target_device is None:
            target_device = self.hardware_detector.recommend_device()
        
        try:
            if TORCH_AVAILABLE and isinstance(model, nn.Module):
                return await self._optimize_torch_model(model, optimization_level, target_device)
            else:
                self.logger.warning("Model optimization not supported for this model type")
                return model
                
        except Exception as e:
            self.logger.error(f"Model optimization failed: {e}")
            return model
    
    async def _optimize_torch_model(
        self,
        model: nn.Module,
        optimization_level: str,
        target_device: str
    ) -> nn.Module:
        """Optimize PyTorch model."""
        
        optimized_model = model
        
        # Apply optimizations based on level
        if optimization_level in ["aggressive", "balanced"]:
            # Quantization
            optimized_model = await self._quantize_model(optimized_model, target_device)
        
        if optimization_level == "aggressive":
            # Pruning
            optimized_model = await self._prune_model(optimized_model)
        
        # Compile for target device
        optimized_model = await self._compile_model(optimized_model, target_device)
        
        return optimized_model
    
    async def _quantize_model(self, model: nn.Module, target_device: str) -> nn.Module:
        """Apply quantization to model."""
        
        try:
            if target_device.startswith("cuda"):
                # Use FP16 for GPU
                model = model.half()
                self.logger.info("Applied FP16 quantization for GPU")
            
            else:
                # Use dynamic quantization for CPU
                quantized_model = quantize_dynamic(
                    model,
                    {nn.Linear, nn.Conv2d},
                    dtype=torch.qint8
                )
                self.logger.info("Applied dynamic INT8 quantization for CPU")
                return quantized_model
            
        except Exception as e:
            self.logger.warning(f"Quantization failed: {e}")
        
        return model
    
    async def _prune_model(self, model: nn.Module, prune_amount: float = 0.3) -> nn.Module:
        """Apply pruning to model."""
        
        try:
            # Global magnitude pruning
            parameters_to_prune = []
            
            for module in model.modules():
                if isinstance(module, (nn.Linear, nn.Conv2d)):
                    parameters_to_prune.append((module, 'weight'))
            
            if parameters_to_prune:
                prune.global_unstructured(
                    parameters_to_prune,
                    pruning_method=prune.L1Unstructured,
                    amount=prune_amount
                )
                
                # Remove pruning reparameterization
                for module, param_name in parameters_to_prune:
                    prune.remove(module, param_name)
                
                self.logger.info(f"Applied {prune_amount*100:.1f}% pruning")
            
        except Exception as e:
            self.logger.warning(f"Pruning failed: {e}")
        
        return model
    
    async def _compile_model(self, model: nn.Module, target_device: str) -> nn.Module:
        """Compile model for target device."""
        
        try:
            # Move to target device
            if target_device.startswith("cuda"):
                model = model.cuda()
            elif target_device == "mps":
                model = model.to("mps")
            else:
                model = model.cpu()
            
            # Enable inference optimizations
            model.eval()
            
            # Try to use torch.jit.script for optimization
            try:
                model = torch.jit.script(model)
                self.logger.info("Applied TorchScript compilation")
            except Exception:
                self.logger.debug("TorchScript compilation not supported")
            
        except Exception as e:
            self.logger.warning(f"Model compilation failed: {e}")
        
        return model


class InferenceEngine(LoggerMixin):
    """Optimized inference engine for edge computing."""
    
    def __init__(
        self,
        max_batch_size: int = 8,
        max_workers: int = None,
        use_threading: bool = True
    ):
        super().__init__()
        self.max_batch_size = max_batch_size
        self.max_workers = max_workers or min(4, multiprocessing.cpu_count())
        self.use_threading = use_threading
        
        # Initialize executor
        if use_threading:
            self.executor = ThreadPoolExecutor(max_workers=self.max_workers)
        else:
            self.executor = ProcessPoolExecutor(max_workers=self.max_workers)
        
        # Batch processing queue
        self._batch_queue = asyncio.Queue()
        self._batch_processor_task = None
    
    async def predict(
        self,
        model: Any,
        inputs: Union[Any, List[Any]],
        batch_processing: bool = True
    ) -> Union[Any, List[Any]]:
        """Run model inference."""
        
        single_input = not isinstance(inputs, list)
        if single_input:
            inputs = [inputs]
        
        try:
            if batch_processing and len(inputs) <= self.max_batch_size:
                # Use batch processing
                results = await self._batch_predict(model, inputs)
            else:
                # Process individually or in chunks
                results = await self._individual_predict(model, inputs)
            
            return results[0] if single_input else results
            
        except Exception as e:
            self.logger.error(f"Inference failed: {e}")
            raise
    
    async def _batch_predict(self, model: Any, inputs: List[Any]) -> List[Any]:
        """Run batch inference."""
        
        try:
            # Execute batch inference in thread pool
            results = await asyncio.get_event_loop().run_in_executor(
                self.executor,
                self._batch_inference_sync,
                model, inputs
            )
            
            return results
            
        except Exception as e:
            self.logger.error(f"Batch inference failed: {e}")
            # Fallback to individual processing
            return await self._individual_predict(model, inputs)
    
    def _batch_inference_sync(self, model: Any, inputs: List[Any]) -> List[Any]:
        """Synchronous batch inference."""
        
        if TORCH_AVAILABLE and isinstance(model, (nn.Module, torch.jit.ScriptModule)):
            return self._torch_batch_inference(model, inputs)
        elif ONNX_AVAILABLE and hasattr(model, 'run'):
            return self._onnx_batch_inference(model, inputs)
        else:
            # Generic batch processing
            return [model(input_data) for input_data in inputs]
    
    def _torch_batch_inference(self, model: nn.Module, inputs: List[Any]) -> List[Any]:
        """PyTorch batch inference."""
        
        try:
            # Stack inputs if they're tensors
            if all(isinstance(inp, torch.Tensor) for inp in inputs):
                batch_input = torch.stack(inputs)
                
                with torch.no_grad():
                    batch_output = model(batch_input)
                
                # Split output back to list
                if isinstance(batch_output, torch.Tensor):
                    return list(torch.unbind(batch_output, dim=0))
                else:
                    return [batch_output] * len(inputs)
            
            else:
                # Process individually if stacking not possible
                results = []
                with torch.no_grad():
                    for inp in inputs:
                        results.append(model(inp))
                return results
                
        except Exception as e:
            self.logger.warning(f"PyTorch batch inference failed: {e}")
            # Fallback to individual processing
            results = []
            with torch.no_grad():
                for inp in inputs:
                    results.append(model(inp))
            return results
    
    def _onnx_batch_inference(self, model, inputs: List[Any]) -> List[Any]:
        """ONNX batch inference."""
        
        try:
            # Get input/output names
            input_names = [inp.name for inp in model.get_inputs()]
            
            if len(input_names) == 1:
                input_name = input_names[0]
                
                # Stack inputs if possible
                if NUMPY_AVAILABLE and all(hasattr(inp, 'shape') for inp in inputs):
                    import numpy as np
                    batch_input = np.stack(inputs)
                    
                    batch_output = model.run(None, {input_name: batch_input})
                    
                    # Split output back to list
                    if len(batch_output) == 1 and len(batch_output[0]) == len(inputs):
                        return list(batch_output[0])
                
            # Fallback to individual processing
            results = []
            for inp in inputs:
                if isinstance(inp, dict):
                    output = model.run(None, inp)
                else:
                    output = model.run(None, {input_names[0]: inp})
                results.append(output)
            
            return results
            
        except Exception as e:
            self.logger.warning(f"ONNX batch inference failed: {e}")
            return [model.run(None, inp) for inp in inputs]
    
    async def _individual_predict(self, model: Any, inputs: List[Any]) -> List[Any]:
        """Process inputs individually."""
        
        # Process in chunks to manage memory
        chunk_size = min(self.max_batch_size, len(inputs))
        results = []
        
        for i in range(0, len(inputs), chunk_size):
            chunk = inputs[i:i + chunk_size]
            
            # Process chunk in parallel
            chunk_tasks = [
                asyncio.get_event_loop().run_in_executor(
                    self.executor,
                    self._single_inference,
                    model, inp
                )
                for inp in chunk
            ]
            
            chunk_results = await asyncio.gather(*chunk_tasks, return_exceptions=True)
            
            # Handle exceptions
            for result in chunk_results:
                if isinstance(result, Exception):
                    self.logger.error(f"Individual inference failed: {result}")
                    results.append(None)
                else:
                    results.append(result)
        
        return results
    
    def _single_inference(self, model: Any, input_data: Any) -> Any:
        """Single inference execution."""
        
        try:
            if TORCH_AVAILABLE and isinstance(model, (nn.Module, torch.jit.ScriptModule)):
                with torch.no_grad():
                    return model(input_data)
            else:
                return model(input_data)
                
        except Exception as e:
            self.logger.error(f"Single inference failed: {e}")
            raise
    
    def close(self):
        """Clean up resources."""
        if hasattr(self, 'executor'):
            self.executor.shutdown(wait=True)


class EdgeComputer(LoggerMixin):
    """Main edge computing orchestrator."""
    
    def __init__(self):
        super().__init__()
        self.hardware_detector = HardwareDetector()
        self.model_optimizer = ModelOptimizer(self.hardware_detector)
        self.inference_engine = None
        
        # Cache for optimized models
        self._optimized_models = {}
    
    async def prepare_model(
        self,
        model: Any,
        optimization_level: str = "balanced",
        batch_size: int = 8
    ) -> str:
        """Prepare model for edge computing."""
        
        try:
            # Generate model key
            model_key = f"{id(model)}_{optimization_level}_{batch_size}"
            
            if model_key in self._optimized_models:
                self.logger.debug(f"Using cached optimized model: {model_key}")
                return model_key
            
            # Detect hardware and recommend device
            capabilities = self.hardware_detector.get_capabilities()
            target_device = self.hardware_detector.recommend_device()
            
            self.logger.info(f"Preparing model for device: {target_device}")
            
            # Optimize model
            optimized_model = await self.model_optimizer.optimize_model(
                model, optimization_level, target_device
            )
            
            # Initialize inference engine
            if self.inference_engine is None:
                self.inference_engine = InferenceEngine(
                    max_batch_size=batch_size,
                    max_workers=capabilities["cpu"]["cores"]
                )
            
            # Cache optimized model
            self._optimized_models[model_key] = {
                "model": optimized_model,
                "device": target_device,
                "capabilities": capabilities,
                "optimization_level": optimization_level
            }
            
            self.logger.info(f"Model prepared for edge computing: {model_key}")
            return model_key
            
        except Exception as e:
            self.logger.error(f"Model preparation failed: {e}")
            raise
    
    async def run_inference(
        self,
        model_key: str,
        inputs: Union[Any, List[Any]],
        **kwargs
    ) -> Union[Any, List[Any]]:
        """Run optimized inference."""
        
        if model_key not in self._optimized_models:
            raise ValueError(f"Model not prepared: {model_key}")
        
        model_info = self._optimized_models[model_key]
        model = model_info["model"]
        
        if self.inference_engine is None:
            raise RuntimeError("Inference engine not initialized")
        
        try:
            start_time = time.time()
            
            results = await self.inference_engine.predict(
                model, inputs, **kwargs
            )
            
            inference_time = time.time() - start_time
            
            self.logger.debug(
                f"Inference completed in {inference_time:.3f}s "
                f"for {len(inputs) if isinstance(inputs, list) else 1} samples"
            )
            
            return results
            
        except Exception as e:
            self.logger.error(f"Edge inference failed: {e}")
            raise
    
    def get_model_info(self, model_key: str) -> Dict[str, Any]:
        """Get information about prepared model."""
        
        if model_key not in self._optimized_models:
            return {"error": "Model not found"}
        
        model_info = self._optimized_models[model_key]
        
        return {
            "device": model_info["device"],
            "optimization_level": model_info["optimization_level"],
            "hardware_capabilities": model_info["capabilities"]
        }
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """Get edge computing performance statistics."""
        
        capabilities = self.hardware_detector.get_capabilities()
        
        stats = {
            "hardware": capabilities,
            "optimized_models": len(self._optimized_models),
            "model_keys": list(self._optimized_models.keys())
        }
        
        if self.inference_engine:
            stats["inference_engine"] = {
                "max_batch_size": self.inference_engine.max_batch_size,
                "max_workers": self.inference_engine.max_workers,
                "use_threading": self.inference_engine.use_threading
            }
        
        return stats
    
    def close(self):
        """Clean up resources."""
        if self.inference_engine:
            self.inference_engine.close()
        
        self._optimized_models.clear()
        self.logger.info("Edge computer closed")
