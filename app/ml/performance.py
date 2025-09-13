"""Performance monitoring and profiling utilities for ML models.

This module provides comprehensive performance monitoring, profiling, and
benchmarking capabilities for ML model inference and training.
"""

import logging
import time
import asyncio
import threading
from typing import Dict, Any, Optional, List, Callable, Union
from dataclasses import dataclass, field
from collections import defaultdict, deque
from datetime import datetime, timedelta
import statistics
import json
from pathlib import Path
from functools import wraps
import traceback
import psutil
import gc

from app.core.logging import LoggerMixin
from app.core.config import settings

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

try:
    import numpy as np
    NUMPY_AVAILABLE = True
except ImportError:
    NUMPY_AVAILABLE = False


@dataclass
class PerformanceMetric:
    """Individual performance metric."""
    
    name: str
    value: float
    unit: str
    timestamp: datetime = field(default_factory=datetime.now)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class InferenceStats:
    """Statistics for model inference."""
    
    total_inferences: int = 0
    total_time: float = 0.0
    min_time: float = float('inf')
    max_time: float = 0.0
    avg_time: float = 0.0
    throughput: float = 0.0  # inferences per second
    memory_usage_mb: float = 0.0
    gpu_utilization: float = 0.0
    error_count: int = 0
    last_updated: datetime = field(default_factory=datetime.now)


class SystemMonitor(LoggerMixin):
    """System resource monitoring."""
    
    def __init__(self, monitor_interval: float = 1.0):
        super().__init__()
        self.monitor_interval = monitor_interval
        self._monitoring = False
        self._monitor_task = None
        self._metrics_history = defaultdict(lambda: deque(maxlen=1000))
        self._lock = threading.Lock()
    
    def start_monitoring(self):
        """Start system monitoring."""
        
        if self._monitoring:
            return
        
        self._monitoring = True
        self._monitor_task = asyncio.create_task(self._monitor_loop())
        self.logger.info("Started system monitoring")
    
    def stop_monitoring(self):
        """Stop system monitoring."""
        
        if not self._monitoring:
            return
        
        self._monitoring = False
        if self._monitor_task:
            self._monitor_task.cancel()
        
        self.logger.info("Stopped system monitoring")
    
    async def _monitor_loop(self):
        """Main monitoring loop."""
        
        try:
            while self._monitoring:
                await self._collect_metrics()
                await asyncio.sleep(self.monitor_interval)
        
        except asyncio.CancelledError:
            pass
        except Exception as e:
            self.logger.error(f"Monitoring error: {e}")
    
    async def _collect_metrics(self):
        """Collect system metrics."""
        
        try:
            timestamp = datetime.now()
            
            # CPU metrics
            cpu_percent = psutil.cpu_percent(interval=None)
            cpu_count = psutil.cpu_count()
            
            # Memory metrics
            memory = psutil.virtual_memory()
            memory_percent = memory.percent
            memory_used_gb = memory.used / (1024**3)
            memory_available_gb = memory.available / (1024**3)
            
            # GPU metrics (if available)
            gpu_metrics = await self._get_gpu_metrics()
            
            # Process metrics
            process = psutil.Process()
            process_memory = process.memory_info().rss / (1024**2)  # MB
            process_cpu = process.cpu_percent()
            
            # Store metrics
            with self._lock:
                self._metrics_history['cpu_percent'].append(
                    PerformanceMetric('cpu_percent', cpu_percent, '%', timestamp)
                )
                self._metrics_history['memory_percent'].append(
                    PerformanceMetric('memory_percent', memory_percent, '%', timestamp)
                )
                self._metrics_history['memory_used_gb'].append(
                    PerformanceMetric('memory_used_gb', memory_used_gb, 'GB', timestamp)
                )
                self._metrics_history['process_memory_mb'].append(
                    PerformanceMetric('process_memory_mb', process_memory, 'MB', timestamp)
                )
                self._metrics_history['process_cpu_percent'].append(
                    PerformanceMetric('process_cpu_percent', process_cpu, '%', timestamp)
                )
                
                # Add GPU metrics if available
                for metric_name, metric_value in gpu_metrics.items():
                    self._metrics_history[metric_name].append(
                        PerformanceMetric(metric_name, metric_value, '%', timestamp)
                    )
        
        except Exception as e:
            self.logger.warning(f"Failed to collect metrics: {e}")
    
    async def _get_gpu_metrics(self) -> Dict[str, float]:
        """Get GPU metrics."""
        
        gpu_metrics = {}
        
        if TORCH_AVAILABLE:
            try:
                if torch.cuda.is_available():
                    for i in range(torch.cuda.device_count()):
                        # Memory usage
                        memory_allocated = torch.cuda.memory_allocated(i) / (1024**3)  # GB
                        memory_reserved = torch.cuda.memory_reserved(i) / (1024**3)   # GB
                        memory_total = torch.cuda.get_device_properties(i).total_memory / (1024**3)
                        
                        gpu_metrics[f'gpu_{i}_memory_allocated_gb'] = memory_allocated
                        gpu_metrics[f'gpu_{i}_memory_reserved_gb'] = memory_reserved
                        gpu_metrics[f'gpu_{i}_memory_percent'] = (memory_reserved / memory_total) * 100
                        
                        # Utilization (simplified - would need nvidia-ml-py for detailed stats)
                        gpu_metrics[f'gpu_{i}_utilization_percent'] = 0.0
            
            except Exception as e:
                self.logger.debug(f"GPU metrics collection failed: {e}")
        
        # Try nvidia-ml-py for detailed GPU stats
        try:
            import pynvml
            pynvml.nvmlInit()
            
            device_count = pynvml.nvmlDeviceGetCount()
            for i in range(device_count):
                handle = pynvml.nvmlDeviceGetHandleByIndex(i)
                
                # GPU utilization
                utilization = pynvml.nvmlDeviceGetUtilizationRates(handle)
                gpu_metrics[f'gpu_{i}_utilization_percent'] = utilization.gpu
                gpu_metrics[f'gpu_{i}_memory_utilization_percent'] = utilization.memory
                
                # Temperature
                temp = pynvml.nvmlDeviceGetTemperature(handle, pynvml.NVML_TEMPERATURE_GPU)
                gpu_metrics[f'gpu_{i}_temperature_c'] = temp
        
        except ImportError:
            pass
        except Exception as e:
            self.logger.debug(f"NVML GPU metrics failed: {e}")
        
        return gpu_metrics
    
    def get_current_metrics(self) -> Dict[str, Any]:
        """Get current system metrics."""
        
        try:
            cpu_percent = psutil.cpu_percent()
            memory = psutil.virtual_memory()
            
            metrics = {
                'cpu_percent': cpu_percent,
                'memory_percent': memory.percent,
                'memory_used_gb': memory.used / (1024**3),
                'memory_available_gb': memory.available / (1024**3)
            }
            
            # Add process metrics
            try:
                process = psutil.Process()
                metrics.update({
                    'process_memory_mb': process.memory_info().rss / (1024**2),
                    'process_cpu_percent': process.cpu_percent()
                })
            except Exception:
                pass
            
            return metrics
            
        except Exception as e:
            self.logger.error(f"Failed to get current metrics: {e}")
            return {}
    
    def get_metrics_summary(self, minutes: int = 10) -> Dict[str, Any]:
        """Get metrics summary for the last N minutes."""
        
        cutoff_time = datetime.now() - timedelta(minutes=minutes)
        summary = {}
        
        with self._lock:
            for metric_name, metric_history in self._metrics_history.items():
                # Filter recent metrics
                recent_metrics = [
                    m for m in metric_history 
                    if m.timestamp >= cutoff_time
                ]
                
                if recent_metrics:
                    values = [m.value for m in recent_metrics]
                    
                    summary[metric_name] = {
                        'count': len(values),
                        'min': min(values),
                        'max': max(values),
                        'mean': statistics.mean(values),
                        'median': statistics.median(values),
                        'std': statistics.stdev(values) if len(values) > 1 else 0.0,
                        'unit': recent_metrics[0].unit
                    }
        
        return summary


class ModelProfiler(LoggerMixin):
    """Model performance profiler."""
    
    def __init__(self):
        super().__init__()
        self._profiles = {}
        self._active_profiles = {}
        self._lock = threading.Lock()
    
    def start_profile(self, profile_name: str, model: Any = None) -> str:
        """Start profiling session."""
        
        profile_id = f"{profile_name}_{int(time.time())}"
        
        profile_data = {
            'name': profile_name,
            'start_time': time.time(),
            'model_info': self._get_model_info(model) if model else {},
            'inference_times': [],
            'memory_snapshots': [],
            'errors': [],
            'metadata': {}
        }
        
        with self._lock:
            self._active_profiles[profile_id] = profile_data
        
        self.logger.info(f"Started profiling: {profile_id}")
        return profile_id
    
    def record_inference(
        self,
        profile_id: str,
        inference_time: float,
        memory_usage: Optional[float] = None,
        batch_size: int = 1,
        metadata: Dict[str, Any] = None
    ):
        """Record inference performance data."""
        
        if profile_id not in self._active_profiles:
            return
        
        with self._lock:
            profile = self._active_profiles[profile_id]
            
            inference_data = {
                'time': inference_time,
                'batch_size': batch_size,
                'throughput': batch_size / inference_time,
                'timestamp': time.time(),
                'metadata': metadata or {}
            }
            
            if memory_usage is not None:
                inference_data['memory_usage'] = memory_usage
            
            profile['inference_times'].append(inference_data)
    
    def record_error(self, profile_id: str, error: Exception, context: Dict[str, Any] = None):
        """Record profiling error."""
        
        if profile_id not in self._active_profiles:
            return
        
        error_data = {
            'error_type': type(error).__name__,
            'error_message': str(error),
            'traceback': traceback.format_exc(),
            'timestamp': time.time(),
            'context': context or {}
        }
        
        with self._lock:
            self._active_profiles[profile_id]['errors'].append(error_data)
    
    def add_memory_snapshot(self, profile_id: str, snapshot_name: str):
        """Add memory usage snapshot."""
        
        if profile_id not in self._active_profiles:
            return
        
        try:
            # Python memory usage
            process = psutil.Process()
            memory_info = process.memory_info()
            
            snapshot = {
                'name': snapshot_name,
                'timestamp': time.time(),
                'rss_mb': memory_info.rss / (1024**2),
                'vms_mb': memory_info.vms / (1024**2),
                'python_objects': len(gc.get_objects())
            }
            
            # GPU memory if available
            if TORCH_AVAILABLE and torch.cuda.is_available():
                for i in range(torch.cuda.device_count()):
                    allocated = torch.cuda.memory_allocated(i) / (1024**2)
                    reserved = torch.cuda.memory_reserved(i) / (1024**2)
                    
                    snapshot[f'gpu_{i}_allocated_mb'] = allocated
                    snapshot[f'gpu_{i}_reserved_mb'] = reserved
            
            with self._lock:
                self._active_profiles[profile_id]['memory_snapshots'].append(snapshot)
        
        except Exception as e:
            self.logger.warning(f"Failed to capture memory snapshot: {e}")
    
    def stop_profile(self, profile_id: str) -> Dict[str, Any]:
        """Stop profiling and return results."""
        
        if profile_id not in self._active_profiles:
            return {}
        
        with self._lock:
            profile = self._active_profiles.pop(profile_id)
            profile['end_time'] = time.time()
            profile['duration'] = profile['end_time'] - profile['start_time']
            
            # Calculate statistics
            if profile['inference_times']:
                times = [inf['time'] for inf in profile['inference_times']]
                throughputs = [inf['throughput'] for inf in profile['inference_times']]
                
                profile['statistics'] = {
                    'total_inferences': len(times),
                    'min_time': min(times),
                    'max_time': max(times),
                    'mean_time': statistics.mean(times),
                    'median_time': statistics.median(times),
                    'std_time': statistics.stdev(times) if len(times) > 1 else 0.0,
                    'mean_throughput': statistics.mean(throughputs),
                    'total_errors': len(profile['errors'])
                }
            else:
                profile['statistics'] = {'total_inferences': 0, 'total_errors': len(profile['errors'])}
            
            # Store completed profile
            self._profiles[profile_id] = profile
        
        self.logger.info(f"Stopped profiling: {profile_id}")
        return profile
    
    def _get_model_info(self, model: Any) -> Dict[str, Any]:
        """Get basic model information."""
        
        try:
            info = {'type': type(model).__name__}
            
            if TORCH_AVAILABLE and hasattr(model, 'parameters'):
                try:
                    total_params = sum(p.numel() for p in model.parameters())
                    info['total_parameters'] = total_params
                except Exception:
                    pass
            
            return info
        
        except Exception:
            return {}
    
    def get_profile_results(self, profile_id: str) -> Optional[Dict[str, Any]]:
        """Get results from completed profile."""
        
        return self._profiles.get(profile_id)
    
    def list_profiles(self) -> List[str]:
        """List all completed profiles."""
        
        return list(self._profiles.keys())


class PerformanceMonitor(LoggerMixin):
    """Comprehensive performance monitoring system."""
    
    def __init__(
        self,
        enable_system_monitoring: bool = True,
        system_monitor_interval: float = 1.0
    ):
        super().__init__()
        
        # Initialize components
        self.system_monitor = SystemMonitor(system_monitor_interval) if enable_system_monitoring else None
        self.model_profiler = ModelProfiler()
        
        # Performance tracking
        self._inference_stats = defaultdict(InferenceStats)
        self._lock = threading.Lock()
        
        # Start system monitoring
        if self.system_monitor:
            self.system_monitor.start_monitoring()
    
    def track_inference(
        self,
        model_name: str,
        inference_time: float,
        batch_size: int = 1,
        memory_usage: Optional[float] = None,
        error: Optional[Exception] = None
    ):
        """Track model inference performance."""
        
        with self._lock:
            stats = self._inference_stats[model_name]
            
            if error:
                stats.error_count += 1
            else:
                stats.total_inferences += 1
                stats.total_time += inference_time
                stats.min_time = min(stats.min_time, inference_time)
                stats.max_time = max(stats.max_time, inference_time)
                stats.avg_time = stats.total_time / stats.total_inferences
                stats.throughput = batch_size / inference_time
                
                if memory_usage is not None:
                    stats.memory_usage_mb = memory_usage
            
            stats.last_updated = datetime.now()
    
    def get_inference_stats(self, model_name: Optional[str] = None) -> Dict[str, Any]:
        """Get inference statistics."""
        
        with self._lock:
            if model_name:
                if model_name in self._inference_stats:
                    return {model_name: self._inference_stats[model_name].__dict__}
                else:
                    return {}
            else:
                return {
                    name: stats.__dict__ 
                    for name, stats in self._inference_stats.items()
                }
    
    def start_profiling(self, profile_name: str, model: Any = None) -> str:
        """Start performance profiling session."""
        
        return self.model_profiler.start_profile(profile_name, model)
    
    def stop_profiling(self, profile_id: str) -> Dict[str, Any]:
        """Stop performance profiling session."""
        
        return self.model_profiler.stop_profile(profile_id)
    
    def get_system_metrics(self) -> Dict[str, Any]:
        """Get current system metrics."""
        
        if self.system_monitor:
            return self.system_monitor.get_current_metrics()
        else:
            return {}
    
    def get_system_summary(self, minutes: int = 10) -> Dict[str, Any]:
        """Get system metrics summary."""
        
        if self.system_monitor:
            return self.system_monitor.get_metrics_summary(minutes)
        else:
            return {}
    
    def generate_report(self, include_system: bool = True) -> Dict[str, Any]:
        """Generate comprehensive performance report."""
        
        report = {
            'timestamp': datetime.now().isoformat(),
            'inference_statistics': self.get_inference_stats()
        }
        
        if include_system and self.system_monitor:
            report['system_metrics'] = self.get_system_metrics()
            report['system_summary'] = self.get_system_summary()
        
        return report
    
    def save_report(self, filepath: Union[str, Path], include_system: bool = True):
        """Save performance report to file."""
        
        try:
            report = self.generate_report(include_system)
            
            with open(filepath, 'w') as f:
                json.dump(report, f, indent=2, default=str)
            
            self.logger.info(f"Performance report saved to: {filepath}")
        
        except Exception as e:
            self.logger.error(f"Failed to save report: {e}")
    
    def close(self):
        """Clean up monitoring resources."""
        
        if self.system_monitor:
            self.system_monitor.stop_monitoring()
        
        self.logger.info("Performance monitor closed")


def performance_monitor(monitor_instance: Optional[PerformanceMonitor] = None):
    """Decorator for automatic performance monitoring."""
    
    def decorator(func: Callable) -> Callable:
        
        @wraps(func)
        async def async_wrapper(*args, **kwargs):
            monitor = monitor_instance or PerformanceMonitor()
            model_name = kwargs.get('model_name', func.__name__)
            
            start_time = time.time()
            memory_before = None
            
            try:
                # Memory snapshot before
                if TORCH_AVAILABLE and torch.cuda.is_available():
                    memory_before = torch.cuda.memory_allocated() / (1024**2)
                
                # Execute function
                result = await func(*args, **kwargs)
                
                # Calculate metrics
                inference_time = time.time() - start_time
                memory_usage = None
                
                if memory_before is not None and TORCH_AVAILABLE and torch.cuda.is_available():
                    memory_after = torch.cuda.memory_allocated() / (1024**2)
                    memory_usage = memory_after - memory_before
                
                # Track performance
                batch_size = kwargs.get('batch_size', 1)
                monitor.track_inference(model_name, inference_time, batch_size, memory_usage)
                
                return result
            
            except Exception as e:
                inference_time = time.time() - start_time
                monitor.track_inference(model_name, inference_time, error=e)
                raise
        
        @wraps(func)
        def sync_wrapper(*args, **kwargs):
            monitor = monitor_instance or PerformanceMonitor()
            model_name = kwargs.get('model_name', func.__name__)
            
            start_time = time.time()
            
            try:
                result = func(*args, **kwargs)
                
                inference_time = time.time() - start_time
                batch_size = kwargs.get('batch_size', 1)
                monitor.track_inference(model_name, inference_time, batch_size)
                
                return result
            
            except Exception as e:
                inference_time = time.time() - start_time
                monitor.track_inference(model_name, inference_time, error=e)
                raise
        
        return async_wrapper if asyncio.iscoroutinefunction(func) else sync_wrapper
    
    return decorator
