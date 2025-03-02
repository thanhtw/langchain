"""
GPU utility module for managing GPU detection and configuration.
Centralizes GPU-related functionality for consistent usage across the application.
"""

import torch
import logging
from typing import Dict, Any, Optional, Tuple

logger = logging.getLogger(__name__)

def is_gpu_available() -> bool:
    """
    Check if a CUDA-compatible GPU is available using PyTorch.
    
    Returns:
        bool: True if GPU is available, False otherwise
    """
    try:
        return torch.cuda.is_available()
    except Exception as e:
        logger.warning(f"Error checking GPU availability: {str(e)}")
        return False

def get_gpu_info() -> Dict[str, Any]:
    """
    Get detailed information about the available GPU(s).
    
    Returns:
        dict: Dictionary containing GPU information
    """
    info = {
        "available": is_gpu_available(),
        "count": 0,
        "name": None,
        "memory_gb": None,
        "cuda_version": None
    }
    
    if info["available"]:
        try:
            info["count"] = torch.cuda.device_count()
            info["name"] = torch.cuda.get_device_name(0)
            info["memory_gb"] = torch.cuda.get_device_properties(0).total_memory / (1024**3)
            info["cuda_version"] = torch.version.cuda
        except Exception as e:
            logger.warning(f"Error getting detailed GPU info: {str(e)}")
    
    return info

def optimize_model_params(params: Dict[str, Any], model_type: str) -> Dict[str, Any]:
    """
    Optimize model parameters based on GPU availability and specifications.
    Different model types require different GPU optimizations.
    
    Args:
        params (dict): Original model parameters
        model_type (str): Type of model ("transformers", "llama.cpp", "ctransformers", etc.)
        
    Returns:
        dict: Optimized parameters
    """
    optimized_params = params.copy()
    
    # If GPU is available, update parameters accordingly
    if is_gpu_available():
        if model_type == "transformers":
            # Set device mapping to auto for transformers
            if "device_map" not in optimized_params:
                optimized_params["device_map"] = "auto"
                
            # Enable quantization if appropriate and not explicitly disabled
            if "load_in_8bit" not in optimized_params and "load_in_4bit" not in optimized_params:
                gpu_memory = get_gpu_info().get("memory_gb", 0)
                
                # If limited GPU memory, use 4-bit quantization
                if gpu_memory and gpu_memory < 8:
                    optimized_params["load_in_4bit"] = True
                # For moderate GPU memory, use 8-bit quantization
                elif gpu_memory and gpu_memory < 16:
                    optimized_params["load_in_8bit"] = True
        
        elif model_type in ["llama.cpp", "ctransformers"]:
            # Set GPU layers for llama.cpp and ctransformers
            # Use a high number for better GPU utilization when available
            if "n_gpu_layers" not in optimized_params:
                optimized_params["n_gpu_layers"] = 35  # Good default for most models
                
            # For ctransformers, ensure GPU usage is enabled
            if model_type == "ctransformers" and "gpu_layers" not in optimized_params:
                optimized_params["gpu_layers"] = 35
                
        elif model_type == "ollama":
            # Ensure Ollama uses GPU if available
            # (Ollama handles GPU internally, but we could add specific parameters here if needed)
            pass
    
    return optimized_params

def get_optimal_batch_size() -> int:
    """
    Get optimal batch size based on available GPU memory.
    Critical for efficient memory usage with large models.
    
    Returns:
        int: Recommended batch size
    """
    if not is_gpu_available():
        return 4  # Conservative default for CPU
        
    try:
        gpu_memory = get_gpu_info().get("memory_gb", 0)
        
        # Simple heuristic based on GPU memory
        if gpu_memory > 24:
            return 32
        elif gpu_memory > 16:
            return 16
        elif gpu_memory > 8:
            return 8
        elif gpu_memory > 4:
            return 4
        else:
            return 2
    except Exception as e:
        logger.warning(f"Error determining optimal batch size: {str(e)}")
        return 4  # Default if we can't determine

def torch_device() -> torch.device:
    """
    Get the appropriate torch device based on availability.
    
    Returns:
        torch.device: CUDA device if available, otherwise CPU
    """
    return torch.device("cuda" if is_gpu_available() else "cpu")

def get_device_string() -> str:
    """
    Get a user-friendly device string.
    
    Returns:
        str: "GPU" or "CPU"
    """
    return "GPU" if is_gpu_available() else "CPU"