"""
Utility for downloading and managing LLM models for local use.
"""

import os
import sys
import requests
import subprocess
import platform
import streamlit as st
from typing import Optional, Dict, Any, List, Tuple
from tqdm.auto import tqdm
from pathlib import Path

# Models directory
DEFAULT_MODELS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "models")

# Recommended models with their HuggingFace URLs
RECOMMENDED_MODELS = {
    # Llama models
    "llama-3-8b-instruct.Q4_K_M.gguf": {
        "url": "https://huggingface.co/TheBloke/Llama-3-8B-Instruct-GGUF/resolve/main/llama-3-8b-instruct.Q4_K_M.gguf",
        "description": "Llama 3 8B Instruct Q4_K_M (4-bit quantized, ~4.5GB)",
        "family": "llama"
    },
    "llama-3-8b-instruct.Q5_K_M.gguf": {
        "url": "https://huggingface.co/TheBloke/Llama-3-8B-Instruct-GGUF/resolve/main/llama-3-8b-instruct.Q5_K_M.gguf",
        "description": "Llama 3 8B Instruct Q5_K_M (5-bit quantized, ~5.3GB)",
        "family": "llama"
    },
    "llama-3-1b-instruct.Q4_K_M.gguf": {
        "url": "https://huggingface.co/TheBloke/Llama-3-1B-Instruct-GGUF/resolve/main/llama-3-1b-instruct.Q4_K_M.gguf",
        "description": "Llama 3 1B Instruct Q4_K_M (4-bit quantized, ~700MB)",
        "family": "llama"
    },
    
    # DeepSeek models
    "deepseek-coder-6.7b-instruct.Q4_K_M.gguf": {
        "url": "https://huggingface.co/TheBloke/deepseek-coder-6.7B-instruct-GGUF/resolve/main/deepseek-coder-6.7b-instruct.Q4_K_M.gguf",
        "description": "DeepSeek Coder 6.7B Instruct Q4_K_M (4-bit quantized, ~4GB)",
        "family": "deepseek"
    },
    "deepseek-llm-7b-chat.Q4_K_M.gguf": {
        "url": "https://huggingface.co/TheBloke/deepseek-llm-7B-chat-GGUF/resolve/main/deepseek-llm-7b-chat.Q4_K_M.gguf",
        "description": "DeepSeek LLM 7B Chat Q4_K_M (4-bit quantized, ~4GB)",
        "family": "deepseek"
    }
}

def download_file(url: str, destination: str, chunk_size: int = 8192) -> bool:
    """
    Download a file with progress bar.
    
    Args:
        url (str): URL to download from
        destination (str): Destination path
        chunk_size (int): Chunk size for download
        
    Returns:
        bool: True if successful, False otherwise
    """
    try:
        response = requests.get(url, stream=True)
        total_size = int(response.headers.get('content-length', 0))
        
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(destination), exist_ok=True)
        
        # Show progress bar
        progress_bar = tqdm(
            total=total_size, 
            unit='B', 
            unit_scale=True, 
            desc=os.path.basename(destination)
        )
        
        with open(destination, 'wb') as f:
            for chunk in response.iter_content(chunk_size=chunk_size):
                if chunk:
                    f.write(chunk)
                    progress_bar.update(len(chunk))
                    
        progress_bar.close()
        return True
        
    except Exception as e:
        print(f"Error downloading file: {str(e)}")
        return False

def check_ollama_running() -> bool:
    """
    Check if Ollama is running.
    
    Returns:
        bool: True if running, False otherwise
    """
    try:
        # Try to connect to Ollama API
        response = requests.get("http://localhost:11434/api/tags")
        return response.status_code == 200
        
    except:
        return False

def start_ollama() -> bool:
    """
    Start Ollama server.
    
    Returns:
        bool: True if started successfully, False otherwise
    """
    try:
        system = platform.system()
        
        if system == "Windows":
            # For Windows, try to start Ollama using the installed executable
            subprocess.Popen(
                ["ollama", "serve"], 
                shell=True, 
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                creationflags=subprocess.CREATE_NO_WINDOW
            )
                
        elif system == "Linux":
            # For Linux, use the systemd service if available
            try:
                subprocess.run(
                    ["systemctl", "--user", "start", "ollama"], 
                    check=True, 
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE
                )
            except (subprocess.SubprocessError, FileNotFoundError):
                # If systemd fails, try running directly
                subprocess.Popen(
                    ["ollama", "serve"], 
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE
                )
                    
        elif system == "Darwin":  # macOS
            # For macOS, try running directly
            subprocess.Popen(
                ["ollama", "serve"], 
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE
            )
            
        # Wait a moment to give Ollama time to start
        import time
        time.sleep(3)
        
        return check_ollama_running()
        
    except Exception as e:
        print(f"Error starting Ollama: {str(e)}")
        return False

def download_model_with_streamlit(model_info: Dict[str, Any], models_dir: str = DEFAULT_MODELS_DIR) -> Optional[str]:
    """
    Download a model with Streamlit progress bar.
    
    Args:
        model_info (dict): Model information
        models_dir (str): Directory to save the model
        
    Returns:
        str: Path to the downloaded model or None if failed
    """
    try:
        # Get model info
        model_name = next(iter(model_info.keys()))
        url = model_info[model_name]["url"]
        
        # Create destination path
        destination = os.path.join(models_dir, model_name)
        
        # Create directory if it doesn't exist
        os.makedirs(models_dir, exist_ok=True)
        
        # Check if file already exists with correct size
        if os.path.exists(destination):
            try:
                # Get expected file size from server
                response = requests.head(url)
                expected_size = int(response.headers.get('content-length', 0))
                current_size = os.path.getsize(destination)
                
                # If file size matches, skip download
                if current_size == expected_size:
                    st.success(f"Model {model_name} already downloaded")
                    return destination
                    
                st.warning(f"Model file exists but with incorrect size. Downloading again.")
            except Exception:
                # If we can't check size, assume we need to download
                pass
        
        # Show download message
        progress_text = f"Downloading {model_name}..."
        progress_bar = st.progress(0, text=progress_text)
        
        # Download in chunks with progress updates
        response = requests.get(url, stream=True)
        total_size = int(response.headers.get('content-length', 0))
        
        if total_size == 0:
            st.error(f"Invalid content length for model {model_name}")
            return None
            
        downloaded_size = 0
        with open(destination, 'wb') as f:
            for chunk in response.iter_content(chunk_size=1024*1024):  # 1MB chunks
                if chunk:
                    f.write(chunk)
                    downloaded_size += len(chunk)
                    progress = int((downloaded_size / total_size) * 100)
                    progress_bar.progress(min(progress/100, 1.0), 
                                         text=f"Downloading {model_name}: {progress}%")
        
        # Verify downloaded file
        if os.path.exists(destination) and os.path.getsize(destination) > 0:
            st.success(f"Successfully downloaded {model_name}")
            return destination
        else:
            st.error(f"Download appears to have failed for {model_name}")
            return None
        
    except Exception as e:
        st.error(f"Failed to download model: {str(e)}")
        return None

def get_model_family(model_path: str) -> str:
    """
    Determine the model family from the model path.
    
    Args:
        model_path (str): Path to the model
        
    Returns:
        str: Model family name
    """
    filename = os.path.basename(model_path).lower()
    
    if "llama" in filename:
        return "llama"
    elif "deepseek" in filename:
        return "deepseek"
    elif "mistral" in filename:
        return "mistral"
    elif "phi" in filename:
        return "phi"
    elif "gemma" in filename:
        return "gemma"
    else:
        return "other"

def get_recommended_model_parameters(model_path: str, provider: str) -> Dict[str, Any]:
    """
    Get recommended parameters for a specific model with GPU optimization.
    
    Args:
        model_path (str): Path to the model or name of the model
        provider (str): Provider name
        
    Returns:
        dict: Recommended parameters
    """
    # Import GPU utilities
    from utils.gpu_utils import is_gpu_available, optimize_model_params
    
    # Base parameters
    params = {
        "temperature": 0.7,
        "max_tokens": 512
    }
    
    # Get model family
    model_family = get_model_family(model_path)
    
    # Parameters based on model family
    if model_family == "llama":
        if provider in ["llama.cpp", "ctransformers"]:
            params.update({
                "model_type": "llama",
                "n_ctx": 4096
            })
    elif model_family == "deepseek":
        if provider in ["llama.cpp", "ctransformers"]:
            params.update({
                "model_type": "deepseek",
                "n_ctx": 4096
            })
    elif model_family == "mistral":
        if provider in ["llama.cpp", "ctransformers"]:
            params.update({
                "model_type": "mistral",
                "n_ctx": 4096
            })
    elif model_family == "phi":
        if provider in ["llama.cpp", "ctransformers"]:
            params.update({
                "model_type": "phi",
                "n_ctx": 2048
            })
    elif model_family == "gemma":
        if provider in ["llama.cpp", "ctransformers"]:
            params.update({
                "model_type": "gemma",
                "n_ctx": 4096
            })
    
    # Optimize parameters based on GPU availability
    return optimize_model_params(params, provider)