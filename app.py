"""
Main application entry point for the RAG-based tool.

This script initializes the Streamlit application with local model support,
optimized for running on both Linux and Windows platforms with GPU acceleration when available.
"""

import streamlit as st
import os
import sys
import platform
from pathlib import Path
from dotenv import load_dotenv
from config_manager import ConfigManager

# Add the current directory to the Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Load environment variables from .env file
load_dotenv()

def ensure_directories():
    """
    Ensure required directories exist.
    """
    # Create models directory if it doesn't exist
    models_dir = os.getenv("MODELS_DIR", os.path.join(os.path.dirname(os.path.abspath(__file__)), "models"))
    os.makedirs(models_dir, exist_ok=True)
    
    # Create .env file if it doesn't exist
    env_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env")
    if not os.path.exists(env_file):
        with open(env_file, "w") as f:
            f.write("# Models directory\n")
            f.write(f"MODELS_DIR={models_dir}\n")
            f.write("\n# Ollama Settings\n")
            f.write("OLLAMA_BASE_URL=http://localhost:11434\n")

def check_system_compatibility():
    """
    Check system compatibility and show relevant information.
    GPU detection is now centralized using PyTorch.
    
    Returns:
        dict: System information
    """
    # Import GPU utilities
    from utils.gpu_utils import is_gpu_available, get_gpu_info
    
    info = {
        "platform": platform.system(),
        "cpu_count": os.cpu_count(),
        "python_version": platform.python_version(),
        "gpu_available": False,
        "ram_gb": None
    }
    
    # Check for GPU using our centralized utility
    gpu_info = get_gpu_info()
    info["gpu_available"] = gpu_info["available"]
    if info["gpu_available"]:
        info["gpu_name"] = gpu_info["name"]
        info["gpu_memory"] = gpu_info["memory_gb"]
    
    # Check RAM
    try:
        import psutil
        info["ram_gb"] = psutil.virtual_memory().total / (1024**3)
    except ImportError:
        pass
    
    return info

def show_system_info(info):
    """
    Display system information.
    
    Args:
        info (dict): System information
    """
    st.sidebar.subheader("System Information")
    
    # Platform info
    st.sidebar.text(f"Platform: {info['platform']}")
    st.sidebar.text(f"CPU Cores: {info['cpu_count']}")
    
    # RAM info
    if info["ram_gb"]:
        st.sidebar.text(f"RAM: {info['ram_gb']:.1f} GB")
    
    # GPU info
    if info["gpu_available"]:
        st.sidebar.text(f"GPU: {info['gpu_name']}")
        st.sidebar.text(f"GPU Memory: {info['gpu_memory']:.1f} GB")
        st.sidebar.text("✅ GPU acceleration available and enabled")
    else:
        st.sidebar.text("❌ No GPU detected - using CPU mode")
    
    # Recommendations based on system
    st.sidebar.subheader("Recommendations")
    
    if info["ram_gb"] and info["ram_gb"] < 8:
        st.sidebar.warning("Low RAM detected. Use smaller models like Gemma 2B or Phi-3 Mini.")
    elif info["ram_gb"] and info["ram_gb"] < 16:
        st.sidebar.info("Medium RAM detected. Use quantized models (Q4) for best performance.")
    
    if info["gpu_available"]:
        if info["gpu_memory"] < 4:
            st.sidebar.warning("Limited GPU memory. Using 4-bit quantization for better efficiency.")
        elif info["gpu_memory"] < 8:
            st.sidebar.info("Medium GPU memory. Using 8-bit quantization for better efficiency.")
        else:
            st.sidebar.success("Good GPU memory available. Maximizing GPU usage.")
    else:
        st.sidebar.info("No GPU detected. CPU-only mode will be slower.")

def check_ollama_status():
    """Check and start Ollama if needed."""
    from utils.model_downloader import check_ollama_running, start_ollama
    
    # Check if Ollama is running
    if not check_ollama_running():
        st.sidebar.warning("Ollama is not running.")
        if st.sidebar.button("Start Ollama"):
            with st.sidebar.spinner("Starting Ollama..."):
                if start_ollama():
                    st.sidebar.success("Ollama started successfully!")
                else:
                    st.sidebar.error("Failed to start Ollama. Please start it manually.")
    else:
        st.sidebar.success("Ollama is running.")

def main():
    """
    Initialize and run the main application.
    
    This function creates a ConfigManager instance and uses it to
    render the sidebar and main content of the application.
    """
    # Set page configuration
    st.set_page_config(
        page_title="Local LLM RAG System",
        page_icon="🧠",
        layout="wide"
    )
    
    # Ensure necessary directories and files exist
    ensure_directories()
    
    # Check system compatibility
    sys_info = check_system_compatibility()
    
    # Show system information in sidebar
    show_system_info(sys_info)
    
    # Check Ollama status
    check_ollama_status()
    
    # Initialize configuration manager
    config_manager = ConfigManager()

    # Render sidebar
    config_manager.render_sidebar()

    # Render main content
    config_manager.render_main_content()

if __name__ == "__main__":
    main()