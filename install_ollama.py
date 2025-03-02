#!/usr/bin/env python3
"""
Standalone script to install and run Ollama in a Docker container on Ubuntu.
This script automatically detects hardware and runs the appropriate container.

Usage:
    python install_ollama.py
"""

import os
import sys
import platform
import argparse
import time

# Add the parent directory to the path so we can import our modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Try to import the docker_utils module
try:
    from utils.docker_utils import (
        check_docker_installed,
        install_docker,
        has_nvidia_gpu,
        has_amd_gpu,
        install_nvidia_toolkit,
        check_ollama_running,
        remove_running_container,
        run_ollama_container,
        pull_ollama_model
    )
except ImportError:
    print("Could not import docker_utils module. Make sure it's in the utils directory.")
    print("This script should be run from the project root directory.")
    sys.exit(1)

def print_status(message):
    """Print a status message with a timestamp."""
    print(f"[{time.strftime('%H:%M:%S')}] {message}")

def main():
    """Main function to run the script."""
    parser = argparse.ArgumentParser(description="Install and run Ollama in a Docker container")
    parser.add_argument("--pull", type=str, help="Pull a specific model after installation (e.g., 'llama3')")
    args = parser.parse_args()
    
    print_status("Starting Ollama container installation...")
    
    # Check if we're on Ubuntu
    if platform.system().lower() != "linux":
        print_status(f"This script is designed for Ubuntu Linux, but detected {platform.system()}.")
        print_status("Please install Docker and run the Ollama container manually.")
        sys.exit(1)
    
    # Check if Docker is installed
    if not check_docker_installed():
        print_status("Docker is not installed. Installing Docker...")
        if not install_docker(position_noti=None):
            print_status("Failed to install Docker. Please install it manually.")
            sys.exit(1)
        print_status("Docker installed successfully!")
    else:
        print_status("Docker is already installed.")
    
    # Check for GPU
    if has_nvidia_gpu():
        print_status("NVIDIA GPU detected!")
        print_status("Installing NVIDIA Container Toolkit...")
        install_nvidia_toolkit()
    elif has_amd_gpu():
        print_status("AMD GPU detected!")
    else:
        print_status("No GPU detected. Will use CPU only.")
    
    # Check if Ollama is already running
    if check_ollama_running():
        print_status("Ollama is already running. Stopping existing container...")
        remove_running_container("ollama", position_noti=None)
    
    # Run Ollama container without Streamlit UI
    print_status("Running Ollama container...")
    container_name = "ollama"
    
    if has_nvidia_gpu():
        print_status("Starting Ollama with NVIDIA GPU support...")
        os.system(f"docker run -d --gpus=all -v ollama:/root/.ollama -p 11434:11434 --name {container_name} ollama/ollama")
    elif has_amd_gpu():
        print_status("Starting Ollama with AMD GPU support...")
        os.system(f"docker run -d --device /dev/kfd --device /dev/dri -v ollama:/root/.ollama -p 11434:11434 --name {container_name} ollama/ollama:rocm")
    else:
        print_status("Starting Ollama with CPU-only support...")
        os.system(f"docker run -d -v ollama:/root/.ollama -p 11434:11434 --name {container_name} ollama/ollama")
    
    # Wait for Ollama to start
    print_status("Waiting for Ollama to initialize...")
    for i in range(15):
        if check_ollama_running():
            print_status("Ollama is now up and running!")
            break
        
        if i == 14:
            print_status("Ollama container was started but is not responding yet.")
            print_status("It might need more time to initialize. Check status with 'docker ps'.")
        else:
            print_status(f"Waiting for Ollama to start... ({i+1}/15)")
            time.sleep(2)
    
    # Pull model if specified
    if args.pull:
        print_status(f"Pulling model {args.pull}. This may take a while...")
        os.system(f"docker exec ollama ollama pull {args.pull}")
        print_status(f"Model {args.pull} pulled successfully!")
    
    print_status("Installation complete!")
    print_status("Ollama API is available at: http://localhost:11434")
    print_status("You can now use Ollama with your RAG application.")

if __name__ == "__main__":
    main()