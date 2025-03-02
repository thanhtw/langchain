"""
Docker utility functions for managing Ollama containers.
Provides functions to run Ollama with appropriate hardware acceleration.
"""

import os
import sys
import subprocess
import platform
import streamlit as st
import time

def check_docker_installed():
    """
    Check if Docker is installed on the system.
    
    Returns:
        bool: True if Docker is installed, False otherwise
    """
    try:
        result = subprocess.run(
            ["docker", "--version"], 
            capture_output=True, 
            text=True
        )
        return result.returncode == 0
    except FileNotFoundError:
        return False

def install_docker(position_noti="content"):
    """
    Install Docker on Ubuntu.
    
    Args:
        position_noti (str): Where to display notifications ("content" or "sidebar")
    
    Returns:
        bool: True if installation was successful, False otherwise
    """
    try:
        if position_noti == "content":
            status_placeholder = st.empty()
            status_placeholder.info("Installing Docker...")
        else:
            status_placeholder = st.sidebar.empty()
            status_placeholder.info("Installing Docker...")
            
        # Update apt and install dependencies
        os.system("sudo apt-get update")
        os.system("sudo apt-get install -y apt-transport-https ca-certificates curl software-properties-common")
        
        # Add Docker's official GPG key
        os.system("curl -fsSL https://download.docker.com/linux/ubuntu/gpg | sudo apt-key add -")
        
        # Add Docker repository
        os.system("sudo add-apt-repository \"deb [arch=$(dpkg --print-architecture)] https://download.docker.com/linux/ubuntu $(lsb_release -cs) stable\"")
        
        # Install Docker
        os.system("sudo apt-get update")
        os.system("sudo apt-get install -y docker-ce docker-ce-cli containerd.io")
        
        # Allow current user to use Docker without sudo
        os.system("sudo groupadd -f docker")
        os.system(f"sudo usermod -aG docker {os.getenv('USER')}")
        
        status_placeholder.success("Docker installed successfully! You may need to log out and back in for group changes to take effect.")
        return True
    except Exception as e:
        if position_noti == "content":
            st.error(f"Error installing Docker: {str(e)}")
        else:
            st.sidebar.error(f"Error installing Docker: {str(e)}")
        return False

def has_nvidia_gpu():
    """
    Check if the system has an NVIDIA GPU.
    
    Returns:
        bool: True if NVIDIA GPU is detected, False otherwise
    """
    try:
        # Try nvidia-smi command
        result = subprocess.run(
            ["nvidia-smi"], 
            capture_output=True, 
            text=True
        )
        return result.returncode == 0
    except FileNotFoundError:
        # Check for CUDA devices using lspci
        try:
            result = subprocess.run(
                ["lspci"], 
                capture_output=True, 
                text=True
            )
            return "NVIDIA" in result.stdout
        except FileNotFoundError:
            return False

def has_amd_gpu():
    """
    Check if the system has an AMD GPU.
    
    Returns:
        bool: True if AMD GPU is detected, False otherwise
    """
    try:
        # Check using lspci
        result = subprocess.run(
            ["lspci"], 
            capture_output=True, 
            text=True
        )
        return "AMD" in result.stdout and "VGA" in result.stdout
    except FileNotFoundError:
        return False

def install_nvidia_toolkit():
    """
    Install NVIDIA Container Toolkit for Docker.
    
    Returns:
        bool: True if installation was successful, False otherwise
    """
    try:
        # Add NVIDIA package repositories
        os.system("distribution=$(. /etc/os-release;echo $ID$VERSION_ID)")
        os.system("curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey | sudo apt-key add -")
        os.system("curl -s -L https://nvidia.github.io/nvidia-docker/$distribution/nvidia-docker.list | sudo tee /etc/apt/sources.list.d/nvidia-docker.list")
        
        # Install nvidia-docker2
        os.system("sudo apt-get update")
        os.system("sudo apt-get install -y nvidia-docker2")
        
        # Restart Docker daemon
        os.system("sudo systemctl restart docker")
        
        return True
    except Exception as e:
        st.error(f"Error installing NVIDIA Container Toolkit: {str(e)}")
        return False

def check_ollama_running():
    """
    Check if Ollama container is running.
    
    Returns:
        bool: True if Ollama is running, False otherwise
    """
    try:
        # Check for running container
        result = subprocess.run(
            ["docker", "ps", "--filter", "name=ollama", "--format", "{{.Names}}"],
            capture_output=True,
            text=True
        )
        
        if "ollama" in result.stdout:
            return True
            
        # If not found in running containers, try to check the API
        import requests
        try:
            response = requests.get("http://localhost:11434/api/tags", timeout=2)
            return response.status_code == 200
        except requests.exceptions.RequestException:
            return False
    except Exception:
        return False

def remove_running_container(container_name, position_noti="content"):
    """
    Remove a running Docker container.
    
    Args:
        container_name (str): Name of the container to remove
        position_noti (str): Where to display notifications ("content" or "sidebar")
    """
    try:
        # Check if container exists
        result = subprocess.run(
            ["docker", "ps", "-a", "--filter", f"name={container_name}", "--format", "{{.Names}}"],
            capture_output=True,
            text=True
        )
        
        if container_name in result.stdout:
            if position_noti == "content":
                st.info(f"Removing existing {container_name} container...")
            else:
                st.sidebar.info(f"Removing existing {container_name} container...")
                
            # Stop the container
            os.system(f"docker stop {container_name}")
            # Remove the container
            os.system(f"docker rm {container_name}")
            
            if position_noti == "content":
                st.success(f"Removed existing {container_name} container.")
            else:
                st.sidebar.success(f"Removed existing {container_name} container.")
    except Exception as e:
        if position_noti == "content":
            st.error(f"Error removing container: {str(e)}")
        else:
            st.sidebar.error(f"Error removing container: {str(e)}")

def run_ollama_container(position_noti="content"):
    """
    Run Ollama container based on the hardware type (NVIDIA GPU, AMD GPU, or CPU-only).
    Automatically installs Docker if needed.
    
    Args:
        position_noti (str): Where to display notifications ("content" or "sidebar")
    
    Returns:
        bool: True if Ollama is running successfully, False otherwise
    """
    system = platform.system().lower()
    container_name = "ollama"
    
    # Check if we're on Linux
    if system != "linux":
        if position_noti == "content":
            st.error(f"This function is designed for Ubuntu Linux, but detected {system}.")
        else:
            st.sidebar.error(f"This function is designed for Ubuntu Linux, but detected {system}.")
        return False
    
    # Check if Docker is installed
    if not check_docker_installed():
        if position_noti == "content":
            st.warning("Docker is not installed. Installing Docker...")
        else:
            st.sidebar.warning("Docker is not installed. Installing Docker...")
        
        if not install_docker(position_noti):
            return False
    
    # Check if docker socket is accessible without sudo
    try:
        result = subprocess.run(
            ["docker", "info"],
            capture_output=True,
            text=True
        )
        
        if result.returncode != 0 and "permission denied" in result.stderr.lower():
            message = """
            Permission denied accessing Docker. Run these commands and then restart:
            
            ```
            sudo groupadd docker
            sudo usermod -aG docker $USER
            newgrp docker
            ```
            """
            if position_noti == "content":
                st.error(message)
            else:
                st.sidebar.error(message)
            return False
    except Exception as e:
        if position_noti == "content":
            st.error(f"Error checking Docker permissions: {str(e)}")
        else:
            st.sidebar.error(f"Error checking Docker permissions: {str(e)}")
        return False
    
    # Remove the container if it's already running
    remove_running_container(container_name, position_noti=position_noti)
    
    # Create status placeholder
    if position_noti == "content":
        status_placeholder = st.empty()
    else:
        status_placeholder = st.sidebar.empty()
    
    # Run container based on GPU availability
    try:
        if has_nvidia_gpu():
            status_placeholder.info("NVIDIA GPU detected. Installing NVIDIA Container Toolkit if necessary...")
            install_nvidia_toolkit()  # Ensure NVIDIA toolkit is installed
            
            # Run Ollama container with NVIDIA GPU
            result = subprocess.run(
                ["docker", "run", "-d", "--gpus=all", "-v", "ollama:/root/.ollama", 
                 "-p", "11434:11434", "--name", container_name, "ollama/ollama"],
                capture_output=True,
                text=True
            )
            
            if result.returncode != 0:
                status_placeholder.error(f"Error starting Ollama container: {result.stderr}")
                return False
                
            status_placeholder.success("Ollama container running with NVIDIA GPU!")
        elif has_amd_gpu():
            status_placeholder.info("AMD GPU detected. Starting Ollama with ROCm support...")
            
            # Run Ollama container with AMD GPU
            result = subprocess.run(
                ["docker", "run", "-d", "--device", "/dev/kfd", "--device", "/dev/dri", 
                 "-v", "ollama:/root/.ollama", "-p", "11434:11434", 
                 "--name", container_name, "ollama/ollama:rocm"],
                capture_output=True,
                text=True
            )
            
            if result.returncode != 0:
                status_placeholder.error(f"Error starting Ollama container: {result.stderr}")
                return False
                
            status_placeholder.success("Ollama container running with AMD GPU!")
        else:
            status_placeholder.info("No GPU detected. Starting Ollama with CPU-only support...")
            
            # Run Ollama container with CPU-only
            result = subprocess.run(
                ["docker", "run", "-d", "-v", "ollama:/root/.ollama", "-p", "11434:11434", 
                 "--name", container_name, "ollama/ollama"],
                capture_output=True,
                text=True
            )
            
            if result.returncode != 0:
                status_placeholder.error(f"Error starting Ollama container: {result.stderr}")
                return False
                
            status_placeholder.success("Ollama container running with CPU-only!")
        
        # Wait for Ollama to start
        status_placeholder.info("Waiting for Ollama to initialize...")
        
        # Wait up to 30 seconds for Ollama to start
        for _ in range(15):
            if check_ollama_running():
                status_placeholder.success("Ollama is up and running!")
                return True
            time.sleep(2)
        
        status_placeholder.warning("Ollama container was started but is not responding yet. It might need more time to initialize.")
        return False
        
    except Exception as e:
        if position_noti == "content":
            st.error(f"Error running Ollama container: {str(e)}")
        else:
            st.sidebar.error(f"Error running Ollama container: {str(e)}")
        return False

def pull_ollama_model(model_name, position_noti="content"):
    """
    Pull an Ollama model.
    
    Args:
        model_name (str): Name of the model to pull
        position_noti (str): Where to display notifications ("content" or "sidebar")
    
    Returns:
        bool: True if successful, False otherwise
    """
    try:
        if not check_ollama_running():
            if position_noti == "content":
                st.error("Ollama is not running. Please start Ollama first.")
            else:
                st.sidebar.error("Ollama is not running. Please start Ollama first.")
            return False
        
        # Create a placeholder for status updates
        if position_noti == "content":
            status_placeholder = st.empty()
        else:
            status_placeholder = st.sidebar.empty()
            
        status_placeholder.info(f"Pulling model {model_name}. This may take a while...")
        
        # Use Docker exec to run the Ollama pull command
        result = subprocess.run(
            ["docker", "exec", "ollama", "ollama", "pull", model_name],
            capture_output=True,
            text=True
        )
        
        if result.returncode != 0:
            status_placeholder.error(f"Error pulling model: {result.stderr}")
            return False
            
        status_placeholder.success(f"Successfully pulled model {model_name}!")
        return True
    except Exception as e:
        if position_noti == "content":
            st.error(f"Error pulling model: {str(e)}")
        else:
            st.sidebar.error(f"Error pulling model: {str(e)}")
        return False