#!/usr/bin/env python3
"""
Docker Permission Fix Helper Script for Ubuntu

This script fixes common Docker permission issues on Ubuntu and
creates a proper Docker environment for running Ollama.

Run with:
    sudo python3 fix_docker_permissions.py
"""

import os
import sys
import subprocess
import getpass
import time

def print_step(step, message):
    """Print a step with formatting."""
    print(f"\n[{step}] {message}")
    print("=" * 60)

def run_command(command, check=True):
    """Run a shell command and print output."""
    print(f"Running: {command}")
    try:
        result = subprocess.run(
            command,
            shell=True,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=check
        )
        if result.stdout:
            print(result.stdout)
        if result.stderr and not check:
            print(f"Warning: {result.stderr}")
        return True
    except subprocess.CalledProcessError as e:
        print(f"Error: {e}")
        if e.stderr:
            print(f"Details: {e.stderr}")
        return False

def is_docker_installed():
    """Check if Docker is installed."""
    try:
        result = subprocess.run(
            ["docker", "--version"],
            capture_output=True,
            text=True
        )
        return result.returncode == 0
    except FileNotFoundError:
        return False

def is_user_in_docker_group(username):
    """Check if user is in the docker group."""
    try:
        result = subprocess.run(
            ["groups", username],
            capture_output=True,
            text=True
        )
        return "docker" in result.stdout
    except Exception:
        return False

def check_docker_socket_permissions():
    """Check permissions on the Docker socket."""
    try:
        result = subprocess.run(
            ["ls", "-la", "/var/run/docker.sock"],
            capture_output=True,
            text=True
        )
        if result.returncode == 0:
            print(f"Docker socket permissions: {result.stdout.strip()}")
            return True
        return False
    except Exception:
        return False

def fix_docker_permissions():
    """Fix Docker permissions issues."""
    # Get the current username
    current_user = os.getenv("SUDO_USER") or getpass.getuser()
    
    print_step(1, "Checking Docker installation")
    
    if not is_docker_installed():
        print("Docker is not installed. Installing Docker...")
        
        # Add Docker's official GPG key
        run_command("apt-get update")
        run_command("apt-get install -y apt-transport-https ca-certificates curl software-properties-common")
        run_command("curl -fsSL https://download.docker.com/linux/ubuntu/gpg | apt-key add -")
        
        # Add Docker repository
        run_command('add-apt-repository "deb [arch=$(dpkg --print-architecture)] https://download.docker.com/linux/ubuntu $(lsb_release -cs) stable"')
        
        # Install Docker
        run_command("apt-get update")
        run_command("apt-get install -y docker-ce docker-ce-cli containerd.io")
    else:
        print("Docker is already installed.")
    
    print_step(2, "Setting up Docker group")
    
    # Create the docker group if it doesn't exist
    run_command("groupadd -f docker")
    
    # Add user to the docker group
    if not is_user_in_docker_group(current_user):
        print(f"Adding user {current_user} to the docker group...")
        run_command(f"usermod -aG docker {current_user}")
    else:
        print(f"User {current_user} is already in the docker group.")
    
    print_step(3, "Fixing Docker socket permissions")
    
    # Check if Docker socket exists
    if not os.path.exists("/var/run/docker.sock"):
        print("Docker socket doesn't exist. Starting Docker service...")
        run_command("systemctl start docker")
        time.sleep(2)
    
    # Ensure Docker socket has the right permissions
    check_docker_socket_permissions()
    run_command("chmod 666 /var/run/docker.sock")
    print("Updated Docker socket permissions:")
    check_docker_socket_permissions()
    
    print_step(4, "Restarting Docker service")
    
    # Restart Docker to apply changes
    run_command("systemctl restart docker")
    time.sleep(2)
    
    print_step(5, "Verifying Docker setup")
    
    # Check if Docker is working properly
    if run_command("docker info", check=False):
        print("Docker is configured correctly!")
    else:
        print("There might still be issues with Docker configuration.")
    
    print("\n" + "=" * 60)
    print("Docker permission fix completed!")
    print(f"For the changes to take full effect, log out and log back in, or run:")
    print(f"  su - {current_user}")
    print("=" * 60)
    print("\nTo test Docker, run as your normal user (not sudo):")
    print("  docker run hello-world")
    print("=" * 60)

if __name__ == "__main__":
    # Check if running as root
    if os.geteuid() != 0:
        print("This script needs to be run with sudo privileges.")
        print("Please run: sudo python3 fix_docker_permissions.py")
        sys.exit(1)
    
    fix_docker_permissions()