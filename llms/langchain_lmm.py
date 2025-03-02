"""
LangChain integration module for managing and interacting with locally run LLM models.
Optimized for Llama and DeepSeek models running on both Linux and Windows platforms.
"""

import os
import sys
import platform
import streamlit as st
import backoff
import glob
import subprocess
import time
import shutil
from pathlib import Path
from dotenv import load_dotenv
from typing import List, Dict, Any, Optional, Union
from llms.base import LLM

# Import LangChain components
from langchain_community.llms import (
    HuggingFaceHub, 
    HuggingFacePipeline, 
    Ollama, 
    LlamaCpp,
    CTransformers
)
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.callbacks.manager import CallbackManager

# Load environment variables
load_dotenv()

class LangChainLLM(LLM):
    """
    Class for interacting with locally run LLM models via LangChain.
    Optimized for Llama and DeepSeek models.
    """
    
    def __init__(self, model_path: str, provider: str, model_params: Dict[str, Any] = None, position_noti: str = "content"):
        """
        Initialize the LangChainLLM instance.
        
        Args:
            model_path (str): Path to the model or name of the model
            provider (str): Provider/backend to use ("ollama", "llama.cpp", "transformers", "ctransformers")
            model_params (dict, optional): Additional parameters for the model
            position_noti (str): Where to display notifications ("content" or "sidebar")
        """
        self.model_path = model_path
        self.provider = provider
        self.model_params = model_params or {}
        self.position_noti = position_noti
        self.callback_manager = CallbackManager([StreamingStdOutCallbackHandler()])
        
        # Apply GPU optimizations before initializing
        from utils.gpu_utils import optimize_model_params, is_gpu_available
        self.gpu_available = is_gpu_available()
        self.optimized_params = optimize_model_params(self.model_params, self.provider)
        
        # Initialize the LLM with optimized parameters
        self.llm = self._initialize_llm()
        
    def _initialize_llm(self) -> Any:
        """
        Initialize the appropriate LLM based on provider with GPU optimization.
        
        Returns:
            The initialized LLM object
        
        Raises:
            Exception: If initialization fails
        """
        try:
            # Use optimized parameters
            temperature = self.optimized_params.get("temperature", 0.7)
            max_tokens = self.optimized_params.get("max_tokens", 512)
            
            # Initialize LLM based on provider with GPU awareness
            if self.provider == "ollama":
                self._show_notification(f"Initializing Ollama model: {self.model_path}", "info")
                base_url = os.getenv("OLLAMA_BASE_URL") or "http://localhost:11434"
                
                # Check if Ollama is running
                self._check_ollama_running(base_url)
                
                # Ollama handles GPU internally
                if self.gpu_available:
                    self._show_notification("GPU detected - Ollama will use GPU automatically", "info")
                
                return Ollama(
                    model=self.model_path,
                    base_url=base_url,
                    temperature=temperature,
                    callback_manager=self.callback_manager
                )
                
            elif self.provider == "llama.cpp":
                self._show_notification(f"Loading model with llama.cpp: {self.model_path}", "info")
                
                # Check if file exists
                if not os.path.exists(self.model_path):
                    self._show_notification(f"Model file not found: {self.model_path}", "error")
                    raise FileNotFoundError(f"Model file not found: {self.model_path}")
                
                # Get model specific parameters from optimized params
                n_ctx = self.optimized_params.get("n_ctx", 2048)
                n_gpu_layers = self.optimized_params.get("n_gpu_layers", 0)
                
                if self.gpu_available and n_gpu_layers > 0:
                    self._show_notification(f"GPU acceleration enabled with {n_gpu_layers} layers", "success")
                
                return LlamaCpp(
                    model_path=self.model_path,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    n_ctx=n_ctx,
                    n_gpu_layers=n_gpu_layers,
                    callback_manager=self.callback_manager,
                    verbose=True
                )
                
            elif self.provider == "transformers":
                self._show_notification(f"Loading model with Transformers: {self.model_path}", "info")
                
                try:
                    from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
                    import torch
                    
                    # Get device information
                    device = "cuda" if self.gpu_available else "cpu"
                    self._show_notification(f"Using device: {device}", "info")
                    
                    # Get model parameters from optimized params
                    load_in_8bit = self.optimized_params.get("load_in_8bit", False)
                    load_in_4bit = self.optimized_params.get("load_in_4bit", False)
                    device_map = self.optimized_params.get("device_map", "auto" if device == "cuda" else None)
                    
                    # Load tokenizer and model
                    tokenizer = AutoTokenizer.from_pretrained(self.model_path)
                    
                    # Prepare model loading kwargs
                    model_kwargs = {
                        "device_map": device_map,
                    }
                    
                    if load_in_8bit and device == "cuda":
                        model_kwargs["load_in_8bit"] = True
                        self._show_notification("Using 8-bit quantization for better GPU memory usage", "info")
                    elif load_in_4bit and device == "cuda":
                        model_kwargs["load_in_4bit"] = True
                        self._show_notification("Using 4-bit quantization for better GPU memory usage", "info")
                        
                    model = AutoModelForCausalLM.from_pretrained(
                        self.model_path,
                        **model_kwargs
                    )
                    
                    # Create pipeline
                    pipe = pipeline(
                        "text-generation",
                        model=model,
                        tokenizer=tokenizer,
                        max_new_tokens=max_tokens,
                        temperature=temperature,
                        top_p=self.optimized_params.get("top_p", 0.95),
                        top_k=self.optimized_params.get("top_k", 50),
                        device=0 if device == "cuda" else -1
                    )
                    
                    return HuggingFacePipeline(
                        pipeline=pipe,
                        model_id=self.model_path
                    )
                    
                except Exception as e:
                    self._show_notification(f"Error loading Transformers model: {str(e)}", "error")
                    raise
                    
            elif self.provider == "ctransformers":
                self._show_notification(f"Loading model with CTransformers: {self.model_path}", "info")
                
                # Check if file exists
                if not os.path.exists(self.model_path):
                    self._show_notification(f"Model file not found: {self.model_path}", "error")
                    raise FileNotFoundError(f"Model file not found: {self.model_path}")
                
                # Get configuration from optimized params
                model_type = self.optimized_params.get("model_type", "llama")
                gpu_layers = self.optimized_params.get("gpu_layers", 0)
                
                if self.gpu_available and gpu_layers > 0:
                    self._show_notification(f"GPU acceleration enabled with {gpu_layers} layers", "success")
                
                config = {
                    "max_new_tokens": max_tokens,
                    "temperature": temperature,
                    "context_length": self.optimized_params.get("context_length", 2048),
                    "gpu_layers": gpu_layers
                }
                
                return CTransformers(
                    model=self.model_path,
                    model_type=model_type,
                    config=config
                )
                
            else:
                self._show_notification(f"Unsupported provider: {self.provider}", "error")
                raise ValueError(f"Unsupported provider: {self.provider}")
                
            self._show_notification(f"Model {self.model_path} initialized successfully.", "success")
            
        except Exception as e:
            self._show_notification(f"Error initializing model: {str(e)}", "error")
            raise
            
    def _check_ollama_running(self, base_url: str):
        """Check if Ollama is running and start it if necessary."""
        import requests
        
        try:
            response = requests.get(f"{base_url}/api/tags")
            if response.status_code != 200:
                self._show_notification("Ollama server returned an error. Please check if it's running correctly.", "error")
                self._start_ollama()
        except requests.ConnectionError:
            self._show_notification("Ollama server is not running. Attempting to start it...", "warning")
            self._start_ollama()
    
    def _start_ollama(self):
        """Attempt to start Ollama server."""
        try:
            # Different startup methods based on platform
            system = platform.system()
            
            if system == "Windows":
                # For Windows, try to start Ollama using the installed executable
                subprocess.Popen(["ollama", "serve"], 
                                shell=True, 
                                stdout=subprocess.PIPE,
                                stderr=subprocess.PIPE,
                                creationflags=subprocess.CREATE_NO_WINDOW)
                self._show_notification("Attempted to start Ollama. Please wait a moment...", "info")
                
            elif system == "Linux":
                # For Linux, use the systemd service if available
                try:
                    subprocess.run(["systemctl", "--user", "start", "ollama"], 
                                check=True, 
                                stdout=subprocess.PIPE,
                                stderr=subprocess.PIPE)
                    self._show_notification("Started Ollama service.", "info")
                except (subprocess.SubprocessError, FileNotFoundError):
                    # If systemd fails, try running directly
                    subprocess.Popen(["ollama", "serve"], 
                                    stdout=subprocess.PIPE,
                                    stderr=subprocess.PIPE)
                    self._show_notification("Attempted to start Ollama directly. Please wait a moment...", "info")
                    
            elif system == "Darwin":  # macOS
                # For macOS, try running directly
                subprocess.Popen(["ollama", "serve"], 
                                stdout=subprocess.PIPE,
                                stderr=subprocess.PIPE)
                self._show_notification("Attempted to start Ollama. Please wait a moment...", "info")
                
            # Wait a moment to give Ollama time to start
            time.sleep(3)
            
        except Exception as e:
            self._show_notification(f"Failed to start Ollama: {str(e)}", "error")
            raise

    def chat(self, messages: List[Dict[str, str]], options: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Send messages to the model and return the assistant's response.
        
        Args:
            messages (list): List of message objects with role and content
            options (dict, optional): Additional options for the model
            
        Returns:
            dict: Response information including content and metadata
        """
        try:
            # Format messages for LangChain
            formatted_messages = []
            
            for message in messages:
                role = message.get('role', '').lower()
                content = message.get('content', '')
                
                if role == 'system':
                    # Add system message as human message with system prefix
                    formatted_messages.append({
                        'role': 'user',
                        'content': f"[System Instruction] {content}"
                    })
                else:
                    formatted_messages.append(message)
            
            # Process options
            temperature = options.get('temperature', 0.7) if options else 0.7
            max_tokens = options.get('max_tokens', 1000) if options else 1000
            
            # Use different approach based on provider
            if self.provider in ["openai", "anthropic", "google"]:
                # These providers use the chat interface
                response = self.llm.invoke(formatted_messages)
                assistant_message = response.content
                
            else:
                # For other providers, concatenate messages and generate completion
                prompt = self._format_prompt(formatted_messages)
                response = self.llm.invoke(prompt)
                assistant_message = response
            
            # Return response in a format similar to the original implementation
            return {
                "content": assistant_message,
                "model": getattr(self, 'model_name', self.model_path),
                "created_at": None,  # LangChain doesn't provide this
                "total_duration": None,  # LangChain doesn't provide this
                "done": True
            }
                
        except Exception as e:
            print(f"Error in chat: {e}")
            return {"content": f"Error: {str(e)}", "error": True}

    def _format_prompt(self, messages: List[Dict[str, str]]) -> str:
        """
        Format a list of messages into a prompt string.
        
        Args:
            messages (list): List of message objects with role and content
            
        Returns:
            str: Formatted prompt string
        """
        prompt = ""
        for message in messages:
            role = message.get('role', '').lower()
            content = message.get('content', '')
            
            if role == 'system':
                prompt += f"System: {content}\n\n"
            elif role == 'user':
                prompt += f"User: {content}\n\n"
            elif role == 'assistant':
                prompt += f"Assistant: {content}\n\n"
            else:
                prompt += f"{content}\n\n"
                
        prompt += "Assistant: "
        return prompt

    @backoff.on_exception(backoff.expo, Exception, max_tries=3)
    def create_agentic_chunker_message(self, system_prompt: str, messages: List[Dict[str, str]], max_tokens: int = 1000, temperature: float = 1) -> str:
        """
        Create a message using the agentic chunker with automatic retry.
        
        Args:
            system_prompt (str): System prompt to guide the model
            messages (list): List of message objects
            max_tokens (int): Maximum number of tokens for context
            temperature (float): Temperature parameter for generation
            
        Returns:
            str: Generated content from the model
            
        Raises:
            Exception: Propagates exceptions after retries are exhausted
        """
        try:
            langchain_messages = [
                {"role": "system", "content": system_prompt}
            ] + messages

            response = self.chat(
                langchain_messages, 
                {"temperature": temperature, "max_tokens": max_tokens}
            )
            
            return response.get("content", "")
            
        except Exception as e:
            print(f"Error occurred: {e}, retrying...")
            raise e
        
    def generate_content(self, prompt: str) -> str:
        """
        Generate content from a prompt using the model.
        
        Args:
            prompt (str): The prompt to generate content from
            
        Returns:
            str: Generated content or empty string on error
        """
        try:
            # Different handling for different providers
            if self.provider in ["openai", "anthropic", "google"]:
                messages = [{"role": "user", "content": prompt}]
                response = self.llm.invoke(messages)
                return response.content
            else:
                response = self.llm.invoke(prompt)
                return response
                
        except Exception as e:
            print(f"Error generating content: {e}")
            return f"Error generating content: {str(e)}"

    def _show_notification(self, message: str, notification_type: str) -> None:
        """
        Show a notification in the Streamlit UI.
        
        Args:
            message (str): Message to display
            notification_type (str): Type of notification (success, error, info)
        """
        if self.position_noti == "content":
            getattr(st, notification_type)(message)
        else:
            getattr(st.sidebar, notification_type)(message)


class LangChainManager:
    """
    Class for managing LangChain model configurations, focused on local models.
    """
    
    def __init__(self):
        """Initialize the LangChain manager."""
        load_dotenv()
        self.models_dir = os.getenv("MODELS_DIR", os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "models"))
        
        # Ensure models directory exists
        os.makedirs(self.models_dir, exist_ok=True)
        # Create Ollama model directory
        os.makedirs(os.path.join(self.models_dir, "ollama"), exist_ok=True)
        
    def get_available_providers(self) -> Dict[str, str]:
        """
        Get available local LLM providers.
        
        Returns:
            dict: Provider names and descriptions
        """
        providers = {
            "ollama": "Ollama (easiest setup, recommended for beginners)",
            "llama.cpp": "Llama.cpp (best performance for GGUF models)",
            "transformers": "Transformers (HuggingFace models, high RAM usage)",
            "ctransformers": "CTransformers (alternative GGUF loader)"
        }
        return providers
        
    def scan_for_local_models(self) -> Dict[str, List[Dict[str, str]]]:
        """
        Scan for locally available models in the models directory.
        
        Returns:
            dict: Dictionary of model types and their available models
        """
        result = {
            "gguf": [],
            "huggingface": [],
            "ollama": []
        }
        
        # Check if models directory exists
        if not os.path.exists(self.models_dir):
            return result
            
        # Scan for GGUF models
        gguf_models = glob.glob(os.path.join(self.models_dir, "**", "*.gguf"), recursive=True)
        for model_path in gguf_models:
            model_name = os.path.basename(model_path)
            result["gguf"].append({
                "id": model_path,
                "name": model_name,
                "description": f"GGUF model ({self._get_size_str(model_path)})"
            })
            
        # Scan for HuggingFace models (look for config.json files)
        for root, dirs, files in os.walk(self.models_dir):
            if "config.json" in files:
                model_path = root
                model_name = os.path.basename(model_path)
                result["huggingface"].append({
                    "id": model_path,
                    "name": model_name,
                    "description": "HuggingFace model"
                })
        
        # Scan for Ollama models in our models directory
        ollama_models_dir = os.path.join(self.models_dir, "ollama")
        if os.path.exists(ollama_models_dir):
            for file in os.listdir(ollama_models_dir):
                if file.endswith(".bin"):
                    model_name = file[:-4]  # Remove .bin extension
                    model_path = os.path.join(ollama_models_dir, file)
                    result["ollama"].append({
                        "id": model_name,
                        "name": model_name,
                        "description": f"Ollama model ({self._get_size_str(model_path)})"
                    })
        
        # Also check Ollama API for models not in our directory
        try:
            import requests
            base_url = os.getenv("OLLAMA_BASE_URL") or "http://localhost:11434"
            response = requests.get(f"{base_url}/api/tags")
            
            if response.status_code == 200:
                api_models = response.json().get("models", [])
                # Add models from API not already in our list
                for model in api_models:
                    model_name = model["name"]
                    # Only add if not already in our list
                    if not any(m["id"] == model_name for m in result["ollama"]):
                        result["ollama"].append({
                            "id": model_name,
                            "name": model_name,
                            "description": f"Ollama API model (Size: {model.get('size', 'Unknown')})"
                        })
        except Exception as e:
            # If API call fails, we still have models from directory scan
            print(f"Error checking Ollama API: {e}")
                
        return result
    
    def get_ollama_models(self) -> List[Dict[str, str]]:
        """
        Get list of available Ollama models, including ones that can be pulled.
        
        Returns:
            list: List of model information dictionaries
        """
        # Standard models that can be pulled
        library_models = [
            {"id": "llama3", "name": "Llama 3", "description": "Meta's Llama 3 model", "pulled": False},
            {"id": "llama3:8b", "name": "Llama 3 (8B)", "description": "Meta's Llama 3 8B model", "pulled": False},
            {"id": "llama3:1b", "name": "Llama 3 (1B)", "description": "Meta's Llama 3 1B model", "pulled": False},
            {"id": "deepseek-coder", "name": "DeepSeek Coder", "description": "Code-specialized LLM", "pulled": False},
            {"id": "deepseek-llm", "name": "DeepSeek LLM", "description": "General purpose LLM", "pulled": False},
            {"id": "phi3:mini", "name": "Phi-3 Mini", "description": "Microsoft Phi-3 model", "pulled": False},
            {"id": "phi3:medium", "name": "Phi-3 Medium", "description": "Microsoft's larger Phi-3 model", "pulled": False},
            {"id": "gemma:2b", "name": "Gemma 2B", "description": "Google's lightweight Gemma model", "pulled": False},
            {"id": "gemma:7b", "name": "Gemma 7B", "description": "Google's Gemma 7B model", "pulled": False},
            {"id": "mistral", "name": "Mistral 7B", "description": "Mistral AI's 7B model", "pulled": False},
            {"id": "mixtral:8x7b", "name": "Mixtral 8x7B", "description": "Mistral's MoE model", "pulled": False},
            {"id": "codellama", "name": "Code Llama", "description": "Meta's code-specialized model", "pulled": False},
            {"id": "qwen:14b", "name": "Qwen 14B", "description": "Alibaba's Qwen model", "pulled": False},
            {"id": "qwen:72b", "name": "Qwen 72B", "description": "Alibaba's large Qwen model", "pulled": False},
            {"id": "neural-chat", "name": "Neural Chat", "description": "Optimized conversational model", "pulled": False},
            {"id": "stablelm:zephyr", "name": "StableLM Zephyr", "description": "Stability AI's Zephyr model", "pulled": False}
        ]
        
        # First, check for locally saved Ollama models in ./models/ollama
        ollama_models_dir = os.path.join(self.models_dir, "ollama")
        if os.path.exists(ollama_models_dir):
            for file in os.listdir(ollama_models_dir):
                if file.endswith(".bin"):
                    model_name = file[:-4]  # Remove .bin extension
                    # Mark as pulled if it exists in library_models
                    for model in library_models:
                        if model["id"] == model_name:
                            model["pulled"] = True
                            break
                    # Add if not in library_models
                    if not any(model["id"] == model_name for model in library_models):
                        library_models.append({
                            "id": model_name,
                            "name": model_name,
                            "description": "Local Ollama model",
                            "pulled": True
                        })
        
        # Then check Ollama API
        try:
            import requests
            
            # Try to get list of already pulled models from Ollama
            base_url = os.getenv("OLLAMA_BASE_URL") or "http://localhost:11434"
            response = requests.get(f"{base_url}/api/tags")
            
            if response.status_code == 200:
                pulled_models = response.json().get("models", [])
                pulled_ids = [model["name"] for model in pulled_models]
                
                # Mark models as pulled if they exist locally
                for model in library_models:
                    if model["id"] in pulled_ids:
                        model["pulled"] = True
                
                # Add any pulled models that aren't in our standard list
                for pulled_model in pulled_models:
                    model_id = pulled_model["name"]
                    if not any(model["id"] == model_id for model in library_models):
                        library_models.append({
                            "id": model_id,
                            "name": model_id,
                            "description": f"Size: {pulled_model.get('size', 'Unknown')}",
                            "pulled": True
                        })
            
            return library_models
                
        except Exception as e:
            print(f"Error connecting to Ollama: {str(e)}")
            # Return list with local models marked as pulled
            return library_models
            
    def download_ollama_model(self, model_name: str) -> bool:
        """
        Download a model using Ollama and save in models directory.
        
        Args:
            model_name (str): Name of the model to download
            
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            import requests
            import time
            
            # Create models directory if it doesn't exist
            os.makedirs(self.models_dir, exist_ok=True)
            
            # Create an Ollama subdirectory
            ollama_models_dir = os.path.join(self.models_dir, "ollama")
            os.makedirs(ollama_models_dir, exist_ok=True)
            
            # Try to pull the model using Ollama API
            base_url = os.getenv("OLLAMA_BASE_URL") or "http://localhost:11434"
            
            # Create progress placeholder
            progress_placeholder = st.empty()
            progress_placeholder.info(f"Starting download of {model_name}...")
            
            # Start the pull operation
            response = requests.post(
                f"{base_url}/api/pull",
                json={"name": model_name, "stream": False}  # Non-streaming for simplicity
            )
            
            if response.status_code != 200:
                st.error(f"Failed to start model download: {response.text}")
                return False
            
            # Monitor progress
            progress_placeholder.info(f"Downloading {model_name}... This may take a while.")
            
            # Poll for completion
            model_ready = False
            start_time = time.time()
            while not model_ready and (time.time() - start_time) < 1800:  # 30 minute timeout
                try:
                    # Check if model exists in list of models
                    check_response = requests.get(f"{base_url}/api/tags")
                    if check_response.status_code == 200:
                        models = check_response.json().get("models", [])
                        if any(model["name"] == model_name for model in models):
                            model_ready = True
                            progress_placeholder.success(f"Model {model_name} downloaded successfully!")
                            break
                    time.sleep(5)  # Check every 5 seconds
                    
                    # Update progress message to show we're still working
                    if (time.time() - start_time) % 15 < 5:  # Change message every 15 seconds
                        progress_placeholder.info(f"Still downloading {model_name}... Please wait.")
                except Exception as e:
                    # Log error but continue polling
                    print(f"Error checking model status: {str(e)}")
                    time.sleep(5)
            
            if not model_ready:
                progress_placeholder.error(f"Download timeout for {model_name}. Model may still be downloading.")
                return False
            
            # Model is now available, try to copy it to our models directory
            try:
                # Get Ollama's model directory (platform-specific)
                if platform.system() == "Linux":
                    ollama_data_dir = os.path.expanduser("~/.ollama/models")
                elif platform.system() == "Darwin":  # macOS
                    ollama_data_dir = os.path.expanduser("~/.ollama/models")
                elif platform.system() == "Windows":
                    ollama_data_dir = os.path.join(os.getenv("LOCALAPPDATA"), "ollama", "models")
                else:
                    progress_placeholder.warning("Unsupported platform for Ollama model copying.")
                    return True  # Model is still usable via Ollama API
                
                # Look for model files
                if os.path.exists(ollama_data_dir):
                    # There might be multiple files with this model name
                    model_files = []
                    for file in os.listdir(ollama_data_dir):
                        if file.startswith(model_name.replace(":", "-")) and file.endswith(".bin"):
                            model_files.append(file)
                    
                    # Copy each model file
                    if model_files:
                        for file in model_files:
                            src_file = os.path.join(ollama_data_dir, file)
                            dest_file = os.path.join(ollama_models_dir, file)
                            
                            # Copy with progress update
                            file_size = os.path.getsize(src_file)
                            progress_placeholder.info(f"Saving {file} to models directory ({file_size/(1024*1024):.1f} MB)...")
                            
                            shutil.copy2(src_file, dest_file)
                        
                        progress_placeholder.success(f"Model {model_name} saved to local models directory.")
                    else:
                        progress_placeholder.warning(f"Could not find model files for {model_name} to copy.")
                else:
                    progress_placeholder.warning(f"Ollama data directory not found at {ollama_data_dir}")
            except Exception as e:
                # Don't fail if we can't copy, just log a warning
                progress_placeholder.warning(f"Could not copy model file to models directory: {str(e)}")
            
            return True
                
        except Exception as e:
            st.error(f"Error downloading model: {str(e)}")
            return False
    
    def get_models_for_provider(self, provider: str) -> List[Dict[str, str]]:
        """
        Get available models for a specific provider.
        
        Args:
            provider (str): The provider name
            
        Returns:
            list: List of model information dictionaries
        """
        if provider == "ollama":
            return self.get_ollama_models()
            
        elif provider == "llama.cpp" or provider == "ctransformers":
            # These providers use GGUF models
            local_models = self.scan_for_local_models()
            return local_models["gguf"]
            
        elif provider == "transformers":
            # Transformers can use either local models or HuggingFace models
            local_models = self.scan_for_local_models()
            huggingface_models = local_models["huggingface"]
            
            # Add some recommended HuggingFace models that can be downloaded
            recommended = [
                {"id": "meta-llama/Meta-Llama-3-8B", "name": "Meta-Llama-3-8B", "description": "Meta's Llama 3 (8B) - HF format"},
                {"id": "deepseek-ai/deepseek-coder-6.7b-instruct", "name": "DeepSeek Coder 6.7B", "description": "Code-specialized model"},
                {"id": "microsoft/phi-3-mini-4k-instruct", "name": "Phi-3 Mini", "description": "Microsoft's compact Phi-3 model"},
                {"id": "google/gemma-2b-it", "name": "Gemma 2B Instruct", "description": "Google's instruction-tuned 2B model"}
            ]
            
            return huggingface_models + recommended
        else:
            return []
    
    def _get_size_str(self, file_path: str) -> str:
        """Get a human-readable file size string."""
        try:
            size_bytes = os.path.getsize(file_path)
            for unit in ['B', 'KB', 'MB', 'GB', 'TB']:
                if size_bytes < 1024.0:
                    return f"{size_bytes:.1f} {unit}"
                size_bytes /= 1024.0
            return f"{size_bytes:.1f} PB"
        except:
            return "Unknown size"
            
    def initialize_llm(self, model_path: str, provider: str, model_params: Dict[str, Any] = None) -> Optional[LangChainLLM]:
        """
        Initialize and return a LangChainLLM instance.
        
        Args:
            model_path (str): Path to the model or name of the model
            provider (str): Provider name
            model_params (dict, optional): Additional parameters for the model
            
        Returns:
            LangChainLLM: Initialized LangChainLLM instance or None if initialization fails
        """
        try:
            return LangChainLLM(model_path, provider, model_params)
        except Exception as e:
            st.error(f"Failed to initialize LLM: {str(e)}")
            return None
            
    def get_recommended_model_parameters(self, model_path: str, provider: str) -> Dict[str, Any]:
        """
        Get recommended parameters for a specific model.
        
        Args:
            model_path (str): Path to the model or name of the model
            provider (str): Provider name
            
        Returns:
            dict: Recommended parameters
        """
        # Base parameters
        params = {
            "temperature": 0.7,
            "max_tokens": 512
        }
        
        # Check if model name contains indicators
        model_name = os.path.basename(model_path).lower()
        
        # Special cases for Llama models
        if "llama" in model_name:
            if provider == "llama.cpp" or provider == "ctransformers":
                params.update({
                    "model_type": "llama",
                    "n_ctx": 4096,
                    "n_gpu_layers": 35 if "7b" in model_name else 0
                })
                
        # Special cases for DeepSeek models
        elif "deepseek" in model_name:
            if provider == "llama.cpp" or provider == "ctransformers":
                params.update({
                    "model_type": "deepseek",  # for ctransformers
                    "n_ctx": 4096,
                    "n_gpu_layers": 35 if "7b" in model_name else 0
                })
                
        # For transformers provider
        if provider == "transformers":
            # Check system memory to decide on quantization
            import psutil
            ram_gb = psutil.virtual_memory().total / (1024**3)
            
            # If less than 16GB RAM, suggest 4-bit quantization
            if ram_gb < 16:
                params["load_in_4bit"] = True
            # If between 16-32GB RAM, suggest 8-bit quantization
            elif ram_gb < 32:
                params["load_in_8bit"] = True
            
        return params
    
    def _initialize_ollama_with_fallback(self, model_path, base_url):
        """
        Initialize Ollama with fallback options to handle common errors.
        
        Args:
            model_path (str): Name of the model
            base_url (str): Ollama API base URL
            
        Returns:
            The initialized Ollama LLM object
        
        Raises:
            Exception: If all fallback attempts fail
        """
        import requests
        import time
        import platform
        import os
        import subprocess
        from langchain_community.llms import Ollama
        
        # Check if model exists
        try:
            response = requests.get(f"{base_url}/api/tags")
            if response.status_code == 200:
                models = response.json().get("models", [])
                model_exists = any(model["name"] == model_path for model in models)
                
                if not model_exists:
                    self._show_notification(f"Model {model_path} not found. Attempting to pull...", "info")
                    pull_response = requests.post(
                        f"{base_url}/api/pull",
                        json={"name": model_path, "stream": False}
                    )
                    
                    if pull_response.status_code != 200:
                        self._show_notification(f"Failed to pull model {model_path}. Status: {pull_response.status_code}", "error")
                        raise ValueError(f"Failed to pull model {model_path}")
                    
                    # Wait for model to be pulled
                    self._show_notification(f"Pulling model {model_path}. This may take some time...", "info")
                    pulled = False
                    for _ in range(30):  # Try for up to 5 minutes
                        time.sleep(10)
                        check_response = requests.get(f"{base_url}/api/tags")
                        if check_response.status_code == 200:
                            models = check_response.json().get("models", [])
                            if any(model["name"] == model_path for model in models):
                                pulled = True
                                break
                    
                    if not pulled:
                        self._show_notification(f"Timeout waiting for model {model_path} to be pulled", "error")
                        raise TimeoutError(f"Timeout waiting for model {model_path} to be pulled")
        except Exception as e:
            self._show_notification(f"Error checking/pulling model: {str(e)}", "error")
        
        # Try multiple initialization approaches
        errors = []
        
        # Approach 1: Standard initialization
        try:
            return Ollama(
                model=model_path,
                base_url=base_url,
                temperature=self.optimized_params.get("temperature", 0.7),
                callback_manager=self.callback_manager
            )
        except Exception as e:
            errors.append(f"Standard init failed: {str(e)}")
        
        # Approach 2: Try with different parameters
        try:
            return Ollama(
                model=model_path,
                base_url=base_url,
                temperature=self.optimized_params.get("temperature", 0.7),
                num_gpu=0,  # Force CPU mode
                num_thread=4  # Limit threads
            )
        except Exception as e:
            errors.append(f"CPU-only init failed: {str(e)}")
        
        # Approach 3: Restart Ollama service
        try:
            self._show_notification("Attempting to restart Ollama service...", "info")
            if platform.system() == "Linux":
                try:
                    # Try systemctl restart
                    subprocess.run(
                        ["systemctl", "--user", "restart", "ollama"],
                        check=False,
                        stdout=subprocess.PIPE,
                        stderr=subprocess.PIPE
                    )
                except:
                    # Try direct restart
                    try:
                        subprocess.run(["pkill", "-f", "ollama"], check=False)
                        time.sleep(2)
                        subprocess.Popen(
                            ["ollama", "serve"],
                            stdout=subprocess.PIPE,
                            stderr=subprocess.PIPE
                        )
                        time.sleep(5)  # Wait for service to start
                    except Exception as e:
                        errors.append(f"Failed to restart Ollama: {str(e)}")
            
            # Try initialization again
            return Ollama(
                model=model_path,
                base_url=base_url,
                temperature=self.optimized_params.get("temperature", 0.7)
            )
        except Exception as e:
            errors.append(f"Init after restart failed: {str(e)}")
        
        # If all approaches fail, suggest solutions
        error_message = "Failed to initialize Ollama model after multiple attempts:\n" + "\n".join(errors)
        self._show_notification(error_message, "error")
        self._show_notification(
            "Please try these solutions:\n"
            "1. Run the Ubuntu troubleshooter script\n"
            "2. Manually restart Ollama: `pkill -f ollama && ollama serve`\n"
            "3. Check logs: `journalctl -u ollama -f`\n"
            "4. Install missing dependencies: `sudo apt install nvidia-cuda-toolkit libnvidia-compute-*`\n"
            "5. Try a smaller model like 'phi:mini' or 'gemma:2b'",
            "info"
        )
        
        raise RuntimeError(error_message)