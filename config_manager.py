"""
Configuration manager for the RAG application.

This module handles application configuration, UI rendering, and state management.
"""

import streamlit as st
import pandas as pd
import json
import os
import re
import numpy as np
from typing import List, Dict, Any, Tuple, Optional
from sentence_transformers import SentenceTransformer
import chromadb
from config.constants import (
    EN, VI, NONE, USER, ASSISTANT, ENGLISH, VIETNAMESE,
    LANGCHAIN_LLM, DEFAULT_LANGCHAIN_MODEL, DEFAULT_LANGCHAIN_PROVIDER,
    DB, VECTOR_SEARCH, LANGCHAIN_PROVIDERS, OLLAMA_MODELS,
    DEFAULT_CHUNK_SIZE, DEFAULT_CHUNK_OVERLAP, DEFAULT_NUM_DOCS_RETRIEVAL, MODEL_TYPES
)
from data.data_processor import DataProcessor
from llms import LangChainLLM, LangChainManager
from utils.model_downloader import download_model_with_streamlit, RECOMMENDED_MODELS
from prompts.prompt_engineering import PromptTemplate, PromptLibrary
# Add quality check widget integrated into chatbot
from utils.auto_quality_check import render_quality_check_widget_in_chatbot

"""
Enhanced vector search with relevance ranking.

This module provides improved vector search capabilities with
result ranking based on semantic similarity and metadata factors.
"""




class ConfigManager:
    """
    Configuration manager for the RAG application.
    Handles UI rendering, state management, and application configuration.
    """
    
    def __init__(self):
        """Initialize the configuration manager."""
        self.initialize_state()
        self.data_processor = DataProcessor(self)
        self.langchain_manager = LangChainManager()
        
        # Create models directory if it doesn't exist
        self.models_dir = os.getenv("MODELS_DIR", os.path.join(os.path.dirname(os.path.abspath(__file__)), "models"))
        os.makedirs(self.models_dir, exist_ok=True)

    def initialize_state(self):
        """Initialize all session state variables with default values."""
        defaults = {
            "language": None,
            "embedding_model": None,
            "embedding_model_name": None,
            "llm_type": LANGCHAIN_LLM,
            "llm_provider": DEFAULT_LANGCHAIN_PROVIDER,
            "llm_name": DEFAULT_LANGCHAIN_MODEL,
            "llm_model": None,
            "client": chromadb.PersistentClient("db"),
            "active_collections": {},
            "search_option": VECTOR_SEARCH,
            "open_dialog": None,
            "source_data": "UPLOAD",
            "chunks_df": pd.DataFrame(),
            "random_collection_name": None,
            "chunk_size": DEFAULT_CHUNK_SIZE,
            "chunk_overlap": DEFAULT_CHUNK_OVERLAP,
            "number_docs_retrieval": DEFAULT_NUM_DOCS_RETRIEVAL,
            "data_saved_success": False,
            "chat_history": [],
            "columns_to_answer": [],
            "preview_collection": None,
            "model_params": {},
        }
        
        for key, value in defaults.items():
            if key not in st.session_state:
                st.session_state[key] = value

    def render_sidebar(self):
        """Render all sidebar components."""
        with st.sidebar:
            self._render_language_settings()
            self._render_settings()
            self._render_configuration_summary()
            
    # 1. In the _render_language_settings method:
    def _render_language_settings(self):
        """Render language selection in sidebar."""
        st.header("1. Setup Language")
        language_choice = st.radio(
            "Select language:",
            [NONE, ENGLISH, VIETNAMESE],
            index=0
        )
        
        # Import GPU utilities for model initialization
        from utils.gpu_utils import torch_device, get_device_string
        device = torch_device()
        
        # Handle language selection
        if language_choice == ENGLISH:
            if st.session_state.get("language") != EN:
                st.session_state.language = EN
                if st.session_state.get("embedding_model_name") != 'all-MiniLM-L6-v2':
                    # Use the device for SentenceTransformer
                    st.session_state.embedding_model = SentenceTransformer('all-MiniLM-L6-v2', device=device)
                    st.session_state.embedding_model_name = 'all-MiniLM-L6-v2'
                    st.success(f"Using English embedding model: all-MiniLM-L6-v2 on {get_device_string()}")
        elif language_choice == VIETNAMESE:
            if st.session_state.get("language") != VI:
                st.session_state.language = VI
                if st.session_state.get("embedding_model_name") != 'keepitreal/vietnamese-sbert':
                    # Use the device for SentenceTransformer
                    st.session_state.embedding_model = SentenceTransformer('keepitreal/vietnamese-sbert', device=device)
                    st.session_state.embedding_model_name = 'keepitreal/vietnamese-sbert'
                    st.success(f"Using Vietnamese embedding model: keepitreal/vietnamese-sbert on {get_device_string()}")

    def _render_settings(self):
        """Render settings section in sidebar."""
        st.header("Settings")
        
        st.session_state.chunk_size = st.number_input(
            "Chunk Size",
            min_value=10,
            max_value=1000,
            value=st.session_state.chunk_size,
            step=10,
            help="Set the size of each chunk in terms of tokens."
        )

        st.session_state.chunk_overlap = st.number_input(
            "Chunk Overlap",
            min_value=0,
            max_value=1000,
            step=10,
            value=st.session_state.chunk_size // 10,
            help="Set the overlap between chunks."
        )

        st.session_state.number_docs_retrieval = st.number_input(
            "Number of documents retrieval",
            min_value=1,
            max_value=50,
            value=st.session_state.number_docs_retrieval,
            step=1,
            help="Set the number of documents to retrieve."
        )

    def _render_configuration_summary(self):
        """Display configuration summary in sidebar."""
        st.subheader("All configurations:")
        active_collection_names = list(st.session_state.active_collections.keys())
        collection_names = ', '.join(active_collection_names) if active_collection_names else 'No collections'
        
        configs = [
            ("Active Collections", collection_names),
            ("LLM Model", st.session_state.llm_name if 'llm_name' in st.session_state else 'Not selected'),
            ("LLM Provider", st.session_state.llm_provider if 'llm_provider' in st.session_state else 'Not specified'),
            ("Language", st.session_state.language),
            ("Embedding Model", st.session_state.embedding_model.__class__.__name__ if st.session_state.embedding_model else 'None'),
            ("Chunk Size", st.session_state.chunk_size),
            ("Number of Documents Retrieval", st.session_state.number_docs_retrieval),
            ("Data Saved", 'Yes' if st.session_state.data_saved_success else 'No')
        ]

        for i, (key, value) in enumerate(configs, 1):
            st.markdown(f"{i}. {key}: **{value}**")

        if st.session_state.get('chunkOption'):
            st.markdown(f"9. Chunking Option: **{st.session_state.chunkOption}**")

    def render_main_content(self):
        """Render main content area of the application."""
        st.header("Local LLM RAG System")
        st.markdown("Powered by local LLMs optimized for Llama and DeepSeek models.")
        
        # Organize sections with proper numbering
        section_num = 1
        
        # Data Source Section
        self.data_processor.render_data_source_section(section_num)
        section_num += 1
        
        # LLM Setup Section
        self._render_llm_setup_section(section_num)
        section_num += 1
        
        # Export Section
        self._render_export_section(section_num)
        section_num += 1
        
        # Interactive Chatbot Section
        self._render_chatbot_section(section_num)
   
    def _render_llm_setup_section(self, section_num):
        """
        Render LLM setup section focused on local models.
        
        Args:
            section_num (int): Section number for header
        """
        st.header(f"{section_num}. Setup Local LLMs")  
        
        # Provider selection
        provider_options = list(LANGCHAIN_PROVIDERS.keys())
        
        st.write("### Select Model Backend:")
        selected_provider = st.selectbox(
            "Choose the provider for running your model:",
            provider_options,
            index=provider_options.index("ollama") if "ollama" in provider_options else 0,
            format_func=lambda x: LANGCHAIN_PROVIDERS.get(x, x)
        )
        
        st.session_state.llm_provider = selected_provider
        
        # Render provider-specific setup
        if selected_provider == "ollama":
            self._render_ollama_setup()
        elif selected_provider in ["llama.cpp", "ctransformers"]:
            self._render_gguf_setup(selected_provider)
        elif selected_provider == "transformers":
            self._render_transformers_setup()
        
    def _render_ollama_setup(self):
        """Render Ollama-specific setup with a dropdown for all models."""
        st.write("### Ollama Models")
        st.write("Select a model from the dropdown. Models with ✅ are already pulled.")
        
        # Get available models
        ollama_models = self.langchain_manager.get_ollama_models()
        
        if not ollama_models:
            st.warning("No Ollama models found. Please make sure Ollama is running.")
            if st.button("Refresh Models"):
                st.rerun()
            return
        
        # Format model names for dropdown - adding indicators for pulled models
        model_display_names = {}
        for model in ollama_models:
            if model.get("pulled", False):
                display_name = f"✅ {model['name']} - {model['description']}"
            else:
                display_name = f"⬇️ {model['name']} - {model['description']}"
            model_display_names[display_name] = model["id"]
        
        # Model selection dropdown
        selected_display_name = st.selectbox(
            "Select Model:",
            list(model_display_names.keys())
        )
        
        if selected_display_name:
            model_id = model_display_names[selected_display_name]
            selected_model = next((m for m in ollama_models if m["id"] == model_id), None)
            
            # Show model information
            st.write(f"**Selected Model:** {selected_model['name']}")
            
            # Check if model is already pulled
            is_pulled = selected_model.get("pulled", False)
            
            if not is_pulled:
                # Show pull button for models that haven't been pulled yet
                if st.button(f"Pull {selected_model['name']} Model"):
                    with st.spinner(f"Pulling {selected_model['name']}..."):
                        if self.langchain_manager.download_ollama_model(model_id):
                            st.success(f"Successfully pulled {selected_model['name']}")
                            st.rerun()
                        else:
                            st.error(f"Failed to pull {selected_model['name']}")
            
            # Initialize button - only enable if model is pulled
            if st.button("Initialize Selected Model", disabled=not is_pulled):
                if not is_pulled:
                    st.warning("Please pull the model first before initializing.")
                else:
                    with st.spinner(f"Initializing {selected_model['name']}..."):
                        llm = self.langchain_manager.initialize_llm(
                            model_id, 
                            "ollama", 
                            st.session_state.model_params
                        )
                        
                        if llm:
                            st.session_state.llm_name = model_id
                            st.session_state.llm_provider = "ollama"
                            st.session_state.llm_type = LANGCHAIN_LLM
                            st.session_state.llm_model = llm
                            st.success(f"Successfully initialized {selected_model['name']}!")
                        else:
                            st.error(f"Failed to initialize {selected_model['name']}.")
            
    # 3. In the _render_gguf_setup method:
    def _render_gguf_setup(self, provider):
        """
        Render GGUF model setup for llama.cpp or ctransformers.
        GPU detection and optimization is now centralized.
        
        Args:
            provider (str): Provider name
        """
        st.write(f"### {provider.capitalize()} with GGUF Models")
        st.write("GGUF models provide excellent performance on consumer hardware.")
        
        # Check for GPU using our utility
        from utils.gpu_utils import get_gpu_info, is_gpu_available
        gpu_available = is_gpu_available()
        gpu_info = get_gpu_info()
        
        if gpu_available:
            st.success(f"GPU acceleration will be enabled ({gpu_info['name']})")
        
        # Rest of the method...
        
        # When configuring GPU layers:
        n_ctx = st.number_input(
            "Context Length:",
            min_value=512,
            max_value=8192,
            value=4096,
            step=512,
            help="Maximum context length. Higher values use more memory."
        )
        
        # GPU layers - automatically set if GPU is available
        if gpu_available:
            max_gpu_layers_default = 35  # Good default value for most GPUs
            gpu_memory = gpu_info.get("memory_gb", 0)
            
            # Adjust based on GPU memory
            if gpu_memory < 4:
                max_gpu_layers_default = 20
            elif gpu_memory > 16:
                max_gpu_layers_default = 50
                
            max_gpu_layers = st.number_input(
                "GPU Layers:",
                min_value=0,
                max_value=100,
                value=max_gpu_layers_default,
                step=5,
                help="Higher values use more GPU memory but provide better performance."
            )
        else:
            max_gpu_layers = 0
            st.info("No GPU detected - GPU layers set to 0 (CPU-only mode)")
        
        # Store parameters
        params = {
            "temperature": temperature,
            "max_tokens": max_tokens,
            "n_ctx": n_ctx,
            "n_gpu_layers": max_gpu_layers
        }
        
        if model_type_value:
            params["model_type"] = model_type_value
        
        st.session_state.model_params = params
        
        # Initialize button            
        if st.button("Initialize Selected Model"):
            with st.spinner(f"Initializing {selected_model_name}..."):
                # Import the optimize_model_params function to ensure proper GPU utilization
                from utils.gpu_utils import optimize_model_params, get_device_string
                
                # Optimize parameters for GPU when available
                optimized_params = optimize_model_params(
                    st.session_state.model_params, 
                    provider
                )
                
                llm = self.langchain_manager.initialize_llm(
                    model_path, 
                    provider, 
                    optimized_params
                )
                
                if llm:
                    st.session_state.llm_name = model_path
                    st.session_state.llm_provider = provider
                    st.session_state.llm_type = LANGCHAIN_LLM
                    st.session_state.llm_model = llm
                    st.success(f"Successfully initialized {selected_model_name} on {get_device_string()}!")
                else:
                    st.error(f"Failed to initialize {selected_model_name}.")

    # 2. In the _render_transformers_setup method:
    def _render_transformers_setup(self):
        """Render setup for HuggingFace Transformers models."""
        st.write("### HuggingFace Transformers")
        st.write("Load models directly from HuggingFace or from local files.")
        
        # Check for GPU using our utility
        from utils.gpu_utils import get_gpu_info, is_gpu_available
        gpu_info = get_gpu_info()
        gpu_available = gpu_info["available"]
        
        if gpu_available:
            st.success(f"GPU detected: {gpu_info['name']} with {gpu_info['memory_gb']:.1f} GB memory")
            st.info("GPU acceleration will be automatically enabled for all models")
        else:
            st.warning("No GPU detected. Model loading and inference will be slower.")
        
        # Tabs for HuggingFace options
        hub_tab, local_tab = st.tabs(["HuggingFace Hub", "Local Models"])
        
        with hub_tab:
            st.write("#### Load from HuggingFace Hub")
            
            # Recommended models
            recommended_models = [
                {"id": "meta-llama/Meta-Llama-3-8B", "name": "Meta-Llama-3-8B", "description": "Meta's Llama 3 (8B)"},
                {"id": "deepseek-ai/deepseek-coder-6.7b-instruct", "name": "DeepSeek Coder 6.7B", "description": "Code-specialized model"},
                {"id": "microsoft/phi-3-mini-4k-instruct", "name": "Phi-3 Mini", "description": "Microsoft's compact Phi-3 model"},
                {"id": "google/gemma-2b-it", "name": "Gemma 2B Instruct", "description": "Google's instruction-tuned 2B model"}
            ]
            
            model_options = {f"{model['name']} - {model['description']}": model["id"] for model in recommended_models}
            
            # Allow custom model ID input
            use_custom = st.checkbox("Enter custom model ID")
            
            if use_custom:
                model_id = st.text_input(
                    "HuggingFace Model ID:",
                    value="meta-llama/Meta-Llama-3-8B",
                    help="Enter the HuggingFace model ID (e.g., 'meta-llama/Meta-Llama-3-8B')"
                )
            else:
                selected_model_name = st.selectbox(
                    "Select Model:",
                    list(model_options.keys())
                )
                model_id = model_options[selected_model_name] if selected_model_name else None
            
            if model_id:
                # Quantization options
                if gpu_available:
                    gpu_memory = gpu_info.get("memory_gb", 0)
                    default_quantization = 1  # 8-bit by default when GPU available
                    
                    if gpu_memory < 8:
                        default_quantization = 2  # 4-bit for limited memory
                        st.info("Limited GPU memory detected. 4-bit quantization recommended.")
                    
                    quantization = st.selectbox(
                        "Quantization:",
                        ["None", "8-bit", "4-bit"],
                        index=default_quantization,
                        help="Quantization reduces memory usage but may slightly impact quality."
                    )
                    
                    load_in_8bit = quantization == "8-bit"
                    load_in_4bit = quantization == "4-bit"
                else:
                    load_in_8bit = False
                    load_in_4bit = False
                    st.info("No GPU detected - quantization disabled")
                
                # Temperature slider
                temperature = st.slider(
                    "Temperature:",
                    min_value=0.0,
                    max_value=1.0,
                    value=0.7,
                    step=0.1,
                    help="Higher values make output more random, lower values more deterministic."
                )
                
                # Max tokens slider
                max_tokens = st.slider(
                    "Max Tokens:",
                    min_value=64,
                    max_value=4096,
                    value=512,
                    step=64,
                    help="Maximum number of tokens to generate."
                )
                
                # Store parameters
                st.session_state.model_params = {
                    "temperature": temperature,
                    "max_tokens": max_tokens,
                    "load_in_8bit": load_in_8bit,
                    "load_in_4bit": load_in_4bit,
                    "device_map": "auto" if gpu_available else None
                }
                
                # Initialize button            
                if st.button("Initialize Selected Model"):
                    with st.spinner(f"Initializing {model_id}..."):
                        # Import the optimize_model_params function to ensure proper GPU utilization
                        from utils.gpu_utils import optimize_model_params
                        
                        # Optimize parameters for GPU when available
                        optimized_params = optimize_model_params(
                            st.session_state.model_params, 
                            "transformers"
                        )
                        
                        llm = self.langchain_manager.initialize_llm(
                            model_id, 
                            "transformers", 
                            optimized_params
                        )
                        
                        if llm:
                            st.session_state.llm_name = model_id
                            st.session_state.llm_provider = "transformers"
                            st.session_state.llm_type = LANGCHAIN_LLM
                            st.session_state.llm_model = llm
                            st.success(f"Successfully initialized {model_id} on {get_device_string()}!")
                        else:
                            st.error(f"Failed to initialize {model_id}.")
                            
    def _render_export_section(self, section_num):
        """
        Render export section.
        
        Args:
            section_num (int): Section number for header
        """
        st.header(f"{section_num}. Export Chatbot")
        if st.button("Export Chatbot"):
            self._export_chatbot()

    def _export_chatbot(self):
        """Export chatbot configuration to JSON file."""
        file_path = "pages/session_state.json"
        
        # Ensure the directory exists
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        
        # Required fields to export
        required_fields = [
            "random_collection_name", 
            "number_docs_retrieval", 
            "embedding_model_name", 
            "llm_type", 
            "llm_name",
            "llm_provider",
            "columns_to_answer",
        ]

        # Check if all required fields are present in session state
        missing_fields = [field for field in required_fields if field not in st.session_state]
        if missing_fields:
            st.error(f"Missing required fields: {', '.join(missing_fields)}")
            return

        # Check if llm_type is LANGCHAIN_LLM
        if st.session_state["llm_type"] != LANGCHAIN_LLM:
            st.error("Only support exporting LangChain LLMs.")
            return
        
        # Filter session state to only include specified fields and serializable types
        session_data = {
            key: value for key, value in st.session_state.items() 
            if key in required_fields and isinstance(value, (str, int, float, bool, list, dict))
        }
        
        # Add model_params if they exist
        if "model_params" in st.session_state and isinstance(st.session_state["model_params"], dict):
            session_data["model_params"] = st.session_state["model_params"]

        # Save to JSON file
        with open(file_path, "w") as file:
            json.dump(session_data, file)
        
        st.success("Chatbot exported successfully!")

    def vector_search(self, model, query, active_collections, columns_to_answer, number_docs_retrieval):
        """
        Perform vector search across multiple collections.
        
        Args:
            model: The embedding model to use
            query (str): Search query
            active_collections (dict): Dictionary of active collections
            columns_to_answer (list): Columns to include in the response
            number_docs_retrieval (int): Number of results to retrieve
            
        Returns:
            tuple: (metadata list, formatted search result string)
        """
        # Validate inputs
        if not model:
            st.error("Embedding model not initialized. Please select a language first.")
            return [], ""
            
        if not active_collections:
            st.error("No collections available for search.")
            return [], ""
            
        if not columns_to_answer:
            st.error("No columns selected for answering.")
            return [], ""

        try:
            all_metadatas = []
            filtered_metadatas = []
            
            # Generate query embeddings
            try:
                query_embeddings = model.encode([query])
            except Exception as e:
                st.error(f"Error generating embeddings: {str(e)}")
                return [], ""
            
            # Search each active collection
            for collection_name, collection in active_collections.items():
                try:
                    results = collection.query(
                        query_embeddings=query_embeddings,
                        n_results=number_docs_retrieval
                    )
                    
                    if results and 'metadatas' in results and results['metadatas']:
                        # Add collection name to each metadata item
                        for meta_list in results['metadatas']:
                            for meta in meta_list:
                                meta['source_collection'] = collection_name
                        
                        # Flatten the nested metadata structure
                        for meta_list in results['metadatas']:
                            all_metadatas.extend(meta_list)
                            
                except Exception as e:
                    st.error(f"Error searching collection {collection_name}: {str(e)}")
                    continue
            
            if not all_metadatas:
                st.info("No relevant results found in any collection.")
                return [], ""
            
            # Filter metadata to only include selected columns plus source collection
            for metadata in all_metadatas:
                filtered_metadata = {
                    'source_collection': metadata.get('source_collection', 'Unknown')
                }
                for column in columns_to_answer:
                    if column in metadata:
                        filtered_metadata[column] = metadata[column]
                filtered_metadatas.append(filtered_metadata)
                
            # Format the search results
            search_result = self._format_search_results(filtered_metadatas, columns_to_answer)
            
            return [filtered_metadatas], search_result
            
        except Exception as e:
            st.error(f"Error in vector search: {str(e)}")
            return [], ""
            
    def _format_search_results(self, metadatas, columns_to_answer):
        """
        Format search results for display.
        
        Args:
            metadatas (list): List of metadata dictionaries
            columns_to_answer (list): Columns to include in the result
            
        Returns:
            str: Formatted search result string
        """
        search_result = ""
        for i, metadata in enumerate(metadatas, 1):
            search_result += f"\n{i}) Source: {metadata.get('source_collection', 'Unknown')}\n"
            for column in columns_to_answer:
                if column in metadata:
                    search_result += f"   {column.capitalize()}: {metadata.get(column)}\n"
        return search_result

    def _render_chatbot_section(self, section_num):
        """
        Render interactive chatbot section.
        
        Args:
            section_num (int): Section number for header
        """
        st.header(f"{section_num}. Interactive Chatbot")

        # Validate prerequisites 
        if not self._validate_chatbot_prerequisites():
            return

        # Allow user to select columns
        self._render_column_selector()
        
        render_quality_check_widget_in_chatbot()

        # Initialize chat history if needed
        if "chat_history" not in st.session_state:
            st.session_state.chat_history = []

        # Display chat history
        self._display_chat_history()

        # Handle new user input
        if prompt := st.chat_input("Ask a question..."):
            self._process_user_input(prompt)
    
    def _validate_chatbot_prerequisites(self):
        """
        Validate that prerequisites for the chatbot are met.
        
        Returns:
            bool: True if prerequisites are met, False otherwise
        """
        if not st.session_state.embedding_model:
            st.error("Please select a language to initialize the embedding model.")
            return False

        if not st.session_state.active_collections:
            st.error("No collection found. Please upload data and save it first.")
            return False
            
        if not st.session_state.llm_model:
            st.error("Please initialize an LLM model first.")
            return False
            
        return True
        
    def _render_column_selector(self):
        """Render the column selector for the chatbot."""
        available_columns = list(st.session_state.chunks_df.columns)
        st.session_state.columns_to_answer = st.multiselect(
            "Select one or more columns LLMs should answer from:", 
            available_columns
        )
        
    def _display_chat_history(self):
        """Display the chat history."""
        for message in st.session_state.chat_history:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])
                
    def _process_user_input(self, prompt):
        """
        Process a new user input in the chatbot.
        
        Args:
            prompt (str): User's prompt
        """
        # Add user message to chat history
        st.session_state.chat_history.append({"role": USER, "content": prompt})
        with st.chat_message(USER):
            st.markdown(prompt)

        with st.chat_message(ASSISTANT):
            if not st.session_state.columns_to_answer:
                st.warning("Please select columns for the chatbot to answer from.")
                return

            # Search status
            search_status = st.empty()
            search_status.info("Searching for relevant information...")
            
            try:
                # Perform vector search
                metadatas, retrieved_data = self.vector_search(
                    st.session_state.embedding_model,
                    prompt,
                    st.session_state.active_collections,
                    st.session_state.columns_to_answer,
                    st.session_state.number_docs_retrieval
                )
                
                if not metadatas or not retrieved_data:
                    search_status.warning("No relevant information found.")
                    return
                
                # Check if this is a code quality analysis request
                is_quality_analysis = any(kw in prompt.lower() for kw in 
                                        ["error", "violation", "quality", "build", "checkstyle", 
                                        "fix", "issue", "improve", "code problem"])
                
                # Handle quality analysis
                if is_quality_analysis:
                    search_status.info("Analyzing code quality information...")
                    
                    # Check for quality report analysis in session state
                    json_analysis = ""
                    if hasattr(st.session_state, 'quality_report_analysis') and st.session_state.quality_report_analysis:
                        json_analysis = st.session_state.quality_report_analysis
                    else:
                        # Look for JSON files in current directory
                        from utils.utils import find_quality_report, analyze_json_data
                        
                        file_path, report_data = find_quality_report()
                        
                        if file_path and report_data:
                            search_status.info(f"Found quality report: {file_path}")
                            json_analysis = analyze_json_data(report_data)
                            
                            # Store in session state for future use
                            st.session_state.quality_report_path = file_path
                            st.session_state.quality_report_analysis = json_analysis
                    
                    if json_analysis:
                        # Create a specialized prompt for quality analysis
                        enhanced_prompt = f"""You are a professional code quality analyst and programming instructor. Analyze the following code quality issues and provide detailed explanations and solutions.

    USER QUERY: {prompt}

    CODE QUALITY ANALYSIS:
    {json_analysis}

    DATA RETRIEVAL:
    {retrieved_data}

    Please provide a comprehensive analysis of the build errors and checkstyle violations found in the report. For each issue:
    1. Explain what the error or violation means in simple terms
    2. Explain why it's important to fix it (impact on code quality, reliability, etc.)
    3. Provide a step-by-step solution to fix the issue
    4. If possible, show both the incorrect code and the corrected code

    Also include a summary of the overall code quality issues and general recommendations for improving the code. 
    Be educational and helpful, as if you're teaching a student how to write better code."""
                    else:
                        # No JSON analysis available, use standard prompt format
                        enhanced_prompt = f"The prompt of the user is: \"{prompt}\". Answer it based on the following retrieved data: \n{retrieved_data}"
                else:
                    # Use standard prompt format for non-quality analysis queries
                    enhanced_prompt = f"The prompt of the user is: \"{prompt}\". Answer it based on the following retrieved data: \n{retrieved_data}"
                
                # Display search results
                search_status.success("Found relevant information!")
                    
                # Show retrieved data in sidebar
                st.sidebar.subheader("Retrieved Data")
                if metadatas and metadatas[0]:
                    st.sidebar.dataframe(pd.DataFrame(metadatas[0]))
                
                # If quality analysis, also show quality report in sidebar
                if is_quality_analysis and hasattr(st.session_state, 'quality_report_analysis') and st.session_state.quality_report_analysis:
                    with st.sidebar.expander("Quality Report Analysis"):
                        st.text_area("Analysis", st.session_state.quality_report_analysis, height=200)
                
                # Call LLM for response
                try:
                    search_status.info("Generating response...")
                    response = st.session_state.llm_model.generate_content(enhanced_prompt)
                    search_status.empty()
                    st.markdown(response)
                    st.session_state.chat_history.append({"role": ASSISTANT, "content": response})
                except Exception as e:
                    search_status.error(f"Error from LLM: {str(e)}")
            
            except Exception as e:
                search_status.error(f"An error occurred: {str(e)}")
                import traceback
                st.error(traceback.format_exc())

    def get_llm_options(self):
        """
        Return LLM options dictionary.
        
        Returns:
            dict: Dictionary of LLM options
        """
        return LANGCHAIN_PROVIDERS
    
    def create_quality_template(self):
        """
        Create and return a specialized prompt template for code quality analysis.
        
        Returns:
            PromptTemplate: A new template for code quality analysis
        """
        return PromptTemplate(
            name="code_quality",
            description="Specialized prompt for code quality analysis",
            template=(
                "You are a professional code quality analyst and programming instructor. "
                "Analyze the following code quality issues and provide detailed explanations and solutions.\n\n"
                "USER QUERY: {query}\n\n"
                "CODE QUALITY ANALYSIS:\n{context}\n\n"
                "Please provide a comprehensive analysis of the build errors and checkstyle violations found in the report. "
                "For each issue:\n"
                "1. Explain what the error or violation means in simple terms\n"
                "2. Explain why it's important to fix it (impact on code quality, reliability, etc.)\n"
                "3. Provide a step-by-step solution to fix the issue\n"
                "4. If possible, show both the incorrect code and the corrected code\n\n"
                "Also include a summary of the overall code quality issues and general recommendations for improving the code. "
                "Be educational and helpful, as if you're teaching a student how to write better code."
            )
        )

    def register_quality_template(self, prompt_library=None):
        """
        Register the code quality template with the prompt library.
        
        Args:
            prompt_library (PromptLibrary, optional): Library to register with
            
        Returns:
            bool: True if registration was successful
        """
        if prompt_library is None:
            prompt_library = PromptLibrary()
        
        # Check if template already exists
        if not prompt_library.get_template("code_quality"):
            template = self.create_quality_template()
            prompt_library.add_template(template)
            return True
        
        return False

    # Function to format quality analysis prompt
    def format_quality_analysis_prompt(self, query, code_analysis, additional_context=""):
        """
        Format a quality analysis prompt with the specialized template.
        
        Args:
            query (str): User's query
            code_analysis (str): Code quality analysis text
            additional_context (str, optional): Additional context information
            
        Returns:
            str: Formatted prompt for quality analysis
        """
        # Ensure template is registered
        prompt_library = PromptLibrary()
        self.register_quality_template(prompt_library)
        
        # Get the template
        template = prompt_library.get_template("code_quality")
        
        # Combine code analysis with additional context
        if additional_context:
            context = f"{code_analysis}\n\nADDITIONAL DOCUMENTATION:\n{additional_context}"
        else:
            context = code_analysis
        
        # Format and return
        return template.format(query=query, context=context)