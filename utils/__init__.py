"""
Utility module for the RAG application.

This module provides general-purpose utility functions and tools
used across the application, including data processing, model
downloading, and string manipulation utilities.
"""
# General utilities
from utils.utils import process_batch, divide_dataframe, clean_collection_name

# Model downloading utilities
from utils.model_downloader import (
    download_file, 
    check_ollama_running,
    start_ollama, 
    download_model_with_streamlit,
    get_model_family,
    get_recommended_model_parameters,
    RECOMMENDED_MODELS
)

__all__ = [
    # General utilities
    'process_batch',
    'divide_dataframe',
    'clean_collection_name',
    
    # Model downloading utilities
    'download_file',
    'check_ollama_running',
    'start_ollama',
    'download_model_with_streamlit',
    'get_model_family',
    'get_recommended_model_parameters',
    'RECOMMENDED_MODELS'
]