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

# GPU utilities
from utils.gpu_utils import (
    is_gpu_available,
    get_gpu_info,
    optimize_model_params,
    get_optimal_batch_size,
    torch_device,
    get_device_string
)

from utils.auto_quality_check import (
    run_quality_check_with_analysis, 
    generate_search_query,
    vector_search,
    format_search_results,
    get_build_checkstyle_dir,
    run_quality_check,
    analyze_quality_report,
    render_quality_check_widget_in_chatbot
    )

from utils.enhanced_vector_search import(
    compute_similarity,
    rank_search_results,
    enhanced_vector_search,
    format_search_results
     
)

from utils.optimized_chunking import (
    get_chunker, 
    BaseChunker, 
    RecursiveTokenChunker, 
    SemanticChunker, 
    NoChunker
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
    'RECOMMENDED_MODELS',
    
    # GPU utilities
    'is_gpu_available',
    'get_gpu_info',
    'optimize_model_params',
    'get_optimal_batch_size',
    'torch_device',
    'get_device_string',

    # Auto quality check
    'run_quality_check_with_analysis', 
    'generate_search_query',
    'vector_search',
    'format_search_results',
    'get_build_checkstyle_dir',
    'run_quality_check',
    'analyze_quality_report',
    'render_quality_check_widget_in_chatbot',

    # Enhanced vector search
    'compute_similarity',
    'rank_search_results',
    'enhanced_vector_search',
    'format_search_results',

    # Chunking 
    'get_chunker', 
    'BaseChunker', 
    'RecursiveTokenChunker', 
    'SemanticChunker', 
    'NoChunker'
]