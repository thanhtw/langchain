"""
Constants used throughout the application.
Focused on local model implementations, especially Llama and DeepSeek.
"""

# Languages
EN = "en"
VI = "vi"
NONE = "None"
ENGLISH = "English"
VIETNAMESE = "Vietnamese"

# Chat roles
USER = "user"
ASSISTANT = "assistant"

# LLM types
LANGCHAIN_LLM = "langchain_llm"
DEFAULT_LANGCHAIN_MODEL = "llama3"
DEFAULT_LANGCHAIN_PROVIDER = "ollama"

# Data sources
UPLOAD = "UPLOAD"
DB = "DB"

# Chunking options
NO_CHUNKING = "No Chunking"
RECURSIVE_CHUNKER = "RecursiveTokenChunker"
SEMANTIC_CHUNKER = "SemanticChunker"
AGENTIC_CHUNKER = "AgenticChunker"

# LangChain local providers
LANGCHAIN_PROVIDERS = {
    "ollama": "Ollama (easiest setup, recommended for beginners)",
    "llama.cpp": "Llama.cpp (best performance for GGUF models)",
    "transformers": "Transformers (HuggingFace models, high RAM usage)",
    "ctransformers": "CTransformers (alternative GGUF loader)"
}

# Default Ollama models
OLLAMA_MODELS = {
    "Llama 3": "llama3",
    "Llama 3 (8B)": "llama3:8b",
    "Llama 3 (1B)": "llama3:1b",
    "DeepSeek Coder": "deepseek-coder",
    "DeepSeek LLM": "deepseek-llm",
    "Phi-3 Mini": "phi3:mini",
    "Gemma 2B": "gemma:2b"
}

# GGUF model types for CTransformers
MODEL_TYPES = {
    "Llama": "llama",
    "Llama 2": "llama",
    "Llama 3": "llama",
    "DeepSeek": "deepseek",
    "Mistral": "mistral",
    "Phi": "phi",
    "Gemma": "gemma",
    "Other": "other"
}

# Search options
VECTOR_SEARCH = "Vector Search"
HYDE_SEARCH = "Hyde Search"

# Default settings
DEFAULT_CHUNK_SIZE = 200
DEFAULT_CHUNK_OVERLAP = 10
DEFAULT_NUM_DOCS_RETRIEVAL = 3