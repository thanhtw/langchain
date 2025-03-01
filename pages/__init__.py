"""
Streamlit pages module for the RAG application.

This module contains additional pages and UI components
for the RAG application, including prompt management and
custom visualization tools.
"""
from pages.custom_prompts import (
    PromptManager,
    create_rag_prompt
)

__all__ = [
    'PromptManager',
    'create_rag_prompt'
]