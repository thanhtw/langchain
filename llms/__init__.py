"""
LLM module for the RAG application.

This module provides interfaces and implementations for working with
Large Language Models in the RAG application.
"""
# Export the base interface
from llms.base import LLM

# Export the LangChain implementation
from llms.langchain_lmm import LangChainLLM, LangChainManager

__all__ = [
    'LLM',
    'LangChainLLM',
    'LangChainManager'
]