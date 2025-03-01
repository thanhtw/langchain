"""
Base chunker interface for the RAG application.
"""

from abc import ABC, abstractmethod
from typing import List

class BaseChunker(ABC):
    """
    Abstract base class for text chunkers.
    All chunker implementations must inherit from this class.
    """
    
    @abstractmethod
    def split_text(self, text: str) -> List[str]:
        """
        Split text into multiple chunks.
        
        Args:
            text (str): Text to split into chunks
            
        Returns:
            List[str]: List of text chunks
        """
        pass