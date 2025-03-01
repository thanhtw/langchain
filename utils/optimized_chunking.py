"""
Optimized text chunking module for the RAG application.

This module provides various strategies for chunking text documents
to optimize retrieval and context usage for LLMs.
"""
from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional
import re
import nltk
from tqdm import tqdm
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
from sentence_transformers import SentenceTransformer

# Load NLTK resources only once at module level
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt', quiet=True)

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

class RecursiveTokenChunker(BaseChunker):
    """
    Recursively splits text based on separators.
    
    This chunker uses a hierarchical approach to split text into semantically
    meaningful chunks, respecting document structure.
    """
    
    def __init__(self, chunk_size: int = 200, chunk_overlap: int = 10):
        """
        Initialize RecursiveTokenChunker.
        
        Args:
            chunk_size (int): Target size for each chunk
            chunk_overlap (int): Number of overlapping tokens between chunks
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.separators = ["\n\n", "\n", ".", "?", "!", " ", ""]
        
    def split_text(self, text: str) -> List[str]:
        """
        Split text into chunks recursively.
        
        Args:
            text (str): Text to split
            
        Returns:
            List[str]: List of text chunks
        """
        if not text or len(text) <= self.chunk_size:
            return [text] if text else []
            
        return self._split_text_recursive(text, self.separators)
        
    def _split_text_recursive(self, text: str, separators: List[str]) -> List[str]:
        """
        Recursively split text using the given separators.
        
        Args:
            text (str): Text to split
            separators (List[str]): List of separators to try
            
        Returns:
            List[str]: List of text chunks
        """
        # Final chunks to return
        chunks = []
        
        # Get appropriate separator
        separator = separators[-1]  # Default to last separator (empty string)
        remaining_separators = []
        
        for i, sep in enumerate(separators):
            if sep == "":
                separator = sep
                break
                
            if sep in text:
                separator = sep
                remaining_separators = separators[i+1:]
                break
        
        # Split text using the chosen separator
        if separator:
            splits = text.split(separator)
            splits = [s for s in splits if s.strip()]
        else:
            # If empty separator, treat each character as a separate chunk
            splits = list(text)
        
        # Process splits
        current_chunk = []
        current_length = 0
        
        for split in splits:
            split_length = len(split)
            
            # If this split alone exceeds chunk size and we have more separators,
            # process it recursively
            if split_length > self.chunk_size and remaining_separators:
                # Process any accumulated chunks first
                if current_chunk:
                    chunks.append(separator.join(current_chunk))
                    current_chunk = []
                    current_length = 0
                
                # Process this split recursively
                recursive_chunks = self._split_text_recursive(split, remaining_separators)
                chunks.extend(recursive_chunks)
                continue
            
            # Check if adding this split would exceed the chunk size
            if current_length + split_length + (len(separator) if current_chunk else 0) > self.chunk_size:
                # Save current chunk and start a new one
                if current_chunk:
                    chunks.append(separator.join(current_chunk))
                    
                    # Handle overlap by keeping some of the last elements
                    overlap_count = max(0, min(len(current_chunk), self.chunk_overlap))
                    if overlap_count > 0:
                        current_chunk = current_chunk[-overlap_count:]
                        current_length = sum(len(s) for s in current_chunk) + (len(separator) * (len(current_chunk) - 1))
                    else:
                        current_chunk = []
                        current_length = 0
            
            # Add the split to the current chunk
            current_chunk.append(split)
            current_length += split_length + (len(separator) if len(current_chunk) > 1 else 0)
        
        # Add the last chunk if there's anything left
        if current_chunk:
            chunks.append(separator.join(current_chunk))
        
        return chunks

class SemanticChunker(BaseChunker):
    """
    Groups text by semantic similarity.
    
    This chunker finds natural semantic boundaries in text to create coherent
    chunks based on meaning rather than just token count.
    """
    
    def __init__(self, threshold: float = 0.3, model_name: str = "all-MiniLM-L6-v2"):
        """
        Initialize SemanticChunker.
        
        Args:
            threshold (float): Similarity threshold (0-1) for chunk boundaries
            model_name (str): SentenceTransformer model name for embeddings
        """
        self.threshold = threshold
        self.model_name = model_name
        self.model = None  # Lazy-load the model when needed
        
    def _get_model(self):
        """Lazy-load the embedding model."""
        if self.model is None:
            self.model = SentenceTransformer(self.model_name)
        return self.model
        
    def split_text(self, text: str) -> List[str]:
        """
        Split text based on semantic similarity.
        
        Args:
            text (str): Text to split
            
        Returns:
            List[str]: List of text chunks
        """
        if not text:
            return []
            
        # Extract sentences
        sentences = nltk.sent_tokenize(text)
        sentences = [s.strip() for s in sentences if s.strip()]
        
        if len(sentences) <= 1:
            return sentences
            
        # Get embeddings for semantic comparison
        model = self._get_model()
        embeddings = model.encode(sentences)
        
        # Group sentences into chunks based on similarity
        chunks = []
        current_chunk = [sentences[0]]
        current_embedding = embeddings[0].reshape(1, -1)
        
        for i in range(1, len(sentences)):
            sentence = sentences[i]
            embedding = embeddings[i].reshape(1, -1)
            
            # Compute similarity with the average of the current chunk
            similarity = cosine_similarity(current_embedding, embedding)[0][0]
            
            if similarity >= self.threshold:
                # Similar enough to add to current chunk
                current_chunk.append(sentence)
                # Update running average of embeddings
                current_embedding = (current_embedding * len(current_chunk) + embedding) / (len(current_chunk) + 1)
            else:
                # Start a new chunk
                chunks.append(" ".join(current_chunk))
                current_chunk = [sentence]
                current_embedding = embedding
        
        # Add the last chunk
        if current_chunk:
            chunks.append(" ".join(current_chunk))
            
        return chunks

class NoChunker(BaseChunker):
    """
    Returns the text as is without chunking.
    
    This is useful when you want to preserve document structure
    or when the documents are already small enough.
    """
    
    def split_text(self, text: str) -> List[str]:
        """
        Return text as a single chunk.
        
        Args:
            text (str): Text to process
            
        Returns:
            List[str]: List containing the original text
        """
        return [text] if text else []

# Factory function to get the appropriate chunker
def get_chunker(chunker_type: str, **kwargs) -> BaseChunker:
    """
    Get a chunker instance based on the specified type.
    
    Args:
        chunker_type (str): Type of chunker
        **kwargs: Additional arguments for the chunker
        
    Returns:
        BaseChunker: Chunker instance
    """
    chunkers = {
        "no_chunking": NoChunker,
        "recursive": RecursiveTokenChunker,
        "semantic": SemanticChunker
    }
    
    # Select the chunker class
    chunker_class = chunkers.get(chunker_type.lower(), RecursiveTokenChunker)
    
    # Create and return the chunker instance
    return chunker_class(**kwargs)