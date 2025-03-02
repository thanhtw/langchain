"""
Semantic chunking implementation.

This module provides text chunking based on semantic similarity
between sentences.
"""

from chunking.base_chunker import BaseChunker
import nltk
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
from sentence_transformers import SentenceTransformer


# Updates to chunking/semantic_chunker.py

class ProtonxSemanticChunker(BaseChunker):
    """
    Chunker that groups text by semantic similarity.
    Now optimized to use GPU when available.
    
    Uses either TF-IDF or transformer embeddings to measure
    semantic similarity between sentences.
    """
    
    def __init__(self, threshold=0.3, embedding_type="tfidf", model="all-MiniLM-L6-v2"):
        """
        Initialize the semantic chunker.
        
        Args:
            threshold (float): Similarity threshold for chunking (0.0-1.0)
            embedding_type (str): Embedding method ("tfidf" or "transformers")
            model (str): Transformer model name (if embedding_type is "transformers")
        """
        self.threshold = threshold
        self.embedding_type = embedding_type
        self.model_name = model
        self.model = None  # Will be initialized when needed

        # Download punkt for sentence tokenization, ensuring it's only done when class is initialized
        nltk.download("punkt", quiet=True)
    
    def _initialize_model(self):
        """Initialize the model with GPU support if available."""
        if self.embedding_type == "transformers" and self.model is None:
            try:
                # Import GPU utilities
                from utils.gpu_utils import torch_device
                device = torch_device()
                
                # Initialize model on appropriate device
                self.model = SentenceTransformer(self.model_name, device=device)
            except Exception as e:
                print(f"Warning: Could not initialize model on GPU: {e}")
                # Fall back to CPU if there's an issue
                self.model = SentenceTransformer(self.model_name)
    
    def embed_function(self, sentences):
        """
        Embeds sentences using the specified embedding method.
        Now GPU-aware for better performance.
        
        Args:
            sentences (List[str]): List of sentences to embed
            
        Returns:
            numpy.ndarray: Embeddings for the sentences
            
        Raises:
            ValueError: If embedding_type is not supported
        """
        if self.embedding_type == "tfidf":
            vectorizer = TfidfVectorizer().fit_transform(sentences)
            return vectorizer.toarray()
        elif self.embedding_type == "transformers":
            # Initialize model if not done yet
            self._initialize_model()
            return self.model.encode(sentences)
        else:
            raise ValueError("Unsupported embedding type. Choose 'tfidf' or 'transformers'.")
        
    def split_text(self, text):
        """
        Split text into semantically coherent chunks.
        
        Args:
            text (str): Text to split
            
        Returns:
            List[str]: List of text chunks
        """
        sentences = nltk.sent_tokenize(text)  # Extract sentences
        sentences = [item for item in sentences if item and item.strip()]
        if not len(sentences):
            return []

        # Vectorize the sentences for similarity checking
        vectors = self.embed_function(sentences)

        # Calculate pairwise cosine similarity between sentences
        similarities = cosine_similarity(vectors)

        # Initialize chunks with the first sentence
        chunks = [[sentences[0]]]

        # Group sentences into chunks based on similarity threshold
        for i in range(1, len(sentences)):
            sim_score = similarities[i-1, i]

            if sim_score >= self.threshold:
                # If the similarity is above the threshold, add to the current chunk
                chunks[-1].append(sentences[i])
            else:
                # Start a new chunk
                chunks.append([sentences[i]])

        # Join the sentences in each chunk to form coherent paragraphs
        return [' '.join(chunk) for chunk in chunks]