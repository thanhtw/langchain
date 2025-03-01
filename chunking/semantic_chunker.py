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


class ProtonxSemanticChunker(BaseChunker):
    """
    Chunker that groups text by semantic similarity.
    
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
        self.model = model

        # Download punkt for sentence tokenization, ensuring it's only done when class is initialized
        nltk.download("punkt", quiet=True)
    
    def embed_function(self, sentences):
        """
        Embeds sentences using the specified embedding method.
        
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
            self.model = SentenceTransformer(self.model)
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