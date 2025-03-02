"""
Enhanced vector search with relevance ranking.

This module provides improved vector search capabilities with
result ranking based on semantic similarity and metadata factors.
"""

import numpy as np
from typing import List, Dict, Any, Tuple, Optional
import pandas as pd
import streamlit as st
from sentence_transformers import SentenceTransformer

def compute_similarity(query_embedding, result_embedding):
    """
    Compute cosine similarity between query and result embeddings.
    
    Args:
        query_embedding: Query embedding vector
        result_embedding: Result embedding vector
        
    Returns:
        float: Cosine similarity score
    """
    # Normalize vectors to unit length
    query_norm = query_embedding / np.linalg.norm(query_embedding)
    result_norm = result_embedding / np.linalg.norm(result_embedding)
    
    # Compute cosine similarity
    return np.dot(query_norm, result_norm)

def rank_search_results(
    metadatas: List[Dict[str, Any]], 
    embeddings: List[np.ndarray], 
    query_embedding: np.ndarray,
    boost_factors: Optional[Dict[str, float]] = None
) -> List[Dict[str, Any]]:
    """
    Rank search results by combining semantic similarity with metadata factors.
    
    Args:
        metadatas: List of metadata dictionaries for each result
        embeddings: List of embedding vectors for each result
        query_embedding: Query embedding vector
        boost_factors: Optional dictionary of metadata fields and their boost weights
        
    Returns:
        List of ranked metadata dictionaries with added scores
    """
    if not metadatas or not embeddings:
        return []
    
    # Default boost factors
    if boost_factors is None:
        boost_factors = {}
    
    # Calculate base similarity scores
    ranked_results = []
    for idx, (metadata, embedding) in enumerate(zip(metadatas, embeddings)):
        # Compute similarity score
        sim_score = compute_similarity(query_embedding, embedding)
        
        # Start with the similarity score
        final_score = sim_score
        
        # Apply metadata boosts if available
        for field, boost in boost_factors.items():
            if field in metadata:
                # If the field exists and is a string, apply boost based on presence
                if isinstance(metadata[field], str) and metadata[field].strip():
                    final_score += boost
                # If the field is a number, normalize and apply boost
                elif isinstance(metadata[field], (int, float)):
                    # Normalize to 0-1 range assuming values between 0-100
                    # This could be refined based on actual data ranges
                    norm_value = min(1.0, max(0.0, metadata[field] / 100.0))
                    final_score += norm_value * boost
        
        # Add score to metadata
        result_with_score = metadata.copy()
        result_with_score['semantic_score'] = sim_score
        result_with_score['final_score'] = final_score
        result_with_score['rank'] = idx + 1  # Original rank
        
        ranked_results.append(result_with_score)
    
    # Sort by final score in descending order
    ranked_results.sort(key=lambda x: x['final_score'], reverse=True)
    
    # Update rank based on new order
    for i, result in enumerate(ranked_results, 1):
        result['rank'] = i
    
    return ranked_results

def enhanced_vector_search(
    model: SentenceTransformer,
    query: str, 
    active_collections: Dict[str, Any], 
    columns_to_answer: List[str],
    number_docs_retrieval: int = 3,
    boost_factors: Optional[Dict[str, float]] = None
) -> Tuple[List[Dict[str, Any]], str]:
    """
    Perform enhanced vector search with result ranking.
    Now GPU-aware for faster encoding and similarity calculations.
    
    Args:
        model: The embedding model to use
        query: Search query
        active_collections: Dictionary of active vector collections
        columns_to_answer: Columns to include in the response
        number_docs_retrieval: Number of results to retrieve
        boost_factors: Optional dictionary of metadata fields and their boost weights
        
    Returns:
        Tuple of (ranked metadata list, formatted search result string)
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
        all_embeddings = []
        
        # Import GPU utilities
        from utils.gpu_utils import is_gpu_available, get_device_string
        gpu_status = f" on {get_device_string()}" if is_gpu_available() else ""
        
        # Generate query embeddings - model should already be on GPU if available
        try:
            with st.spinner(f"Generating embeddings{gpu_status}..."):
                query_embedding = model.encode([query])[0]
        except Exception as e:
            st.error(f"Error generating embeddings: {str(e)}")
            return [], ""
        
        # Search each active collection
        for collection_name, collection in active_collections.items():
            try:
                # Fetch more results than needed for better ranking
                fetch_count = min(number_docs_retrieval * 2, 10)  
                results = collection.query(
                    query_embeddings=[query_embedding],
                    n_results=fetch_count,
                    include=['metadatas', 'embeddings']
                )
                
                if results and 'metadatas' in results and results['metadatas']:
                    # Flatten the nested metadata structure and add collection name
                    for i, meta_list in enumerate(results['metadatas']):
                        for j, meta in enumerate(meta_list):
                            meta['source_collection'] = collection_name
                            # Get corresponding embedding
                            if 'embeddings' in results and len(results['embeddings']) > i and len(results['embeddings'][i]) > j:
                                all_metadatas.append(meta)
                                all_embeddings.append(results['embeddings'][i][j])
                            
            except Exception as e:
                st.error(f"Error searching collection {collection_name}: {str(e)}")
                continue
        
        if not all_metadatas:
            st.info("No relevant results found in any collection.")
            return [], ""
        
        # Rank the results
        with st.spinner(f"Ranking results{gpu_status}..."):
            ranked_results = rank_search_results(
                all_metadatas, 
                all_embeddings, 
                query_embedding,
                boost_factors
            )
        
        # Limit to requested number after ranking
        ranked_results = ranked_results[:number_docs_retrieval]
        
        # Filter ranked results to only include selected columns plus score and source
        filtered_results = []
        for result in ranked_results:
            filtered_result = {
                'source_collection': result.get('source_collection', 'Unknown'),
                'rank': result.get('rank', 0),
                'score': round(result.get('final_score', 0), 3)
            }
            
            for column in columns_to_answer:
                if column in result:
                    filtered_result[column] = result[column]
                    
            filtered_results.append(filtered_result)
            
        # Format the search results
        search_result = format_search_results(filtered_results, columns_to_answer)
        
        return filtered_results, search_result
        
    except Exception as e:
        st.error(f"Error in vector search: {str(e)}")
        return [], ""

def format_search_results(ranked_results: List[Dict[str, Any]], columns_to_answer: List[str]) -> str:
    """
    Format ranked search results for display.
    
    Args:
        ranked_results: List of ranked and filtered result dictionaries
        columns_to_answer: Columns to include in the result
        
    Returns:
        str: Formatted search result string
    """
    search_result = ""
    for result in ranked_results:
        rank = result.get('rank', 0)
        score = result.get('score', 0)
        collection = result.get('source_collection', 'Unknown')
        
        search_result += f"\n{rank}) Source: {collection} (Score: {score})\n"
        
        for column in columns_to_answer:
            if column in result:
                content = result.get(column, '')
                # Truncate very long content for display
                if isinstance(content, str) and len(content) > 500:
                    content = content[:500] + "... [truncated]"
                search_result += f"   {column.capitalize()}: {content}\n"
                
    return search_result