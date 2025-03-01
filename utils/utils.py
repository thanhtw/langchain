"""
General utility functions for the RAG application.
"""

import os
import math
import uuid
import re
import streamlit as st


def process_batch(batch_df, model, collection):
    """
    Encode and save a batch of data to Chroma.
    
    Args:
        batch_df (DataFrame): DataFrame containing batch data
        model: Embedding model to use
        collection: Chroma collection to add data to
    """
    try:
        # Encode column data to vectors for this batch
        embeddings = model.encode(batch_df['chunk'].tolist())

        # Collect all metadata in one list
        metadatas = [row.to_dict() for _, row in batch_df.iterrows()]

        # Generate unique ids for the batch
        batch_ids = [str(uuid.uuid4()) for _ in range(len(batch_df))]

        # Add the batch to Chroma
        collection.add(
            ids=batch_ids,
            embeddings=embeddings,
            metadatas=metadatas
        )
    except Exception as e:
        if str(e) == "'NoneType' object has no attribute 'encode'":
            raise RuntimeError("Please set up the language model before running the processing.")
        raise RuntimeError(f"Error saving data to Chroma for a batch: {str(e)}")

    
def divide_dataframe(df, batch_size):
    """
    Divide DataFrame into smaller chunks based on the specified batch size.
    
    Args:
        df (DataFrame): DataFrame to divide
        batch_size (int): Size of each batch
        
    Returns:
        list: List of DataFrame batches
    """
    num_batches = math.ceil(len(df) / batch_size)
    return [df.iloc[i * batch_size:(i + 1) * batch_size] for i in range(num_batches)]


def clean_collection_name(name):
    """
    Clean a collection name to ensure it's valid for Chroma.
    
    Args:
        name (str): Original collection name
        
    Returns:
        str: Cleaned collection name or None if invalid
    """
    if not name:
        return None
        
    # Allow only alphanumeric, underscores, hyphens, and single periods
    cleaned_name = re.sub(r'[^a-zA-Z0-9_.-]', '', name)
    cleaned_name = re.sub(r'\.{2,}', '.', cleaned_name)
    cleaned_name = re.sub(r'^[^a-zA-Z0-9]+|[^a-zA-Z0-9]+$', '', cleaned_name)

    # Ensure the cleaned name meets length constraints (3-63 chars)
    if 3 <= len(cleaned_name) <= 63:
        return cleaned_name
    elif len(cleaned_name) > 63:
        return cleaned_name[:63]
    else:
        return f"collection_{uuid.uuid4().hex[:8]}"  # Fallback to random name