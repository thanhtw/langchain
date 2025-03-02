"""
Data processor module for handling file uploads, chunking, and database operations.

This module provides functionalities for processing various file types,
chunking text, and storing data in the vector database.
"""

import streamlit as st
import pandas as pd
import json
import uuid
import io
import os
import time
import re
from pathlib import Path
from typing import List, Dict, Any, Tuple, Optional, Union
from docx import Document
import pdfplumber
import chromadb

# Import our optimized chunking module
from utils.optimized_chunking import get_chunker
from utils.enhanced_vector_search import enhanced_vector_search

class DataProcessor:
    """
    Handles data processing, chunking, and database operations.
    """
    
    def __init__(self, config_manager):
        """
        Initialize DataProcessor with reference to ConfigManager.
        
        Args:
            config_manager: Reference to ConfigManager instance
        """
        self.config_manager = config_manager
        self.chroma_client = chromadb.PersistentClient("db")
        self._initialize_state()

    def _initialize_state(self):
        """Initialize required session state variables."""
        if "active_collections" not in st.session_state:
            st.session_state.active_collections = {}
        if "preview_collection" not in st.session_state:
            st.session_state.preview_collection = None
        if "collection" not in st.session_state:
            st.session_state.collection = None
        if "chunks_df" not in st.session_state:
            st.session_state.chunks_df = pd.DataFrame()
        if "doc_ids" not in st.session_state:
            st.session_state.doc_ids = []

    def render_data_source_section(self, section_num):
        """
        Render data source setup section.
        
        Args:
            section_num (int): Section number for the header
        """
        if not self._check_prerequisites():
            return

        st.header(f"{section_num}. Setup Data Source")
        
        # Tab selection for upload or load
        tab1, tab2 = st.tabs(["Upload New Data", "Load Existing Collections"])
        
        with tab1:
            self._render_upload_tab()

        with tab2:
            self._render_collection_manager()

    def _check_prerequisites(self):
        """
        Check if all required prerequisites are met.
        
        Returns:
            bool: True if prerequisites are met, False otherwise
        """
        if not st.session_state.get("language"):
            st.warning("Please select a language first.")
            return False
        if not st.session_state.get("embedding_model"):
            st.warning("Please wait for embedding model to load.")
            return False
        return True

    def _render_upload_tab(self):
        """Render the upload tab content."""
        st.subheader("Upload Data", divider=True)
        uploaded_files = st.file_uploader(
            "Upload CSV, JSON, PDF, or DOCX files", 
            accept_multiple_files=True
        )

        if uploaded_files:
            self._process_uploaded_files(uploaded_files)

        if len(st.session_state.chunks_df) > 0:
            st.write("Number of chunks:", len(st.session_state.chunks_df))
            st.dataframe(st.session_state.chunks_df.head(10))

        if st.button("Save Data", disabled=len(st.session_state.chunks_df) == 0):
            self._save_data_to_collections(uploaded_files)

    def _process_uploaded_files(self, uploaded_files):
        """
        Process uploaded files and create DataFrame.
        
        Args:
            uploaded_files (list): List of uploaded file objects
        """
        all_data = []
        
        for uploaded_file in uploaded_files:
            df = self._parse_file(uploaded_file)
            if df is not None:
                # Add source file name to dataframe for tracking
                df['source_file'] = uploaded_file.name
                all_data.append(df)

        if all_data:
            # Combine all data
            df = pd.concat(all_data, ignore_index=True)
            
            # Display file stats
            st.write(f"Loaded {len(uploaded_files)} files with {len(df)} total rows.")
            
            # Preview the data
            with st.expander("Preview Raw Data"):
                st.dataframe(df.head(10))

            # Generate document IDs
            doc_ids = [str(uuid.uuid4()) for _ in range(len(df))]
            st.session_state.doc_ids = doc_ids
            df['doc_id'] = doc_ids

            # Handle chunking
            self._setup_chunking(df)

    def _parse_file(self, uploaded_file):
        """
        Parse different file types into DataFrame.
        
        Args:
            uploaded_file: File object to parse
            
        Returns:
            DataFrame or None: Parsed data or None if parsing failed
        """
        try:
            file_name = uploaded_file.name.lower()
            
            if file_name.endswith(".csv"):
                return pd.read_csv(uploaded_file)
                
            elif file_name.endswith(".json"):
                json_data = json.load(uploaded_file)
                # Handle different JSON structures
                if isinstance(json_data, list):
                    return pd.json_normalize(json_data)
                elif isinstance(json_data, dict):
                    # Check if it's a nested structure or simple key-value
                    if any(isinstance(v, (dict, list)) for v in json_data.values()):
                        return pd.json_normalize(json_data)
                    else:
                        return pd.DataFrame([json_data])
                else:
                    st.error(f"Unsupported JSON structure in {file_name}")
                    return None
                
            elif file_name.endswith(".pdf"):
                return self._parse_pdf_file(uploaded_file)
                
            elif file_name.endswith((".docx", ".doc")):
                return self._parse_docx_file(uploaded_file)
                
            elif file_name.endswith(".xlsx") or uploaded_file.type == "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet":
                return pd.read_excel(uploaded_file, engine="openpyxl")
                
            else:
                st.error(f"Unsupported file format: {file_name}")
                return None
                
        except Exception as e:
            st.error(f"Error reading file {uploaded_file.name}: {str(e)}")
            return None
            
    def _parse_pdf_file(self, uploaded_file):
        """
        Parse PDF file into DataFrame.
        
        Args:
            uploaded_file: PDF file to parse
            
        Returns:
            DataFrame: DataFrame with extracted text
        """
        pdf_text = []
        page_numbers = []
        
        with pdfplumber.open(io.BytesIO(uploaded_file.read())) as pdf:
            for i, page in enumerate(pdf.pages):
                text = page.extract_text()
                if text:
                    pdf_text.append(text)
                    page_numbers.append(i + 1)
        
        return pd.DataFrame({
            "content": pdf_text,
            "page_number": page_numbers
        })
    
    def _parse_docx_file(self, uploaded_file):
        """
        Parse DOCX file into DataFrame.
        
        Args:
            uploaded_file: DOCX file to parse
            
        Returns:
            DataFrame: DataFrame with extracted text
        """
        doc = Document(io.BytesIO(uploaded_file.read()))
        
        # Extract paragraphs and their styles
        paragraphs = []
        styles = []
        
        for para in doc.paragraphs:
            if para.text.strip():
                paragraphs.append(para.text)
                styles.append(para.style.name if para.style else "Normal")
        
        return pd.DataFrame({
            "content": paragraphs,
            "style": styles
        })

    def _save_data_to_collections(self, uploaded_files):
        """
        Save processed data to Chroma collections.
        
        Args:
            uploaded_files: List of uploaded file objects
        """
        try:
            if st.session_state.chunks_df.empty:
                st.warning("No data available to process.")
                return

            # Generate collection name
            collection_name = self._generate_collection_name(uploaded_files)
            st.session_state.random_collection_name = collection_name

            # Create or get collection
            collection = self.chroma_client.get_or_create_collection(
                name=collection_name,
                metadata={"description": "Collection for RAG system"}
            )

            # Add to active collections
            st.session_state.active_collections[collection_name] = collection
            
            # Process data in batches
            self._save_data_in_batches(collection)

            st.success(f"Data saved to collection: {collection_name}")
            st.session_state.data_saved_success = True

        except Exception as e:
            st.error(f"Error saving data to Chroma: {str(e)}")

    def _save_data_in_batches(self, collection):
        """
        Process dataframe batches with progress bar.
        Optimized for GPU usage with dynamic batch sizing.
        
        Args:
            collection: Chroma collection to save data to
        """
        # Get optimal batch size based on GPU availability and memory
        from utils.gpu_utils import get_optimal_batch_size, is_gpu_available, get_device_string
        
        # Larger batches for embedding generation when using GPU
        base_batch_size = get_optimal_batch_size() * 8 if is_gpu_available() else 64
        batch_size = min(base_batch_size, 256)  # Cap at 256 to avoid memory issues
        
        df_batches = self._divide_dataframe(st.session_state.chunks_df, batch_size)
        
        if not df_batches:
            st.warning("No data available to process.")
            return
            
        num_batches = len(df_batches)
        device_info = f"on {get_device_string()}" if is_gpu_available() else "on CPU"
        progress_text = f"Saving data to Chroma {device_info}. Please wait..."
        progress_bar = st.progress(0, text=progress_text)

        for i, batch_df in enumerate(df_batches):
            if not batch_df.empty:
                self._process_batch(batch_df, st.session_state.embedding_model, collection)
                progress_percentage = int(((i + 1) / num_batches) * 100)
                progress_bar.progress(
                    progress_percentage / 100, 
                    text=f"Processing batch {i + 1}/{num_batches} {device_info}"
                )
                # No need for sleep when processing is intensive

        progress_bar.empty()
        
    def _process_batch(self, batch_df, model, collection):
        """
        Encode and save a batch of data to Chroma.
        Now with GPU awareness for better performance.
        
        Args:
            batch_df (DataFrame): DataFrame containing batch data
            model: Embedding model to use
            collection: Chroma collection to add data to
        """
        try:
            # Using model on appropriate device (GPU/CPU) as configured earlier
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

    def _divide_dataframe(self, df, batch_size):
        """
        Divide DataFrame into smaller chunks based on the specified batch size.
        
        Args:
            df (DataFrame): DataFrame to divide
            batch_size (int): Size of each batch
            
        Returns:
            list: List of DataFrame batches
        """
        if df.empty:
            return []
            
        total_size = len(df)
        return [df.iloc[i:i+batch_size] for i in range(0, total_size, batch_size)]

    def _generate_collection_name(self, uploaded_files):
        """
        Generate a name for the collection.
        
        Args:
            uploaded_files: List of uploaded file objects
            
        Returns:
            str: Generated collection name
        """
        if uploaded_files:
            first_file_name = os.path.splitext(uploaded_files[0].name)[0]
            return f"rag_collection_{self._clean_collection_name(first_file_name)}"
        return f"rag_collection_{uuid.uuid4().hex[:8]}"
        
    def _clean_collection_name(self, name):
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

    def _setup_chunking(self, df):
        """
        Set up chunking options and process text chunks.
        
        Args:
            df (DataFrame): DataFrame to process
        """
        st.subheader("Chunking Options")

        if not df.empty:
            # Column selection for vector search
            index_column = st.selectbox(
                "Choose the column to index (for vector search):", 
                df.columns
            )
            st.write(f"Selected column for indexing: {index_column}")

            # Chunking options
            chunk_options = {
                "no_chunking": "No Chunking (keep original document)",
                "recursive": "Recursive Token Chunker (splits by paragraph, sentence, etc.)",
                "semantic": "Semantic Chunker (groups by meaning)"
            }

            selected_option = st.radio(
                "Select a chunking strategy:",
                options=list(chunk_options.keys()),
                format_func=lambda x: chunk_options[x],
                key="chunking_strategy"
            )
            
            # Chunking parameters
            with st.expander("Chunking Parameters"):
                if selected_option == "no_chunking":
                    st.info("No parameters needed for this chunking strategy.")
                elif selected_option == "recursive":
                    chunk_size = st.slider(
                        "Chunk Size (tokens)", 
                        min_value=50, 
                        max_value=1000, 
                        value=st.session_state.get("chunk_size", 200),
                        step=50,
                        help="Target size for each chunk"
                    )
                    chunk_overlap = st.slider(
                        "Chunk Overlap", 
                        min_value=0, 
                        max_value=100, 
                        value=st.session_state.get("chunk_overlap", 20),
                        step=5,
                        help="Number of tokens that overlap between chunks"
                    )
                    st.session_state.chunk_size = chunk_size
                    st.session_state.chunk_overlap = chunk_overlap
                elif selected_option == "semantic":
                    similarity_threshold = st.slider(
                        "Similarity Threshold", 
                        min_value=0.0, 
                        max_value=1.0, 
                        value=0.3,
                        step=0.05,
                        help="Threshold for determining chunk boundaries (higher = fewer chunks)"
                    )
                    st.session_state.similarity_threshold = similarity_threshold

            # Process button
            if st.button("Process Chunks"):
                with st.spinner("Processing chunks..."):
                    chunks_df = self._process_chunks(df, index_column, selected_option)
                    if chunks_df is not None and not chunks_df.empty:
                        st.session_state.chunks_df = chunks_df
                        st.success(f"Generated {len(chunks_df)} chunks!")
                    else:
                        st.error("No chunks were generated. Please check your data and settings.")

    def _process_chunks(self, df, index_column, chunker_type):
        """
        Process text chunks based on selected chunking option.
        
        Args:
            df (DataFrame): DataFrame to process
            index_column (str): Column to use for chunking
            chunker_type (str): Type of chunker to use
            
        Returns:
            DataFrame or None: DataFrame with chunks or None if no chunks
        """
        # Prepare chunker parameters
        chunker_params = {}
        
        if chunker_type == "recursive":
            chunker_params = {
                "chunk_size": st.session_state.get("chunk_size", 200),
                "chunk_overlap": st.session_state.get("chunk_overlap", 20)
            }
        elif chunker_type == "semantic":
            chunker_params = {
                "threshold": st.session_state.get("similarity_threshold", 0.3)
            }
        
        # Get the appropriate chunker
        chunker = get_chunker(chunker_type, **chunker_params)
        
        # Process chunks
        chunk_records = []
        progress_bar = st.progress(0)
        
        for i, (_, row) in enumerate(df.iterrows()):
            # Update progress
            progress_bar.progress(min((i + 1) / len(df), 1.0))
            
            # Get text to chunk
            text_to_chunk = row.get(index_column)
            if not isinstance(text_to_chunk, str) or not text_to_chunk.strip():
                continue
                
            # Split text into chunks
            chunks = chunker.split_text(text_to_chunk)
            
            # Create records for each chunk
            for chunk in chunks:
                if not chunk.strip():
                    continue
                    
                # Create a record with all original metadata plus the chunk
                record = {
                    'chunk': chunk,
                    'chunk_index': len(chunk_records),  # Track chunk order
                    **{k: v for k, v in row.to_dict().items() if k not in ['chunk', 'chunk_index']}
                }
                chunk_records.append(record)
        
        progress_bar.empty()
        
        # Create DataFrame if we have records
        if chunk_records:
            result_df = pd.DataFrame(chunk_records)
            # Add chunk length information
            result_df['chunk_length'] = result_df['chunk'].apply(len)
            return result_df
        
        return None

    def _render_collection_manager(self):
        """Render the collection management interface."""
        st.subheader("Manage Collections", divider=True)
        
        # Get all available collections
        collections = self._get_all_collections()
        
        if not collections:
            st.info("No collections available. Please upload data first.")
            return
            
        # Render collection list with form
        self._render_collection_list(collections)
        
        # Actions for selected collections
        self._render_collection_actions()

    def _get_all_collections(self):
        """
        Get information about all available collections.
        
        Returns:
            list: List of collection information dictionaries
        """
        collections = self.chroma_client.list_collections()
        collection_info = []
        
        # Gather information about each collection
        for collection in collections:
            try:
                count = collection.count()
                metadata = collection.get(include=["metadatas"], limit=1)
                sample_metadata = metadata["metadatas"][0] if metadata["metadatas"] else {}
                collection_info.append({
                    "name": collection.name,
                    "count": count,
                    "fields": list(sample_metadata.keys()) if sample_metadata else []
                })
            except Exception as e:
                st.error(f"Error accessing collection {collection.name}: {str(e)}")
                
        return collection_info

    def _render_collection_list(self, collection_info):
        """
        Render the list of collections with selection checkboxes using a form.
        
        Args:
            collection_info (list): List of collection information dictionaries
        """
        # Create a form to prevent reloading after each checkbox selection
        with st.form(key="collection_selection_form"):
            st.write("Select collections to load:")
            
            # Initialize checkbox values in session state if needed
            for info in collection_info:
                if f"select_{info['name']}" not in st.session_state:
                    st.session_state[f"select_{info['name']}"] = info["name"] in st.session_state.active_collections
            
            # Display collections in a table format
            for idx, info in enumerate(collection_info):
                col1, col2, col3 = st.columns([0.1, 0.45, 0.45])
                
                # Checkbox for selection
                with col1:
                    st.checkbox(
                        "", 
                        key=f"select_{info['name']}", 
                        value=info["name"] in st.session_state.active_collections
                    )
                
                # Collection name
                with col2:
                    st.write(f"**{info['name']}**")
                
                # Document count and fields
                with col3:
                    st.write(f"{info['count']} documents, {len(info['fields'])} fields")
            
            # Submit button for the form
            submitted = st.form_submit_button("Apply Selections")
            
            # Process selections when form is submitted
            if submitted:
                # Update active collections based on checkboxes
                for info in collection_info:
                    collection_name = info["name"]
                    is_selected = st.session_state.get(f"select_{collection_name}", False)
                    
                    if is_selected and collection_name not in st.session_state.active_collections:
                        self._add_collection(collection_name)
                    elif not is_selected and collection_name in st.session_state.active_collections:
                        self._remove_collection(collection_name)
        
        # Preview section outside the form
        if collection_info:
            # Create a selectbox for previewing collections
            preview_options = ["None"] + [info["name"] for info in collection_info]
            selected_preview = st.selectbox(
                "Preview collection:",
                preview_options,
                index=0
            )
            
            if selected_preview != "None":
                st.session_state.preview_collection = selected_preview
                self._preview_collection(selected_preview)

    def _render_collection_actions(self):
        """Render buttons for collection actions."""
        st.divider()
        col1, col2 = st.columns(2)
        
        with col1:
            if st.button("Load Selected Collections", 
                        disabled=len(st.session_state.active_collections) == 0,
                        use_container_width=True):
                self._load_selected_collections()
        
        with col2:
            if st.button("Clear Selection", 
                        disabled=len(st.session_state.active_collections) == 0,
                        use_container_width=True):
                self._clear_selection()

    def _add_collection(self, collection_name):
        """
        Add a collection to active collections.
        
        Args:
            collection_name (str): Name of collection to add
        """
        collection = self.chroma_client.get_collection(name=collection_name)
        st.session_state.active_collections[collection_name] = collection

    def _remove_collection(self, collection_name):
        """
        Remove a collection from active collections.
        
        Args:
            collection_name (str): Name of collection to remove
        """
        if collection_name in st.session_state.active_collections:
            del st.session_state.active_collections[collection_name]

    def _preview_collection(self, collection_name):
        """
        Preview the contents of a collection.
        
        Args:
            collection_name (str): Name of collection to preview
        """
        try:
            collection = self.chroma_client.get_collection(collection_name)
            if not collection:
                st.error(f"Collection {collection_name} not found.")
                return

            # Get sample documents with error handling
            try:
                results = collection.get(
                    limit=5,
                    include=['metadatas', 'documents']
                )
            except Exception as e:
                st.error(f"Failed to fetch documents: {str(e)}")
                return

            # Check if results exist and have the expected structure
            if not results or 'metadatas' not in results:
                st.info("No metadata available in this collection.")
                return

            metadatas = results['metadatas']
            documents = results.get('documents', [])

            if not metadatas:
                st.info("Collection is empty.")
                return

            # Display documents alongside metadata if available
            st.write("Sample documents:")
            for i, (meta, doc) in enumerate(zip(metadatas[:5], documents[:5] if documents else [None] * len(metadatas[:5]))):
                with st.expander(f"Document {i+1}"):
                    if doc:
                        st.text_area("Content", value=doc[:1000] + ("..." if len(doc) > 1000 else ""), height=100)
                    
                    st.write("Metadata:")
                    st.json(meta)

        except Exception as e:
            st.error(f"Error previewing collection: {str(e)}")

    def _load_selected_collections(self):
        """Load all selected collections into the main dataframe."""
        if not st.session_state.active_collections:
            st.warning("Please select at least one collection to load.")
            return
            
        all_metadatas = []
        
        with st.spinner("Loading selected collections..."):
            for collection_name, collection in st.session_state.active_collections.items():
                try:
                    data = collection.get(include=["documents", "metadatas"])
                    metadatas = data["metadatas"]
                    documents = data["documents"]
                    
                    # Add documents as chunks if not already in metadata
                    for i, metadata in enumerate(metadatas):
                        metadata['collection_source'] = collection_name
                        if 'chunk' not in metadata and i < len(documents):
                            metadata['chunk'] = documents[i]
                            
                    all_metadatas.extend(metadatas)
                except Exception as e:
                    st.error(f"Error loading collection {collection_name}: {str(e)}")
        
        if all_metadatas:
            # Get union of all column names
            column_names = list(set().union(*(meta.keys() for meta in all_metadatas)))
            
            # Create DataFrame with all columns, handling missing values
            st.session_state.chunks_df = pd.DataFrame(all_metadatas)
            
            # Set data source flag and success indicator
            st.session_state.data_saved_success = True
            st.toast(f"Successfully loaded {len(st.session_state.active_collections)} collections with {len(st.session_state.chunks_df)} documents", icon="✅")
        else:          
            st.toast("No data found in the selected collections.", icon="❌")

    def _clear_selection(self):
        """Clear all selected collections."""
        st.session_state.active_collections = {}
        st.session_state.chunks_df = pd.DataFrame()
        st.toast("Selection cleared", icon="✅")
    
    def search_documents(self, query, columns_to_answer, number_docs_retrieval=3):
        """
        Search documents using enhanced vector search with ranking.
        
        Args:
            query (str): Search query
            columns_to_answer (list): Columns to include in results
            number_docs_retrieval (int): Number of documents to retrieve
            
        Returns:
            tuple: (results, context text for LLM)
        """
        if not st.session_state.embedding_model:
            return [], "Error: Embedding model not initialized."
            
        if not st.session_state.active_collections:
            return [], "Error: No collections loaded."
        
        # Define boost factors for ranking - adjust as needed
        boost_factors = {
            "source_file": 0.1,      # Small boost for source file relevance
            "chunk_length": 0.05     # Small boost for longer chunks
        }
        
        # Perform enhanced vector search
        results, formatted_results = enhanced_vector_search(
            st.session_state.embedding_model,
            query,
            st.session_state.active_collections,
            columns_to_answer,
            number_docs_retrieval,
            boost_factors
        )
        
        return results, formatted_results