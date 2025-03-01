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
from docx import Document
import pdfplumber
from chunking import RecursiveTokenChunker, LLMAgenticChunkerv2, ProtonxSemanticChunker
from utils import process_batch, divide_dataframe, clean_collection_name
from config.constants import NO_CHUNKING, DB


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
        self.llm_options = config_manager.get_llm_options()
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

        st.header(f"{section_num}. Setup data source")
        
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
        st.subheader("Upload data", divider=True)
        uploaded_files = st.file_uploader(
            "Upload CSV, JSON, PDF, or DOCX files", 
            accept_multiple_files=True
        )

        if uploaded_files:
            self._process_uploaded_files(uploaded_files)

        if len(st.session_state.chunks_df) > 0:
            st.write("Number of chunks:", len(st.session_state.chunks_df))
            st.dataframe(st.session_state.chunks_df)

        if st.button("Save Data"):
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
                all_data.append(df)

        if all_data:
            # Combine all data
            df = pd.concat(all_data, ignore_index=True)
            st.dataframe(df)

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
                return pd.json_normalize(json_data)
                
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
        with pdfplumber.open(io.BytesIO(uploaded_file.read())) as pdf:
            for page in pdf.pages:
                text = page.extract_text()
                if text:
                    pdf_text.append(text)
        return pd.DataFrame({"content": pdf_text})
    
    def _parse_docx_file(self, uploaded_file):
        """
        Parse DOCX file into DataFrame.
        
        Args:
            uploaded_file: DOCX file to parse
            
        Returns:
            DataFrame: DataFrame with extracted text
        """
        doc = Document(io.BytesIO(uploaded_file.read()))
        docx_text = [para.text for para in doc.paragraphs if para.text]
        return pd.DataFrame({"content": docx_text})

    def _save_data_to_collections(self, uploaded_files):
        """
        Save processed data to Chroma collections.
        
        Args:
            uploaded_files: List of uploaded file objects
        """
        try:
            if not len(st.session_state.chunks_df):
                st.warning("No data available to process.")
                return

            # Generate collection name
            collection_name = self._generate_collection_name(uploaded_files)
            st.session_state.random_collection_name = collection_name

            # Create or get collection
            collection = st.session_state.client.get_or_create_collection(
                name=collection_name,
                metadata={"description": "A collection for RAG system"}
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
        
        Args:
            collection: Chroma collection to save data to
        """
        batch_size = 256
        df_batches = divide_dataframe(st.session_state.chunks_df, batch_size)
        
        if not df_batches:
            st.warning("No data available to process.")
            return
            
        num_batches = len(df_batches)
        progress_text = "Saving data to Chroma. Please wait..."
        progress_bar = st.progress(0, text=progress_text)

        for i, batch_df in enumerate(df_batches):
            if not batch_df.empty:
                process_batch(batch_df, st.session_state.embedding_model, collection)
                progress_percentage = int(((i + 1) / num_batches) * 100)
                progress_bar.progress(
                    progress_percentage / 100, 
                    text=f"Processing batch {i + 1}/{num_batches}"
                )
                time.sleep(0.1)

        progress_bar.empty()

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
            return f"rag_collection_{clean_collection_name(first_file_name)}"
        return f"rag_collection_{uuid.uuid4().hex[:8]}"

    def _setup_chunking(self, df):
        """
        Set up chunking options and process text chunks.
        
        Args:
            df (DataFrame): DataFrame to process
        """
        st.subheader("Chunking")

        if not df.empty:
            # Column selection for vector search
            index_column = st.selectbox(
                "Choose the column to index (for vector search):", 
                df.columns
            )
            st.write(f"Selected column for indexing: {index_column}")

            # Chunking options
            chunk_options = [
                NO_CHUNKING,
                "RecursiveTokenChunker", 
                "SemanticChunker",
                "AgenticChunker",
            ]

            # Handle chunking option selection
            currentChunkerIdx = self._get_chunker_index(chunk_options)
            
            selected_option = st.radio(
                "Please select one of the options below.",
                chunk_options,
                captions=[
                    "Keep the original document",
                    "Recursively chunks text into smaller, meaningful token groups",
                    "Chunking with semantic comparison between chunks",
                    "Let LLM decide chunking"
                ],
                key="chunkOption",
                index=currentChunkerIdx
            )

            # Process chunks based on selection
            chunks_df = self._process_chunks(df, index_column, selected_option)
            if chunks_df is not None:
                st.session_state.chunks_df = chunks_df

    def _get_chunker_index(self, chunk_options):
        """
        Determine the index for chunking options.
        
        Args:
            chunk_options (list): List of chunking options
            
        Returns:
            int: Index of selected chunking option
        """
        if not st.session_state.get("chunkOption"):
            st.session_state.chunkOption = NO_CHUNKING
            return 0
        else:
            return chunk_options.index(st.session_state.get("chunkOption"))

    def _process_chunks(self, df, index_column, chunk_option):
        """
        Process text chunks based on selected chunking option.
        
        Args:
            df (DataFrame): DataFrame to process
            index_column (str): Column to use for chunking
            chunk_option (str): Selected chunking option
            
        Returns:
            DataFrame or None: DataFrame with chunks or None if no chunks
        """
        chunk_records = []
        embedding_option = None

        if chunk_option == "SemanticChunker":
            embedding_option = st.selectbox(
                "Choose the embedding method for Semantic Chunker:",
                ["TF-IDF", "Sentence-Transformers"]
            )

        for index, row in df.iterrows():
            selected_column_value = row[index_column]
            if not(isinstance(selected_column_value, str) and len(selected_column_value) > 0):
                continue

            chunks = self._get_chunks(selected_column_value, chunk_option, embedding_option)
            for chunk in chunks:
                chunk_record = {
                    'chunk': chunk,
                    **{k: v for k, v in row.to_dict().items() if k not in ['chunk', '_id']}
                }
                chunk_records.append(chunk_record)

        return pd.DataFrame(chunk_records) if chunk_records else None

    def _get_chunks(self, text, chunk_option, embedding_option=None):
        """
        Get chunks based on selected chunking option.
        
        Args:
            text (str): Text to chunk
            chunk_option (str): Selected chunking option
            embedding_option (str, optional): Embedding option for semantic chunker
            
        Returns:
            list: List of text chunks
        """
        if chunk_option == NO_CHUNKING:
            return [text]
            
        elif chunk_option == "RecursiveTokenChunker":
            chunker = RecursiveTokenChunker(
                chunk_size=st.session_state.chunk_size,
                chunk_overlap=st.session_state.chunk_overlap
            )
            return chunker.split_text(text)
            
        elif chunk_option == "SemanticChunker":
            if embedding_option == "TF-IDF":
                chunker = ProtonxSemanticChunker(embedding_type="tfidf")
            else:
                chunker = ProtonxSemanticChunker(
                    embedding_type="transformers",
                    model="all-MiniLM-L6-v2"
                )
            return chunker.split_text(text)
            
        elif chunk_option == "AgenticChunker":
            llm_model = st.session_state.get("llm_model")
            if llm_model:
                chunker = LLMAgenticChunkerv2(llm_model)
                return chunker.split_text(text)
            return [text]
            
        return [text]

    def _render_collection_manager(self):
        """Render the collection management interface."""
        st.subheader("Manage Collections", divider=True)
        
        # Initialize preview state if not exists
        if 'preview_collection' not in st.session_state:
            st.session_state.preview_collection = None
        
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
        collections = st.session_state.client.list_collections()
        collection_info = []
        
        # Gather information about each collection
        for collection in collections:
            try:
                count = collection.count()
                metadata = collection.get(include=["metadatas"])
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
        collection = st.session_state.client.get_collection(name=collection_name)
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
            collection = st.session_state.client.get_collection(collection_name)
            if not collection:
                st.error(f"Collection {collection_name} not found.")
                return

            # Get sample documents with error handling
            try:
                results = collection.get(
                    limit=5,
                    include=['metadatas']  # Only get metadatas, no documents
                )
            except Exception as e:
                st.error(f"Failed to fetch documents: {str(e)}")
                return

            # Check if results exist and have the expected structure
            if not results or 'metadatas' not in results:
                st.info("No metadata available in this collection.")
                return

            metadatas = results['metadatas']

            if not metadatas:
                st.info("Collection is empty.")
                return

            # Create data for the dataframe using only metadata
            data = [meta for meta in metadatas if meta]  # Only add if metadata exists

            # Display the dataframe
            if data:
                st.dataframe(data, use_container_width=True)
            else:
                st.info("No metadata to display.")

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
                    for metadata in metadatas:
                        metadata['collection_source'] = collection_name
                    all_metadatas.extend(metadatas)
                except Exception as e:
                    st.error(f"Error loading collection {collection_name}: {str(e)}")
        
        if all_metadatas:
            # Get union of all column names
            column_names = list(set().union(*(meta.keys() for meta in all_metadatas)))
            st.session_state.chunks_df = pd.DataFrame(all_metadatas, columns=column_names)
            st.session_state.data_saved_success = True
            st.session_state.source_data = DB
            st.toast(f"Successfully loaded {len(st.session_state.active_collections)} collections", icon="✅")
        else:          
            st.toast("No data found in the selected collections.", icon="❌")

    def _clear_selection(self):
        """Clear all selected collections."""
        st.session_state.active_collections = {}
        st.session_state.chunks_df = pd.DataFrame()
        st.toast("Selection cleared", icon="✅")