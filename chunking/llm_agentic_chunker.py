"""
LLM-based agentic chunking implementation.

This module provides text chunking that uses LLMs to intelligently determine
chunk boundaries based on content understanding.
GPU-accelerated when available for faster processing.
"""

import re
import logging
from tqdm import tqdm
from chunking.base_chunker import BaseChunker
from chunking.recursive_token_chunker import RecursiveTokenChunker
from llms.base import LLM

# Configure logger
logger = logging.getLogger(__name__)

class LLMAgenticChunkerv2(BaseChunker):
    """
    Chunker that uses an LLM to determine chunk boundaries.
    
    This chunker leverages an LLM to identify semantically meaningful
    chunk boundaries based on content understanding.
    """
    
    def __init__(self, llm: LLM):
        """
        Initialize the LLM-based agentic chunker.
        
        Args:
            llm (LLM): LLM instance to use for chunking decisions
        """
        self.client = llm
        
        # Use a small chunk size for initial division
        self.splitter = RecursiveTokenChunker(
            chunk_size=50,
            chunk_overlap=0,
            length_function=len  # Simple length function
        )
        
        # Check if GPU is being used by the LLM
        from utils.gpu_utils import is_gpu_available
        self.gpu_available = is_gpu_available()
        if self.gpu_available:
            logger.info("GPU acceleration available for LLM-based chunking")
    
    def get_prompt(self, chunked_input, current_chunk=0, invalid_response=None):
        """
        Generate the prompt for LLM to determine chunk boundaries.
        
        Args:
            chunked_input (str): Text with chunk markers
            current_chunk (int): Current chunk index
            invalid_response (Any, optional): Previous invalid response for refinement
            
        Returns:
            List[Dict]: Messages for the LLM
        """
        messages = [
            {
                "role": "system", 
                "content": (
                    "You are an assistant specialized in splitting text into thematically consistent sections. "
                    "The text has been divided into chunks, each marked with <|start_chunk_X|> and <|end_chunk_X|> tags, where X is the chunk number. "
                    "Your task is to identify the points where splits should occur, such that consecutive chunks of similar themes stay together. "
                    "Respond with a list of chunk IDs where you believe a split should be made. For example, if chunks 1 and 2 belong together but chunk 3 starts a new topic, you would suggest a split after chunk 2. THE CHUNKS MUST BE IN ASCENDING ORDER."
                    "Your response should be in the form: 'split_after: 3, 5'."
                )
            },
            {
                "role": "user", 
                "content": (
                    "CHUNKED_TEXT: " + chunked_input + "\n\n"
                    "Respond only with the IDs of the chunks where you believe a split should occur. YOU MUST RESPOND WITH AT LEAST ONE SPLIT. THESE SPLITS MUST BE IN ASCENDING ORDER AND EQUAL OR LARGER THAN: " + str(current_chunk)+"." + (f"\n\The previous response of {invalid_response} was invalid. DO NOT REPEAT THIS ARRAY OF NUMBERS. Please try again." if invalid_response else "")
                )
            },
        ]
        return messages
        
    def split_text(self, text):
        """
        Split the text using LLM to determine chunk boundaries.
        
        Args:
            text (str): Text to split
            
        Returns:
            List[str]: List of text chunks
        """
        # Get initial chunks with a simple token-based approach
        chunks = self.splitter.split_text(text)
        split_indices = []
        current_chunk = 0

        # Process the chunks with the LLM to find semantic boundaries
        device_info = " on GPU" if self.gpu_available else " on CPU"
        with tqdm(total=len(chunks), desc=f"Processing chunks{device_info}") as pbar:
            while current_chunk < len(chunks) - 4:  # Stop when we're close to the end
                token_count = 0
                chunked_input = ''

                # Prepare input for LLM with chunk markers
                for i in range(current_chunk, len(chunks)):
                    # Simple estimate of token count - can be improved
                    token_count += len(chunks[i].split())
                    chunked_input += f"<|start_chunk_{i+1}|>{chunks[i]}<|end_chunk_{i+1}|>"
                    if token_count > 800:  # Limit context size
                        break

                # Generate the prompt for the LLM
                messages = self.get_prompt(chunked_input, current_chunk)
                
                # Get LLM response and parse it
                max_retries = 3
                retry_count = 0
                
                while retry_count < max_retries:
                    try:
                        # Set a lower temperature for more deterministic results - better for chunking decisions
                        result_string = self.client.create_agentic_chunker_message(
                            messages[0]['content'], 
                            messages[1:], 
                            max_tokens=200, 
                            temperature=0.2
                        )
                        
                        # Extract the split indices from the response
                        split_after_line = [line for line in result_string.split('\n') if 'split_after:' in line][0]
                        numbers = re.findall(r'\d+', split_after_line)
                        numbers = list(map(int, numbers))
                        
                        # Validate that numbers are in ascending order and >= current_chunk
                        if numbers == sorted(numbers) and all(number >= current_chunk for number in numbers):
                            break
                        else:
                            retry_count += 1
                            messages = self.get_prompt(chunked_input, current_chunk, numbers)
                    except (IndexError, ValueError) as e:
                        # Handle parsing errors by regenerating response
                        retry_count += 1
                        logger.warning(f"Error parsing LLM response: {str(e)}. Retry {retry_count}/{max_retries}")
                        messages = self.get_prompt(chunked_input, current_chunk, "invalid format")
                        continue
                
                if retry_count == max_retries:
                    # After max retries, use fallback approach
                    logger.warning("Max retries reached, using fallback chunking method")
                    numbers = [current_chunk + 4]  # Simple fallback: advance by 4 chunks
                
                # Add valid split indices
                split_indices.extend(numbers)
                
                # Move to the next chunk after the last split
                current_chunk = numbers[-1]
                
                # Update progress bar
                pbar.update(current_chunk - pbar.n)

        # Convert split indices to 0-based indices for our chunks array
        chunks_to_split_after = [i - 1 for i in split_indices]

        # Build the final chunks based on split indices
        docs = []
        current_chunk = ''
        for i, chunk in enumerate(chunks):
            current_chunk += chunk + ' '
            if i in chunks_to_split_after:
                docs.append(current_chunk.strip())
                current_chunk = ''
        
        # Add the last chunk if any content remains
        if current_chunk:
            docs.append(current_chunk.strip())

        return docs