"""
Recursive token chunking implementation.

This module provides a text splitter that recursively splits text based on separator
tokens, keeping the semantic structure of the document.
"""

from typing import Any, List, Optional
import re
from chunking.fixed_token_chunker import TextSplitter

def _split_text_with_regex(
    text: str, separator: str, keep_separator: bool
) -> List[str]:
    """
    Split text based on regex separator.
    
    Args:
        text (str): Text to split
        separator (str): Regex separator pattern
        keep_separator (bool): Whether to keep the separator in the result
        
    Returns:
        List[str]: Split text chunks
    """
    # Split the text based on the separator
    if separator:
        if keep_separator:
            # The parentheses in the pattern keep the delimiters in the result.
            _splits = re.split(f"({separator})", text)
            splits = [_splits[i] + _splits[i + 1] for i in range(1, len(_splits), 2)]
            if len(_splits) % 2 == 0:
                splits += _splits[-1:]
            splits = [_splits[0]] + splits
        else:
            splits = re.split(separator, text)
    else:
        splits = list(text)
    return [s for s in splits if s != ""]

class RecursiveTokenChunker(TextSplitter):
    """
    Splitting text by recursively looking at characters.
    
    Recursively tries to split by different characters to find one
    that works, ensuring chunks stay within size limits.
    """

    def __init__(
        self,
        chunk_size: int = 4000,
        chunk_overlap: int = 200,
        separators: Optional[List[str]] = None,
        keep_separator: bool = True,
        is_separator_regex: bool = False,
        **kwargs: Any,
    ) -> None:
        """
        Create a new RecursiveTokenChunker.
        
        Args:
            chunk_size (int): Maximum size of chunks to return
            chunk_overlap (int): Overlap between chunks
            separators (List[str], optional): List of separators to use for splitting
            keep_separator (bool): Whether to keep separators in the chunks
            is_separator_regex (bool): Whether the separators are regex patterns
            **kwargs: Additional arguments for the TextSplitter
        """
        super().__init__(chunk_size=chunk_size, chunk_overlap=chunk_overlap, keep_separator=keep_separator, **kwargs)
        self._separators = separators or ["\n\n", "\n", ".", "?", "!", " ", ""]
        self._is_separator_regex = is_separator_regex

    def _split_text(self, text: str, separators: List[str]) -> List[str]:
        """
        Split incoming text and return chunks.
        
        Args:
            text (str): Text to split
            separators (List[str]): List of separators to try
            
        Returns:
            List[str]: List of text chunks
        """
        final_chunks = []
        # Get appropriate separator to use
        separator = separators[-1]
        new_separators = []
        for i, _s in enumerate(separators):
            _separator = _s if self._is_separator_regex else re.escape(_s)
            if _s == "":
                separator = _s
                break
            if re.search(_separator, text):
                separator = _s
                new_separators = separators[i + 1 :]
                break

        _separator = separator if self._is_separator_regex else re.escape(separator)
        splits = _split_text_with_regex(text, _separator, self._keep_separator)

        # Now go merging things, recursively splitting longer texts.
        _good_splits = []
        _separator = "" if self._keep_separator else separator
        for s in splits:
            if self._length_function(s) < self._chunk_size:
                _good_splits.append(s)
            else:
                if _good_splits:
                    merged_text = self._merge_splits(_good_splits, _separator)
                    final_chunks.extend(merged_text)
                    _good_splits = []
                if not new_separators:
                    final_chunks.append(s)
                else:
                    other_info = self._split_text(s, new_separators)
                    final_chunks.extend(other_info)
        if _good_splits:
            merged_text = self._merge_splits(_good_splits, _separator)
            final_chunks.extend(merged_text)
        return final_chunks

    def split_text(self, text: str) -> List[str]:
        """
        Split text into multiple components.
        
        Args:
            text (str): Text to split
            
        Returns:
            List[str]: List of text chunks
        """
        return self._split_text(text, self._separators)