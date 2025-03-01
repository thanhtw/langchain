"""
Base LLM interface for the RAG application.
"""

from abc import ABC, abstractmethod

class LLM(ABC):
    @abstractmethod
    def create_agentic_chunker_message(self, system_prompt, messages, max_tokens=1000, temperature=1.0):
        """
        Create message method for using agentic chunker.
        
        Args:
            system_prompt (str): System prompt to guide the model
            messages (list): List of message objects
            max_tokens (int): Maximum number of tokens for context
            temperature (float): Temperature parameter for generation
            
        Returns:
            str: Generated content from the model
        """
        pass
        
    @abstractmethod
    def generate_content(self, prompt: str):
        """
        Generate content with given prompt.
        
        Args:
            prompt (str): The prompt to generate content from
            
        Returns:
            str: Generated content
        """
        pass