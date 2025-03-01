"""
Custom prompts for the RAG system.

This module provides a collection of prompts for different use cases
and a mechanism to select and customize them.
"""

from typing import Dict, Optional, List, Any, Union


class PromptManager:
    """
    Manages prompt templates for the RAG system.
    """
    
    def __init__(self, default_prompt_name: str = "general"):
        """
        Initialize the PromptManager.
        
        Args:
            default_prompt_name (str): Name of the default prompt template
        """
        self.default_prompt_name = default_prompt_name
        self.prompt_templates = self._get_default_prompts()
        
    def _get_default_prompts(self) -> Dict[str, str]:
        """
        Get default prompt templates.
        
        Returns:
            dict: Dictionary of prompt name to template
        """
        return {
            "general": "The prompt of the user is: \"{query}\". Answer it based on the following retrieved data: \n\n{context}",
            
            "code_review": """You are a code review assistant. The user has asked: "{query}"
            
Answer based on these code snippets:

{context}

Provide a detailed and helpful analysis of the code, focusing on:
1. Code quality and best practices
2. Potential bugs or issues
3. Security considerations
4. Performance implications
5. Suggestions for improvement

Use specific examples from the provided code in your explanation.""",
            
            "document_summary": """You are a document analysis assistant. The user has asked: "{query}"
            
Based on these document sections:

{context}

Provide a clear, concise summary that addresses the user's question. Include:
1. Key information related to the query
2. Important facts and figures
3. Any relevant context from the documents

Be factual and only use information present in the provided context.""",
            
            "expert": """You are an expert assistant with deep knowledge in software development and RAG systems. 
            
USER QUERY: "{query}"

RELEVANT INFORMATION:
{context}

Provide a comprehensive and authoritative answer based on the information provided. Draw on your expertise to give insightful analysis. Be precise, technical, and thorough in your response.""",
            
            "concise": """Provide a brief and direct response to: "{query}"

Based on this information:
{context}

Keep your answer short and to the point (maximum 3 sentences).""",

            "deepseek_code": """You are an AI assistant powered by DeepSeek Coder, specialized in code analysis, review, and explanation.

The user is asking: "{query}"

Consider the following code context:
{context}

Provide a detailed technical analysis with code examples where appropriate. Focus on the specific elements of the code that are relevant to the user's question.""",

            "llama_custom": """<|system|>
You are a helpful AI assistant specialized in answering questions based on the provided context information. Always be factual, comprehensive, and accurate in your responses.
</|system|>

<|user|>
Context information:
{context}

My question is: {query}
</|user|>

<|assistant|>"""
        }
        
    def get_prompt(self, prompt_name: str) -> str:
        """
        Get a prompt template by name.
        
        Args:
            prompt_name (str): Name of the prompt template
            
        Returns:
            str: Prompt template string
        """
        return self.prompt_templates.get(prompt_name, self.prompt_templates[self.default_prompt_name])
        
    def add_prompt(self, name: str, template: str) -> None:
        """
        Add a new prompt template.
        
        Args:
            name (str): Name of the prompt template
            template (str): Prompt template string
        """
        self.prompt_templates[name] = template
        
    def modify_prompt(self, name: str, template: str) -> bool:
        """
        Modify an existing prompt template.
        
        Args:
            name (str): Name of the prompt template
            template (str): New prompt template string
            
        Returns:
            bool: True if prompt was modified, False if it doesn't exist
        """
        if name in self.prompt_templates:
            self.prompt_templates[name] = template
            return True
        return False
        
    def delete_prompt(self, name: str) -> bool:
        """
        Delete a prompt template.
        
        Args:
            name (str): Name of the prompt template
            
        Returns:
            bool: True if prompt was deleted, False if it doesn't exist
        """
        if name in self.prompt_templates and name != self.default_prompt_name:
            del self.prompt_templates[name]
            return True
        return False
        
    def format_prompt(self, prompt_name: str, query: str, context: str, **kwargs) -> str:
        """
        Format a prompt template with variables.
        
        Args:
            prompt_name (str): Name of the prompt template
            query (str): User query
            context (str): Retrieved context
            **kwargs: Additional variables to format into the template
            
        Returns:
            str: Formatted prompt string
        """
        template = self.get_prompt(prompt_name)
        
        # Base formatting with query and context
        formatted_prompt = template.format(query=query, context=context)
        
        # Apply additional formatting if any
        if kwargs:
            try:
                formatted_prompt = formatted_prompt.format(**kwargs)
            except KeyError:
                # Ignore formatting errors with additional kwargs
                pass
                
        return formatted_prompt
        
    def list_available_prompts(self) -> List[str]:
        """
        Get a list of available prompt template names.
        
        Returns:
            list: List of prompt template names
        """
        return list(self.prompt_templates.keys())
        
    def get_prompt_preview(self, prompt_name: str) -> str:
        """
        Get a preview of a prompt template.
        
        Args:
            prompt_name (str): Name of the prompt template
            
        Returns:
            str: Preview of the prompt template
        """
        template = self.get_prompt(prompt_name)
        return template[:100] + "..." if len(template) > 100 else template


def create_rag_prompt(query: str, context: str, style: str = "default", model_family: Optional[str] = None) -> str:
    """
    Create a RAG prompt based on style and model family.
    
    Args:
        query (str): User query
        context (str): Retrieved context
        style (str): Style of the prompt (default, detailed, concise, expert)
        model_family (str, optional): Model family for optimization
        
    Returns:
        str: Formatted prompt string
    """
    # Basic styles
    styles = {
        "default": f"The user asked: \"{query}\". Based on the following information, provide a helpful answer:\n\n{context}",
        
        "detailed": f"""The user has asked: "{query}"
        
Based on the following information:

{context}

Provide a detailed and comprehensive answer. Include specific references to the context when relevant. Be thorough and informative in your response.""",
        
        "concise": f"""Question: {query}
        
Context:
{context}

Provide a concise answer using only the information in the context. Be brief but thorough.""",
        
        "expert": f"""As an expert AI assistant, answer the following question:

"{query}"

Using this reference information:
{context}

Provide a detailed, authoritative response that demonstrates deep understanding.""",
        
        "code": f"""Analyze the following code in response to the query: "{query}"

CODE CONTEXT:
{context}

Provide a technical analysis focusing on best practices, potential issues, and any improvements that could be made."""
    }
    
    # Get base prompt from style
    prompt = styles.get(style, styles["default"])
    
    # Apply model-specific formatting if provided
    if model_family:
        model_family = model_family.lower()
        
        # Apply model-specific additions or modifications
        if "llama" in model_family:
            prompt = f"<|system|>\nYou are a helpful assistant.\n</|system|>\n\n<|user|>\n{prompt}\n</|user|>\n\n<|assistant|>"
        elif "deepseek" in model_family:
            if "coder" in model_family:
                prompt = f"You are DeepSeek Coder, an expert in code analysis.\n\n{prompt}"
            else:
                prompt = f"You are DeepSeek, a helpful AI assistant.\n\n{prompt}"
        elif "gemma" in model_family:
            prompt = f"<start_of_turn>user\n{prompt}\n<end_of_turn>\n\n<start_of_turn>model"
            
    return prompt