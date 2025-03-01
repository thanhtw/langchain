"""
Advanced prompt engineering module with function calling capabilities.

This module provides tools for creating sophisticated prompts with:
1. Placeholders for retrieved data
2. Function calling within prompts
3. Dynamic data transformation
"""

import re
import os
import json
import importlib
import inspect
from typing import Dict, Any, List, Callable, Optional, Union
import datetime

# Registry for functions that can be called from prompts
FUNCTION_REGISTRY = {}

def register_function(func=None, *, name=None):
    """
    Decorator to register a function for use in prompts.
    
    Args:
        func: The function to register
        name: Optional custom name for the function
        
    Returns:
        The original function
    """
    def decorator(f):
        func_name = name or f.__name__
        FUNCTION_REGISTRY[func_name] = f
        return f
        
    if func is None:
        return decorator
    return decorator(func)

def load_function_modules(module_paths: List[str]):
    """
    Load modules containing registered functions.
    
    Args:
        module_paths: List of module paths to import
    """
    for module_path in module_paths:
        try:
            importlib.import_module(module_path)
        except ImportError as e:
            print(f"Failed to import module {module_path}: {e}")

class PromptTemplate:
    """Advanced prompt template with function calling and data insertion."""
    
    def __init__(self, template: str, name: str = "custom", description: str = ""):
        """
        Initialize a prompt template.
        
        Args:
            template: The template string with placeholders and function calls
            name: Name of the template
            description: Description of the template
        """
        self.template = template
        self.name = name
        self.description = description
        
        # Extract function calls during initialization
        self.function_calls = self._extract_function_calls(template)
        
    def _extract_function_calls(self, template: str) -> List[Dict[str, Any]]:
        """
        Extract function calls from the template.
        
        Args:
            template: The template string
            
        Returns:
            List of function call information
        """
        # Pattern matches {{function_name(arg1="value", arg2=123)}}
        pattern = r'\{\{(\w+)\((.*?)\)\}\}'
        matches = re.findall(pattern, template)
        
        function_calls = []
        for func_name, args_str in matches:
            # Parse arguments
            args_dict = {}
            if args_str.strip():
                # Split by commas but not within quotes
                args_parts = re.findall(r'(\w+)=(".*?"|\'.*?\'|\d+|\w+)', args_str)
                for arg_name, arg_value in args_parts:
                    # Remove quotes if present
                    if arg_value.startswith('"') and arg_value.endswith('"'):
                        arg_value = arg_value[1:-1]
                    elif arg_value.startswith("'") and arg_value.endswith("'"):
                        arg_value = arg_value[1:-1]
                    # Convert numbers if needed
                    elif arg_value.isdigit():
                        arg_value = int(arg_value)
                    args_dict[arg_name] = arg_value
            
            function_calls.append({
                "name": func_name,
                "args": args_dict,
                "pattern": f"{{{{{func_name}({args_str})}}}}",
            })
        
        return function_calls
    
    def format(self, **kwargs) -> str:
        """
        Format the template with the provided values and execute function calls.
        
        Args:
            **kwargs: Values to insert into the template
            
        Returns:
            Formatted prompt string
        """
        formatted_template = self.template
        
        # First process any function calls
        for func_info in self.function_calls:
            func_name = func_info["name"]
            
            if func_name in FUNCTION_REGISTRY:
                func = FUNCTION_REGISTRY[func_name]
                
                # Execute the function with arguments
                try:
                    # Update any placeholders in the arguments
                    processed_args = {}
                    for arg_name, arg_value in func_info["args"].items():
                        if isinstance(arg_value, str) and "{" in arg_value and "}" in arg_value:
                            # This is a placeholder that needs to be filled with kwargs
                            placeholder = arg_value.strip("{}")
                            if placeholder in kwargs:
                                processed_args[arg_name] = kwargs[placeholder]
                            else:
                                processed_args[arg_name] = arg_value
                        else:
                            processed_args[arg_name] = arg_value
                    
                    # Call the function with processed arguments
                    result = func(**processed_args)
                    
                    # Replace the function call with its result
                    formatted_template = formatted_template.replace(
                        func_info["pattern"], str(result)
                    )
                except Exception as e:
                    error_msg = f"[Error executing {func_name}: {str(e)}]"
                    formatted_template = formatted_template.replace(
                        func_info["pattern"], error_msg
                    )
            else:
                error_msg = f"[Unknown function: {func_name}]"
                formatted_template = formatted_template.replace(
                    func_info["pattern"], error_msg
                )
        
        # Then replace any remaining placeholders
        try:
            formatted_template = formatted_template.format(**kwargs)
        except KeyError as e:
            # Handle missing placeholders gracefully
            print(f"Warning: Missing placeholder in template: {e}")
            # Replace missing placeholders with a marker
            missing_key = str(e).strip("'")
            formatted_template = formatted_template.replace(
                "{" + missing_key + "}", f"[Missing: {missing_key}]"
            )
        
        return formatted_template

class PromptLibrary:
    """Library for managing prompt templates."""
    
    def __init__(self, storage_path: str = "data/prompt_templates.json"):
        """
        Initialize the prompt library.
        
        Args:
            storage_path: Path to the JSON file for storing templates
        """
        self.storage_path = storage_path
        self.templates = {}
        self.load_templates()
        
        # Add built-in templates if no templates exist
        if not self.templates:
            self._add_default_templates()
    
    def _add_default_templates(self):
        """Add default templates to the library."""
        default_templates = {
            "standard": PromptTemplate(
                name="standard",
                description="Standard RAG prompt with query and context",
                template=(
                    "Answer the following question based on the provided context information.\n\n"
                    "Question: {query}\n\n"
                    "Context:\n{context}\n\n"
                    "Answer:"
                )
            ),
            "detailed": PromptTemplate(
                name="detailed",
                description="Detailed prompt with explicit instructions",
                template=(
                    "You are a knowledgeable assistant tasked with answering questions based on provided information.\n\n"
                    "USER QUERY: {query}\n\n"
                    "RETRIEVED INFORMATION:\n{context}\n\n"
                    "INSTRUCTIONS:\n"
                    "1. Only use information from the provided context\n"
                    "2. If the context doesn't contain relevant information, acknowledge this limitation\n"
                    "3. Provide a comprehensive response addressing all aspects of the query\n"
                    "4. Format your response clearly with sections and bullet points as needed\n\n"
                    "ANSWER:"
                )
            ),
            "code_analysis": PromptTemplate(
                name="code_analysis",
                description="Specialized prompt for code analysis",
                template=(
                    "You are a professional code reviewer. Analyze the following code snippet in response to the query.\n\n"
                    "QUERY: {query}\n\n"
                    "CODE CONTEXT:\n{context}\n\n"
                    "Provide a detailed analysis including:\n"
                    "- Code quality assessment\n"
                    "- Potential bugs or issues\n"
                    "- Performance considerations\n"
                    "- Security implications\n"
                    "- Suggestions for improvement\n\n"
                    "Include specific examples from the code in your explanation."
                )
            ),
            "llama_optimized": PromptTemplate(
                name="llama_optimized",
                description="Optimized prompt for Llama models",
                template=(
                    "<|system|>\n"
                    "You are a helpful assistant that provides accurate information based on the context provided.\n"
                    "</|system|>\n\n"
                    "<|user|>\n"
                    "CONTEXT INFORMATION:\n"
                    "{context}\n\n"
                    "QUESTION: {query}\n"
                    "</|user|>\n\n"
                    "<|assistant|>"
                )
            ),
            "function_example": PromptTemplate(
                name="function_example",
                description="Example template with function calls",
                template=(
                    "Today is {{get_date()}}.\n\n"
                    "User query: {query}\n\n"
                    "Retrieved information:\n{context}\n\n"
                    "Additional information from external source: {{fetch_data(topic=query)}}\n\n"
                    "Please provide a comprehensive answer taking into account all the information above."
                )
            )
        }
        
        # Add to instance and save
        for name, template in default_templates.items():
            self.templates[name] = template
        
        self.save_templates()
    
    def add_template(self, template: PromptTemplate) -> None:
        """
        Add a template to the library.
        
        Args:
            template: The template to add
        """
        self.templates[template.name] = template
        self.save_templates()
    
    def get_template(self, name: str) -> Optional[PromptTemplate]:
        """
        Get a template by name.
        
        Args:
            name: Name of the template
            
        Returns:
            The template if found, None otherwise
        """
        return self.templates.get(name)
    
    def remove_template(self, name: str) -> bool:
        """
        Remove a template from the library.
        
        Args:
            name: Name of the template
            
        Returns:
            True if removed, False if not found
        """
        if name in self.templates:
            del self.templates[name]
            self.save_templates()
            return True
        return False
    
    def list_templates(self) -> List[Dict[str, str]]:
        """
        List all templates in the library.
        
        Returns:
            List of template information
        """
        return [
            {"name": name, "description": template.description}
            for name, template in self.templates.items()
        ]
    
    def save_templates(self) -> bool:
        """
        Save templates to storage.
        
        Returns:
            True if successful, False otherwise
        """
        try:
            # Create directory if it doesn't exist
            os.makedirs(os.path.dirname(self.storage_path), exist_ok=True)
            
            # Convert templates to serializable format
            serialized = {
                name: {
                    "template": template.template,
                    "name": template.name,
                    "description": template.description
                }
                for name, template in self.templates.items()
            }
            
            with open(self.storage_path, "w") as f:
                json.dump(serialized, f, indent=2)
            
            return True
        except Exception as e:
            print(f"Error saving templates: {e}")
            return False
    
    def load_templates(self) -> bool:
        """
        Load templates from storage.
        
        Returns:
            True if successful, False otherwise
        """
        if not os.path.exists(self.storage_path):
            return False
        
        try:
            with open(self.storage_path, "r") as f:
                serialized = json.load(f)
            
            self.templates = {
                name: PromptTemplate(
                    template=data["template"],
                    name=data["name"],
                    description=data["description"]
                )
                for name, data in serialized.items()
            }
            
            return True
        except Exception as e:
            print(f"Error loading templates: {e}")
            return False


# Register some built-in functions

@register_function
def get_date(format_str="%Y-%m-%d"):
    """Get the current date."""
    return datetime.datetime.now().strftime(format_str)

@register_function
def fetch_data(topic="", source="general"):
    """
    Fetch additional data based on a topic.
    This is a placeholder - implement actual data fetching logic.
    """
    # This is a placeholder implementation - replace with actual data fetching
    sources = {
        "general": f"General information about {topic} from external API.",
        "weather": f"Current weather data related to {topic}.",
        "news": f"Latest news about {topic}.",
    }
    
    return sources.get(source, f"Information about {topic} from {source}.")

@register_function
def summarize(text, max_length=200):
    """
    Summarize text to the specified maximum length.
    This is a placeholder - implement actual summarization logic.
    """
    if len(text) <= max_length:
        return text
    
    # Simple truncation with ellipsis - replace with actual summarization
    return text[:max_length-3] + "..."

@register_function
def count_tokens(text):
    """
    Count approximate tokens in text.
    This is a basic approximation - replace with a proper tokenizer.
    """
    # Rough approximation - 4 chars per token
    return len(text) // 4

@register_function
def transform_data(data, format_type="bullet"):
    """
    Transform data into different formats.
    """
    formats = {
        "bullet": lambda d: "\n".join([f"• {line.strip()}" for line in d.split("\n") if line.strip()]),
        "numbered": lambda d: "\n".join([f"{i+1}. {line.strip()}" for i, line in enumerate(d.split("\n")) if line.strip()]),
        "json": lambda d: json.dumps({"content": d}),
    }
    
    transformer = formats.get(format_type, lambda d: d)
    return transformer(data)