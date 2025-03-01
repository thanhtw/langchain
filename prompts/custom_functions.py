"""
Custom functions for use in prompt templates.

This module defines functions that can be called from within prompt templates
to dynamically fetch or transform data before sending to the LLM.
"""

import requests
import json
import os
import datetime
import re
from typing import Dict, Any, List, Optional

from prompts.prompt_engineering import register_function

@register_function
def get_current_weather(location: str) -> str:
    """
    Get current weather information for a location.
    
    Args:
        location: The location to get weather for
        
    Returns:
        str: Weather information
    """
    # This is a placeholder - replace with actual API call
    return f"The current weather in {location} is 72°F and sunny."

@register_function
def search_documentation(topic: str, max_results: int = 3) -> str:
    """
    Search project documentation for information about a topic.
    
    Args:
        topic: The topic to search for
        max_results: Maximum number of results to return
        
    Returns:
        str: Search results
    """
    # Placeholder for actual documentation search
    docs = {
        "rag": [
            "RAG (Retrieval-Augmented Generation) combines retrieval and generation for enhanced LLM responses.",
            "The RAG implementation supports multiple chunking strategies including semantic and recursive.",
            "RAG systems use vector databases like Chroma to store and retrieve relevant context."
        ],
        "llm": [
            "The system supports multiple LLM backends including Ollama, llama.cpp, and Transformers.",
            "Local LLMs are more private but generally less powerful than cloud-based alternatives.",
            "LLM responses can be improved through careful prompt engineering."
        ],
        "database": [
            "Chroma is used as the vector database for storing embeddings.",
            "Each document is split into chunks before being stored in the database.",
            "Vector similarity search is used to retrieve relevant chunks for answering queries."
        ]
    }
    
    # Get matches for the topic
    matches = []
    for key, entries in docs.items():
        if topic.lower() in key.lower() or key.lower() in topic.lower():
            matches.extend(entries)
        else:
            # Check each entry
            for entry in entries:
                if topic.lower() in entry.lower():
                    matches.append(entry)
    
    # Format results
    if not matches:
        return f"No information found for '{topic}' in the documentation."
    
    results = matches[:max_results]
    formatted = "\n".join([f"- {item}" for item in results])
    return f"Documentation about '{topic}':\n{formatted}"

@register_function
def format_as_markdown(text: str, style: str = "default") -> str:
    """
    Format text as Markdown with different styles.
    
    Args:
        text: The text to format
        style: Style to apply (default, bullet, numbered, heading)
        
    Returns:
        str: Formatted Markdown text
    """
    styles = {
        "default": lambda t: t,
        "bullet": lambda t: "\n".join([f"- {line.strip()}" for line in t.split("\n") if line.strip()]),
        "numbered": lambda t: "\n".join([f"{i+1}. {line.strip()}" for i, line in enumerate(t.split("\n")) if line.strip()]),
        "heading": lambda t: f"## {t}"
    }
    
    formatter = styles.get(style.lower(), styles["default"])
    return formatter(text)

@register_function
def extract_entities(text: str, entity_type: str = "all") -> str:
    """
    Extract named entities from text.
    
    Args:
        text: The text to extract entities from
        entity_type: Type of entities to extract (person, organization, location, all)
        
    Returns:
        str: Extracted entities
    """
    # This is a simplified placeholder - use a proper NER model in production
    entities = {
        "person": ["John Doe", "Jane Smith", "Alex Johnson"],
        "organization": ["Acme Corp", "Globex", "Initech"],
        "location": ["New York", "London", "Tokyo"]
    }
    
    # Simplistic entity extraction based on text matching
    results = {}
    for type_name, entity_list in entities.items():
        if entity_type.lower() in ["all", type_name]:
            found = [entity for entity in entity_list if entity.lower() in text.lower()]
            if found:
                results[type_name] = found
    
    # Format results
    if not results:
        return "No entities found."
    
    formatted = []
    for type_name, found in results.items():
        formatted.append(f"{type_name.capitalize()}: {', '.join(found)}")
    
    return "\n".join(formatted)

@register_function
def generate_examples(topic: str, num_examples: int = 2) -> str:
    """
    Generate examples related to a topic.
    
    Args:
        topic: The topic to generate examples for
        num_examples: Number of examples to generate
        
    Returns:
        str: Generated examples
    """
    examples = {
        "rag": [
            "Question answering system that retrieves documents before generating answers",
            "Legal research tool that searches case law before providing legal advice",
            "Medical diagnosis system that retrieves patient records before suggesting diagnoses",
            "Technical support bot that searches documentation before answering user questions"
        ],
        "prompt engineering": [
            "Using few-shot examples to guide model behavior",
            "Crafting system instructions to influence model personality",
            "Structuring prompts with clear sections and formatting",
            "Adding verification steps to improve reasoning"
        ],
        "vector database": [
            "Storing document embeddings for semantic search",
            "Implementing KNN search for finding similar items",
            "Using vector indexes to speed up similarity queries",
            "Clustering vectors to identify document groups"
        ]
    }
    
    # Find the closest topic
    best_match = None
    best_score = 0
    
    for key in examples:
        # Simple string matching score
        if topic.lower() in key.lower() or key.lower() in topic.lower():
            score = len(set(topic.lower().split()) & set(key.lower().split())) + 1
            if score > best_score:
                best_score = score
                best_match = key
    
    # Get examples for the matched topic
    if best_match and best_score > 0:
        selected = examples[best_match][:num_examples]
        formatted = "\n".join([f"{i+1}. {example}" for i, example in enumerate(selected)])
        return f"Examples of {topic}:\n{formatted}"
    else:
        return f"No examples available for '{topic}'."

@register_function
def summarize_text(text: str, max_length: int = 100) -> str:
    """
    Summarize text to the specified maximum length.
    
    Args:
        text: The text to summarize
        max_length: Maximum length of the summary
        
    Returns:
        str: Summarized text
    """
    # This is a very simplistic summarization - use a proper algorithm in production
    if len(text) <= max_length:
        return text
        
    # Extract the first sentence and truncate if needed
    sentences = text.split('.')
    if sentences:
        first_sentence = sentences[0].strip()
        if len(first_sentence) > max_length:
            return first_sentence[:max_length-3] + "..."
        
        # Add sentences until we reach max length
        summary = first_sentence
        i = 1
        while i < len(sentences) and len(summary) + len(sentences[i]) + 1 < max_length:
            summary += ". " + sentences[i].strip()
            i += 1
            
        return summary + "." if not summary.endswith(".") else summary
    
    # Fallback to truncation
    return text[:max_length-3] + "..."

@register_function
def find_code_examples(language: str, concept: str) -> str:
    """
    Find code examples for a programming concept.
    
    Args:
        language: Programming language
        concept: Programming concept to find examples for
        
    Returns:
        str: Code examples
    """
    examples = {
        "python": {
            "list comprehension": "squares = [x**2 for x in range(10)]",
            "dictionary": "user = {'name': 'John', 'age': 30, 'city': 'New York'}",
            "function": "def greet(name):\n    return f'Hello, {name}!'",
            "class": "class Person:\n    def __init__(self, name):\n        self.name = name\n    def greet(self):\n        return f'Hello, my name is {self.name}'"
        },
        "javascript": {
            "arrow function": "const add = (a, b) => a + b;",
            "map": "const doubled = numbers.map(x => x * 2);",
            "object": "const user = {name: 'John', age: 30, city: 'New York'};",
            "promise": "fetch('https://api.example.com/data')\n  .then(response => response.json())\n  .then(data => console.log(data))\n  .catch(error => console.error(error));"
        }
    }
    
    # Find examples for the specified language and concept
    language = language.lower()
    concept = concept.lower()
    
    if language in examples:
        # Look for exact match
        if concept in examples[language]:
            return f"```{language}\n{examples[language][concept]}\n```"
        
        # Look for partial match
        for key, example in examples[language].items():
            if concept in key or key in concept:
                return f"```{language}\n{example}\n```"
        
        # No match found
        return f"No examples found for '{concept}' in {language}."
    else:
        return f"No examples available for {language}."

@register_function
def get_definition(term: str) -> str:
    """
    Get the definition of a term from a glossary.
    
    Args:
        term: The term to define
        
    Returns:
        str: Definition of the term
    """
    glossary = {
        "rag": "Retrieval-Augmented Generation (RAG) is a technique that combines retrieval of relevant documents with text generation to produce more accurate and contextually relevant responses.",
        "llm": "Large Language Model (LLM) is an AI model trained on vast amounts of text data to generate human-like text and perform various language tasks.",
        "vector database": "A database optimized for storing and querying vector embeddings, typically used for semantic search and similarity matching.",
        "embedding": "A numerical representation of data (such as text) in a high-dimensional vector space, capturing semantic meaning.",
        "prompt engineering": "The practice of designing and optimizing inputs to language models to elicit desired outputs or behaviors.",
        "tokenization": "The process of breaking text into smaller units called tokens, which are processed by language models.",
        "fine-tuning": "The process of further training a pre-trained model on a specific dataset to adapt it for particular tasks or domains.",
        "chunking": "The process of dividing documents into smaller pieces for processing or storage in a vector database."
    }
    
    term = term.lower()
    
    # Check for exact match
    if term in glossary:
        return glossary[term]
    
    # Check for partial matches
    matches = []
    for key, definition in glossary.items():
        if term in key or key in term:
            matches.append((key, definition))
    
    if matches:
        # Return the best match
        matches.sort(key=lambda x: abs(len(x[0]) - len(term)))
        return f"Definition of '{matches[0][0]}': {matches[0][1]}"
    
    return f"No definition found for '{term}'."