"""
Streamlit UI for the advanced prompt engineering system.

This module provides a user interface for creating, editing, and testing
prompt templates with function calling capabilities.
"""

import os
import sys
import json
import streamlit as st
from pathlib import Path

# Add parent directory to path
parent_dir = Path(__file__).parent.parent
sys.path.append(str(parent_dir))

from prompts.prompt_engineering import (
    PromptTemplate, 
    PromptLibrary, 
    FUNCTION_REGISTRY,
    register_function
)

# Example data for testing prompts
SAMPLE_DATA = {
    "query": "How does the RAG system handle PDF documents?",
    "context": (
        "The RAG system processes PDF documents using the pdfplumber library. "
        "When a PDF is uploaded, the system extracts text from each page. "
        "The extracted text is then divided into chunks according to the selected "
        "chunking strategy (recursive, semantic, or agentic). "
        "These chunks are embedded and stored in the Chroma vector database "
        "for efficient retrieval during question answering."
    )
}

def main():
    """Main function for the prompt engineering page."""
    st.title("Advanced Prompt Engineering")
    
    # Initialize prompt library
    prompt_library = PromptLibrary()
    
    # Get the list of available templates
    templates = prompt_library.list_templates()
    
    # Sidebar - Function Registry
    with st.sidebar:
        st.subheader("Available Functions")
        
        if FUNCTION_REGISTRY:
            for func_name, func in FUNCTION_REGISTRY.items():
                with st.expander(func_name):
                    doc = func.__doc__ or "No documentation"
                    st.markdown(f"**{func_name}**")
                    st.markdown(doc)
                    
                    # Show function signature
                    sig = inspect_function(func)
                    st.code(f"{{{{custom_fn(param1='value', param2='another value')}}}}", language="markdown")
        else:
            st.warning("No functions registered.")
            
        st.markdown("---")
        st.markdown("### Function Call Syntax")
        st.markdown("""
            To call functions in your prompt, use the syntax:
            ```
            {{function_name(param1="value", param2=123)}}
            ```
            
            The function call will be replaced with its return value when the prompt is processed.
            
            You can also use placeholders in function arguments:
            ```
            {{fetch_data(topic=query)}}
            ```
            
            This will pass the value of the 'query' parameter to the function.
        """)
        
        st.markdown("---")
        st.markdown("### Placeholders")
        st.markdown("""
            Use the following placeholders in your templates:
            
            - `{query}` - User's question
            - `{context}` - Retrieved information from the database
            - `{custom_param}` - Any custom parameter
        """)
    
    # Main content with tabs
    tab_browse, tab_create, tab_edit, tab_test = st.tabs([
        "Browse Templates", "Create Template", "Edit Template", "Test Template"
    ])
    
    # Browse Templates Tab
    with tab_browse:
        st.subheader("Available Templates")
        
        if not templates:
            st.info("No templates available. Create one in the 'Create Template' tab.")
        else:
            for i, template_info in enumerate(templates):
                name = template_info["name"]
                description = template_info["description"]
                
                with st.expander(f"{name} - {description}"):
                    # Get the full template object
                    template = prompt_library.get_template(name)
                    
                    if template:
                        st.text_area(
                            "Template Content:",
                            value=template.template,
                            height=200,
                            disabled=True
                        )
                        
                        # Show detected function calls
                        if template.function_calls:
                            st.markdown("**Function Calls:**")
                            for func_call in template.function_calls:
                                st.markdown(f"- `{func_call['pattern']}`")
                        else:
                            st.markdown("**Function Calls:** None")
    
    # Create Template Tab
    with tab_create:
        st.subheader("Create New Template")
        
        new_name = st.text_input("Template Name:")
        new_description = st.text_input("Description:")
        
        # Starter template options
        starter_option = st.radio(
            "Select a starting point:",
            ["Empty", "Basic RAG", "With Function Call Example", "Copy Existing"]
        )
        
        if starter_option == "Empty":
            new_template_content = "{query}\n\n{context}"
        elif starter_option == "Basic RAG":
            new_template_content = (
                "Answer the following question based on the provided context information.\n\n"
                "Question: {query}\n\n"
                "Context:\n{context}\n\n"
                "Answer:"
            )
        elif starter_option == "With Function Call Example":
            new_template_content = (
                "Today is {{get_date()}}.\n\n"
                "User query: {query}\n\n"
                "Retrieved information:\n{context}\n\n"
                "Additional information: {{fetch_data(topic=query)}}\n\n"
                "Please provide a comprehensive answer taking into account all the information above."
            )
        elif starter_option == "Copy Existing" and templates:
            template_names = [t["name"] for t in templates]
            selected_template = st.selectbox("Select template to copy:", template_names)
            template = prompt_library.get_template(selected_template)
            new_template_content = template.template if template else ""
        else:
            new_template_content = ""
        
        # Template editor
        new_template_content = st.text_area(
            "Template Content:",
            value=new_template_content,
            height=400
        )
        
        # Create button
        if st.button("Create Template"):
            if not new_name:
                st.error("Please provide a template name.")
            elif not new_template_content:
                st.error("Template content cannot be empty.")
            elif prompt_library.get_template(new_name):
                st.error(f"Template '{new_name}' already exists. Choose a different name.")
            else:
                # Create and add the template
                new_template = PromptTemplate(
                    template=new_template_content,
                    name=new_name,
                    description=new_description
                )
                
                prompt_library.add_template(new_template)
                st.success(f"Template '{new_name}' created successfully!")
                st.experimental_rerun()
    
    # Edit Template Tab
    with tab_edit:
        st.subheader("Edit Template")
        
        if not templates:
            st.info("No templates available to edit.")
        else:
            template_names = [t["name"] for t in templates]
            selected_template = st.selectbox("Select template to edit:", template_names)
            
            template = prompt_library.get_template(selected_template)
            
            if template:
                edit_name = st.text_input("Template Name:", value=template.name)
                edit_description = st.text_input("Description:", value=template.description)
                
                edit_content = st.text_area(
                    "Template Content:",
                    value=template.template,
                    height=400
                )
                
                col1, col2 = st.columns(2)
                
                with col1:
                    if st.button("Update Template"):
                        if not edit_name:
                            st.error("Template name cannot be empty.")
                        elif not edit_content:
                            st.error("Template content cannot be empty.")
                        elif edit_name != template.name and prompt_library.get_template(edit_name):
                            st.error(f"Template '{edit_name}' already exists. Choose a different name.")
                        else:
                            # Remove old template if name changed
                            if edit_name != template.name:
                                prompt_library.remove_template(template.name)
                            
                            # Create and add updated template
                            updated_template = PromptTemplate(
                                template=edit_content,
                                name=edit_name,
                                description=edit_description
                            )
                            
                            prompt_library.add_template(updated_template)
                            st.success(f"Template '{edit_name}' updated successfully!")
                            st.experimental_rerun()
                
                with col2:
                    if st.button("Delete Template"):
                        if prompt_library.remove_template(template.name):
                            st.success(f"Template '{template.name}' deleted successfully!")
                            st.experimental_rerun()
                        else:
                            st.error(f"Failed to delete template '{template.name}'.")
    
    # Test Template Tab
    with tab_test:
        st.subheader("Test Template")
        
        if not templates:
            st.info("No templates available to test.")
        else:
            template_names = [t["name"] for t in templates]
            selected_template = st.selectbox("Select template to test:", template_names, key="test_template_select")
            
            template = prompt_library.get_template(selected_template)
            
            if template:
                # Sample data for testing
                st.write("#### Test Data")
                col1, col2 = st.columns(2)
                
                with col1:
                    test_query = st.text_area("Query:", value=SAMPLE_DATA["query"], height=100)
                
                with col2:
                    test_context = st.text_area("Context:", value=SAMPLE_DATA["context"], height=200)
                
                # Custom parameters
                st.write("#### Custom Parameters (Optional)")
                param1_name = st.text_input("Parameter 1 Name:", value="custom_param1")
                param1_value = st.text_input("Parameter 1 Value:", value="Custom value 1")
                
                param2_name = st.text_input("Parameter 2 Name:", value="custom_param2")
                param2_value = st.text_input("Parameter 2 Value:", value="Custom value 2")
                
                # Run test
                if st.button("Test Template"):
                    try:
                        # Prepare test data
                        test_data = {
                            "query": test_query,
                            "context": test_context,
                            param1_name: param1_value,
                            param2_name: param2_value
                        }
                        
                        # Format template with test data
                        formatted_prompt = template.format(**test_data)
                        
                        st.write("#### Formatted Prompt")
                        st.text_area(
                            "Result (ready to send to LLM):",
                            value=formatted_prompt,
                            height=400
                        )
                        
                        # Function call summary
                        if template.function_calls:
                            st.write("#### Function Calls Executed")
                            for i, func_call in enumerate(template.function_calls, 1):
                                st.markdown(f"{i}. `{func_call['pattern']}`")
                                
                    except Exception as e:
                        st.error(f"Error formatting template: {str(e)}")


def inspect_function(func):
    """
    Get a human-readable representation of a function's signature.
    
    Args:
        func: The function to inspect
        
    Returns:
        str: String representation of the function signature
    """
    import inspect
    
    try:
        signature = inspect.signature(func)
        params = []
        
        for name, param in signature.parameters.items():
            if param.default != inspect.Parameter.empty:
                params.append(f"{name}={repr(param.default)}")
            else:
                params.append(name)
        
        return f"{func.__name__}({', '.join(params)})"
    except Exception:
        return f"{func.__name__}(...)"


if __name__ == "__main__":
    main()