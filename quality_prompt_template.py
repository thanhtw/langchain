"""
Custom prompt template for code quality analysis.
This module extends the existing prompt system with specialized templates
for analyzing build errors and checkstyle violations.
"""

from prompts.prompt_engineering import PromptTemplate, PromptLibrary

def create_quality_template():
    """
    Create and return a specialized prompt template for code quality analysis.
    
    Returns:
        PromptTemplate: A new template for code quality analysis
    """
    return PromptTemplate(
        name="code_quality",
        description="Specialized prompt for code quality analysis",
        template=(
            "You are a professional code quality analyst and programming instructor. "
            "Analyze the following code quality issues and provide detailed explanations and solutions.\n\n"
            "USER QUERY: {query}\n\n"
            "CODE QUALITY ANALYSIS:\n{context}\n\n"
            "Please provide a comprehensive analysis of the build errors and checkstyle violations found in the report. "
            "For each issue:\n"
            "1. Explain what the error or violation means in simple terms\n"
            "2. Explain why it's important to fix it (impact on code quality, reliability, etc.)\n"
            "3. Provide a step-by-step solution to fix the issue\n"
            "4. If possible, show both the incorrect code and the corrected code\n\n"
            "Also include a summary of the overall code quality issues and general recommendations for improving the code. "
            "Be educational and helpful, as if you're teaching a student how to write better code."
        )
    )

def register_quality_template(prompt_library=None):
    """
    Register the code quality template with the prompt library.
    
    Args:
        prompt_library (PromptLibrary, optional): Library to register with
        
    Returns:
        bool: True if registration was successful
    """
    if prompt_library is None:
        prompt_library = PromptLibrary()
    
    # Check if template already exists
    if not prompt_library.get_template("code_quality"):
        template = create_quality_template()
        prompt_library.add_template(template)
        return True
    
    return False

# Function to format quality analysis prompt
def format_quality_analysis_prompt(query, code_analysis, additional_context=""):
    """
    Format a quality analysis prompt with the specialized template.
    
    Args:
        query (str): User's query
        code_analysis (str): Code quality analysis text
        additional_context (str, optional): Additional context information
        
    Returns:
        str: Formatted prompt for quality analysis
    """
    # Ensure template is registered
    prompt_library = PromptLibrary()
    register_quality_template(prompt_library)
    
    # Get the template
    template = prompt_library.get_template("code_quality")
    
    # Combine code analysis with additional context
    if additional_context:
        context = f"{code_analysis}\n\nADDITIONAL DOCUMENTATION:\n{additional_context}"
    else:
        context = code_analysis
    
    # Format and return
    return template.format(query=query, context=context)