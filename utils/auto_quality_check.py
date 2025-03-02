"""
Enhanced quality check runner with automatic LLM analysis.
Integrated directly into the chatbot without a separate section.
"""

import os
import sys
import subprocess
import yaml
import platform
import shutil
import json
import tempfile
from pathlib import Path
import streamlit as st
from datetime import datetime
import time
import re
import pandas as pd

"""
Enhanced quality check runner with automatic LLM analysis.
Optimized vector search for code quality violations.
"""
from typing import List, Dict, Any, Tuple, Optional, Union


def run_quality_check_with_analysis(project_id, hw_number, config_path=None, embedding_model=None, 
                                   active_collections=None, columns_to_answer=None, 
                                   llm_model=None, num_docs_retrieval=3):
    """
    Run quality check and automatically analyze results with LLM.
    Extracts error messages from quality-check-report.json for targeted vector search.
    
    Args:
        project_id (str): Project ID to analyze
        hw_number (str): Homework number
        config_path (str): Path to config file
        embedding_model: Embedding model for vector search
        active_collections (dict): Active vector collections
        columns_to_answer (list): Columns to include in search
        llm_model: LLM model for analysis
        num_docs_retrieval (int): Number of documents to retrieve
        
    Returns:
        tuple: (success, message, analysis_result)
    """
    # Validate required parameters
    if not embedding_model:
        return False, "Embedding model not set. Please select a language first.", None
    
    if not active_collections:
        return False, "No vector collections active. Please upload and index data first.", None
    
    if not columns_to_answer:
        return False, "No columns selected for search. Please select columns first.", None
    
    if not llm_model:
        return False, "LLM model not initialized. Please set up a model first.", None
    
    # Run the quality check
    status_placeholder = st.empty()
    status_placeholder.info("Running quality check...")
    
    success, message, report_path = run_quality_check(project_id, hw_number, config_path)
    
    if not success:
        status_placeholder.error(message)
        return False, message, None
    
    if not report_path:
        status_placeholder.warning("Quality check completed but no report was generated.")
        return False, "No report generated", None
    
    # Analyze the quality report
    status_placeholder.info("Analyzing quality report...")
    analysis_text, json_data = analyze_quality_report(report_path)
    
    # Store in session state for future use
    st.session_state.quality_report_path = report_path
    st.session_state.quality_report_analysis = analysis_text
    st.session_state.quality_report_data = json_data
    
    # Perform targeted vector search for relevant information
    status_placeholder.info("Retrieving relevant documentation for violations...")
    
    # Create targeted queries based on the quality report findings
    queries = generate_targeted_queries(json_data)
    
    # Search vector database for each query
    all_metadatas = []
    all_retrieved_data = []
    
    for query_type, query in queries.items():
        if not query:
            continue
            
        try:
            status_placeholder.info(f"Searching for {query_type} information...")
            metadatas, retrieved_data = vector_search(
                embedding_model,
                query,
                active_collections,
                columns_to_answer,
                num_docs_retrieval // len(queries)  # Divide docs between queries
            )
            
            if metadatas and metadatas[0]:
                # Add query type to metadata for reference
                for metadata in metadatas[0]:
                    metadata['query_type'] = query_type
                all_metadatas.extend(metadatas[0])
                
            if retrieved_data:
                all_retrieved_data.append(f"--- {query_type.upper()} DOCUMENTATION ---\n{retrieved_data}")
                
        except Exception as e:
            status_placeholder.warning(f"Error searching for {query_type}: {str(e)}")
    
    # Combine the retrieved data
    combined_data = "\n\n".join(all_retrieved_data) if all_retrieved_data else "No relevant documentation found."
    
    # Prepare prompt for LLM
    status_placeholder.info("Generating analysis with LLM...")
    prompt = create_analysis_prompt(analysis_text, combined_data, project_id, hw_number)
    
    # Generate analysis with LLM
    try:
        llm_response = llm_model.generate_content(prompt)
        status_placeholder.success("Analysis complete!")
        
        # Display sidebar information if needed
        if all_metadatas:
            # Create dataframe with relevant columns only
            display_df = pd.DataFrame([{
                'Source': m.get('source_collection', 'Unknown'),
                'Type': m.get('query_type', 'General'),
                'Score': round(m.get('score', 0), 2) if 'score' in m else 'N/A'
            } for m in all_metadatas])
            
            st.sidebar.subheader("Retrieved References")
            st.sidebar.dataframe(display_df)
        
        # Return the result
        return True, "Quality check and analysis completed successfully", llm_response
    except Exception as e:
        status_placeholder.error(f"Error generating analysis: {str(e)}")
        return False, f"LLM analysis failed: {str(e)}", None

def generate_targeted_queries(json_data: Dict[str, Any]) -> Dict[str, str]:
    """
    Generate targeted search queries based on quality report findings.
    
    Args:
        json_data (dict): Parsed quality report JSON
        
    Returns:
        dict: Dictionary of query types and their search strings
    """
    queries = {
        "build_errors": "",
        "checkstyle_violations": "",
        "general_practices": "java code best practices and common errors"
    }
    
    # Extract build errors - specifically from logs.build.errors.details.message
    if "logs" in json_data and "build" in json_data["logs"]:
        build_logs = json_data["logs"]["build"]
        if "errors" in build_logs and "details" in build_logs["errors"]:
            errors = build_logs["errors"]["details"]
            error_messages = []
            
            for error in errors:
                message = error.get("message", "")
                if message:
                    # Clean the message for search
                    clean_message = message.replace("'", "").replace("\"", "")
                    # Extract key phrases from the error message
                    key_phrases = []
                    
                    # Look for specific error patterns
                    if "cannot find symbol" in clean_message.lower():
                        # Extract the symbol name
                        symbol_match = re.search(r'symbol:\s+(\w+)', clean_message)
                        if symbol_match:
                            key_phrases.append(f"java cannot find symbol {symbol_match.group(1)}")
                    elif "incompatible types" in clean_message.lower():
                        # Extract the type information
                        types_match = re.search(r'(\w+)\s+cannot be converted to\s+(\w+)', clean_message)
                        if types_match:
                            key_phrases.append(f"java convert {types_match.group(1)} to {types_match.group(2)}")
                    elif "missing return statement" in clean_message.lower():
                        key_phrases.append("java missing return statement")
                    else:
                        # For other errors, extract the first meaningful phrase
                        first_line = clean_message.split('\n')[0].strip()
                        # Limit to first 6 words for better search results
                        words = first_line.split()[:6]
                        if words:
                            key_phrases.append("java " + " ".join(words))
                    
                    # Add all valid phrases to our search terms
                    error_messages.extend([phrase for phrase in key_phrases if phrase])
            
            if error_messages:
                queries["build_errors"] = " OR ".join(error_messages)
    
    # Extract checkstyle violations - specifically from logs.checkstyle.violations.details.message
    if "logs" in json_data and "checkstyle" in json_data["logs"]:
        checkstyle_logs = json_data["logs"]["checkstyle"]
        if "violations" in checkstyle_logs and "details" in checkstyle_logs["violations"]:
            violations = checkstyle_logs["violations"]["details"]
            violation_messages = set()  # Use set to avoid duplicates
            
            for violation in violations:
                # Extract the actual violation message
                message = violation.get("message", "")
                rule = violation.get("rule", "")
                
                if message:
                    # Clean and extract key terms from the message
                    clean_message = message.replace("'", "").replace("\"", "")
                    # Take first 5-8 words for better search
                    words = clean_message.split()[:8]
                    if words:
                        violation_messages.add("checkstyle " + " ".join(words))
                
                # Also include the rule name for better context
                if rule:
                    # Format the rule name for better search
                    clean_rule = rule.replace("Check", "")
                    # Split camel case into words
                    words = re.findall(r'[A-Z][a-z]*', clean_rule)
                    if words:
                        violation_messages.add("checkstyle " + " ".join(words).lower())
            
            if violation_messages:
                queries["checkstyle_violations"] = " OR ".join(violation_messages)
    
    return queries

def vector_search(model, query, active_collections, columns_to_answer, number_docs_retrieval):
    """
    Perform targeted vector search across multiple collections.
    Optimized for code quality analysis queries.
    
    Args:
        model: The embedding model to use
        query (str): Search query
        active_collections (dict): Dictionary of active collections
        columns_to_answer (list): Columns to include in the response
        number_docs_retrieval (int): Number of results to retrieve
        
    Returns:
        tuple: (metadata list, formatted search result string)
    """
    all_metadatas = []
    filtered_metadatas = []
    
    if not query:
        return [], "No valid search query provided."
    
    # Generate query embeddings
    query_embeddings = model.encode([query])
    
    # Search each active collection
    for collection_name, collection in active_collections.items():
        try:
            results = collection.query(
                query_embeddings=query_embeddings,
                n_results=number_docs_retrieval
            )
            
            if results and 'metadatas' in results and results['metadatas']:
                # Add collection name to each metadata item
                for meta_list in results['metadatas']:
                    for meta in meta_list:
                        meta['source_collection'] = collection_name
                
                # Flatten the nested metadata structure
                for meta_list in results['metadatas']:
                    all_metadatas.extend(meta_list)
                    
        except Exception as e:
            print(f"Error searching collection {collection_name}: {str(e)}")
            continue
    
    if not all_metadatas:
        return [], "No relevant results found in any collection."
    
    # Filter metadata to only include selected columns plus source collection
    for metadata in all_metadatas:
        filtered_metadata = {
            'source_collection': metadata.get('source_collection', 'Unknown')
        }
        for column in columns_to_answer:
            if column in metadata:
                filtered_metadata[column] = metadata[column]
        filtered_metadatas.append(filtered_metadata)
        
    # Format the search results - keep content more compact for LLM context
    search_result = format_search_results_compact(filtered_metadatas, columns_to_answer)
    
    return [filtered_metadatas], search_result

def format_search_results_compact(metadatas, columns_to_answer):
    """
    Format search results in a more compact way for LLM processing.
    
    Args:
        metadatas (list): List of metadata dictionaries
        columns_to_answer (list): Columns to include in the result
        
    Returns:
        str: Formatted search result string
    """
    search_result = ""
    for i, metadata in enumerate(metadatas, 1):
        source = metadata.get('source_collection', 'Unknown')
        search_result += f"[{i}] Source: {source}\n"
        
        for column in columns_to_answer:
            if column in metadata:
                content = metadata.get(column)
                
                # Handle different content types
                if isinstance(content, str):
                    # Truncate content if it's extremely long
                    if len(content) > 1000:
                        content = content[:1000] + "... [truncated]"
                    
                    # Replace multiple newlines with single newlines
                    content = re.sub(r'\n+', '\n', content)
                    
                    search_result += f"{column}: {content}\n"
                else:
                    search_result += f"{column}: {content}\n"
        
        # Add separator between entries
        search_result += "-" * 40 + "\n"
    
    return search_result

def format_search_results(metadatas, columns_to_answer):
    """
    Format search results for display.
    
    Args:
        metadatas (list): List of metadata dictionaries
        columns_to_answer (list): Columns to include in the result
        
    Returns:
        str: Formatted search result string
    """
    search_result = ""
    for i, metadata in enumerate(metadatas, 1):
        search_result += f"\n{i}) Source: {metadata.get('source_collection', 'Unknown')}\n"
        for column in columns_to_answer:
            if column in metadata:
                search_result += f"   {column.capitalize()}: {metadata.get(column)}\n"
    return search_result

def create_analysis_prompt(analysis_text, retrieved_data, project_id, hw_number):
    """
    Create a comprehensive prompt for LLM analysis.
    Optimized for targeted code quality review using extracted error messages.
    
    Args:
        analysis_text (str): Quality report analysis text
        retrieved_data (str): Retrieved vector data
        project_id (str): Project ID
        hw_number (str): Homework number
        
    Returns:
        str: Formatted prompt
    """
    prompt = f"""You are a professional code quality analyst and programming instructor. 
Analyze the following code quality issues for Project {project_id}, {hw_number}, and provide detailed explanations and solutions.

CODE QUALITY ANALYSIS:
{analysis_text}

RELEVANT DOCUMENTATION:
{retrieved_data}

Provide a targeted analysis focusing on the specific error messages from the report:
1. For each build error message in the report, explain what it means and how to fix it
2. For each checkstyle violation message, explain what coding standard is being violated and how to correct it
3. Provide clear example code showing both incorrect and corrected versions where possible

Include a brief summary of key issues and general recommendations at the end.
Be clear and concise in your explanations - focus on practical guidance.
"""

    return prompt

def get_build_checkstyle_dir(config_path=None):
    """
    Get the build-checkstyle directory path from config.
    
    Args:
        config_path (str): Path to config file
        
    Returns:
        str: Path to build-checkstyle directory
    """
    # Default config path
    if not config_path:
        config_path = "config.yaml"
    
    # Default path as fallback
    default_path = os.path.abspath(os.path.join(os.getcwd(), '..', 'build-checkstyle'))
    try:
        # Check if config file exists
        if not os.path.exists(config_path):
            return default_path
            
        # Load config
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        
        # Get build-checkstyle path from config
        if 'quality' in config and 'build_checkstyle_dir' in config['quality']:
            path = config['quality']['build_checkstyle_dir']
            
            # Handle relative paths
            if not os.path.isabs(path):
                path = os.path.abspath(os.path.join(os.getcwd(), path))
                
            return path
            
        return default_path
    except Exception as e:
        print(f"Error reading config: {str(e)}")
        return default_path

def run_quality_check(project_id, hw_number, config_path=None):
    """
    Run code quality check using the external tool.
    
    Args:
        project_id (str): Project ID to analyze
        hw_number (str): Homework number
        config_path (str, optional): Path to custom config.yaml
        
    Returns:
        tuple: (success, message, report_path)
    """
    # Default config path
    if not config_path:
        config_path = "config.yaml"
    
    # Check if config file exists
    if not os.path.exists(config_path):
        return False, f"Config file not found: {config_path}", None
    
    try:
        # Read the existing config
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        
        # Update project ID and HW
        if 'project' not in config:
            config['project'] = {}
        
        config['project']['id'] = project_id
        config['project']['hw'] = hw_number
        
        # Create a temporary config file with the updated values
        temp_config = tempfile.NamedTemporaryFile(delete=False, suffix='.yaml')
        temp_config_path = temp_config.name
        
        with open(temp_config_path, 'w') as f:
            yaml.dump(config, f)
        
        # Get the build-checkstyle directory path from config
        build_checkstyle_dir = get_build_checkstyle_dir(config_path)
        
        # Check if the directory exists
        if not os.path.exists(build_checkstyle_dir):
            return False, f"Build checkstyle directory not found: {build_checkstyle_dir}", None
        
        # Change to the build checkstyle directory
        original_dir = os.getcwd()
        os.chdir(build_checkstyle_dir)
        
        # Run the quality check command
        command = [sys.executable, "run_checker.py", "--config", temp_config_path]
        
        # Execute with platform-specific settings
        try:
            if platform.system() == "Windows":
                result = subprocess.run(
                    command,
                    capture_output=True,
                    text=True,
                    check=False,
                    creationflags=subprocess.CREATE_NO_WINDOW
                )
            else:
                result = subprocess.run(
                    command,
                    capture_output=True,
                    text=True,
                    check=False
                )
            
            # Check for successful execution
            if result.returncode != 0:
                return False, f"Checker failed with error:\n{result.stderr}", None
            
            # Find the generated report
            timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
            report_pattern = f"quality-check-report-{timestamp}*.json"
            
            # Look for a matching report file
            report_files = []
            os.chdir("quality-check-results") 
            for file in os.listdir():
                if file.startswith("quality-check-report-") and file.endswith(".json"):
                    report_files.append(file)
            
            if not report_files:
                return True, "Quality check completed but no report file found.", None
            
            # Get the most recent report
            latest_report = max(report_files, key=os.path.getmtime)
            
            # Copy the report to the original directory
            report_path = os.path.join(original_dir, latest_report)
            shutil.copy2(latest_report, report_path)
            
            # Return to original directory
            os.chdir(original_dir)
            
            return True, "Quality check completed successfully.", report_path
            
        except Exception as e:
            os.chdir(original_dir)
            return False, f"Error running quality check: {str(e)}", None
            
    except Exception as e:
        return False, f"Error in quality check: {str(e)}", None
    finally:
        # Clean up the temporary config file
        if 'temp_config_path' in locals():
            try:
                os.unlink(temp_config_path)
            except:
                pass
        
        # Ensure we return to the original directory
        if 'original_dir' in locals():
            os.chdir(original_dir)

def analyze_quality_report(json_path):
    """
    Analyze a quality report JSON file.
    
    Args:
        json_path (str): Path to the JSON report file
        
    Returns:
        tuple: (formatted analysis text, json_data)
    """
    try:
        with open(json_path, 'r') as f:
            json_data = json.load(f)
        
        analysis = []
        
        # Extract meta information
        if "meta" in json_data:
            meta = json_data["meta"]
            project_id = meta.get("project_id", "Unknown")
            project_hw = meta.get("project_hw", "Unknown")
            timestamp = meta.get("timestamp", "Unknown")
            analysis.append(f"Quality check report for Project {project_id}, HW{project_hw} (Timestamp: {timestamp})")
        
        # Check for build errors
        if "results" in json_data and "build_success" in json_data["results"]:
            build_success = json_data["results"]["build_success"]
            if not build_success:
                analysis.append("BUILD: FAILED")
                
                # Check for detailed errors
                if "logs" in json_data and "build" in json_data["logs"]:
                    build_logs = json_data["logs"]["build"]
                    if "errors" in build_logs and "details" in build_logs["errors"]:
                        errors = build_logs["errors"]["details"]
                        analysis.append(f"Found {len(errors)} build errors:")
                        for i, error in enumerate(errors, 1):
                            file = error.get("file", "Unknown file")
                            line = error.get("line", "Unknown line")
                            message = error.get("message", "Unknown error")
                            analysis.append(f"{i}. Error in {file} at line {line}: {message}")
                            
                            # Include code context if available
                            if "code_context" in error:
                                analysis.append(f"   Code context: {error['code_context']}")
            else:
                analysis.append("BUILD: PASSED")
        
        # Check for checkstyle violations
        if "results" in json_data and "checkstyle_success" in json_data["results"]:
            checkstyle_success = json_data["results"]["checkstyle_success"]
            if not checkstyle_success:
                analysis.append("CHECKSTYLE: FAILED")
                
                # Check for detailed violations
                if "logs" in json_data and "checkstyle" in json_data["logs"]:
                    checkstyle_logs = json_data["logs"]["checkstyle"]
                    if "violations" in checkstyle_logs and "details" in checkstyle_logs["violations"]:
                        violations = checkstyle_logs["violations"]["details"]
                        total_violations = len(violations)
                        analysis.append(f"Found {total_violations} checkstyle violations:")
                        
                        # Group violations by file
                        violations_by_file = {}
                        for violation in violations:
                            file = violation.get("file", "Unknown file")
                            file_name = os.path.basename(file) if file != "Unknown file" else file
                            if file_name not in violations_by_file:
                                violations_by_file[file_name] = []
                            violations_by_file[file_name].append(violation)
                        
                        # Report violations by file
                        for file, file_violations in violations_by_file.items():
                            analysis.append(f"File: {file}")
                            for i, violation in enumerate(file_violations[:5], 1):
                                line = violation.get("line", "Unknown line")
                                message = violation.get("message", "Unknown violation")
                                rule = violation.get("rule", "Unknown rule")
                                analysis.append(f"  {i}. Line {line}: {message} (Rule: {rule})")
                            
                            if len(file_violations) > 5:
                                analysis.append(f"  ... and {len(file_violations) - 5} more violations in this file.")
            else:
                analysis.append("CHECKSTYLE: PASSED")
        
        # Check overall quality
        if "results" in json_data and "overall_quality_success" in json_data["results"]:
            overall_success = json_data["results"]["overall_quality_success"]
            if not overall_success:
                analysis.append("OVERALL QUALITY: FAILED")
            else:
                analysis.append("OVERALL QUALITY: PASSED")
        
        return "\n".join(analysis), json_data
        
    except Exception as e:
        return f"Error analyzing quality report: {str(e)}", None

def render_quality_check_widget_in_chatbot():
    """
    Add a quality check widget integrated into the chatbot UI.
    Compact version for better UI experience.
    """
    with st.expander("Run Code Quality Check", expanded=False):
        col1, col2 = st.columns(2)
        
        with col1:
            project_id = st.text_input("Student ID:", value="D0948363")
        
        with col2:
            hw_number = st.text_input("Homework Number:", value="HW3")
        
        if st.button("Run Quality Check and Analysis"):
            # Check prerequisites
            if not hasattr(st.session_state, 'embedding_model') or not st.session_state.embedding_model:
                st.error("Please select a language to initialize the embedding model.")
                return
                
            if not hasattr(st.session_state, 'active_collections') or not st.session_state.active_collections:
                st.error("No collection found. Please upload data and save it first.")
                return
                
            if not hasattr(st.session_state, 'llm_model') or not st.session_state.llm_model:
                st.error("Please initialize an LLM model first.")
                return
                
            if not hasattr(st.session_state, 'columns_to_answer') or not st.session_state.columns_to_answer:
                st.error("Please select columns for the chatbot to answer from.")
                return
            
            # Run the quality check and analysis
            with st.spinner("Running quality check and generating analysis..."):
                success, message, analysis_result = run_quality_check_with_analysis(
                    project_id,
                    hw_number,
                    config_path="../build-checkstyle/config.yaml",
                    embedding_model=st.session_state.embedding_model,
                    active_collections=st.session_state.active_collections,
                    columns_to_answer=st.session_state.columns_to_answer,
                    llm_model=st.session_state.llm_model,
                    num_docs_retrieval=st.session_state.number_docs_retrieval
                )
                
                if success and analysis_result:
                    # Add to chat history in a compact form
                    if "chat_history" not in st.session_state:
                        st.session_state.chat_history = []
                    
                    # Create a more compact version of the user query
                    user_query = f"Analyze code quality for Project {project_id}, {hw_number} and provide solutions"
                    st.session_state.chat_history.append({
                        "role": "user", 
                        "content": user_query
                    })
                    
                    # Add LLM response to chat history with full content
                    st.session_state.chat_history.append({
                        "role": "assistant", 
                        "content": analysis_result
                    })
                    
                    # Force a rerun to show the updated chat history
                    st.rerun()
                else:
                    st.error(message)