"""
General utility functions for the RAG application.
"""

import os
import sys
import json
import subprocess
import platform
import math
import uuid
import re
from pathlib import Path
import streamlit as st



"""
Utilities for analyzing code quality reports and executing external tools.
"""

def process_batch(batch_df, model, collection):
    """
    Encode and save a batch of data to Chroma.
    
    Args:
        batch_df (DataFrame): DataFrame containing batch data
        model: Embedding model to use
        collection: Chroma collection to add data to
    """
    try:
        # Encode column data to vectors for this batch
        embeddings = model.encode(batch_df['chunk'].tolist())

        # Collect all metadata in one list
        metadatas = [row.to_dict() for _, row in batch_df.iterrows()]

        # Generate unique ids for the batch
        batch_ids = [str(uuid.uuid4()) for _ in range(len(batch_df))]

        # Add the batch to Chroma
        collection.add(
            ids=batch_ids,
            embeddings=embeddings,
            metadatas=metadatas
        )
    except Exception as e:
        if str(e) == "'NoneType' object has no attribute 'encode'":
            raise RuntimeError("Please set up the language model before running the processing.")
        raise RuntimeError(f"Error saving data to Chroma for a batch: {str(e)}")

def divide_dataframe(df, batch_size):
    """
    Divide DataFrame into smaller chunks based on the specified batch size.
    
    Args:
        df (DataFrame): DataFrame to divide
        batch_size (int): Size of each batch
        
    Returns:
        list: List of DataFrame batches
    """
    num_batches = math.ceil(len(df) / batch_size)
    return [df.iloc[i * batch_size:(i + 1) * batch_size] for i in range(num_batches)]

def clean_collection_name(name):
    """
    Clean a collection name to ensure it's valid for Chroma.
    
    Args:
        name (str): Original collection name
        
    Returns:
        str: Cleaned collection name or None if invalid
    """
    if not name:
        return None
        
    # Allow only alphanumeric, underscores, hyphens, and single periods
    cleaned_name = re.sub(r'[^a-zA-Z0-9_.-]', '', name)
    cleaned_name = re.sub(r'\.{2,}', '.', cleaned_name)
    cleaned_name = re.sub(r'^[^a-zA-Z0-9]+|[^a-zA-Z0-9]+$', '', cleaned_name)

    # Ensure the cleaned name meets length constraints (3-63 chars)
    if 3 <= len(cleaned_name) <= 63:
        return cleaned_name
    elif len(cleaned_name) > 63:
        return cleaned_name[:63]
    else:
        return f"collection_{uuid.uuid4().hex[:8]}"  # Fallback to random name
    
def analyze_json_data(json_data):
    """
    Analyze JSON data for errors and violations from a quality check report.
    
    Args:
        json_data (dict): JSON data to analyze
        
    Returns:
        str: Analysis result
    """
    if not json_data:
        return "No JSON data to analyze."
    
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
    
    return "\n".join(analysis)

def execute_python_file(file_path, *args, capture_output=True):
    """
    Execute an external Python file in a platform-independent way.
    
    Args:
        file_path (str): Path to the Python file to execute
        *args: Additional arguments to pass to the Python file
        capture_output (bool): Whether to capture stdout/stderr
        
    Returns:
        subprocess.CompletedProcess: Result of the execution
    """
    try:
        # Ensure the path is properly formatted for the current OS
        file_path = os.path.normpath(file_path)
        
        # Construct the command
        command = [sys.executable, file_path]
        if args:
            command.extend(args)
        
        # Execute the command with platform-specific settings
        if platform.system() == "Windows":
            # Windows-specific settings
            if capture_output:
                result = subprocess.run(
                    command,
                    capture_output=True,
                    text=True,
                    check=False,  # Don't raise an exception on non-zero return code
                    creationflags=subprocess.CREATE_NO_WINDOW  # Hide console window
                )
            else:
                result = subprocess.run(
                    command,
                    check=False,
                    creationflags=subprocess.CREATE_NO_WINDOW
                )
        else:
            # Linux/Unix settings
            if capture_output:
                result = subprocess.run(
                    command,
                    capture_output=True,
                    text=True,
                    check=False
                )
            else:
                result = subprocess.run(
                    command,
                    check=False
                )
        
        return result
    except Exception as e:
        print(f"Error executing Python file: {str(e)}")
        return None

def find_quality_report():
    """
    Find the most recent quality report JSON file in the current directory.
    
    Returns:
        tuple: (file_path, json_data) or (None, None) if no file found
    """
    try:
        # Look for JSON files that might be quality reports
        json_files = [f for f in os.listdir() if f.endswith('.json') and 
                     ('quality' in f.lower() or 'report' in f.lower())]
        
        if not json_files:
            return None, None
        
        # Use the most recent file
        latest_file = max(json_files, key=os.path.getmtime)
        
        with open(latest_file, 'r') as f:
            json_data = json.load(f)
            
        return latest_file, json_data
    except Exception as e:
        print(f"Error finding quality report: {str(e)}")
        return None, None