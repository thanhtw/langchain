"""
Utility for running code quality checks using external tools.
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
        config_path = "../../build-checkstyle/config.yaml"
    
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
        
        # Define the checker script path
        build_checkstyle_dir = os.path.abspath(os.path.join(os.getcwd(), '..', '..', 'build-checkstyle'))
        
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
        str: Formatted analysis text
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
# Function to render quality check widget (streamlit component)
def render_quality_check_widget():
    """
    Add a quality check widget to the Streamlit UI.
    """
    with st.expander("Run Code Quality Check", expanded=False):
        col1, col2 = st.columns(2)
        
        with col1:
            project_id = st.text_input("Project ID:", value="D0948363")
        
        with col2:
            hw_number = st.text_input("Homework Number:", value="HW3")
        
        if st.button("Run Quality Check"):
            with st.spinner("Running quality check..."):
                success, message, report_path = run_quality_check(project_id, hw_number)
                
                if success:
                    st.success(message)
                    if report_path:
                        st.success(f"Report generated: {report_path}")
                        
                        # Analyze and display the report
                        analysis_text, json_data = analyze_quality_report(report_path)
                        
                        # Store in session state for chatbot
                        st.session_state.quality_report_path = report_path
                        st.session_state.quality_report_analysis = analysis_text
                        st.session_state.quality_report_data = json_data
                        
                        # Display analysis
                        st.text_area("Quality Analysis", analysis_text, height=300)
                else:
                    st.error(message)