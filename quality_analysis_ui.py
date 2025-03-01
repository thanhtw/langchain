"""
UI component for code quality analysis.
"""

import os
import streamlit as st
import json
from utils.utils import analyze_json_data, execute_python_file, find_quality_report

def render_quality_analysis_section(section_num):
    """
    Render quality analysis section in the UI.
    
    Args:
        section_num (int): Section number for the header
    """
    st.header(f"{section_num}. Code Quality Analysis")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Input for analysis script path
        analysis_script = st.text_input(
            "Path to quality analysis script:",
            value="",
            placeholder="e.g., /path/to/analysis_script.py",
            help="Enter the path to the script that performs code quality analysis"
        )
    
    with col2:
        # Input for project path
        project_path = st.text_input(
            "Path to project directory:",
            value="",
            placeholder="e.g., /path/to/project",
            help="Enter the path to the project directory to analyze"
        )
    
    # Run analysis button
    if st.button("Run Quality Analysis"):
        if not analysis_script:
            st.error("Please provide the path to the analysis script.")
            return
        
        with st.spinner("Running quality analysis..."):
            # Convert paths to platform-independent formats
            analysis_script = os.path.normpath(analysis_script)
            if project_path:
                project_path = os.path.normpath(project_path)
            
            # Execute the analysis script
            args = [project_path] if project_path else []
            result = execute_python_file(analysis_script, *args)
            
            if not result or result.returncode != 0:
                st.error(f"Error running quality analysis: {result.stderr if result else 'No result'}")
                return
            
            st.success(f"Quality analysis completed successfully.")
            
            # Look for output JSON files
            file_path, report_data = find_quality_report()
            
            if file_path and report_data:
                st.info(f"Found quality report: {file_path}")
                
                # Display report summary
                with st.expander("View Report Summary", expanded=True):
                    try:
                        report_analysis = analyze_json_data(report_data)
                        st.text_area("Report Analysis", report_analysis, height=300)
                    except Exception as e:
                        st.error(f"Error reading report: {str(e)}")
            else:
                st.warning("No quality report found after running analysis.")
    
    # Display existing reports section
    st.subheader("Existing Quality Reports")
    file_path, report_data = find_quality_report()
    
    if file_path and report_data:
        st.info(f"Found existing report: {file_path}")
        
        if st.button("Analyze Existing Report"):
            with st.spinner("Analyzing report..."):
                try:
                    report_analysis = analyze_json_data(report_data)
                    st.text_area("Report Analysis", report_analysis, height=300)
                    
                    # Store in session state for use in chatbot
                    st.session_state.quality_report_path = file_path
                    st.session_state.quality_report_analysis = report_analysis
                    
                    st.success("Analysis complete. You can now ask questions about the quality issues in the chatbot.")
                except Exception as e:
                    st.error(f"Error analyzing report: {str(e)}")
    else:
        st.info("No existing quality reports found. Run a quality analysis or upload a report.")
        
        # Add file uploader for quality reports
        uploaded_file = st.file_uploader("Upload a quality report (JSON)", type=["json"])
        if uploaded_file:
            try:
                report_data = json.load(uploaded_file)
                report_analysis = analyze_json_data(report_data)
                
                st.text_area("Report Analysis", report_analysis, height=300)
                
                # Save the uploaded file
                with open(f"uploaded_quality_report_{os.path.basename(uploaded_file.name)}", "wb") as f:
                    f.write(uploaded_file.getbuffer())
                
                # Store in session state for use in chatbot
                st.session_state.quality_report_path = f"uploaded_quality_report_{os.path.basename(uploaded_file.name)}"
                st.session_state.quality_report_analysis = report_analysis
                
                st.success("Report uploaded and analyzed successfully. You can now ask questions about the quality issues in the chatbot.")
            except Exception as e:
                st.error(f"Error processing uploaded file: {str(e)}")