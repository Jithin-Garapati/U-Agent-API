"""
General-Purpose Flight Log Analysis Tools

This module provides truly general-purpose tools for flight log analysis that make
no assumptions about specific metrics or data structures. These tools allow the agent
to have full flexibility in discovering, retrieving, and analyzing flight log data.

The tools include:
1. get_topic_fields - List all fields in a specific topic
2. get_data - Retrieve raw data with minimal processing
3. run_computation - Run arbitrary computations on data

Usage:
    from tools.general_purpose_tools import get_topic_fields, get_data, run_computation
    
    # List all fields in a topic
    fields = get_topic_fields("vehicle_local_position")
    
    # Get data for specific fields
    data = get_data("vehicle_local_position", ["timestamp", "x", "y", "z"])
    
    # Run a computation on the data
    result = run_computation(data, "max(df['z'])")
"""

import os
import glob
import pandas as pd
import numpy as np
from typing import List, Dict, Any, Union, Optional
import ast
from pathlib import Path
import re
import builtins
import io         # Added for capturing stdout
import contextlib # Added for redirect_stdout
import math
import time
import traceback
import datetime # <-- ADDED IMPORT
from contextlib import redirect_stdout, redirect_stderr
import uuid # Added for unique filenames
import base64 # <-- ADDED IMPORT

# Add color constants for console output
try:
    from constants import RED, YELLOW, RESET, GREEN
except ImportError:
    # Fallback if constants can't be imported
    RED = "\033[91m"
    YELLOW = "\033[93m"
    GREEN = "\033[92m"
    RESET = "\033[0m"

# Add Import for SessionContext
# Assuming session.py is accessible via the adjusted sys.path in main.py
# This might need adjustment based on how tools are imported/run
try:
    from The_agent_api.api.session import SessionContext
except ImportError:
    # Fallback or error handling if SessionContext can't be imported directly
    # This indicates a potential issue with how the environment/PYTHONPATH is set up
    # when this tool module is loaded.
    print("WARNING: Could not import SessionContext. 'run_computation' might fail.")
    SessionContext = None # Define as None to avoid NameError, but tool will likely fail

# Default directory for CSV topic files
DEFAULT_CSV_DIR = "csv_topics"

# Cache to store topic to file path mappings
_TOPIC_FILE_CACHE = {}

# --- Attempt to import plotting libraries --- 
# These are optional for the compute tool but needed for plotting
_MATPLOTLIB_AVAILABLE = False
_SEABORN_AVAILABLE = False
try:
    import matplotlib
    matplotlib.use('Agg') # Use non-interactive backend suitable for servers
    import matplotlib.pyplot as plt
    _MATPLOTLIB_AVAILABLE = True
except ImportError:
    print("WARNING: matplotlib not found. Plotting tool will not work.")
try:
    import seaborn as sns
    _SEABORN_AVAILABLE = True
except ImportError:
    print("WARNING: seaborn not found. Some plotting styles may not work.")
# --- End Plotting Imports --- 

def get_topic_fields(topic_name: str, csv_dir: str = DEFAULT_CSV_DIR, file_path: str = None) -> Dict[str, Any]:
    """
    List all fields (columns) in a specific topic.
    
    Args:
        topic_name: Name of the topic to get fields from
        csv_dir: Directory containing CSV files (default: csv_topics)
        file_path: Exact file path (overrides topic_name if provided)
        
    Returns:
        Dictionary with 'success' flag and either 'fields' list or 'error' message
    """
    try:
        # If file_path is provided, use it directly
        if file_path:
            if os.path.exists(file_path):
                topic_file = file_path
                # Cache this file path for future use
                _TOPIC_FILE_CACHE[topic_name] = file_path
            else:
                return {
                    "success": False,
                    "error": f"Provided file path '{file_path}' does not exist"
                }
        else:
            # Check if we have this topic in our cache
            if topic_name in _TOPIC_FILE_CACHE:
                topic_file = _TOPIC_FILE_CACHE[topic_name]
            else:
                # Try both direct pattern and with flight_log prefix
                topic_files = glob.glob(f"{csv_dir}/{topic_name}_*.csv")
                
                # If not found, try with flight_log prefix
                if not topic_files:
                    topic_files = glob.glob(f"{csv_dir}/flight_log_{topic_name}_*.csv")
                
                # If still not found, try a more flexible pattern
                if not topic_files:
                    topic_files = glob.glob(f"{csv_dir}/*{topic_name}*.csv")
                
                if not topic_files:
                    return {
                        "success": False,
                        "error": f"No CSV files found for topic '{topic_name}' in '{csv_dir}'"
                    }
                
                # Use the first file (most topics only have one file anyway)
                topic_file = topic_files[0]
                # Cache this for future use
                _TOPIC_FILE_CACHE[topic_name] = topic_file
        
        # Read the CSV header to get field names
        df = pd.read_csv(topic_file, nrows=0)
        fields = df.columns.tolist()
        
        return {
            "success": True,
            "topic": topic_name,
            "fields": fields,
            "file": topic_file
        }
    except Exception as e:
        return {
            "success": False,
            "error": f"Error getting fields for topic '{topic_name}': {str(e)}"
        }

def get_data(topic_name: str, fields: Optional[List[str]] = None, 
             filters: Optional[Dict[str, Any]] = None, 
             csv_dir: str = DEFAULT_CSV_DIR,
             file_path: str = None) -> Dict[str, Any]:
    """
    Retrieve raw data from a topic with minimal processing.
    
    Args:
        topic_name: Name of the topic to get data from
        fields: List of field names to retrieve (IMPORTANT: Always specify specific fields 
               rather than None to improve performance and clarity)
        filters: Dictionary of filters to apply {field_name: value} or {field_name: (operator, value)}
        csv_dir: Directory containing CSV files (default: csv_topics)
        file_path: Exact file path (overrides topic_name if provided)
        
    Returns:
        Dictionary with 'success' flag and either 'data' DataFrame or 'error' message
    """
    try:
        # If file_path is provided, use it directly
        if file_path:
            if os.path.exists(file_path):
                topic_file = file_path
                # Cache this file path for future use
                _TOPIC_FILE_CACHE[topic_name] = file_path
            else:
                return {
                    "success": False,
                    "error": f"Provided file path '{file_path}' does not exist"
                }
        else:
            # Check if we have this topic in our cache
            if topic_name in _TOPIC_FILE_CACHE:
                topic_file = _TOPIC_FILE_CACHE[topic_name]
            else:
                # Try both direct pattern and with flight_log prefix
                topic_files = glob.glob(f"{csv_dir}/{topic_name}_*.csv")
                
                # If not found, try with flight_log prefix
                if not topic_files:
                    topic_files = glob.glob(f"{csv_dir}/flight_log_{topic_name}_*.csv")
                
                # If still not found, try a more flexible pattern
                if not topic_files:
                    topic_files = glob.glob(f"{csv_dir}/*{topic_name}*.csv")
                
                if not topic_files:
                    return {
                        "success": False,
                        "error": f"No CSV files found for topic '{topic_name}' in '{csv_dir}'"
                    }
                
                # Use the first file (most topics only have one file anyway)
                topic_file = topic_files[0]
                # Cache this for future use
                _TOPIC_FILE_CACHE[topic_name] = topic_file
        
        # Read the CSV data
        df = pd.read_csv(topic_file)
        
        # Select specific fields if requested
        if fields:
            # Check if all requested fields exist
            missing_fields = [f for f in fields if f not in df.columns]
            if missing_fields:
                return {
                    "success": False,
                    "error": f"Missing fields in topic '{topic_name}': {missing_fields}"
                }
            
            df = df[fields]
        else:
            # Warning message if no fields specified (selecting all fields)
            print(f"WARNING: No specific fields selected for topic '{topic_name}'. "
                  f"Consider specifying fields for better performance. "
                  f"Available fields: {', '.join(df.columns[:5])}... and {len(df.columns)-5} more")
        
        # Apply filters if provided
        if filters:
            for field, condition in filters.items():
                if field not in df.columns:
                    return {
                        "success": False,
                        "error": f"Filter field '{field}' not found in topic '{topic_name}'"
                    }
                
                if isinstance(condition, tuple) and len(condition) == 2:
                    operator, value = condition
                    if operator == '==':
                        df = df[df[field] == value]
                    elif operator == '!=':
                        df = df[df[field] != value]
                    elif operator == '>':
                        df = df[df[field] > value]
                    elif operator == '>=':
                        df = df[df[field] >= value]
                    elif operator == '<':
                        df = df[df[field] < value]
                    elif operator == '<=':
                        df = df[df[field] <= value]
                    else:
                        return {
                            "success": False,
                            "error": f"Unsupported operator '{operator}' in filter"
                        }
                else:
                    # Assume equality for simple value
                    df = df[df[field] == condition]
        
        # Create a preview of the data (first few rows and statistics) that is JSON-serializable
        preview_rows = 3
        
        # Store the entire DataFrame in memory for potential calculations
        # But only send a preview to the LLM
        result = {
            "success": True,
            "topic": topic_name,
            "rows": len(df),
            "columns": df.columns.tolist(),
            "file_path": topic_file,
            "preview": df.head(preview_rows).values.tolist(),
            "column_names": df.columns.tolist(),
            "statistics": {
                col: {
                    "min": float(df[col].min()) if pd.api.types.is_numeric_dtype(df[col]) else None,
                    "max": float(df[col].max()) if pd.api.types.is_numeric_dtype(df[col]) else None,
                    "mean": float(df[col].mean()) if pd.api.types.is_numeric_dtype(df[col]) else None,
                    "std": float(df[col].std()) if pd.api.types.is_numeric_dtype(df[col]) else None,
                } for col in df.columns
            },
            "_dataframe": df  # This is for internal use and will be removed before serialization
        }
        
        return result
    except Exception as e:
        return {
            "success": False,
            "error": f"Error getting data for topic '{topic_name}': {str(e)}"
        }

def run_computation(session: SessionContext, 
                    data_ids: Optional[List[str]], # Changed from data_id
                    computation: str,
                    comment: Optional[str] = None) -> Dict[str, Any]:
    """
    Executes a Python computation string. If data_ids are provided, loads the cached 
    DataFrames into variables named 'df1', 'df2', etc. Otherwise, runs the code directly.
    
    Args:
        session: The SessionContext object.
        data_ids: List of IDs of cached DataFrames to load (e.g., [id1, id2]). If None/empty, no DataFrames are loaded.
        computation: The Python code string to execute.
        comment: Optional comment about the computation.
        
    Returns:
        Dictionary with success status, result summary (stdout/stderr), and error message.
    """
    print(f"run_computation: Received request with data_ids='{data_ids}', computation='{computation}'")
    
    # Define a restricted global scope with allowed modules
    allowed_globals = {
        '__builtins__': __builtins__, # Allow standard built-ins
        'pd': pd,              # Allow pandas
        'np': np,              # Allow numpy
        'math': math,            # Allow math
        'datetime': datetime,      # Allow datetime
        # Add other safe modules as needed
        # df variables will be added dynamically below
    }

    # Load DataFrames if data_ids are provided
    if data_ids: 
        print(f"run_computation: Attempting to load DataFrames for data_ids {data_ids} from session '{session.session_id}'")
        loaded_dfs = {}
        for i, data_id in enumerate(data_ids):
            df_var_name = f"df{i+1}" # Create names df1, df2, ...
            df = session.get_from_cache(data_id) 
            
            if df is None:
                error_msg = f"Data ID '{data_id}' (index {i}, requested as {df_var_name}) not found or failed to load in session '{session.session_id}'. Cache data first."
                print(f"run_computation: Failed loading: {error_msg}")
                return {
                    "success": False,
                    "error": error_msg,
                    "result_summary": None
                }
            else:
                print(f"run_computation: Loaded DataFrame for '{data_id}' into '{df_var_name}' with shape {df.shape}")
                # Make the loaded DataFrame available in the execution scope
                allowed_globals[df_var_name] = df
                loaded_dfs[df_var_name] = df # Keep track for potential future use/inspection
        
        if not loaded_dfs: # Should not happen if loop ran unless data_ids was empty but not None
             print("run_computation: data_ids provided but no DataFrames were loaded successfully.")
             # This case might indicate an issue, but proceed to execution for now?
             # Or return error? Let's return error for clarity.
             return {
                    "success": False,
                    "error": "data_ids provided, but failed to load any DataFrames.",
                    "result_summary": None
                }   
    else:
        print("run_computation: No data_ids provided, executing general Python code.")

    # Capture stdout and stderr
    stdout_capture = io.StringIO()
    stderr_capture = io.StringIO()
    
    start_time = time.time() # Track execution time
    success = False
    error_message = None
    
    try:
        # Execute the computation string within the restricted scope
        print(f"Executing computation: {computation}")
        with redirect_stdout(stdout_capture), redirect_stderr(stderr_capture):
            exec(computation, allowed_globals)
        success = True
        print("Computation execution finished. Output source: stdout")
    except Exception as e:
        print(f"Computation execution error: {e}")
        error_message = f"Computation execution error: {traceback.format_exc()}"
        success = False
        print("Computation execution finished. Output source: stderr")
    
    end_time = time.time()
    execution_time = end_time - start_time
    print(f"Computation execution time: {execution_time:.4f} seconds")

    # Get the captured output
    stdout_result = stdout_capture.getvalue()
    stderr_result = stderr_capture.getvalue()
    
    # Combine stdout and stderr for summary, prioritize stderr if error occurred
    result_summary = f"stdout:\n{stdout_result}\nstderr:\n{stderr_result}" if not error_message else f"stderr:\n{stderr_result}\nstdout:\n{stdout_result}"
    
    # Return results
    return {
        "success": success,
        "result_summary": result_summary.strip(),
        "error": error_message,
        "execution_time_seconds": execution_time
    }

def update_topic_file_cache(topic_name: str, file_path: str):
    """
    Update the cache with a topic name to file path mapping.
    This is useful when the dynamic parameter search provides file paths.
    
    Args:
        topic_name: The topic name
        file_path: The file path associated with the topic
    """
    if os.path.exists(file_path):
        _TOPIC_FILE_CACHE[topic_name] = file_path
        return True
    return False

def list_available_topics(csv_dir: str = DEFAULT_CSV_DIR) -> Dict[str, Any]:
    """
    List all available topics in the CSV directory.
    
    Args:
        csv_dir: Directory containing CSV files (default: csv_topics)
        
    Returns:
        Dictionary with topic information
    """
    if not os.path.exists(csv_dir):
        return {
            "success": False,
            "error": f"CSV directory '{csv_dir}' does not exist"
        }
    
    try:
        topic_info = {}
        
        # Get all CSV files
        csv_files = glob.glob(f"{csv_dir}/*.csv")
        
        for file_path in csv_files:
            file_name = os.path.basename(file_path)
            
            # Extract topic name - handle both with and without flight_log prefix
            if file_name.startswith("flight_log_"):
                # Remove 'flight_log_' prefix and strip numbers and extension
                topic_parts = file_name[11:].split('_')
                # Remove the number part at the end (if any)
                if topic_parts[-1].split('.')[0].isdigit():
                    topic_name = '_'.join(topic_parts[:-1])
                else:
                    topic_name = '_'.join(topic_parts).split('.')[0]
            else:
                # Just strip numbers and extension
                topic_parts = file_name.split('_')
                if topic_parts[-1].split('.')[0].isdigit():
                    topic_name = '_'.join(topic_parts[:-1])
                else:
                    topic_name = '_'.join(topic_parts).split('.')[0]
            
            # Store in cache and info dictionary
            _TOPIC_FILE_CACHE[topic_name] = file_path
            
            if topic_name not in topic_info:
                topic_info[topic_name] = {
                    "name": topic_name,
                    "files": []
                }
            
            topic_info[topic_name]["files"].append(file_path)
        
        return {
            "success": True,
            "topics": topic_info,
            "count": len(topic_info)
        }
    except Exception as e:
        return {
            "success": False,
            "error": f"Error listing topics: {str(e)}"
        }

# Simplified function names for easier access
def get_fields(topic_name: str, csv_dir: str = DEFAULT_CSV_DIR, file_path: str = None) -> Dict[str, Any]:
    """Alias for get_topic_fields for easier access."""
    return get_topic_fields(topic_name, csv_dir, file_path)

def get_topic_data(topic_name: str, fields: Optional[List[str]] = None, 
                  filters: Optional[Dict[str, Any]] = None, 
                  csv_dir: str = DEFAULT_CSV_DIR,
                  file_path: str = None) -> Dict[str, Any]:
    """Alias for get_data for easier access."""
    return get_data(topic_name, fields, filters, csv_dir, file_path)

def compute(data: Union[Dict[str, Any], pd.DataFrame], 
           computation: str) -> Dict[str, Any]:
    """Alias for run_computation for easier access."""
    return run_computation(data, computation)

# Helper function to extract file paths from dynamic parameter search results
def extract_file_paths_from_dp_results(dp_results: List[Dict[str, Any]]) -> Dict[str, str]:
    """
    Extract file paths from dynamic parameter search results and update the cache.
    
    Args:
        dp_results: Results from dynamic_param_search tool
        
    Returns:
        Dictionary mapping topic names to file paths
    """
    paths = {}
    for result in dp_results:
        if 'name' in result and 'file_path' in result and result['file_path']:
            topic_name = result['name']
            file_path = result['file_path']
            if update_topic_file_cache(topic_name, file_path):
                paths[topic_name] = file_path
    
    return paths

# --- NEW Plotting Tool Function ---
def run_plotting(session: 'SessionContext', 
                 data_ids: Optional[List[str]], 
                 code: str,
                 comment: Optional[str] = None) -> Dict[str, Any]:
    """
    Executes Python code to generate a plot, save it temporarily, read it, 
    and return the Base64 encoded content.
    
    Args:
        session: The SessionContext object.
        data_ids: List of IDs of cached DataFrames (loaded as df1, df2, ...).
        code: Python code to execute. MUST generate a plot and call plt.savefig(save_path).
        comment: Optional comment.
        
    Returns:
        Dictionary with success status, plot filename, base64 content, error message, and execution summary.
    """
    print(f"run_plotting: Received request with data_ids='{data_ids}', comment='{comment}'")

    if not _MATPLOTLIB_AVAILABLE:
        return {"success": False, "error": "matplotlib is not installed on the server. Plotting is disabled."}

    plots_dir = session.session_dir / "plots"
    try:
        plots_dir.mkdir(parents=True, exist_ok=True)
    except OSError as e:
        return {"success": False, "error": f"Failed to create plots directory '{plots_dir}': {e}"}

    plot_filename = f"plot_{uuid.uuid4()}.png"
    save_path = plots_dir / plot_filename

    allowed_globals = {
        '__builtins__': __builtins__,
        'pd': pd,
        'np': np,
        'math': math,
        'datetime': datetime,
        'plt': plt if _MATPLOTLIB_AVAILABLE else None,
        'sns': sns if _SEABORN_AVAILABLE else None,
        'save_path': str(save_path),
        # df variables added below
    }

    if data_ids: 
        print(f"run_plotting: Loading DataFrames for {data_ids}")
        for i, data_id in enumerate(data_ids):
            df_var_name = f"df{i+1}"
            df = session.get_from_cache(data_id) 
            if df is None:
                error_msg = f"Data ID '{data_id}' (index {i}, as {df_var_name}) not found or failed to load."
                print(f"run_plotting: Failed loading: {error_msg}")
                return {"success": False, "error": error_msg}
            else:
                print(f"run_plotting: Loaded '{df_var_name}' (shape {df.shape})")
                allowed_globals[df_var_name] = df
    else:
        print("run_plotting: No data_ids provided, executing general plotting code.")

    stdout_capture = io.StringIO()
    stderr_capture = io.StringIO()
    start_time = time.time()
    success = False
    error_message = None
    plot_content_base64 = None
    
    plt.close('all') 

    try:
        print(f"Executing plotting code... Target path: {save_path}")
        with redirect_stdout(stdout_capture), redirect_stderr(stderr_capture):
            exec(code, allowed_globals)
        
        if not save_path.is_file():
             error_message = "Plot execution finished, but the expected plot file was not saved. Did you forget to call `plt.savefig(save_path)` in your code?"
             print(f"run_plotting: Error - {error_message}")
             success = False
        else:
             print(f"Plot successfully saved locally to: {save_path}. Reading and encoding...")
             try:
                 with open(save_path, "rb") as image_file:
                     plot_content_base64 = base64.b64encode(image_file.read()).decode('utf-8')
                 print(f"Successfully encoded plot content.")
                 success = True
             except Exception as read_error:
                 error_message = f"Plot file created, but failed to read or encode it: {read_error}"
                 print(f"run_plotting: Error - {error_message}")
                 success = False

    except Exception as e:
        print(f"Plotting code execution error: {e}")
        error_message = f"Plotting code execution error: {traceback.format_exc()}"
        success = False
        if save_path.exists():
             try: save_path.unlink()
             except OSError: pass

    finally:
         # Ensure matplotlib figures are closed after execution to free memory
         plt.close('all') 
         # --- REMOVED Cleanup of local file ---
         # # Optionally delete local file after encoding/failure? 
         # # Decide if we keep the local plot file for debugging or clean it up.
         # # Let's clean it up for now to avoid filling temp storage.
         # if save_path.exists():
         #      try: 
         #          save_path.unlink()
         #          print(f"Cleaned up local plot file: {save_path}")
         #      except OSError as del_err:
         #          print(f"Warning: Failed to clean up local plot file {save_path}: {del_err}")
         # --- End REMOVED Cleanup --- 
         
    end_time = time.time()
    execution_time = end_time - start_time
    print(f"Plotting execution time: {execution_time:.4f} seconds")
    stdout_result = stdout_capture.getvalue()
    stderr_result = stderr_capture.getvalue()
    result_summary = f"stdout:\n{stdout_result}\nstderr:\n{stderr_result}".strip()
    
    return {
        "success": success,
        "plot_filename": plot_filename if success else None,
        "plot_content_base64": plot_content_base64 if success else None,
        "result_summary": result_summary,
        "error": error_message,
        "execution_time_seconds": execution_time
    }

if __name__ == "__main__":
    # Example usage
    print("Testing General-Purpose Flight Log Analysis Tools")
    print("=" * 50)
    
    # List topics in the csv_topics directory
    csv_dir = "csv_topics"
    if os.path.exists(csv_dir):
        # List all available topics
        topics_result = list_available_topics(csv_dir)
        if topics_result["success"]:
            print(f"Found {topics_result['count']} topics in {csv_dir}:")
            for topic_name, info in topics_result["topics"].items():
                print(f" - {topic_name} ({len(info['files'])} files)")
            
            # Test with a real topic if available
            if topics_result["count"] > 0:
                test_topic = next(iter(topics_result["topics"].keys()))
                test_file = topics_result["topics"][test_topic]["files"][0]
                print(f"\nTesting with topic: {test_topic}")
                print(f"File path: {test_file}")
                
                # Get fields directly using file path
                fields_result = get_topic_fields(test_topic, file_path=test_file)
                if fields_result["success"]:
                    print(f"Fields in {test_topic}: {fields_result['fields']}")
                    
                    # Get data
                    if fields_result["fields"]:
                        test_fields = fields_result["fields"][:3]  # First 3 fields
                        print(f"\nGetting data for fields: {test_fields}")
                        data_result = get_data(test_topic, test_fields, file_path=test_file)
                        
                        if data_result["success"]:
                            df = data_result["data"]
                            print(f"Retrieved {len(df)} rows")
                            
                            if not df.empty:
                                # Run a computation
                                print("\nRunning a sample computation...")
                                comp_result = run_computation(df, "df.describe()")
                                if comp_result["success"]:
                                    print("Computation result:")
                                    print(comp_result["result"])
                                else:
                                    print(f"Computation error: {comp_result['error']}")
                        else:
                            print(f"Data error: {data_result['error']}")
                else:
                    print(f"Fields error: {fields_result['error']}")
        else:
            print(f"Topics error: {topics_result['error']}")
    else:
        print(f"Directory {csv_dir} not found. Create it by converting ULog files to CSV.") 