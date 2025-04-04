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

# Add color constants for console output
try:
    from constants import RED, YELLOW, RESET, GREEN
except ImportError:
    # Fallback if constants can't be imported
    RED = "\033[91m"
    YELLOW = "\033[93m"
    GREEN = "\033[92m"
    RESET = "\033[0m"

# Default directory for CSV topic files
DEFAULT_CSV_DIR = "csv_topics"

# Cache to store topic to file path mappings
_TOPIC_FILE_CACHE = {}

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

def run_computation(data: Union[Dict[str, Any], pd.DataFrame, str] = None, 
                   computation: str = None,
                   comment: str = None,
                   data_id: str = None,
                   data_cache: Dict[str, Any] = None) -> Dict[str, Any]:
    """
    Run arbitrary computation on data.
    
    Args:
        data: Either the complete data object returned from get_data()
              or custom data that can be converted to a DataFrame
        computation: A valid Python expression operating on a DataFrame 'df'
        comment: A brief explanation of what this computation is doing (optional)
        data_id: ID of the data to retrieve from cache (alternative to data parameter)
        data_cache: External data cache to use (passed from calling module)
        
    Returns:
        Dictionary with computation results or error
    """
    try:
        # Check parameters
        if not computation:
            return {
                "success": False,
                "error": "Computation expression is required but was not provided."
            }
            
        df = None
        data_source = "direct"
        
        # Prioritize data_id if provided
        if data_id:
            # The data should already be loaded by the calling module
            # and passed in the data parameter
            data_source = "data_id"
            print(f"Using data with ID: {data_id} for computation")
            
            # Data should already be loaded and passed in the data parameter
            if data is not None:
                if isinstance(data, dict) and '_dataframe' in data:
                    df = data['_dataframe']
                elif isinstance(data, pd.DataFrame):
                    df = data
                else:
                    return {
                        "success": False,
                        "error": f"Data format for ID {data_id} is not supported. Expected DataFrame or dict with _dataframe."
                    }
            else:
                return {
                    "success": False,
                    "error": f"No data provided for ID {data_id}. Data must be loaded before calling run_computation."
                }
        # Check for function reference pattern - this is a common pattern where models
        # attempt to reference a previous function call rather than using the data directly
        elif isinstance(data, dict) and 'function_name' in data and 'args' in data:
            data_source = "function_reference"
            # We've received a function reference - execute the referenced function to get data
            try:
                print(f"Detected function reference pattern. Executing referenced function: {data['function_name']}")
                
                if data['function_name'] == 'topic_data' or data['function_name'] == 'get_data':
                    # Extract arguments from the reference
                    topic_name = None
                    fields = None
                    
                    # Process args which might be in different formats
                    if isinstance(data['args'], list):
                        for arg in data['args']:
                            if isinstance(arg, dict):
                                if 'topic_name' in arg:
                                    topic_name = arg['topic_name']
                                if 'fields' in arg:
                                    fields = arg['fields']
                    elif isinstance(data['args'], dict):
                        topic_name = data['args'].get('topic_name')
                        fields = data['args'].get('fields')
                    
                    if topic_name:
                        # Execute topic_data/get_data to get the actual data
                        result = get_data(topic_name, fields)
                        if result.get('success', False) and '_dataframe' in result:
                            df = result['_dataframe']
                            print(f"Successfully retrieved data for topic '{topic_name}' with fields {fields}")
                        else:
                            return {
                                "success": False,
                                "error": f"Error retrieving data for topic '{topic_name}': {result.get('error', 'Unknown error')}"
                            }
                    else:
                        return {
                            "success": False,
                            "error": "Missing topic_name in function reference"
                        }
                else:
                    return {
                        "success": False,
                        "error": f"Unsupported function reference: {data['function_name']}"
                    }
            except Exception as e:
                return {
                    "success": False,
                    "error": f"Error executing referenced function: {str(e)}"
                }
        # Handle direct data input
        elif data is not None:
            data_source = "direct"
            # Case 1: Direct data object from get_data() - extract the _dataframe field
            if isinstance(data, dict) and '_dataframe' in data:
                df = data['_dataframe']
            
            # Case 2: Custom data from the agent (various formats)
            elif isinstance(data, dict):
                # Handle various dictionary formats that can be converted to DataFrame
                if 'data' in data and isinstance(data['data'], list):
                    if 'columns' in data and isinstance(data['columns'], list):
                        df = pd.DataFrame(data['data'], columns=data['columns'])
                    else:
                        df = pd.DataFrame(data['data'])
                else:
                    # Try to create DataFrame directly from the dictionary
                    df = pd.DataFrame(data)
            
            # Case 3: Already a DataFrame
            elif isinstance(data, pd.DataFrame):
                df = data
            else:
                return {
                    "success": False,
                    "error": f"Provided data is in an unsupported format: {type(data)}"
                }
        else:
            return {
                "success": False,
                "error": "No data or data_id provided. You must provide either data or data_id."
            }
            
        # Check if we have valid data to work with
        if df is None:
            return {
                "success": False,
                "error": "Could not extract a valid DataFrame from the provided data."
            }
        
        # Normalize the DataFrame to ensure it's properly structured
        if df.empty:
            return {
                "success": False,
                "error": "The DataFrame is empty. There is no data to compute on."
            }
            
        # Handle single-column DataFrame in a way that prevents 'must pass index' errors
        if len(df.columns) == 1:
            # If there's only one column and computation refers to it by name,
            # convert to Series access pattern to avoid index errors
            col_name = df.columns[0]
            
            # Check if computation attempts to access column by name
            if f'df["{col_name}"]' in computation or f"df['{col_name}']" in computation:
                # Create a special computation environment with the Series
                series = df[col_name]
                namespace = {"series": series, "pd": pd, "np": np}
                
                # Modify computation to use series directly
                modified_comp = computation.replace(f'df["{col_name}"]', 'series')
                modified_comp = modified_comp.replace(f"df['{col_name}']", 'series')
                
                try:
                    result = eval(modified_comp, namespace)
                    return {
                        "success": True,
                        "computation": "Computation performed successfully",
                        "result": result,
                        "data_source": data_source,
                        "note": "Used Series access for single-column DataFrame"
                    }
                except Exception:
                    # If modified computation fails, fall back to standard approach
                    pass
        
        # Reset index to ensure proper structure regardless of original format
        df = df.reset_index(drop=True)
        
        # Verify column names
        if any(not isinstance(col, str) for col in df.columns):
            # Convert all column names to strings for consistent access
            df.columns = [str(col) for col in df.columns]
            
        # Execute the computation in a safe environment with the normalized DataFrame
        namespace = {"df": df, "pd": pd, "np": np}
        
        try:
            # Detect if the computation is multi-line (needs exec instead of eval)
            is_multiline = '\n' in computation.strip()
            
            # Special case for quaternion to Euler conversion (common in flight logs)
            if "quaternion_to_euler" in computation or ("quaternion" in computation and "euler" in computation):
                print(f"{GREEN}Detected quaternion to Euler conversion. Using specialized handling...{RESET}")
                # Add quaternion_to_euler function to namespace
                namespace["quaternion_to_euler"] = lambda q: (
                    np.degrees(np.arctan2(2.0 * (q[0] * q[1] + q[2] * q[3]), 1.0 - 2.0 * (q[1]**2 + q[2]**2))),  # roll
                    np.degrees(np.arcsin(2.0 * (q[0] * q[2] - q[3] * q[1]))),  # pitch
                    np.degrees(np.arctan2(2.0 * (q[0] * q[3] + q[1] * q[2]), 1.0 - 2.0 * (q[2]**2 + q[3]**2)))   # yaw
                )
                
                # If this is a multi-line computation with quaternion conversion, make sure it works
                if is_multiline:
                    # Ensure the computation has access to the quaternion_to_euler function
                    computation = "from numpy import degrees, arctan2, arcsin\n" + computation
            
            if is_multiline:
                # For multi-line code, we need to use exec and capture output
                # Create a StringIO object to capture print statements
                import sys
                from io import StringIO
                
                # Create a variable to store the result
                namespace['_result'] = None
                old_stdout = sys.stdout
                
                try:
                    # Redirect stdout to capture prints
                    sys.stdout = mystdout = StringIO()
                    
                    # Add the code to store the last expression's result in _result
                    # We'll wrap the user's code in a try-except block
                    wrapped_code = "try:\n"
                    
                    # Indent each line of the original code
                    for line in computation.split('\n'):
                        wrapped_code += "    " + line + "\n"
                    
                    # If needed, capture result
                    wrapped_code += "except Exception as e:\n"
                    wrapped_code += "    print(f'Error in computation: {str(e)}')\n"
                    
                    # Execute the wrapped code
                    exec(wrapped_code, namespace)
                    
                    # Get printed output
                    printed_output = mystdout.getvalue()
                    
                    # If there's explicit output from print statements, use that
                    if printed_output.strip():
                        result = printed_output.strip()
                    # Otherwise try to get the stored _result
                    elif '_result' in namespace and namespace['_result'] is not None:
                        result = namespace['_result']
                    else:
                        # If no explicit output, the "result" is that the code executed successfully
                        result = "Computation executed successfully (no explicit output)"
                        
                    return {
                        "success": True,
                        "computation": "Computation performed successfully",
                        "result": result,
                        "data_source": data_source,
                        "execution_method": "exec (multi-line code)"
                    }
                finally:
                    # Restore stdout
                    sys.stdout = old_stdout
            else:
                # For simple one-line expressions, use eval
                result = eval(computation, namespace)
                
                # Prepare the result
                response = {
                    "success": True,
                    "computation": "Computation performed successfully",
                    "result": result,
                    "data_source": data_source,
                    "execution_method": "eval (single-line expression)"
                }
                
                # Add the comment if provided
                if comment:
                    response["comment"] = comment
                    
                return response
                
        except Exception as e:
            return {
                "success": False,
                "error": f"Error executing computation '{computation}': {str(e)}"
            }
            
    except Exception as e:
        return {
            "success": False,
            "error": f"Error processing data for computation: {str(e)}"
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