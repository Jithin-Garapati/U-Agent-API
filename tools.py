import json
from typing import List, Dict, Any, Optional
import uuid
import pandas as pd
import os
import time
import shutil

# Import existing tool implementations
from tools.extract_dynamic_param_tool import DP
from tools.extract_static_param_tool import SP, get_param_by_name
from tools.general_purpose_tools import get_topic_fields, get_data, run_computation
from constants import RED, YELLOW, RESET, GREEN

# Create cache directory for persistent storage
CACHE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data_cache')
if not os.path.exists(CACHE_DIR):
    os.makedirs(CACHE_DIR, exist_ok=True)

# Create a global data cache to store DataFrames (memory index of what's on disk)
DATA_CACHE = {}

# Cache management settings
CACHE_MAX_AGE_SECONDS = 3600  # 1 hour expiration

def _clean_expired_cache():
    """Remove cache files older than CACHE_MAX_AGE_SECONDS"""
    current_time = time.time()
    for filename in os.listdir(CACHE_DIR):
        file_path = os.path.join(CACHE_DIR, filename)
        if os.path.isfile(file_path):
            # Get file creation/modification time
            file_time = os.path.getmtime(file_path)
            if current_time - file_time > CACHE_MAX_AGE_SECONDS:
                try:
                    os.remove(file_path)
                    print(f"Removed expired cache file: {filename}")
                    # Also remove from memory cache if it's there
                    data_id = os.path.splitext(filename)[0]
                    if data_id in DATA_CACHE:
                        del DATA_CACHE[data_id]
                except Exception as e:
                    print(f"Error removing cache file {filename}: {e}")

def _get_cache_path(data_id: str) -> str:
    """Get the file path for a cache entry"""
    return os.path.join(CACHE_DIR, f"{data_id}.pkl")

def _save_to_cache(data_id: str, data: Any) -> bool:
    """Save data to cache file and memory index"""
    try:
        # Clean expired cache files occasionally
        if uuid.uuid4().int % 10 == 0:  # ~10% chance to trigger cleanup
            _clean_expired_cache()
            
        # Save to memory index
        DATA_CACHE[data_id] = {'timestamp': time.time(), 'in_memory': True}
        
        # Save DataFrame to disk
        cache_path = _get_cache_path(data_id)
        
        # Handle different data types
        if isinstance(data, dict) and '_dataframe' in data:
            # If it's a dict with a DataFrame, save the DataFrame
            df = data['_dataframe']
            df.to_pickle(cache_path)
            print(f"{GREEN}Saved DataFrame to cache: {data_id}{RESET}")
            return True
        elif isinstance(data, pd.DataFrame):
            # If it's a DataFrame, save it directly
            data.to_pickle(cache_path)
            print(f"{GREEN}Saved DataFrame to cache: {data_id}{RESET}")
            return True
        else:
            print(f"{YELLOW}Data is not a DataFrame, cannot cache: {data_id}{RESET}")
            return False
    except Exception as e:
        print(f"{RED}Error saving to cache: {e}{RESET}")
        return False

def _load_from_cache(data_id: str) -> Any:
    """Load data from cache file"""
    cache_path = _get_cache_path(data_id)
    
    if not os.path.exists(cache_path):
        print(f"{YELLOW}Cache file not found: {cache_path}{RESET}")
        # Also remove from memory index if it's there
        if data_id in DATA_CACHE:
            del DATA_CACHE[data_id]
        return None
    
    try:
        # Load DataFrame from file
        df = pd.read_pickle(cache_path)
        
        # Update access timestamp
        os.utime(cache_path, None)
        
        print(f"{GREEN}Loaded DataFrame from cache: {data_id}{RESET}")
        return df
    except Exception as e:
        print(f"{RED}Error loading from cache: {e}{RESET}")
        # Remove corrupt cache file
        try:
            os.remove(cache_path)
            print(f"Removed corrupt cache file: {cache_path}")
        except:
            pass
        
        # Also remove from memory index
        if data_id in DATA_CACHE:
            del DATA_CACHE[data_id]
            
        return None

# Function implementations
def dynamic_param_search(query: str) -> List[Dict[str, Any]]:
    """Search for dynamic parameters from flight logs."""
    print(f"Searching for dynamic parameters: {query}")
    
    # Get the full results from the DP tool
    full_results = DP(query)
    
    # Simplify the results to reduce token usage
    simplified_results = []
    
    # Process first 5 parameters - include description and all field names
    for result in full_results[:5]:
        # Get all field names from the full fields dictionary
        field_names = list(result.get('fields', {}).keys())
        
        # Create a simplified result with name, description and all field names
        simplified_result = {
            'name': result.get('name', 'Unknown'),
            'description': result.get('description', 'No description'),
            'fields': field_names,  # Include ALL field names
            'total_field_count': len(field_names)
        }
        
        simplified_results.append(simplified_result)
    
    # Process next 5 parameters - include only the parameter name
    for result in full_results[5:10]:
        # Create a simplified result with only the name
        simplified_result = {
            'name': result.get('name', 'Unknown'),
        }
        
        simplified_results.append(simplified_result)
    
    return simplified_results

def static_param_search(query: str) -> List[Dict[str, Any]]:
    """Search for static parameters from configuration files."""
    print(f"Searching for static parameters: {query}")
    return SP(query)

def static_param_lookup(param_name: str) -> Dict[str, Any]:

    """Get the exact row of a specific static parameter by name."""
    result = get_param_by_name(param_name)
    if result:
        return result
    else:
        return {"error": f"Parameter {param_name} not found"}

def topic_fields(topic_name: str) -> Dict[str, Any]:
    """List all fields in a specific topic."""
    print(f"Getting fields for topic: {topic_name}")
    return get_topic_fields(topic_name)

def topic_data(topic_name: str, fields: Optional[List[str]] = None, file_path: Optional[str] = None) -> Dict[str, Any]:
    """Retrieve data from a topic with minimal processing."""
    print(f"Getting data for topic: {topic_name}, fields: {fields}")
    
    # Get the data
    result = get_data(topic_name, fields, file_path=file_path)
    
    # Check if the data was successfully retrieved
    if not result.get("success", False):
        print(f"{RED}Error retrieving data: {result.get('error', 'Unknown error')}{RESET}")
        return result
    
    # Check if we have any data
    rows = result.get("rows", 0)
    if rows == 0:
        message = f"WARNING: No data found for topic '{topic_name}'"
        if fields:
            message += f" with fields {fields}"
        
        print(f"{YELLOW}{message}{RESET}")
        return {
            "success": True,
            "warning": message,
            "rows": 0,
            "topic": topic_name,
            "columns": fields or []
        }
    
    # Get column information
    columns = result.get("columns", [])
    
    # Generate a unique ID for this data
    data_id = str(uuid.uuid4())
    
    # Save the full result to the persistent cache
    cache_success = _save_to_cache(data_id, result)
    
    # Remove _dataframe from the result to reduce token size
    if "_dataframe" in result:
        # Keep the dataframe in memory index but remove from response
        df = result.pop("_dataframe")
        
        # Create simplified statistics
        statistics = {}
        preview_data = []
        
        # Process preview data (first few rows)
        preview_rows = min(3, len(df))
        for i in range(preview_rows):
            row_data = []
            for col in columns:
                if i < len(df) and col in df:
                    # Convert numpy values to plain Python types
                    val = df[col].iloc[i]
                    if hasattr(val, 'item'):
                        try:
                            val = val.item()
                        except:
                            val = str(val)
                    row_data.append(val)
                else:
                    row_data.append(None)
            preview_data.append(row_data)
            
        # Calculate statistics for numeric columns
        for col in columns:
            if col in df and pd.api.types.is_numeric_dtype(df[col]):
                try:
                    statistics[col] = {
                        "min": float(df[col].min()) if not df[col].isna().all() else None,
                        "max": float(df[col].max()) if not df[col].isna().all() else None,
                        "mean": float(df[col].mean()) if not df[col].isna().all() else None,
                        "std": float(df[col].std()) if not df[col].isna().all() else None,
                    }
                except Exception as e:
                    print(f"Error calculating statistics for column {col}: {e}")
                    statistics[col] = {"error": str(e)}
    
    # Prepare the simplified response with data_id and cache info
    simplified_response = {
        "success": True,
        "topic": topic_name,
        "rows": rows,
        "columns": columns,
        "preview": preview_data if 'preview_data' in locals() else [],
        "column_names": columns,
        "data_id": data_id,  # Add the data_id to the response
        "is_cached": cache_success,   # Mark as cached only if successful
        "message": f"Retrieved {rows} rows, {len(columns)} columns. Use data_id with computation tool."
    }
    
    # Add statistics if available
    if 'statistics' in locals():
        simplified_response["statistics"] = statistics
    
    return simplified_response

def computation(data_id: Optional[str] = None, computation_expr: Optional[str] = None, comment: Optional[str] = None, data: Any = None) -> Dict[str, Any]:
    """Run a computation on data using the provided expression."""
    print(f"Running computation: {computation_expr}")
    if comment:
        print(f"Comment: {comment}")

    # Input validation
    if not computation_expr:
        return {
            "success": False,
            "error": "ERROR: No computation expression provided. Please specify a computation_expr parameter."
        }

    # Check for common data_id errors
    common_invalid_data_ids = ["topic_data_result", "data", "data_id", "previous_data", "result", 
                              "data_from_topic_data", "data_from_previous_call", "topic_data"]
    
    if data_id in common_invalid_data_ids:
        return {
            "success": False,
            "error": f"ERROR: Invalid data_id '{data_id}'. You must use the exact UUID string returned by topic_data, not a hardcoded value. Look for the 'data_id' field in the topic_data response, which contains a string like '2b1ac157-f27c-40f1-8ecc-e44a6fa6c67a'."
        }
    
    # Check if data_id looks like a UUID (basic validation)
    if data_id and not (len(data_id) > 8 and "-" in data_id):
        return {
            "success": False, 
            "error": f"ERROR: The data_id '{data_id}' does not appear to be a valid UUID. You must use the exact data_id string from the topic_data response, which looks like: '2b1ac157-f27c-40f1-8ecc-e44a6fa6c67a'."
        }

    # Prioritize data_id if provided (this is the preferred approach)
    if data_id:
        print(f"Using data_id: {data_id}")
        
        # First try to load from file cache
        cached_data = _load_from_cache(data_id)
        
        if cached_data is not None:
            print(f"{GREEN}Found data in cache with ID: {data_id}{RESET}")
            
            # Create a dict to match the expected structure if it's just a DataFrame
            if isinstance(cached_data, pd.DataFrame):
                data_obj = {'_dataframe': cached_data}
            else:
                data_obj = cached_data
                
            # Pass both the data and data_id to run_computation
            return run_computation(data_obj, computation_expr, comment, data_id, DATA_CACHE)
        else:
            return {
                "success": False,
                "error": f"ERROR: Data with ID {data_id} not found in cache. The data may have expired. Please retrieve the data again using topic_data."
            }
    
    # Fallback to direct data if provided (deprecated approach)
    if data is None:
        return {
            "success": False,
            "error": "ERROR: No data or data_id provided for computation. Retrieve data first using topic_data."
        }
    
    # Check if this is cached data (has a data_id in the object)
    if isinstance(data, dict) and "data_id" in data and data.get("is_cached", False):
        existing_data_id = data["data_id"]
        
        # Try to load from file cache
        cached_data = _load_from_cache(existing_data_id)
        
        if cached_data is not None:
            print(f"{GREEN}Using cached data with ID: {existing_data_id}{RESET}")
            
            # Create a dict to match the expected structure if it's just a DataFrame
            if isinstance(cached_data, pd.DataFrame):
                data_obj = {'_dataframe': cached_data}
            else:
                data_obj = cached_data
                
            # Pass data to run_computation
            return run_computation(data_obj, computation_expr, comment, existing_data_id, DATA_CACHE)
        else:
            return {
                "success": False,
                "error": f"ERROR: Data with ID {existing_data_id} not found in cache. The data may have expired. Please retrieve the data again using topic_data."
            }
    
    # If not cached data, validate it's from a real source
    if isinstance(data, dict):
        # Check if this is likely fabricated data (simple list of values)
        is_likely_hallucinated = False
        if not any(key in data for key in ['_dataframe', 'topic', 'rows', 'columns', 'is_cached']):
            # Simple list hallucination check (manually created arrays)
            if any(isinstance(v, list) and len(v) > 0 and all(isinstance(x, (int, float)) for x in v) for v in data.values()):
                is_likely_hallucinated = True
                
        if is_likely_hallucinated:
            print("WARNING: This appears to be manually constructed data, not data retrieved from topic_data!")
            return {
                "success": False,
                "error": "ERROR: Data appears to be manually constructed, not retrieved from topic_data tool. Please use topic_data to get real flight data first."
            }
            
    # Pass the direct data to run_computation
    return run_computation(data, computation_expr, comment, None, DATA_CACHE)

def submit_final_answer(answer: str, explanation: Optional[str] = None) -> Dict[str, Any]:
    """Submit a final answer to the user's query.
    
    Args:
        answer: The complete, final answer to provide to the user
        explanation: Optional explanation of how this answer was derived
        
    Returns:
        A dictionary containing the final answer and metadata
    """
    print("Submitting final answer...")
    
    # ANTI-HALLUCINATION SAFEGUARD
    # Check global data tracking to ensure we're not submitting hallucinated values
    from dataclasses import dataclass
    import inspect
    import sys
    
    # Get the call stack to analyze recent tool calls
    frame = inspect.currentframe()
    valid_submission = False
    
    # Look through call history to check if we had valid data
    stack = inspect.stack()
    
    # Find the most recent data retrieval call results
    last_data_result = None
    for frame_info in stack[1:10]:  # Check up to 10 frames back
        if 'messages' in frame_info.frame.f_locals:
            # Found the agent's message history
            messages = frame_info.frame.f_locals['messages']
            # Check the last few messages for tool results
            for msg in reversed(messages[-5:]):
                if msg.get('role') == 'tool' and 'content' in msg:
                    try:
                        import json
                        content = json.loads(msg['content'])
                        # Check for empty results or errors in topic_data or other tool responses
                        if 'error' in content or 'warning' in content:
                            print(f"{RED}WARNING: Last tool call had an error or warning. Blocking potentially hallucinated answer.{RESET}")
                            return {
                                "final_answer": "ERROR: Cannot provide a final answer due to insufficient or invalid data.",
                                "is_final": False,
                                "hallucination_detected": True,
                                "error": "The last tool call returned an error or warning. You must obtain valid data before submitting a final answer."
                            }
                        
                        # For topic_data specifically, check rows
                        if 'rows' in content and content['rows'] == 0:
                            print(f"{RED}WARNING: Last topic_data call returned 0 rows. Blocking potentially hallucinated answer.{RESET}")
                            return {
                                "final_answer": "ERROR: Cannot provide a final answer because the last data retrieval returned 0 rows.",
                                "is_final": False,
                                "hallucination_detected": True,
                                "error": "No data was retrieved in the last topic_data call. You must obtain valid data before submitting a final answer."
                            }
                            
                        # If we got here, there was at least some valid data
                        valid_submission = True
                        break
                    except Exception as e:
                        # If we can't parse the content, continue checking
                        continue
            # If we already checked the messages, no need to keep looking
            if valid_submission:
                break
    
    # Only allow submission if we found evidence of valid data
    if not valid_submission:
        print(f"{RED}WARNING: No evidence of valid data retrieval found. Blocking potentially hallucinated answer.{RESET}")
        return {
            "final_answer": "ERROR: Cannot provide a final answer due to insufficient or invalid data.",
            "is_final": False,
            "hallucination_detected": True,
            "error": "You must use tools to retrieve valid data before submitting a final answer."
        }
    
    result = {
        "final_answer": answer,
        "is_final": True
    }
    
    if explanation:
        result["explanation"] = explanation
    
    return result

# Map function names to their implementations
AVAILABLE_FUNCTIONS = {
    "dynamic_param_search": dynamic_param_search,
    "static_param_search": static_param_search,
    "static_param_lookup": static_param_lookup,
    "topic_fields": topic_fields,
    "topic_data": topic_data,
    "computation": computation,
    "submit_final_answer": submit_final_answer
}

# Make sure the cache is cleaned up when the module exits
import atexit

def _cleanup_cache():
    """Clean up cache when the module exits"""
    try:
        # Run full cleanup on exit
        _clean_expired_cache()
    except:
        pass

# Register cleanup function
atexit.register(_cleanup_cache) 