# The_agent_api/main.py
import sys
import os
import shutil
import asyncio
from pathlib import Path
from fastapi import FastAPI, File, UploadFile, HTTPException, Path as FastApiPath, Depends, Query
from fastapi.responses import JSONResponse

# --- Adjust Python Path to Find Modules in Parent Directory ---
# Get the directory containing this file (API_ROOT)
current_dir = Path(__file__).parent.resolve()
# Get the parent directory (WORKSPACE_ROOT)
parent_dir = current_dir.parent
# Add the parent directory to sys.path
if str(parent_dir) not in sys.path:
    sys.path.insert(0, str(parent_dir))

# --- Project Imports ---
try:
    # API specific imports (using absolute imports from package root)
    from The_agent_api.api import models
    from The_agent_api.api.session import InMemorySessionManager, SessionContext
    from The_agent_api import config # Changed from relative

    # Imports from parent directory (original workspace)
    # Ensure these modules/functions exist at the expected location
    from ulog_utils import convert_ulog_to_csv # Function from parent/ulog_utils.py
    # Import tool classes (adjust paths/names as needed based on actual files)
    from tools.extract_dynamic_param_tool import DynamicParameterTool
    from tools.extract_static_param_tool import StaticParameterTool
    from tools.general_purpose_tools import get_topic_fields, get_data, run_computation
except ImportError as e:
    print(f"Error importing project modules: {e}")
    print("Ensure the script is run from the correct directory or PYTHONPATH is set.")
    # Optionally re-raise or exit if imports are critical
    raise

# --- FastAPI App Initialization ---
app = FastAPI(
    title="ULog Analysis Agent API",
    description="API backend for analyzing ULog files.",
    version="0.1.0"
)

# --- Global Instances ---
# Initialize session manager using the base temp directory from config
session_manager = InMemorySessionManager(base_temp_dir=config.TEMP_BASE_DIR)

# Optional: Load Sentence Transformer model globally once for efficiency
# try:
#     from sentence_transformers import SentenceTransformer
#     shared_sentence_model = SentenceTransformer(config.SENTENCE_MODEL_NAME)
#     print(f"Loaded Sentence Transformer model: {config.SENTENCE_MODEL_NAME}")
# except ImportError:
#     shared_sentence_model = None
#     print("SentenceTransformer not installed or model not found. Tools will load models individually.")

# --- Helper Functions --- 
async def get_valid_session(session_id: str = FastApiPath(..., description="The ID of the session")):
    """Dependency to validate session_id and return session context."""
    session = session_manager.get_session(session_id)
    if not session:
        raise HTTPException(status_code=404, detail=f"Session ID '{session_id}' not found. Please upload a file first using POST /upload/{{{session_id}}}")
    return session

# --- New Formatting Helper ---
def _print_formatted_tool_results(results: list, result_type: str = "matches"):
    """Prints a formatted summary of tool results to the console."""
    try:
        count = len(results)
        print(f"\n--- Server Console: Formatted Tool Result ---")
        print(f"Found {count} {result_type}.")
        if count > 0:
            print("\nTop matches:")
            for item in results:
                name = item.get('name', 'N/A')
                desc = item.get('description', 'No description available.')
                fields_dict = item.get('fields', {})
                score = item.get('score')

                # Extract field names
                field_names = list(fields_dict.keys()) if isinstance(fields_dict, dict) else []
                field_count = len(field_names)
                fields_str = ', '.join(field_names) if field_names else "Not available"

                print(f"\n- {name}")
                print(f"  Description: {desc}")
                print(f"  Fields: {fields_str}")
                print(f"  Total fields: {field_count}")
                if score is not None:
                    print(f"  Relevance Score: {score:.2f}") # Format score
        print(f"--- End Formatted Tool Result ---\n")
    except Exception as e:
        print(f"\n--- Error formatting tool results for printing: {e} ---\n")

# --- New Helper for Conversion and Renaming ---
def _convert_and_rename_ulog_csvs(ulog_filepath_str: str, csv_output_dir_str: str, ulog_original_filename: str) -> bool:
    """
    Converts ULog to CSV and then renames files to remove the ULog filename prefix.

    Args:
        ulog_filepath_str: Path to the saved ULog file.
        csv_output_dir_str: Directory where CSVs should be placed.
        ulog_original_filename: The original filename of the uploaded ULog.

    Returns:
        True if conversion was successful, False otherwise. Renaming errors are logged but don't cause a False return.
    """
    # 1. Convert ULog to CSV
    conversion_success = convert_ulog_to_csv(
        ulog_filepath=ulog_filepath_str,
        output_folder=csv_output_dir_str
    )

    if not conversion_success:
        return False # Conversion failed, nothing to rename

    # 2. Rename files if conversion succeeded
    try:
        ulog_basename = Path(ulog_original_filename).stem
        prefix_to_remove = f"{ulog_basename}_"
        output_dir = Path(csv_output_dir_str)

        print(f"Renaming CSV files in {output_dir} to replace prefix '{prefix_to_remove}' with 'flight_log_'...")
        renamed_count = 0
        for item in os.listdir(output_dir):
            if item.startswith(prefix_to_remove) and item.lower().endswith('.csv'):
                old_path = output_dir / item
                # Remove old prefix, then prepend 'flight_log_'
                new_filename = f"flight_log_{item.replace(prefix_to_remove, '', 1)}"
                new_path = output_dir / new_filename
                if old_path != new_path:
                    try:
                        os.rename(old_path, new_path)
                        renamed_count += 1
                    except OSError as rename_error:
                        print(f"Error renaming file {old_path} to {new_path}: {rename_error}")
        print(f"Finished renaming. Renamed {renamed_count} files.")

    except Exception as rename_exception:
        # Log renaming errors but don't fail the overall operation
        # as the core conversion still succeeded.
        print(f"An error occurred during CSV renaming: {rename_exception}")

    return True # Return True because conversion was successful

# --- API Endpoints ---

@app.post("/upload/{session_id}",
            response_model=models.UploadResponse,
            summary="Upload ULog File for a Session",
            description="Upload a .ulg file for a specific session ID. "
                        "If the session ID exists, its previous data (CSVs, cache) will be cleared first. "
                        "If it doesn't exist, a new session is created with this ID.")
async def upload_ulog_for_session(
    session_id: str = FastApiPath(..., description="The client-provided session ID (e.g., chat ID)."),
    file: UploadFile = File(..., description="The .ulg flight log file to upload.")
):
    """Handles ULog file upload for a specific session, converting and preparing it."""
    if not file.filename.lower().endswith('.ulg'):
        raise HTTPException(status_code=400, detail="Invalid file type. Please upload a .ulg file.")

    # Get or create session with the provided ID. This also resets if it exists.
    session = session_manager.get_or_create_session(session_id)

    # Define paths for this session
    temp_ulog_path = session.session_dir / file.filename # Save original log in session dir
    csv_output_dir = session.csv_dir # Use the session's dedicated CSV dir

    try:
        # Save the uploaded file temporarily (overwrites if same filename uploaded again)
        print(f"Saving uploaded file to: {temp_ulog_path}")
        with temp_ulog_path.open("wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        print(f"Successfully saved {file.filename} for session {session.session_id}")

        # Store the path to the original ulog file in the session context
        session.set_ulog_path(temp_ulog_path)

        # --- Call new helper for conversion AND renaming ---
        print(f"Starting conversion and renaming: {temp_ulog_path} -> {csv_output_dir}")
        processing_success = await asyncio.to_thread(
            _convert_and_rename_ulog_csvs, # Call the new wrapper function
            ulog_filepath_str=str(temp_ulog_path),
            csv_output_dir_str=str(csv_output_dir),
            ulog_original_filename=file.filename # Pass original filename for prefix calculation
        )
        # --- End change ---

        if not processing_success: # Check the result of the wrapper
            raise HTTPException(status_code=500, detail="ULog to CSV conversion failed. Check server logs.")

        print(f"Processing successful for session {session.session_id}")
        return models.UploadResponse(
            session_id=session.session_id, # Return the session_id used/created
            message=f"File '{file.filename}' uploaded and processed successfully for session {session.session_id}."
        )

    except Exception as e:
        # Log error but don't necessarily delete session on generic error
        print(f"Error during upload for session {session.session_id}: {e}")
        raise HTTPException(status_code=500, detail=f"An unexpected error occurred: {e}")
    finally:
        # Ensure the uploaded file object is closed
        if file and hasattr(file, 'file') and not file.file.closed:
             await file.close()

@app.delete("/session/{session_id}",
            summary="End Session",
            description="End an analysis session and delete its temporary data.")
async def end_session(
    session_id: str = FastApiPath(..., description="The ID of the session to end.")
):
    """Ends a session and cleans up its resources."""
    session = session_manager.get_session(session_id)
    if not session:
         # Return 200 even if session not found, as the desired state (no session) is achieved
        return JSONResponse(content={"message": f"Session {session_id} not found or already ended."}, status_code=200)
        
    session_manager.end_session(session_id)
    return JSONResponse(content={"message": f"Session {session_id} ended successfully and data cleaned up."}, status_code=200)


# --- Placeholder Tool Endpoints --- 
# (Implementations to follow)

@app.post("/tools/dynamic_search/{session_id}", 
            response_model=models.QueryToolResponse, 
            summary="Dynamic Parameter Search")
async def dynamic_search(
    request: models.SearchRequest,
    session: SessionContext = Depends(get_valid_session) # Use SessionContext type hint
):
    """Performs semantic search over dynamic ULog topics using the DynamicParameterTool."""
    print(f"--- API: ENTERING dynamic_search endpoint for session {session.session_id} ---") # <-- DIAGNOSTIC
    # print(f"Received dynamic search for session {session.session_id}: '{request.query}'") # Keep original print commented for now
    
    csv_dir = str(session.csv_dir)
    kb_path = str(config.KNOWLEDGE_BASE_JSON_PATH)

    if not session.csv_dir.exists():
        raise HTTPException(status_code=404, detail=f"CSV directory for session {session.session_id} not found.")
    if not config.KNOWLEDGE_BASE_JSON_PATH.exists():
        raise HTTPException(status_code=500, detail="Knowledge base file not found on server.")

    try:
        print(f"--- API: About to instantiate DynamicParameterTool ---") # <-- DIAGNOSTIC
        print(f"--- API: Using csv_dir='{csv_dir}', kb_file='{kb_path}' ---") # <-- DIAGNOSTIC
        # Instantiate the tool
        tool = DynamicParameterTool(csv_dir=csv_dir, kb_file=kb_path)
        print(f"--- API: DynamicParameterTool instantiation COMPLETE ---") # <-- DIAGNOSTIC

        print(f"--- API: About to call tool.query('{request.query}') in thread ---") # <-- DIAGNOSTIC
        # Run the potentially blocking query method in a thread pool
        # The query method should return a list of results or raise an exception
        search_results = await asyncio.to_thread(tool.query, request.query)
        print(f"--- API: tool.query call COMPLETE ---") # <-- DIAGNOSTIC
        
        # print(f"Dynamic search completed for session {session.session_id}. Found {len(search_results)} results.") # Original print
        print(f"--- API: Dynamic search completed. Found {len(search_results)} results. Preparing response. ---") # <-- DIAGNOSTIC
        
        # --- Add Formatted Print --- 
        _print_formatted_tool_results(search_results, result_type="dynamic parameter matches")
        # --- End Add ---
        
        # Assume tool.query returns the list directly on success
        return models.QueryToolResponse(success=True, results=search_results)

    except FileNotFoundError as e:
        print(f"Error during dynamic search for session {session.session_id}: {e}")
        raise HTTPException(status_code=404, detail=f"Required file not found: {e}")
    except Exception as e:
        print(f"Error during dynamic search for session {session.session_id}: {e}")
        # Log the full error for debugging
        # import traceback
        # traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"An error occurred during dynamic parameter search: {e}")

@app.post("/tools/static_search/{session_id}", 
            response_model=models.QueryToolResponse, 
            summary="Static Parameter Search")
async def static_search(
    request: models.SearchRequest,
    session: SessionContext = Depends(get_valid_session) # Use SessionContext type hint
):
    """Performs semantic search over static parameters using the StaticParameterTool."""
    print(f"Received static search for session {session.session_id}: '{request.query}'")

    static_csv_path = str(config.STATIC_PARAMS_CSV_PATH)
    ulog_file_path = str(session.ulog_path) if session.ulog_path else None

    # Check required files exist
    if not config.STATIC_PARAMS_CSV_PATH.exists():
        raise HTTPException(status_code=500, detail="Static parameters CSV file not found on server.")
    if ulog_file_path and not session.ulog_path.exists():
        # This shouldn't happen if upload worked, but check anyway
        raise HTTPException(status_code=404, detail=f"Original ULog file for session {session.session_id} not found.")

    try:
        # Instantiate the tool, providing the static CSV definitions 
        # and the session's specific ULog file for potential intersection
        tool = StaticParameterTool(csv_path=static_csv_path, ulog_file=ulog_file_path)
        
        # Run the potentially blocking query method in a thread pool
        search_results = await asyncio.to_thread(tool.query, request.query)
        
        print(f"Static search completed for session {session.session_id}. Found {len(search_results)} results.")
        
        # Assume tool.query returns the list directly on success
        return models.QueryToolResponse(success=True, results=search_results)

    except FileNotFoundError as e:
        print(f"Error during static search for session {session.session_id}: {e}")
        raise HTTPException(status_code=404, detail=f"Required file not found: {e}")
    except Exception as e:
        print(f"Error during static search for session {session.session_id}: {e}")
        # Log the full error for debugging
        # import traceback
        # traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"An error occurred during static parameter search: {e}")

@app.post("/tools/topic_fields/{session_id}", 
            response_model=models.TopicFieldsResponse,
            summary="Get Topic Fields")
async def topic_fields(
    request: models.TopicFieldsRequest,
    session: SessionContext = Depends(get_valid_session) # Use SessionContext type hint
):
    """Lists the fields (columns) available in a specific topic CSV file."""
    print(f"Received topic fields request for session {session.session_id}")

    csv_dir = str(session.csv_dir)
    topic_name = request.topic_name
    file_path_hint = request.file_path_hint
    
    # The underlying tool needs a topic name even if using file_path, 
    # use filename as topic name if topic_name not provided but file path is.
    if file_path_hint and not topic_name:
        topic_name = Path(file_path_hint).stem # Extract name from path hint if needed

    # Determine primary identifier: Hint > Filename > Topic Name
    # The tool logic itself handles searching based on topic_name in csv_dir if path isn't absolute
    if file_path_hint:
        print(f"Using file path hint: {file_path_hint}")
        call_args = {"topic_name": topic_name, "file_path": file_path_hint, "csv_dir": csv_dir}
    elif request.topic_filename:
        # Construct path from filename and session dir if filename is given
        file_path_from_filename = session.csv_dir / request.topic_filename
        if not file_path_from_filename.exists():
             raise HTTPException(status_code=404, detail=f"Specified topic filename '{request.topic_filename}' not found in session directory.")
        print(f"Using specific filename: {request.topic_filename}")
        call_args = {"topic_name": topic_name or Path(request.topic_filename).stem, "file_path": str(file_path_from_filename), "csv_dir": csv_dir}
    elif topic_name:
        print(f"Using topic name: {topic_name}")
        # Let the tool find the file in csv_dir based on topic_name
        call_args = {"topic_name": topic_name, "csv_dir": csv_dir, "file_path": None}
    else:
        raise HTTPException(status_code=400, detail="Request must include topic_name, topic_filename, or file_path_hint.")

    try:
        # Call the get_topic_fields function in a thread pool
        result_dict = await asyncio.to_thread(
            get_topic_fields, 
            **call_args
        )
        
        if not result_dict.get("success"):
            error_msg = result_dict.get("error", "Unknown error in get_topic_fields")
            print(f"get_topic_fields failed for session {session.session_id}: {error_msg}")
            # Determine appropriate status code (404 if not found, 500 otherwise)
            status_code = 404 if "not found" in error_msg.lower() else 500
            raise HTTPException(status_code=status_code, detail=error_msg)

        print(f"Topic fields retrieved successfully for session {session.session_id}")
        return models.TopicFieldsResponse(
            success=True,
            topic=result_dict.get("topic"),
            fields=result_dict.get("fields"),
            file=result_dict.get("file")
        )

    except Exception as e:
        print(f"Error during topic_fields execution for session {session.session_id}: {e}")
        raise HTTPException(status_code=500, detail=f"An unexpected error occurred: {e}")

@app.post("/tools/topic_data/{session_id}", 
            response_model=models.TopicDataResponse,
            summary="Get Topic Data (and Cache)")
async def topic_data(
    request: models.TopicDataRequest,
    session: SessionContext = Depends(get_valid_session) # Use SessionContext type hint
):
    """Retrieves data for specified fields from a topic CSV and caches the DataFrame."""
    print(f"Received topic data request for session {session.session_id}")

    csv_dir = str(session.csv_dir)
    topic_name = request.topic_name
    file_path_hint = request.file_path_hint
    requested_fields = request.columns
    
    # Similar logic to topic_fields to determine the target file
    if file_path_hint and not topic_name:
        topic_name = Path(file_path_hint).stem 

    if file_path_hint:
        print(f"Using file path hint: {file_path_hint}")
        call_args = {"topic_name": topic_name, "file_path": file_path_hint, "csv_dir": csv_dir}
    elif request.topic_filename:
        file_path_from_filename = session.csv_dir / request.topic_filename
        if not file_path_from_filename.exists():
             raise HTTPException(status_code=404, detail=f"Specified topic filename '{request.topic_filename}' not found in session directory.")
        print(f"Using specific filename: {request.topic_filename}")
        call_args = {"topic_name": topic_name or Path(request.topic_filename).stem, "file_path": str(file_path_from_filename), "csv_dir": csv_dir, "fields": requested_fields}
    elif topic_name:
        print(f"Using topic name: {topic_name}")
        call_args = {"topic_name": topic_name, "csv_dir": csv_dir, "file_path": None, "fields": requested_fields}
    else:
        raise HTTPException(status_code=400, detail="Request must include topic_name, topic_filename, or file_path_hint.")

    try:
        # Call the get_data function in a thread pool
        # It returns a dictionary including the DataFrame if successful
        result_dict = await asyncio.to_thread(
            get_data,
            **call_args
        )

        if not result_dict.get("success"):
            error_msg = result_dict.get("error", "Unknown error in get_data")
            print(f"get_data failed for session {session.session_id}: {error_msg}")
            status_code = 404 if "not found" in error_msg.lower() or "missing fields" in error_msg.lower() else 500
            raise HTTPException(status_code=status_code, detail=error_msg)

        # Extract the DataFrame from the result (key might vary, check general_purpose_tools.py - assuming '_dataframe')
        dataframe = result_dict.get("_dataframe")
        if dataframe is None or not hasattr(dataframe, 'columns'): # Basic check if it's DataFrame-like
             print(f"Error extracting DataFrame from get_data result for session {session.session_id}")
             raise HTTPException(status_code=500, detail="Failed to retrieve DataFrame from get_data tool result.")

        # Cache the DataFrame using the session manager
        data_id = session_manager.cache_data(session.session_id, dataframe)
        if not data_id:
             # This indicates an issue within the session manager itself
             raise HTTPException(status_code=500, detail="Failed to cache data in session.")

        print(f"Data retrieved and cached successfully for session {session.session_id} with data_id {data_id}")
        return models.TopicDataResponse(
            data_id=data_id,
            rows_cached=len(dataframe),
            columns_cached=dataframe.columns.tolist()
            # message field has default in Pydantic model
        )

    except Exception as e:
        print(f"Error during topic_data execution for session {session.session_id}: {e}")
        raise HTTPException(status_code=500, detail=f"An unexpected error occurred: {e}")

@app.post("/tools/compute/{session_id}", 
            response_model=models.ComputeResponse,
            summary="Run Computation on Cached Data")
async def compute(
    request: models.ComputeRequest,
    session: SessionContext = Depends(get_valid_session)
):
    """Executes Python code, optionally against a cached DataFrame."""
    print(f"Received compute request for session {session.session_id} (data_id: {request.data_id or 'NONE'})") # Updated log
    code_to_run = request.code
    comment = request.comment
    dataframe = None # Initialize DataFrame to None
    data_id_used = request.data_id

    # --- SECURITY CHECK --- (Keep removed as per previous request)

    # --- Retrieve Cached Data (ONLY if data_id is provided) ---
    if request.data_id:
        # Specific data_id provided
        print(f"Attempting to retrieve specific DataFrame: {request.data_id}")
        dataframe = session_manager.get_cached_data(session.session_id, request.data_id)
        if dataframe is None:
            raise HTTPException(status_code=404, detail=f"Specified Data ID '{request.data_id}' not found in cache for session {session.session_id}.")
        print(f"Retrieved DataFrame {data_id_used} for computation (shape: {dataframe.shape})")
    else:
        # No data_id provided, proceed without a pre-loaded DataFrame
        print("No specific data_id provided. Proceeding without pre-loading 'df'.")
        data_id_used = None # Ensure this is None if not provided

    try:
        # --- Execute Computation --- 
        # Pass the retrieved DataFrame (which might be None) to the tool
        print(f"WARNING: Executing potentially unsafe code for session {session.session_id} without sandboxing!")
        computation_result = await asyncio.to_thread(
            run_computation,
            data=dataframe, # Pass DataFrame OR None
            computation=code_to_run,
            comment=comment,
        )
        
        # --- Process Result --- 
        print(f"Computation executed for session {session.session_id} (data_id used: {data_id_used})") # Use actual id used
        # ... (rest of result processing logic remains the same) ...
        if isinstance(computation_result, dict):
            success = computation_result.get("success", False)
            result_summary = computation_result.get("result", computation_result.get("preview"))
            error_msg = computation_result.get("error")
            if not success:
                print(f"Computation failed for session {session.session_id}, data {data_id_used}: {error_msg}")
                return models.ComputeResponse(success=False, error=error_msg or "Computation tool reported failure.", result_summary=None)
            else:
                return models.ComputeResponse(success=True, error=None, result_summary=result_summary)
        else:
            return models.ComputeResponse(success=True, error=None, result_summary=computation_result)

    except Exception as e:
        print(f"Error during compute execution for session {session.session_id} (data_id used: {data_id_used}): {e}") # Use actual id used
        raise HTTPException(status_code=500, detail=f"An unexpected error occurred during computation: {e}")

@app.get("/tools/static_param_lookup/{session_id}",
            response_model=models.StaticParamLookupResponse,
            summary="Static Parameter Lookup by Name",
            description="Retrieves details for a specific static parameter by its exact name, "
                        "optionally overlaying the value found in the session's ULog file.")
async def static_param_lookup(
    session: SessionContext = Depends(get_valid_session), # Use SessionContext type hint
    param_name: str = Query(..., description="The exact name of the static parameter to look up.")
):
    """Looks up a static parameter by its exact name."""
    print(f"Received static param lookup for session {session.session_id}, param_name='{param_name}'")

    static_csv_path = str(config.STATIC_PARAMS_CSV_PATH)
    ulog_file_path = str(session.ulog_path) if session.ulog_path else None

    # Check required files exist
    if not config.STATIC_PARAMS_CSV_PATH.exists():
        raise HTTPException(status_code=500, detail="Static parameters CSV file not found on server.")
    # No need to check ulog path here, the tool handles it being None

    try:
        # Instantiate the tool
        tool = StaticParameterTool(csv_path=static_csv_path, ulog_file=ulog_file_path)

        # Call the get_param_by_name method in a thread pool
        # Assume it returns a dictionary on success, None if not found
        param_details = await asyncio.to_thread(tool.get_param_by_name, param_name)

        if param_details is None:
            print(f"Parameter '{param_name}' not found for session {session.session_id}.")
            raise HTTPException(status_code=404, detail=f"Static parameter '{param_name}' not found.")

        print(f"Parameter '{param_name}' found for session {session.session_id}.")
        # Validate and return using the Pydantic model
        return models.StaticParamLookupResponse(success=True, parameter=models.StaticParamDetail(**param_details))

    except Exception as e:
        print(f"Error during static param lookup for session {session.session_id}: {e}")
        raise HTTPException(status_code=500, detail=f"An error occurred during static parameter lookup: {e}")

# --- Optional: Run with Uvicorn (for local development) ---
if __name__ == "__main__":
    import uvicorn
    print("Starting Uvicorn server...")
    # Run from the directory containing 'The_agent_api' folder
    # Example: python -m The_agent_api.main
    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=True, app_dir=str(current_dir)) 