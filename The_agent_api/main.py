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
    from tools.general_purpose_tools import get_topic_fields, get_data, run_computation, run_plotting
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
    """Handles ULog file upload for a specific session, converting and preparing it.
    Checks if the session already has processed CSV data and skips reprocessing if found.
    """
    if not file.filename.lower().endswith('.ulg'):
        raise HTTPException(status_code=400, detail="Invalid file type. Please upload a .ulg file.")

    # --- Check if session exists and has already been processed ---
    existing_session = session_manager.get_session(session_id)
    if existing_session:
        # Check if the CSV directory exists and contains CSV files
        try:
            if existing_session.csv_dir.exists() and any(existing_session.csv_dir.glob('*.csv')):
                print(f"Session {session_id} already has processed CSV data. Skipping upload.")
                # Close the uploaded file handle as we are not using it
                if file and hasattr(file, 'file') and not file.file.closed:
                     await file.close()
                return models.UploadResponse(
                    session_id=session_id,
                    message=f"Session {session_id} data already processed. Upload skipped."
                )
        except Exception as check_err:
            # Log the error but proceed as if data doesn't exist, allowing reprocessing
            print(f"Error checking existing session {session_id} data, proceeding with upload: {check_err}")
    # --- End check ---

    # If session doesn't exist, or exists but has no CSV data, proceed with get_or_create (which resets if needed)
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

        # --- Conversion and Renaming ---
        print(f"Starting ULog to CSV conversion for {file.filename}...")
        conversion_success = await asyncio.to_thread(
            _convert_and_rename_ulog_csvs,
            str(temp_ulog_path),          # Pass ulog path as string
            str(csv_output_dir),          # Pass csv output dir as string
            file.filename                 # Pass original filename
        )
        # --- End Conversion ---

        if not conversion_success:
            print(f"Conversion failed for {file.filename}. Session {session_id} might be incomplete.")
            raise HTTPException(
                status_code=500,
                detail=f"Failed to convert ULog file '{file.filename}' to CSV format."
            )
        else:
            print(f"Conversion successful for {file.filename}. CSVs in {csv_output_dir}")
            # Create the success marker file AFTER successful conversion
            marker_filename = "_SUCCESS" # Consistent marker file name
            marker_file_path = session.session_dir / marker_filename
            try:
                marker_file_path.touch() # Create an empty _SUCCESS file
                print(f"Created success marker file: {marker_file_path}")
            except Exception as marker_err:
                print(f"Warning: Could not create success marker file {marker_file_path}: {marker_err}")


        return models.UploadResponse(
            session_id=session.session_id,
            message=f"Successfully uploaded and processed '{file.filename}'. CSV data ready."
        )

    except HTTPException as http_exc:
        # Re-raise HTTPExceptions directly
        raise http_exc
    except Exception as e:
        # Log the error and return a 500 response for other errors
        print(f"Error during upload/processing for session {session_id}: {e}")
        # Clean up potentially partially saved file? Depends on desired atomicity.
        # if temp_ulog_path.exists():
        #     temp_ulog_path.unlink(missing_ok=True)
        raise HTTPException(status_code=500, detail=f"An unexpected error occurred: {e}")
    finally:
        # Ensure the uploaded file handle is closed even if errors occur
        if file and hasattr(file, 'file') and not file.file.closed:
             await file.close()

@app.delete("/session/{session_id}",
            response_model=models.DeleteResponse,
            summary="Delete Session Data",
            description="Deletes all temporary data (ULog, CSVs, cache) associated with a session ID.")
async def delete_session(
    session_id: str = FastApiPath(..., description="The ID of the session to delete.")
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
    print(f"DEBUG: dynamic_search using session '{session.session_id}', csv_dir: '{session.csv_dir}'") # <-- ADDED DEBUG
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
    print(f"DEBUG: static_search using session '{session.session_id}', ulog_path: '{session.ulog_path}'") # <-- ADDED DEBUG

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
    print(f"DEBUG: topic_fields using session '{session.session_id}', csv_dir: '{session.csv_dir}'") # <-- ADDED DEBUG

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
    print(f"DEBUG: topic_data using session '{session.session_id}', csv_dir: '{session.csv_dir}'") # <-- ADDED DEBUG

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
            summary="Run Computation on Cached Data",
            description="Executes a Python computation string on previously cached data using its data_id.")
async def run_computation_endpoint(
    request: models.ComputeRequest,
    session: SessionContext = Depends(get_valid_session)
):
    """Endpoint to execute computations using the run_computation tool."""
    print(f"DEBUG: run_computation_endpoint using session '{session.session_id}', cache_dir: '{session.cache_dir}'") # <-- ADDED DEBUG
    # --- Use data_ids list from request --- 
    data_ids_to_use = request.data_ids # This is now Optional[List[str]]

    # Remove the logic for automatically using the most recent cache item.
    # If data_ids_to_use is None or empty, it signifies general execution.
    if not data_ids_to_use: 
         print("No data_ids provided. Proceeding with general Python execution.")
         data_ids_to_use = None # Ensure it's None if empty list was passed
    else:
        print(f"Received data_ids: {data_ids_to_use}")

    # --- Execute Computation --- 
    # Pass the list of data IDs (or None) to the tool.
    print(f"Calling run_computation tool for session {session.session_id} with data_ids {data_ids_to_use}")
    try:
        result = await asyncio.to_thread(
            run_computation,
            session=session, # Pass the SessionContext object
            data_ids=data_ids_to_use, # Pass the list of data_ids (or None)
            computation=request.code,
            comment=request.comment
        )
    except Exception as tool_exec_err:
        # Catch errors originating from the tool function execution itself
        print(f"Error executing run_computation tool via asyncio.to_thread: {tool_exec_err}")
        raise HTTPException(
            status_code=500,
            detail=f"Internal error executing computation tool: {tool_exec_err}"
        )
    # --- End Execution ---

    if not result or not result.get("success"):
        error_detail = result.get("error", "Computation tool failed.") if result else "Computation tool failed."
        # Check if it was a user code error or tool error
        status_code = 400 if "Computation execution error" in error_detail else 500
        raise HTTPException(
            status_code=status_code, 
            detail=error_detail
        )

    # Return the summary from the computation result
    # Ensure the response model matches the actual structure returned by the tool
    return models.ComputeResponse(
        success=True,
        result_summary=result.get("result_summary"),
        error=result.get("error", None) # Include error field if tool provides one even on success (e.g., warnings)
    )

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
    print(f"DEBUG: static_param_lookup using session '{session.session_id}', ulog_path: '{session.ulog_path}'") # <-- ADDED DEBUG

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

# --- NEW ENDPOINT ---
@app.get("/check_session/{session_id}",
           response_model=models.SessionStatusResponse, # Define this model in api/models.py
           summary="Check Session Processing Status",
           description="Check if the data processing for a given session ID has successfully completed.")
async def check_session_status(
    session_id: str = FastApiPath(..., description="The session ID to check.")
):
    """
    Checks if the session directory and a success marker file exist.
    """
    processed = False
    marker_filename = "_SUCCESS" # Marker file indicating successful completion
    session_dir_path = None

    try:
        # Use config.TEMP_BASE_DIR to construct the path to the specific session directory
        # Ensure session_id is treated as a string component
        session_dir_path = os.path.join(config.TEMP_BASE_DIR, str(session_id))

        # Construct the path to the marker file within that directory
        marker_file_path = os.path.join(session_dir_path, marker_filename)

        # Check if the session directory exists AND the marker file exists within it
        if os.path.isdir(session_dir_path) and os.path.isfile(marker_file_path):
            processed = True

        return models.SessionStatusResponse(processed=processed)

    except Exception as e:
        print(f"Error checking session status for {session_id} at path {session_dir_path}: {e}")
        # If any error occurs (permissions, invalid path, etc.), assume not processed
        # Return 500 to indicate a server-side check failure
        raise HTTPException(
            status_code=500,
            detail=f"Error checking status for session {session_id}: {e}"
        )
# --- END NEW ENDPOINT ---

# --- NEW Plotting Endpoint --- 
@app.post("/tools/plot/{session_id}",
            response_model=models.PlotResponse,
            summary="Generate Plot from Data",
            description="Executes Python code to generate and save a plot from cached data.")
async def run_plot_endpoint(
    request: models.PlotRequest,
    session: SessionContext = Depends(get_valid_session)
):
    """Endpoint to generate plots using the run_plotting tool."""
    print(f"DEBUG: run_plot_endpoint using session '{session.session_id}'")

    try:
        # Call the run_plotting function in a thread pool
        print(f"Calling run_plotting tool for session {session.session_id} with data_ids {request.data_ids}")
        result = await asyncio.to_thread(
            run_plotting,
            session=session, 
            data_ids=request.data_ids, 
            code=request.code,
            comment=request.comment
        )
        
        if not result or not result.get("success"):
            error_detail = result.get("error", "Plotting tool failed.") if result else "Plotting tool failed."
            summary = result.get("result_summary", "") # Get summary regardless of error type
            
            # Determine status code (keep existing logic)
            status_code = 400 
            if error_detail:
                 if "matplotlib is not installed" in error_detail:
                     status_code = 501 
                 elif "Failed to create plots directory" in error_detail:
                     status_code = 500 
            
            # Always include the summary in the detail if it exists, for better debugging
            if summary:
                error_detail += f"\n--- Execution Output ---\n{summary}"
            else: # Add a note if summary was empty
                 error_detail += "\n(No stdout/stderr captured from execution)"

            raise HTTPException(
                status_code=status_code, 
                detail=error_detail # Now includes summary
            )

        # Return the filename and base64 content
        return models.PlotResponse(
            success=True,
            plot_filename=result.get("plot_filename"),
            plot_content_base64=result.get("plot_content_base64"),
            result_summary=result.get("result_summary") # Include for debugging if needed
        )

    except HTTPException as http_exc:
         raise http_exc # Re-raise existing HTTP exceptions
    except Exception as e:
         # Catch unexpected errors during endpoint execution/tool call
         print(f"Error during run_plot_endpoint for session {session.session_id}: {e}")
         import traceback
         traceback.print_exc() # Log full traceback for server debugging
         raise HTTPException(
             status_code=500,
             detail=f"An unexpected error occurred during plot generation: {e}"
         )
# --- END NEW Plotting Endpoint --- 

# --- Optional: Run with Uvicorn (for local development) ---
if __name__ == "__main__":
    import uvicorn
    print("Starting Uvicorn server...")
    # Run from the directory containing 'The_agent_api' folder
    # Example: python -m The_agent_api.main
    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=True, app_dir=str(current_dir)) 