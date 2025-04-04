# The_agent_api/api/models.py
from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional

# --- Upload --- 
class UploadResponse(BaseModel):
    session_id: str
    message: str

# --- Tool Requests --- 
class SearchRequest(BaseModel):
    query: str

class TopicFieldsRequest(BaseModel):
    # Allow identifying the topic CSV via name (preferred) or direct filename
    topic_name: Optional[str] = Field(None, description="Logical name of the topic (e.g., vehicle_local_position)")
    topic_filename: Optional[str] = Field(None, description="Specific CSV filename (e.g., vehicle_local_position_0.csv)")
    file_path_hint: Optional[str] = Field(None, description="Full path hint if known (e.g., from dynamic search results)")

class TopicDataRequest(BaseModel):
    topic_name: Optional[str] = Field(None, description="Logical name of the topic")
    topic_filename: Optional[str] = Field(None, description="Specific CSV filename")
    file_path_hint: Optional[str] = Field(None, description="Full path hint if known")
    columns: Optional[List[str]] = Field(None, description="Specific columns to retrieve. Recommended for performance.")
    # Consider adding filter support later based on get_data capabilities
    # filters: Optional[Dict[str, Any]] = None 

class TopicDataResponse(BaseModel):
    data_id: str = Field(..., description="ID assigned to the cached DataFrame")
    rows_cached: int
    columns_cached: List[str]
    message: str = "Data cached successfully."

class ComputeRequest(BaseModel):
    data_id: Optional[str] = Field(None, description="ID of the cached DataFrame (optional; uses most recent if omitted)")
    code: str = Field(..., description="Python code string to execute. IMPORTANT: Execution is not sandboxed by default!")
    comment: Optional[str] = Field(None, description="Optional comment about the computation")

# --- Generic Tool Responses --- 
# Basic success/failure structure
class BaseToolResponse(BaseModel):
    success: bool
    error: Optional[str] = None

# More detailed response for tools returning structured data
class QueryToolResponse(BaseToolResponse):
    # Example structure for dynamic/static search results
    # Adapt based on actual tool output format
    results: Optional[List[Dict[str, Any]]] = None 

class TopicFieldsResponse(BaseToolResponse):
    topic: Optional[str] = None
    fields: Optional[List[str]] = None
    file: Optional[str] = None 

class ComputeResponse(BaseToolResponse):
    # Adapt based on what run_computation actually returns
    # Could be a value, a status message, serialized data/plot, etc.
    result_summary: Optional[Any] = None 
    # Potentially include updated data_id if computation modifies the cache in place? 

class StaticParamDetail(BaseModel):
    # Define fields based on what StaticParameterTool.get_param_by_name returns
    # Example fields (adjust as needed):
    id: Optional[str] = None
    name: str
    group: Optional[str] = None
    value: Any # Value can be different types
    type: Optional[str] = None
    min: Optional[Any] = None
    max: Optional[Any] = None
    unit: Optional[str] = None
    shortDesc: Optional[str] = None
    longDesc: Optional[str] = None
    decimal: Optional[int] = None
    increment: Optional[float] = None
    reboot_required: Optional[bool] = None
    category: Optional[str] = None
    access_info: Optional[str] = None # If value came from ULog
    source: Optional[str] = None # e.g., 'csv', 'ulog', 'intersection'

class StaticParamLookupResponse(BaseToolResponse): # Inherits success/error
    parameter: Optional[StaticParamDetail] = None 