# The_agent_api/config.py
import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv() # Load .env file if present (e.g., for secrets not in source control)

# --- Core Paths ---
# Use Path objects for better path manipulation

# Path to the root of this FastAPI application ('The_agent_api')
API_ROOT = Path(__file__).parent.resolve()

# Path to the original workspace root (one level up)
WORKSPACE_ROOT = API_ROOT.parent 

# Base directory for temporary user data within the API structure
TEMP_BASE_DIR = API_ROOT / "temp_data" 

# Paths to knowledge base files located in the parent workspace root
KNOWLEDGE_BASE_JSON_PATH = WORKSPACE_ROOT / "formatted_knowledge_base.json"
STATIC_PARAMS_CSV_PATH = WORKSPACE_ROOT / "static_parameters.csv"

# Path to the ulog_utils module (assuming it stays in the parent directory)
# If moved, this needs updating. For imports, ensure parent is in PYTHONPATH or restructure.
ULOG_UTILS_PATH = WORKSPACE_ROOT 

# --- Model Configuration ---
SENTENCE_MODEL_NAME = 'all-MiniLM-L6-v2' # Model tools will load/use

# --- Cache Configuration ---
# Add settings later if needed (e.g., cache TTL)

# --- Security Configuration ---
COMPUTATION_SANDBOXED = False # Flag for computation tool safety (initially False)

# --- Initialization Checks ---
TEMP_BASE_DIR.mkdir(parents=True, exist_ok=True) # Ensure temp directory exists

# Optional: Check if KB files exist at expected locations
if not KNOWLEDGE_BASE_JSON_PATH.is_file():
    print(f"Warning: Knowledge base JSON not found at {KNOWLEDGE_BASE_JSON_PATH}")
if not STATIC_PARAMS_CSV_PATH.is_file():
    print(f"Warning: Static parameters CSV not found at {STATIC_PARAMS_CSV_PATH}")

print("--- API Config Loaded ---")
print(f"API Root: {API_ROOT}")
print(f"Workspace Root: {WORKSPACE_ROOT}")
print(f"Temp Base Dir: {TEMP_BASE_DIR}")
print(f"KB JSON Path: {KNOWLEDGE_BASE_JSON_PATH}")
print(f"Static Params CSV Path: {STATIC_PARAMS_CSV_PATH}")
print(f"ULog Utils Dir: {ULOG_UTILS_PATH}")
print("-------------------------") 