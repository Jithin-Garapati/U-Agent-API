# The_agent_api/api/session.py
import uuid
import shutil
from pathlib import Path
from typing import Dict, Any, Optional, Tuple, List
import threading
import pandas as pd
import os # Import os for listdir and remove
import time # Added for modification time check

# Use a thread-safe lock for modifying shared session/cache data
_lock = threading.Lock()

class SessionContext:
    """Holds context information for a single user session."""
    def __init__(self, session_id: str, base_temp_dir: Path):
        self.session_id: str = session_id
        self.base_temp_dir: Path = base_temp_dir
        self.session_dir: Path = self.base_temp_dir / self.session_id
        self.csv_dir: Path = self.session_dir / "csv_topics"
        self.cache_dir: Path = self.session_dir / "dataframe_cache" # Directory for Pickle cache
        self.ulog_path: Optional[Path] = None

        # Create session-specific directories if they don't exist
        self.session_dir.mkdir(parents=True, exist_ok=True)
        self.csv_dir.mkdir(parents=True, exist_ok=True)
        self.cache_dir.mkdir(parents=True, exist_ok=True) # Create cache directory

    def set_ulog_path(self, ulog_path: Path):
        self.ulog_path = ulog_path

    def add_to_cache(self, data_id: str, df: pd.DataFrame):
        """Saves a DataFrame to the disk cache as a Pickle file."""
        cache_file_path = self.cache_dir / f"{data_id}.pkl" # Changed extension
        try:
            df.to_pickle(cache_file_path)
            print(f"Saved DataFrame cache to: {cache_file_path}")
        except Exception as e:
            print(f"Error saving DataFrame {data_id} to Pickle file {cache_file_path}: {e}")
            raise # Re-raise the exception so the caller knows

    def get_from_cache(self, data_id: str) -> Optional[pd.DataFrame]:
        """Loads a DataFrame from the Pickle disk cache."""
        cache_file_path = self.cache_dir / f"{data_id}.pkl" # Changed extension
        if cache_file_path.is_file():
            try:
                df = pd.read_pickle(cache_file_path)
                print(f"Loaded DataFrame cache from: {cache_file_path}")
                return df
            except Exception as e:
                print(f"Error reading Pickle cache file {cache_file_path}: {e}")
                return None
        else:
            print(f"Cache file not found: {cache_file_path}")
            return None

    # --- Modified Method: Find latest pickle file by mtime ---
    def get_most_recent_cache_item(self) -> Optional[Tuple[str, pd.DataFrame]]:
         """Returns the data_id and DataFrame of the most recently modified Pickle file."""
         latest_file = None
         latest_mtime = 0

         try:
             if not self.cache_dir.exists():
                 return None

             for item in os.listdir(self.cache_dir):
                 if item.lower().endswith('.pkl'): # Changed extension check
                     file_path = self.cache_dir / item
                     try:
                         mtime = os.path.getmtime(file_path)
                         if mtime > latest_mtime:
                             latest_mtime = mtime
                             latest_file = file_path
                     except OSError:
                         continue

             if latest_file:
                 data_id = latest_file.stem # Get filename without extension
                 df = self.get_from_cache(data_id) # Use existing method to load
                 if df is not None:
                     return data_id, df
                 else:
                     return None
             else:
                 # No pickle files found
                 return None

         except Exception as e:
             print(f"Error finding most recent cache item in {self.cache_dir}: {e}")
             return None
    # --- End Modified Method ---

    def reset(self):
        """Clears cached dataframes, temporary files (CSVs, ULog) for this session."""
        print(f"Resetting session: {self.session_id}")
        self.ulog_path = None # Clear ulog path reference

        # Clear contents of csv_dir
        if self.csv_dir.exists():
            try:
                for item in os.listdir(self.csv_dir):
                    item_path = self.csv_dir / item
                    if item_path.is_file():
                        item_path.unlink()
                    elif item_path.is_dir():
                        shutil.rmtree(item_path)
                print(f"Cleared CSV directory: {self.csv_dir}")
            except OSError as e:
                print(f"Error clearing CSV directory {self.csv_dir}: {e}")

        # Clear contents of cache_dir (dataframe cache)
        if self.cache_dir.exists():
            try:
                for item in os.listdir(self.cache_dir):
                    item_path = self.cache_dir / item
                    if item_path.is_file():
                        item_path.unlink()
                    elif item_path.is_dir():
                        shutil.rmtree(item_path) # Should not have dirs here, but just in case
                print(f"Cleared DataFrame cache directory: {self.cache_dir}")
            except OSError as e:
                print(f"Error clearing DataFrame cache directory {self.cache_dir}: {e}")

        # Optionally clear other files in session_dir if needed, be careful not to delete the dir itself
        if self.session_dir.exists():
            try:
                for item in os.listdir(self.session_dir):
                    item_path = self.session_dir / item
                    # Only remove files, keep the csv_dir and cache_dir itself
                    if item_path.is_file():
                         item_path.unlink()
                print(f"Cleared base files in session directory: {self.session_dir}")
            except OSError as e:
                print(f"Error clearing base files in session directory {self.session_dir}: {e}")

    def cleanup(self):
        """Completely remove temporary directory for this session."""
        print(f"Cleaning up session: {self.session_id}")
        if self.session_dir.exists():
            try:
                shutil.rmtree(self.session_dir) # This removes session_dir and everything inside it
                print(f"Removed session directory: {self.session_dir}")
            except OSError as e:
                print(f"Error removing session directory {self.session_dir}: {e}")

class InMemorySessionManager:
    """Manages user sessions and their associated data (now mostly on disk)."""
    def __init__(self, base_temp_dir: Path):
        self.base_temp_dir = base_temp_dir
        self._sessions: Dict[str, SessionContext] = {}
        # Ensure base temp dir exists
        self.base_temp_dir.mkdir(parents=True, exist_ok=True)

    def _prune_old_sessions(self, max_sessions: int = 10):
        """Removes the oldest session directories if the count exceeds max_sessions."""
        print(f"DEBUG: Checking session count against max_sessions={max_sessions}")
        try:
            session_dirs = []
            # List items directly in the base temp directory
            for item_name in os.listdir(self.base_temp_dir):
                item_path = self.base_temp_dir / item_name
                # Check if it's a directory
                if item_path.is_dir():
                    try:
                        # Get modification time
                        mtime = os.path.getmtime(item_path)
                        session_dirs.append((mtime, item_path, item_name)) # Store mtime, path, and session_id (dir name)
                    except OSError as e:
                        # Log if getting mtime fails for a directory (permissions, etc.)
                        print(f"Warning: Could not get mtime for directory {item_path}: {e}")

            if len(session_dirs) > max_sessions:
                print(f"DEBUG: Found {len(session_dirs)} session directories, exceeding limit of {max_sessions}. Pruning...")
                # Sort by modification time (oldest first)
                session_dirs.sort(key=lambda x: x[0])
                # Calculate how many to delete
                num_to_delete = len(session_dirs) - max_sessions

                # Iterate through the oldest sessions to delete
                for i in range(num_to_delete):
                    mtime, dir_path, session_id = session_dirs[i]
                    print(f"DEBUG: Pruning session '{session_id}' (modified: {time.ctime(mtime)}) path: {dir_path}")
                    try:
                        # Delete the directory tree
                        shutil.rmtree(dir_path)
                        print(f"DEBUG: Successfully removed directory {dir_path}")
                        # Also remove from the in-memory dictionary (if present)
                        with _lock:
                            removed_session = self._sessions.pop(session_id, None)
                            if removed_session:
                                print(f"DEBUG: Removed session '{session_id}' from in-memory store.")
                            else:
                                print(f"DEBUG: Session '{session_id}' not found in in-memory store (already removed or from previous run?).")
                    except OSError as e:
                        # Log errors during deletion (e.g., file in use, permissions)
                        print(f"Error: Failed to remove directory {dir_path}: {e}")
                    except Exception as e:
                         # Catch other potential errors during pruning of a specific session
                         print(f"Error: An unexpected error occurred while pruning session {session_id}: {e}")
            else:
                 # Log if count is within limits
                 print(f"DEBUG: Session directory count ({len(session_dirs)}) is within limit ({max_sessions}).")

        except Exception as e:
            # Log errors during the initial listing/checking phase
            print(f"Error during session pruning check: {e}")

    def create_session(self) -> SessionContext:
        """Creates a new session with a unique ID and directory."""
        # --- ADDED: Prune old sessions --- 
        self._prune_old_sessions()
        # --- END ADDED --- 
        session_id = str(uuid.uuid4())
        with _lock:
            if session_id in self._sessions:
                # Extremely unlikely, but handle collision just in case
                return self.create_session()
            session = SessionContext(session_id, self.base_temp_dir)
            self._sessions[session_id] = session
            print(f"Created session: {session_id}")
        return session

    def get_session(self, session_id: str) -> Optional[SessionContext]:
        """Retrieves an existing session context."""
        # --- ADDED: Prune old sessions --- 
        # Decide if pruning should happen only on creation/reset or also on get?
        # Let's prune on get_or_create for now, not just get.
        # self._prune_old_sessions()
        # --- END ADDED --- 
        with _lock:
            return self._sessions.get(session_id)

    def get_or_create_session(self, session_id: str) -> SessionContext:
        """Gets a session by ID or creates it if it doesn't exist. Resets if it exists."""
        # --- ADDED: Prune old sessions before getting/creating the current one ---
        self._prune_old_sessions()
        # --- END ADDED ---

        with _lock:
            session = self._sessions.get(session_id)
            if session:
                print(f"Session {session_id} already exists. Resetting it for new upload.")
                # Reset existing session data (clear cache, CSVs, etc.)
                session.reset()
            else:
                print(f"Creating new session with provided ID: {session_id}")
                session = SessionContext(session_id, self.base_temp_dir)
                self._sessions[session_id] = session
            return session

    def end_session(self, session_id: str):
        """Ends a session and cleans up its resources."""
        with _lock:
            session = self._sessions.pop(session_id, None)
        if session:
            session.cleanup()
        else:
            print(f"Attempted to end non-existent session: {session_id}")

    # --- Modified: Delegate cache operations to SessionContext --- 
    def get_most_recent_cached_data(self, session_id: str) -> Optional[Tuple[str, pd.DataFrame]]:
         session = self.get_session(session_id)
         if not session:
             print(f"Error retrieving most recent cached data: Session {session_id} not found.")
             return None
         # No lock needed here as get_most_recent_cache_item in SessionContext handles file ops
         return session.get_most_recent_cache_item()

    def cache_data(self, session_id: str, df: pd.DataFrame) -> Optional[str]:
        """Caches a DataFrame for a session (to disk) and returns a data_id."""
        session = self.get_session(session_id)
        if not session:
            print(f"Error caching data: Session {session_id} not found.")
            return None

        data_id = f"data_{uuid.uuid4()}"
        try:
            # Delegate saving to the session context
            session.add_to_cache(data_id, df)
            print(f"Cached data for session {session_id} with id {data_id} ({len(df)} rows) to disk.")
            return data_id
        except Exception as e:
            # Catch error from add_to_cache if it re-raises
            print(f"Failed to cache data {data_id} for session {session_id}: {e}")
            return None

    def get_cached_data(self, session_id: str, data_id: str) -> Optional[pd.DataFrame]:
        """Retrieves a cached DataFrame for a session (from disk)."""
        session = self.get_session(session_id)
        if not session:
            print(f"Error retrieving cached data: Session {session_id} not found.")
            return None

        # Delegate loading to the session context
        df = session.get_from_cache(data_id)

        if df is None:
            print(f"Error retrieving cached data: Data ID {data_id} not found or failed to load in session {session_id}.")
        return df 