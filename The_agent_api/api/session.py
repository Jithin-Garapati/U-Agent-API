# The_agent_api/api/session.py
import uuid
import shutil
from pathlib import Path
from typing import Dict, Any, Optional, Tuple, List
import threading
import pandas as pd
import os # Import os for listdir and remove

# Use a thread-safe lock for modifying shared session/cache data
_lock = threading.Lock()

class SessionContext:
    """Holds context information for a single user session."""
    def __init__(self, session_id: str, base_temp_dir: Path):
        self.session_id: str = session_id
        self.base_temp_dir: Path = base_temp_dir
        self.session_dir: Path = self.base_temp_dir / self.session_id
        self.csv_dir: Path = self.session_dir / "csv_topics"
        self.ulog_path: Optional[Path] = None
        self.data_cache: Dict[str, pd.DataFrame] = {}
        self._cache_order: List[str] = [] # New: Track cache insertion order

        # Create session-specific directories if they don't exist
        self.session_dir.mkdir(parents=True, exist_ok=True)
        self.csv_dir.mkdir(parents=True, exist_ok=True)

    def set_ulog_path(self, ulog_path: Path):
        self.ulog_path = ulog_path

    def add_to_cache(self, data_id: str, df: pd.DataFrame):
        self.data_cache[data_id] = df
        # Add to end of order list, remove if already exists to ensure it's last
        if data_id in self._cache_order:
             self._cache_order.remove(data_id)
        self._cache_order.append(data_id)

    def get_from_cache(self, data_id: str) -> Optional[pd.DataFrame]:
        return self.data_cache.get(data_id)

    # --- New Method ---
    def get_most_recent_cache_item(self) -> Optional[Tuple[str, pd.DataFrame]]:
         """Returns the data_id and DataFrame of the most recently added item."""
         if not self._cache_order:
             return None
         last_data_id = self._cache_order[-1]
         df = self.get_from_cache(last_data_id)
         if df is None:
             # Should not happen if cache order is synced, but handle defensively
             try:
                self._cache_order.pop() # Remove bad id
             except IndexError:
                pass # List was already empty
             return self.get_most_recent_cache_item() # Try again, might return None if empty now
         return last_data_id, df
    # --- End New Method ---

    def reset(self):
        """Clears cached data and temporary files (CSVs, ULog) for this session."""
        print(f"Resetting session: {self.session_id}")
        self.data_cache.clear()
        self._cache_order.clear() # Clear order on reset
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

        # Optionally clear other files in session_dir if needed, be careful not to delete the dir itself
        if self.session_dir.exists():
            try:
                for item in os.listdir(self.session_dir):
                    item_path = self.session_dir / item
                    # Only remove files, keep the csv_dir itself
                    if item_path.is_file():
                         item_path.unlink()
                print(f"Cleared base files in session directory: {self.session_dir}")
            except OSError as e:
                print(f"Error clearing base files in session directory {self.session_dir}: {e}")

    def cleanup(self):
        """Completely remove temporary directory and clear cache for this session."""
        print(f"Cleaning up session: {self.session_id}")
        self.data_cache.clear()
        self._cache_order.clear() # Clear order on cleanup
        if self.session_dir.exists():
            try:
                shutil.rmtree(self.session_dir)
                print(f"Removed session directory: {self.session_dir}")
            except OSError as e:
                print(f"Error removing session directory {self.session_dir}: {e}")

class InMemorySessionManager:
    """Manages user sessions and their associated data in memory."""
    def __init__(self, base_temp_dir: Path):
        self.base_temp_dir = base_temp_dir
        self._sessions: Dict[str, SessionContext] = {}

    def create_session(self) -> SessionContext:
        """Creates a new session with a unique ID and directory."""
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
        with _lock:
            return self._sessions.get(session_id)

    def get_or_create_session(self, session_id: str) -> SessionContext:
        """Gets a session by ID or creates it if it doesn't exist. Resets if it exists."""
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

    # --- Modified: Add method to expose get_most_recent_cache_item --- 
    def get_most_recent_cached_data(self, session_id: str) -> Optional[Tuple[str, pd.DataFrame]]:
         session = self.get_session(session_id)
         if not session:
             print(f"Error retrieving most recent cached data: Session {session_id} not found.")
             return None
         # Lock needed here if accessing session's internal state
         with _lock: 
             return session.get_most_recent_cache_item()
    # --- End Modified --- 

    # --- Convenience methods for data caching --- 

    def cache_data(self, session_id: str, df: pd.DataFrame) -> Optional[str]:
        """Caches a DataFrame for a session and returns a data_id."""
        session = self.get_session(session_id)
        if not session:
            print(f"Error caching data: Session {session_id} not found.")
            return None
        
        data_id = f"data_{uuid.uuid4()}"
        with _lock: # Lock specifically for session cache modification
            session.add_to_cache(data_id, df)
        print(f"Cached data for session {session_id} with id {data_id} ({len(df)} rows)")
        return data_id

    def get_cached_data(self, session_id: str, data_id: str) -> Optional[pd.DataFrame]:
        """Retrieves a cached DataFrame for a session."""
        session = self.get_session(session_id)
        if not session:
            print(f"Error retrieving cached data: Session {session_id} not found.")
            return None
            
        with _lock: # Lock specifically for session cache access
            df = session.get_from_cache(data_id)
            
        if df is None:
            print(f"Error retrieving cached data: Data ID {data_id} not found in session {session_id}.")
        return df 