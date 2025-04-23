"""
Dynamic Parameter Extraction Tool

This module provides functionality to extract and query dynamic parameters from flight logs
using retrieval-augmented generation (RAG).

Usage:
    from tools.extract_dynamic_param_tool import DynamicParameterTool
    
    # Initialize the tool
    dp_tool = DynamicParameterTool(csv_dir="path/to/csv_topics", kb_file="path/to/knowledge_base.json")
    
    # Query for parameters
    results = dp_tool.query("What was the maximum speed?")
    
    # Access the results
    for param in results:
        print(f"Parameter: {param['name']}")
        print(f"Description: {param['description']}")
        print(f"Score: {param['score']}")
        print("Fields:", param['fields'])
        print()
"""

import os
import re
import json
import csv
from typing import Dict, List, Tuple, Any, Optional
from difflib import SequenceMatcher
from sentence_transformers import SentenceTransformer
import numpy as np
import collections

class DynamicParameterTool:
    """Tool for extracting and querying dynamic parameters from flight logs."""
    
    # Define motor-related synonyms
    MOTOR_SYNS = {"motor", "throttle", "esc", "actuator"}
    
    def __init__(self, csv_dir: str, kb_file: str = "formatted_knowledge_base.json"):
        """
        Initialize the Dynamic Parameter Tool.
        
        Args:
            csv_dir: Directory containing CSV files from flight logs
            kb_file: Path to the knowledge base JSON file
        """
        self.csv_dir = csv_dir
        self.kb_file = kb_file
        self.knowledge_base = None
        self.model = SentenceTransformer("all-MiniLM-L6-v2")
        
        # Build the knowledge base on initialization
        self._build_knowledge_base()
    
    def _build_knowledge_base(self):
        """Build the internal knowledge base from KB file and CSV files."""
        print("Building knowledge base from csv_topics and formatted_knowledge_base.json...")

        # Load the formatted knowledge base (contains predefined topics)
        formatted_kb = self._load_formatted_knowledge_base(self.kb_file)

        # Get mapping of raw CSV topic names to their corresponding filenames
        topic_to_files = self._get_csv_topic_to_files_mapping() # e.g., {'vehicle_status': ['...status_0.csv'], 'vehicle_status_flags': ['...flags_0.csv']}
        csv_topics_found = list(topic_to_files.keys())
        print(f"Found {len(csv_topics_found)} unique raw topics in {self.csv_dir}")

        # --- REVISED MAPPING ---
        # Map CSV topics to KB entries STRICTLY (prefer exact match)
        # Returns: kb_key_to_best_csv_topic, kb_key_to_confidence, kb_key_to_method, unmapped_csv_topics
        kb_key_to_best_csv_topic, kb_key_to_confidence, kb_key_to_method, unmapped_csv_topics = self._map_csv_topics_to_kb_strict(
            csv_topics_found, topic_to_files, formatted_kb
        )
        print(f"Mapped {len(kb_key_to_best_csv_topic)} KB keys to specific CSV topics.")
        print(f"{len(unmapped_csv_topics)} CSV topics remain unmapped (will be loaded dynamically).")
        # --- END REVISED MAPPING ---

        # Construct the knowledge base
        kb = {}

        # First, include entries from the formatted KB using the strict mapping
        for kb_key, csv_topic in kb_key_to_best_csv_topic.items():
            if kb_key in formatted_kb:
                kb_entry = formatted_kb[kb_key].copy()
                kb_entry['source'] = 'combined' # Mark as combined source

                # Add file paths using the uniquely mapped CSV topic
                if csv_topic in topic_to_files:
                    file_paths = [os.path.join(self.csv_dir, f) for f in topic_to_files[csv_topic]]
                    if file_paths:
                         # Select primary based *only* on files for this specific topic
                         primary_path = self._select_primary_file_simple(csv_topic, file_paths)
                         if primary_path:
                              kb_entry['file_path'] = primary_path
                              kb_entry['all_file_paths'] = file_paths
                              # Update fields ONLY from the primary file of the BEST matching CSV topic
                              self._update_fields_from_csv(kb_entry, primary_path)
                         else:
                              print(f"Warning: Could not select primary file for mapped CSV topic {csv_topic} for KB key {kb_key}")
                    else:
                         print(f"Warning: No files found for mapped CSV topic {csv_topic} for KB key {kb_key}")
                else:
                     print(f"Warning: CSV topic {csv_topic} mapped to KB key {kb_key} not found in topic_to_files")

                # Add the populated entry to the knowledge base
                kb[kb_key] = kb_entry
            else:
                 print(f"Warning: KB key {kb_key} from mapping not found in formatted_kb")


        # Now load dynamic topics from the list of UNMAPPED CSV topics
        dynamic_topics_data = self._load_dynamic_topics(unmapped_csv_topics, topic_to_files)

        # Add dynamic topics to the knowledge base, ensuring no key collision
        for topic, topic_info in dynamic_topics_data.items():
            # Generate a unique key for dynamic topics
            dynamic_key = f"dynamic_{topic}"
            if dynamic_key in kb:
                 print(f"Warning: Dynamic key collision for {dynamic_key}. Skipping.")
                 continue
            if 'file_path' in topic_info and os.path.exists(topic_info['file_path']):
                 kb[dynamic_key] = topic_info
            else:
                 print(f"Warning: Dynamic topic {topic} lacks a valid file path. Skipping.")


        # Final KB is built
        self.knowledge_base = kb # Directly use the built kb
        print(f"Built knowledge base with {len(self.knowledge_base)} entries")

        return self.knowledge_base
    
    def _load_formatted_knowledge_base(self, kb_filepath: str) -> Dict[str, Any]:
        """Load the formatted knowledge base from a JSON file."""
        try:
            with open(kb_filepath, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception as e:
            print(f"Error loading formatted knowledge base: {e}")
            return {}
    
    def _get_csv_topic_to_files_mapping(self) -> Dict[str, List[str]]:
        """Create a mapping of RAW CSV topics to their filenames."""
        topic_to_files = collections.defaultdict(list)
        try:
            for filename in os.listdir(self.csv_dir):
                 if filename.endswith('.csv'):
                     topic_name = self._parse_topic_from_filename(filename)
                     if topic_name:
                          topic_to_files[topic_name].append(filename)
        except Exception as e:
            print(f"Error building CSV topic-to-file mapping: {e}")
        # Sort file lists for consistency
        for topic in topic_to_files:
            topic_to_files[topic].sort()
        return dict(topic_to_files)

    def _parse_topic_from_filename(self, filename: str) -> Optional[str]:
        # Extract the true topic name, handling the "flight_log_" prefix
        if filename.startswith("flight_log_"):
            # Remove 'flight_log_' prefix
            name_parts = filename[11:].split('_')
            
            # Remove numeric suffix if present
            if name_parts and name_parts[-1].split('.')[0].isdigit():
                topic_name = '_'.join(name_parts[:-1])
            else:
                topic_name = '_'.join(name_parts).split('.')[0]
        else:
            # Handle normal CSV files without prefix
            name_parts = filename.split('_')
            if name_parts and name_parts[-1].split('.')[0].isdigit():
                topic_name = '_'.join(name_parts[:-1])
            else:
                topic_name = os.path.splitext(filename)[0]
        
        return topic_name

    def _string_similarity(self, a: str, b: str) -> float:
        """Calculate string similarity ratio between two strings."""
        return SequenceMatcher(None, a.lower(), b.lower()).ratio()
    
    def _find_best_match(self, csv_topic: str, kb: Dict[str, Any]) -> Tuple[Optional[str], float]:
        """Find the best matching knowledge base entry for a CSV topic."""
        best_match = None
        best_score = 0
        
        for kb_key, kb_value in kb.items():
            kb_name = kb_value.get('name', '').lower()
            
            # Try exact match first (ignoring case)
            if csv_topic.lower() == kb_name:
                return kb_key, 1.0
                
            # Try topic name vs KB name
            similarity = self._string_similarity(csv_topic, kb_name)
            if similarity > best_score:
                best_score = similarity
                best_match = kb_key
        
        # Return best match if confidence is high enough
        if best_score >= 0.6:  # Threshold for acceptable match
            return best_match, best_score
        
        return None, 0
    
    def _analyze_csv_header_fields(self, csv_file: str, kb: Dict[str, Any]) -> Optional[Tuple[str, str, float, int]]:
        """Analyze CSV headers to find potential matches based on field names."""
        try:
            with open(os.path.join(self.csv_dir, csv_file), 'r', encoding='utf-8') as f:
                reader = csv.reader(f)
                headers = next(reader, [])
            
            if not headers:
                return None
            
            csv_fields = set(h.lower() for h in headers)
            
            best_match = None
            best_score = 0
            best_common = 0
            best_kb_name = ""
            
            for kb_key, kb_value in kb.items():
                if 'fields' not in kb_value:
                    continue
                    
                kb_fields = set(field.lower() for field in kb_value['fields'].keys())
                
                if not kb_fields:
                    continue
                
                # Count common fields
                common_fields = len(csv_fields.intersection(kb_fields))
                
                # Calculate field similarity as percentage of common fields
                if common_fields > 0:
                    field_similarity = common_fields / min(len(csv_fields), len(kb_fields))
                    
                    if field_similarity > best_score or (field_similarity == best_score and common_fields > best_common):
                        best_score = field_similarity
                        best_match = kb_key
                        best_common = common_fields
                        best_kb_name = kb_value.get('name', kb_key)
            
            # Return best match if score is good enough and we have enough common fields
            if best_score >= 0.4 and best_common >= 3:
                return best_match, best_kb_name, best_score, best_common
            
        except Exception as e:
            print(f"Error analyzing CSV headers: {e}")
        
        return None
    
    def _map_csv_topics_to_kb_strict(self,
                                     csv_topics: List[str],
                                     topic_to_files: Dict[str, List[str]],
                                     kb: Dict[str, Any]) -> Tuple[Dict[str, str], Dict[str, float], Dict[str, str], List[str]]:
        """
        Strictly map CSV topics to knowledge base entries.
        Ensures each KB key maps to at most ONE csv_topic. Prioritizes exact matches.

        Returns:
            kb_key_to_best_csv_topic: Mapping from KB key to the best matching CSV topic name.
            kb_key_to_confidence: Confidence score for each mapping.
            kb_key_to_method: Method used for each mapping ('exact', 'similarity', 'header').
            unmapped_csv_topics: List of CSV topics that couldn't be mapped confidently.
        """
        kb_key_to_best_csv_topic = {}
        kb_key_to_confidence = {}
        kb_key_to_method = {}
        mapped_csv_topics = set()

        # --- Pass 1: Exact Name Matching ---
        print("Mapping KB Keys: Pass 1 (Exact Name Match)")
        kb_names_to_keys = {details.get('name', '').lower(): key for key, details in kb.items() if details.get('name')}
        for csv_topic in csv_topics:
            csv_topic_lower = csv_topic.lower()
            if csv_topic_lower in kb_names_to_keys:
                kb_key = kb_names_to_keys[csv_topic_lower]
                # Check if this KB key is already mapped better
                if kb_key not in kb_key_to_best_csv_topic or kb_key_to_confidence[kb_key] < 1.0:
                     kb_key_to_best_csv_topic[kb_key] = csv_topic
                     kb_key_to_confidence[kb_key] = 1.0
                     kb_key_to_method[kb_key] = 'exact'
                     mapped_csv_topics.add(csv_topic)
                     # print(f"  Exact match: CSV '{csv_topic}' -> KB Key '{kb_key}'")


        # --- Pass 2: Similarity / Header Analysis (Optional, High Threshold) ---
        # Only consider topics not yet mapped
        remaining_csv_topics = [t for t in csv_topics if t not in mapped_csv_topics]
        # Only consider KB keys not yet mapped exactly
        remaining_kb_keys = [k for k in kb.keys() if k not in kb_key_to_best_csv_topic]

        print(f"Mapping KB Keys: Pass 2 (Similarity/Header on {len(remaining_csv_topics)} topics, {len(remaining_kb_keys)} keys)")
        SIMILARITY_THRESHOLD = 0.9 # Example high threshold
        HEADER_CONFIDENCE_THRESHOLD = 0.8 # Example high threshold

        # Store potential matches temporarily to pick the best one per KB key later
        potential_matches = collections.defaultdict(list) # kb_key -> list of (confidence, method, csv_topic)

        for csv_topic in remaining_csv_topics:
            # Similarity check against remaining KB entries
            best_sim_kb_key, best_sim_score = self._find_best_match(csv_topic, {k: kb[k] for k in remaining_kb_keys})
            if best_sim_kb_key and best_sim_score >= SIMILARITY_THRESHOLD:
                 potential_matches[best_sim_kb_key].append((best_sim_score, 'similarity', csv_topic))
                 # print(f"  Potential similarity match: CSV '{csv_topic}' -> KB Key '{best_sim_kb_key}' (Score: {best_sim_score:.2f})")

            # Header analysis check against remaining KB entries
            files_for_topic = topic_to_files.get(csv_topic, [])
            if files_for_topic:
                 primary_file_for_topic = self._select_primary_file_simple(csv_topic, [os.path.join(self.csv_dir, f) for f in files_for_topic])
                 if primary_file_for_topic:
                      header_match_result = self._analyze_csv_header_fields(primary_file_for_topic, {k: kb[k] for k in remaining_kb_keys})
                      if header_match_result:
                           header_kb_key, _, header_confidence, _ = header_match_result
                           if header_confidence >= HEADER_CONFIDENCE_THRESHOLD:
                                potential_matches[header_kb_key].append((header_confidence, 'header', csv_topic))
                                # print(f"  Potential header match: CSV '{csv_topic}' -> KB Key '{header_kb_key}' (Score: {header_confidence:.2f})")


        # Resolve potential matches: pick the best match for each remaining KB key
        for kb_key in remaining_kb_keys:
             if kb_key in potential_matches:
                  # Sort matches by confidence descending, method preference (header > similarity?)
                  # For simplicity, just take highest confidence
                  potential_matches[kb_key].sort(key=lambda x: x[0], reverse=True)
                  best_confidence, best_method, best_csv_topic = potential_matches[kb_key][0]

                  # Ensure this CSV topic hasn't been mapped exactly already
                  if best_csv_topic not in mapped_csv_topics:
                       # Check if another CSV topic already non-exactly mapped to this kb_key with higher confidence
                       if kb_key not in kb_key_to_best_csv_topic or best_confidence > kb_key_to_confidence[kb_key]:
                            # If another csv mapped non-exactly previously, mark it as unmapped now
                            if kb_key in kb_key_to_best_csv_topic:
                                 old_csv_topic = kb_key_to_best_csv_topic[kb_key]
                                 if old_csv_topic not in mapped_csv_topics: # Check it wasn't an exact match originally
                                      pass # It will be added to unmapped later
                            
                            kb_key_to_best_csv_topic[kb_key] = best_csv_topic
                            kb_key_to_confidence[kb_key] = best_confidence
                            kb_key_to_method[kb_key] = best_method
                            mapped_csv_topics.add(best_csv_topic) # Mark this csv topic as mapped
                            # print(f"  Selected non-exact match: CSV '{best_csv_topic}' -> KB Key '{kb_key}' (Score: {best_confidence:.2f}, Method: {best_method})")


        # Final list of unmapped topics
        final_unmapped_csv_topics = [t for t in csv_topics if t not in mapped_csv_topics]

        return kb_key_to_best_csv_topic, kb_key_to_confidence, kb_key_to_method, final_unmapped_csv_topics
    
    def _select_primary_file_simple(self, topic: str, file_paths: List[str]) -> Optional[str]:
        """Select the primary file path - simply take the first one for now."""
        # ToDO: Implement more robust selection if needed (e.g., prefer _0 suffix)
        if not file_paths:
             return None
        # Simple: return the first path in the list (assuming lists are sorted)
        return file_paths[0]
    
    def _load_dynamic_topics(self,
                             dynamic_csv_topics: List[str],
                             topic_to_files: Dict[str, List[str]]) -> Dict[str, Dict[str, Any]]:
        """
        Load topics directly from CSV files for topics that were *not* mapped to the formatted KB.
        Args:
            dynamic_csv_topics: List of raw CSV topic names confirmed to be unmapped.
            topic_to_files: Mapping of raw topic names to filenames.
        """
        dynamic_topics = {}
        print(f"Loading {len(dynamic_csv_topics)} topics dynamically...")

        for topic in dynamic_csv_topics:
            if topic not in topic_to_files:
                print(f"Warning: Unmapped topic '{topic}' not found in topic_to_files. Skipping dynamic load.")
                continue

            files = topic_to_files[topic]
            all_paths = [os.path.join(self.csv_dir, f) for f in files]
            primary_path = self._select_primary_file_simple(topic, all_paths) # Use consistent selection

            if primary_path and os.path.exists(primary_path):
                topic_info = {
                    'name': topic, # Use the raw topic name
                    'description': f'Dynamically loaded data for {topic}', # Basic description
                    'fields': {}, # Placeholder, filled by _update_fields_from_csv
                    'file_path': primary_path,
                    'all_file_paths': all_paths,
                    'source': 'dynamic_csv'
                }
                # Extract fields from the primary CSV
                self._update_fields_from_csv(topic_info, primary_path)
                dynamic_topics[topic] = topic_info # Use topic name as key for dynamic entries
            else:
                print(f"Warning: No valid primary file found for dynamic topic '{topic}'. Skipping.")

        return dynamic_topics
    
    def _update_fields_from_csv(self, kb_entry: Dict[str, Any], csv_file_path: str) -> None:
        """
        Update the fields dictionary in the KB entry with all fields from CSV file.
        This ensures we have all actual fields available, not just those in knowledge_base.json.
        
        Args:
            kb_entry: The knowledge base entry to update
            csv_file_path: Path to the CSV file
        """
        try:
            with open(csv_file_path, 'r', encoding='utf-8') as f:
                reader = csv.reader(f)
                headers = next(reader, [])
                
                # Create fields dictionary if it doesn't exist
                if 'fields' not in kb_entry:
                    kb_entry['fields'] = {}
                
                # Update with all CSV headers
                for header in headers:
                    # Only add if not already present
                    if header not in kb_entry['fields']:
                        kb_entry['fields'][header] = {
                            'description': f"Field {header} in {kb_entry.get('name', 'topic')}"
                        }
        except Exception as e:
            print(f"Error updating fields from CSV file {csv_file_path}: {e}")
    
    def _extract_query_keywords(self, query: str) -> List[str]:
        """Extract keywords from user query to identify query type."""
        query = query.lower()
        
        # Define comprehensive keyword categories with expanded terms
        keyword_categories = {
            'altitude': [
                'altitude', 'height', 'elevation', 'how high', 'highest point', 'lowest point',
                'ceiling', 'agl', 'amsl', 'flight level', 'vertical', 'climb', 'descend', 'rising',
                'falling', 'hover', 'hovering', 'above ground', 'z-axis'
            ],
            'position': [
                'position', 'location', 'coordinates', 'where', 'place', 'spot', 'point',
                'lat', 'latitude', 'lon', 'longitude', 'gps', 'coordinate', 'xyz', 'waypoint',
                'origin', 'destination', 'path', 'trajectory', 'route', 'x-y', 'horizontal'
            ],
            'speed': [
                'speed', 'velocity', 'how fast', 'pace', 'rate', 'knots', 'mph', 'kph',
                'meters per second', 'm/s', 'ft/s', 'acceleration', 'deceleration', 'thrust',
                'rapid', 'quick', 'slow', 'fastest', 'slowest', 'cruising', 'sprint', 'dash'
            ],
            'attitude': [
                'attitude', 'orientation', 'rotation', 'angle', 'tilt', 'pitch', 'roll', 'yaw',
                'heading', 'direction', 'facing', 'turn', 'turning', 'rotate', 'flip', 'inverted',
                'upright', 'level', 'balanced', 'imbalanced', 'steady', 'unstable', 'wobble'
            ],
            'battery': [
                'battery', 'power', 'charge', 'energy', 'fuel', 'voltage', 'current', 'capacity',
                'drain', 'consumption', 'percentage', 'level', 'low battery', 'empty', 'full',
                'remaining', 'endurance', 'runtime', 'discharge', 'recharge'
            ],
            'distance': [
                'distance', 'range', 'how far', 'proximity', 'separation', 'gap', 'span',
                'meters', 'kilometers', 'miles', 'feet', 'yards', 'reach', 'radius', 'perimeter'
            ],
            'time': [
                'time', 'duration', 'how long', 'period', 'interval', 'elapsed', 'seconds',
                'minutes', 'hours', 'timestamp', 'moment', 'instant', 'timing', 'clock',
                'schedule', 'flight time', 'air time'
            ],
            'temperature': [
                'temperature', 'temp', 'heat', 'cooling', 'thermal', 'celsius', 'fahrenheit',
                'kelvin', 'degrees', 'warm', 'hot', 'cold', 'freeze', 'overheat'
            ],
            'pressure': [
                'pressure', 'barometer', 'barometric', 'atmospheric', 'pascal', 'psi',
                'air pressure', 'ambient', 'compression', 'vacuum', 'suction'
            ],
            'wind': [
                'wind', 'air flow', 'gust', 'breeze', 'headwind', 'tailwind', 'crosswind',
                'air speed', 'knots', 'turbulence', 'draft', 'airstream', 'wind direction'
            ],
            'command': [
                'command', 'instruction', 'order', 'directive', 'control', 'input', 'commanded',
                'requested', 'setpoint', 'target', 'goal', 'objective', 'mission', 'task',
                'assignment', 'prescription', 'parameter', 'setting', 'configuration'
            ],
            'error': [
                'error', 'mistake', 'fault', 'failure', 'issue', 'problem', 'glitch', 'bug',
                'deviation', 'discrepancy', 'difference', 'offset', 'variance', 'anomaly',
                'irregularity', 'malfunction', 'incorrect', 'wrong', 'inaccurate', 'mismatch'
            ],
            'status': [
                'status', 'state', 'condition', 'health', 'diagnostics', 'mode', 'phase',
                'stage', 'step', 'progress', 'operation', 'functioning', 'working', 'active',
                'inactive', 'standby', 'ready', 'busy', 'idle', 'operational'
            ]
        }
        
        found_keywords = []
        for category, terms in keyword_categories.items():
            for term in terms:
                # Use word boundary for single words, otherwise exact match
                if len(term.split()) == 1:
                    pattern = r'\b' + re.escape(term) + r'\b'
                    if re.search(pattern, query):
                        found_keywords.append(category)
                        break
                elif term in query:
                    found_keywords.append(category)
                    break
        
        # Make sure we return at least 'general' if no specific category matched
        return found_keywords if found_keywords else ['general']
    
    def _retrieve_candidates(self, query: str, top_k: int = 5) -> List[Tuple[str, float]]:
        """Retrieve candidate entries from knowledge base using semantic search and boosting."""
        
        query_keywords = self._extract_query_keywords(query)
        
        # --- Motor Query Augmentation ---
        is_motor_query = bool(self.MOTOR_SYNS.intersection(query_keywords))
        augmented_query = query # Start with original query
        augmentation_applied_log = False # For logging within this method
        if is_motor_query:
            print(f"MOTOR_QUERY_LOG: Motor query detected (keywords: {self.MOTOR_SYNS.intersection(query_keywords)}). Augmenting query.")
            augmented_query += " actuator_outputs"
            augmentation_applied_log = True
            # NOTE: Omitted disable_parent_topic_boost() as it doesn't map clearly.
        # --- End Motor Query Augmentation ---

        # Generate query embedding using potentially augmented query
        query_embedding = self.model.encode([augmented_query])
        
        # Ensure knowledge base is built
        if not self.knowledge_base:
            self._build_knowledge_base()
        
        # Calculate cosine similarity
        def cosine_similarity(vec1, vec2):
            return float(np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2) + 1e-10))
        
        # Score each candidate in the knowledge base
        candidate_scores = []
        for key, details in self.knowledge_base.items():
            # Basic context from description and group
            context = f"{details.get('description', '')} {details.get('group', '')} {details.get('type', '')}"
            
            # Calculate base similarity score
            base_score = cosine_similarity(query_embedding, self.model.encode(context))
            
            # Initialize boosted score
            boosted_score = base_score
            
            # Boost score based on field relevance
            if 'fields' in details:
                # Check if any fields are directly relevant to the query type
                fields_context = ""
                field_relevance_boost = 0.0
                
                for field_name, field_info in details.get('fields', {}).items():
                    field_desc = field_info.get('description', '').lower()
                    field_name_lower = field_name.lower()
                    
                    # Add field info to context
                    fields_context += f" {field_name} {field_desc}"
                    
                    # Apply per-category boosting
                    for keyword in query_keywords:
                        if keyword == 'altitude' and any(term in field_desc or term in field_name_lower 
                                                         for term in ['altitude', 'height', 'elevation', 'alt', 'z']):
                            field_relevance_boost = max(field_relevance_boost, 0.25)
                        
                        elif keyword == 'position' and any(term in field_desc or term in field_name_lower 
                                                           for term in ['position', 'location', 'coordinate', 'lat', 'lon', 'x', 'y', 'z']):
                            field_relevance_boost = max(field_relevance_boost, 0.25)
                        
                        elif keyword == 'speed' and any(term in field_desc or term in field_name_lower 
                                                        for term in ['speed', 'velocity', 'vel', 'vx', 'vy', 'vz']):
                            field_relevance_boost = max(field_relevance_boost, 0.25)
                        
                        elif keyword == 'attitude' and any(term in field_desc or term in field_name_lower 
                                                           for term in ['attitude', 'orientation', 'roll', 'pitch', 'yaw', 'angle']):
                            field_relevance_boost = max(field_relevance_boost, 0.25)
                        
                        # Add boosting for additional categories
                        elif keyword == 'battery' and any(term in field_desc or term in field_name_lower 
                                                          for term in ['battery', 'power', 'voltage', 'current', 'capacity']):
                            field_relevance_boost = max(field_relevance_boost, 0.25)
                            
                        elif keyword == 'distance' and any(term in field_desc or term in field_name_lower 
                                                           for term in ['distance', 'range', 'proximity', 'separation']):
                            field_relevance_boost = max(field_relevance_boost, 0.25)
                            
                        elif keyword == 'time' and any(term in field_desc or term in field_name_lower 
                                                       for term in ['time', 'duration', 'timestamp', 'elapsed']):
                            field_relevance_boost = max(field_relevance_boost, 0.25)
                            
                        elif keyword == 'temperature' and any(term in field_desc or term in field_name_lower 
                                                              for term in ['temperature', 'temp', 'heat', 'celsius']):
                            field_relevance_boost = max(field_relevance_boost, 0.25)
                            
                        elif keyword == 'pressure' and any(term in field_desc or term in field_name_lower 
                                                           for term in ['pressure', 'baro', 'atmospheric']):
                            field_relevance_boost = max(field_relevance_boost, 0.25)
                            
                        elif keyword == 'wind' and any(term in field_desc or term in field_name_lower 
                                                       for term in ['wind', 'airflow', 'gust', 'airspeed']):
                            field_relevance_boost = max(field_relevance_boost, 0.25)
                            
                        elif keyword == 'error' and any(term in field_desc or term in field_name_lower 
                                                        for term in ['error', 'fault', 'issue', 'deviation']):
                            field_relevance_boost = max(field_relevance_boost, 0.25)
                            
                        elif keyword == 'status' and any(term in field_desc or term in field_name_lower 
                                                         for term in ['status', 'state', 'mode', 'condition']):
                            field_relevance_boost = max(field_relevance_boost, 0.25)
                
                # Add fields context for secondary ranking
                if fields_context:
                    fields_similarity = cosine_similarity(query_embedding, self.model.encode(fields_context))
                    # Weighted average of base score and fields similarity
                    boosted_score = base_score * 0.7 + fields_similarity * 0.3
                
                # Apply direct field relevance boost
                boosted_score += field_relevance_boost
            
            # Data quality boost for dynamic sources
            if details.get('source') == 'dynamic':
                try:
                    # Data quality assessment based on row count
                    line_count = details.get('line_count', 0)
                    if isinstance(line_count, str) and line_count.startswith('>'):
                        # Very large datasets are reliable for most queries
                        boosted_score += 0.2
                    elif line_count > 20:
                        boosted_score += 0.1
                    # Penalize for very low data count - suggests incomplete data
                    elif line_count < 5:
                        boosted_score -= 0.2
                except Exception:
                    # Ignore errors in data quality assessment
                    pass
            
            # Boost primary sources (actual measurements)
            key_lower = key.lower()
            
            # Boost vehicle position/state sources - these are usually primary measurements
            if 'vehicle_' in key_lower or 'sensor_' in key_lower:
                boosted_score += 0.1
                
                # Extra boost for known primary sources for common queries
                if any(k in ['altitude', 'position'] for k in query_keywords) and any(term in key_lower for term in ['position', 'local', 'global', 'gps', 'baro']):
                    boosted_score += 0.15
                elif any(k in ['speed', 'velocity'] for k in query_keywords) and any(term in key_lower for term in ['velocity', 'speed', 'gps', 'odometry']):
                    boosted_score += 0.15
                elif any(k in ['attitude', 'orientation'] for k in query_keywords) and any(term in key_lower for term in ['attitude', 'angular', 'gyro']):
                    boosted_score += 0.15
            
            # Extra boost for dynamic topics for queries likely requiring real-time data
            if details.get('source') == 'dynamic':
                # Identify queries that benefit from dynamic data
                dynamic_data_keywords = ['altitude', 'position', 'speed', 'attitude', 'battery', 
                                        'temperature', 'pressure', 'wind', 'distance', 'time']
                
                if any(k in dynamic_data_keywords for k in query_keywords):
                    boosted_score += 0.1
                    
                    # Extra boost for error-related queries with dynamic data (flight anomalies)
                    if 'error' in query_keywords:
                        boosted_score += 0.15
            
            # Penalize command topics, setpoints, and secondary sources for measurement queries
            if any(term in key_lower for term in ['command', 'cmd', 'setpoint', 'transponder', 'radio', 'remote']):
                # These keywords typically need actual measurements not commands
                measurement_keywords = ['altitude', 'position', 'speed', 'attitude', 'temperature', 
                                       'pressure', 'battery', 'distance']
                
                if any(k in measurement_keywords for k in query_keywords):
                    boosted_score -= 0.15
                
                # Less penalization for status queries where commands might be relevant
                elif any(k in ['status', 'state', 'mode'] for k in query_keywords):
                    boosted_score -= 0.05
            
            # Record the final score
            candidate_scores.append((key, boosted_score))
        
        # Sort by score descending and return top-k
        sorted_candidates = sorted(candidate_scores, key=lambda x: x[1], reverse=True)
        
        # Log if augmentation was applied in this specific call
        if augmentation_applied_log:
             print(f"VISIBILITY_METRIC_LOG: Initial query augmented with 'actuator_outputs' for query: '{query}'")

        return sorted_candidates[:top_k]
    
    def query(self, query_text: str, top_k: int = 10) -> List[Dict[str, Any]]:
        """
        Query the dynamic parameters using natural language.
        
        Args:
            query_text: The natural language query (e.g., "What was the maximum speed?")
            top_k: Number of top results to return (default 10)
            
        Returns:
            List of dictionaries containing parameter information
        """
        if not self.knowledge_base:
            self._build_knowledge_base()
        
        # Get initial top candidates (fetch more to allow filtering)
        initial_candidates = self._retrieve_candidates(query_text, top_k * 2)

        # --- Auto-fallback logic ---
        final_candidates = initial_candidates # Assume initial results are used unless fallback happens
        scores = [score for _, score in initial_candidates]
        
        # Check if original query would have been augmented
        query_keywords_for_fallback = self._extract_query_keywords(query_text)
        is_motor_query_for_fallback = bool(self.MOTOR_SYNS.intersection(query_keywords_for_fallback))
        original_query_was_motor_augmented = is_motor_query_for_fallback 

        # Check if fallback is needed: 
        # 1. No candidates OR max score is low (< 0.4)
        # 2. AND the original query was NOT already augmented due to motor keywords
        if (not initial_candidates or (scores and max(scores) < 0.4)) and not original_query_was_motor_augmented:
             print(f"FALLBACK_LOG: Initial results poor (max score < 0.4 or none) and motor augmentation not initially applied. Auto-retrying with '+ actuator_outputs'.")
             fallback_query = query_text + " actuator_outputs"
             # Re-run candidate retrieval with the explicitly augmented query
             # Use original top_k * 2 for consistency
             final_candidates = self._retrieve_candidates(fallback_query, top_k * 2)
             # Log metric: Note that fallback happened
             print(f"VISIBILITY_METRIC_LOG: Fallback augmentation triggered for query: '{query_text}'")
        # --- End Auto-fallback logic ---

        # Format results using the final candidate list (either initial or fallback)
        results = []
        for key, score in final_candidates:
            if key in self.knowledge_base:
                entry = self.knowledge_base[key]
                
                # Skip entries without a valid file path
                if 'file_path' not in entry or not os.path.exists(entry['file_path']):
                    continue
                
                # Create a more usable result structure
                result = {
                    'name': entry.get('name', key),
                    'key': key,
                    'description': entry.get('description', ''),
                    'score': score,
                    'source': entry.get('source', 'unknown'),
                    'fields': entry.get('fields', {}),
                    'file_path': entry.get('file_path', '')
                }
                
                # Add all file paths if available
                if 'all_file_paths' in entry:
                    result['all_file_paths'] = entry['all_file_paths']
                
                results.append(result)
                
                # Stop once we have enough valid results
                if len(results) >= top_k:
                    break
        
        return results

def DP(query: str) -> List[Dict[str, Any]]:
    """
    Utility function to search for dynamic parameters based on a query.
    
    This is a convenience function that creates a DynamicParameterTool instance
    with default paths or attempts to find the files in the current directory.
    
    Args:
        query: The query text to search for dynamic parameters
        
    Returns:
        A list of matching dynamic parameters with their details
    """
    # Try various possible locations for the required files
    
    # Default relative paths
    default_csv_dir = "../csv_topics"
    default_kb_file = "../formatted_knowledge_base.json"
    
    # Current directory paths
    current_csv_dir = "csv_topics"
    current_kb_file = "formatted_knowledge_base.json"
    
    # Environment paths if available
    env_csv_dir = os.environ.get("CSV_TOPICS_DIR")
    env_kb_file = os.environ.get("KNOWLEDGE_BASE_FILE")
    
    # Find the first existing CSV directory
    csv_dir = None
    for dir_path in [env_csv_dir, default_csv_dir, current_csv_dir]:
        if dir_path and os.path.exists(dir_path) and os.path.isdir(dir_path):
            csv_dir = dir_path
            break
    
    # Find the first existing knowledge base file
    kb_file = None
    for file_path in [env_kb_file, default_kb_file, current_kb_file]:
        if file_path and os.path.exists(file_path):
            kb_file = file_path
            break
    
    if not csv_dir:
        print("Warning: csv_topics directory not found in any location")
        return []
    
    if not kb_file:
        print("Warning: formatted_knowledge_base.json not found in any location")
        # Still proceed as the tool can build a knowledge base from CSV files
    
    # Initialize the tool with the found paths
    print(f"Using CSV directory: {csv_dir}")
    print(f"Using knowledge base file: {kb_file or 'None (will build from scratch)'}")
    
    dp_tool = DynamicParameterTool(csv_dir=csv_dir, kb_file=kb_file or "formatted_knowledge_base.json")
    
    # No need to rebuild the knowledge base - it's already built in the constructor
    # dp_tool._build_knowledge_base()
    
    # Query and return results
    return dp_tool.query(query)

# Example usage
if __name__ == "__main__":
    # Example query
    results = DP("What was the maximum speed during the flight?")
    
    # Display results
    print("\nResults for: What was the maximum speed during the flight?")
    print("=" * 60)
    
    for i, param in enumerate(results, 1):
        print(f"{i}. {param['name']} (Score: {param['score']:.4f})")
        print(f"   Description: {param['description']}")
        print(f"   Key: {param['key']}")
        print("   Fields:")
        for field_name, field_info in list(param['fields'].items())[:5]:  # Show first 5 fields
            print(f"     - {field_name}: {field_info.get('description', '')}")
        if len(param['fields']) > 5:
            print(f"     ... and {len(param['fields']) - 5} more fields")
        print() 