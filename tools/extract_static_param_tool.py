"""
Static Parameter Extraction Tool

This module provides functionality to extract and query static parameters from
configuration files like vehicle parameter settings, constraints, and limits.

Usage:
    from tools.extract_static_param_tool import StaticParameterTool
    
    # Initialize the tool
    sp_tool = StaticParameterTool(csv_path="path/to/static_parameters.csv", ulog_file="path/to/file.ulg")
    
    # Query for parameters
    results = sp_tool.query("What is the maximum allowed speed?")
    
    # Access the results
    for param in results:
        print(f"Parameter: {param['name']}")
        print(f"Description: {param['longDesc']}")
        print(f"Value: {param['value']} {param['unit']}")
        print(f"Score: {param['score']}")
        print()
"""

import os
import re
import csv
from typing import Dict, List, Tuple, Any, Set, Optional
import pandas as pd
from difflib import SequenceMatcher
from sentence_transformers import SentenceTransformer
import numpy as np
from pyulog import ULog  # Import ULog for parsing ULog files

class StaticParameterTool:
    """Tool for extracting and querying static parameters from configuration files."""
    
    def __init__(self, csv_path: str = "../static_parameters.csv", ulog_file: str = None):
        """
        Initialize the Static Parameter Tool.
        
        Args:
            csv_path: Path to the static parameters CSV file
            ulog_file: Path to the ULog file for intersection (optional)
        """
        self.csv_path = csv_path
        self.ulog_file = ulog_file
        self.static_params = None
        self.ulog_params = None
        self.intersected_params = None
        self.model = SentenceTransformer("all-MiniLM-L6-v2")
        
        # Load the static parameters from CSV
        self._load_static_parameters()
        
        # Load parameters from ULog file if provided
        if ulog_file:
            self._load_ulog_parameters()
            
        # Calculate the intersection if both CSV and ULog are loaded
        if self.static_params and self.ulog_params:
            self._calculate_intersection()
    
    def _load_static_parameters(self):
        """Load static parameters from the CSV file."""
        try:
            print(f"Loading static parameters from {self.csv_path}...")
            
            # Load the CSV using pandas for easier handling
            df = pd.read_csv(self.csv_path)
            
            # Convert to dictionary format for easier access
            self.static_params = {}
            
            for index, row in df.iterrows():
                param_id = str(row.get('id', index))
                
                # Convert row to dictionary with consistent keys (handle both camelCase and snake_case)
                # Note: pandas Series.get() method is used rather than dict.get()
                param_data = {
                    'id': param_id,
                    'name': row['name'] if 'name' in row else '',
                    'group': row['group'] if 'group' in row else '',
                    'value': row['default'] if 'default' in row else '',
                    'type': row['type'] if 'type' in row else '',
                    'min': row['min'] if 'min' in row else '',
                    'max': row['max'] if 'max' in row else '',
                    'unit': row['units'] if 'units' in row else '',
                    'shortDesc': row['shortDesc'] if 'shortDesc' in row else '',
                    'longDesc': row['longDesc'] if 'longDesc' in row else (row['shortDesc'] if 'shortDesc' in row else ''),
                    'decimal': row['decimalPlaces'] if 'decimalPlaces' in row else 0,
                    'increment': row['increment'] if 'increment' in row else 0,
                    'reboot_required': row['rebootRequired'] if 'rebootRequired' in row else False,
                    'category': self._determine_parameter_category(row)
                }
                
                self.static_params[param_id] = param_data
            
            print(f"Loaded {len(self.static_params)} static parameters from CSV")
            
        except Exception as e:
            print(f"Error loading static parameters from CSV: {e}")
            self.static_params = {}
            
    def _load_ulog_parameters(self):
        """Load static parameters from the ULog file."""
        try:
            print(f"Loading parameters from ULog file: {self.ulog_file}")
            
            if not os.path.exists(self.ulog_file):
                print(f"ULog file not found: {self.ulog_file}")
                self.ulog_params = {}
                return
                
            # Load the ULog file
            ulog = ULog(self.ulog_file)
            
            # Extract the initial_parameters dictionary
            self.ulog_params = {}
            
            # Add each parameter to our dictionary
            for param_name, param_value in ulog.initial_parameters.items():
                self.ulog_params[param_name] = {
                    'name': param_name,
                    'value': param_value,
                    'access_info': f"ulog.initial_parameters['{param_name}']"
                }
                
            print(f"Loaded {len(self.ulog_params)} parameters from ULog file")
            
        except Exception as e:
            print(f"Error loading parameters from ULog file: {e}")
            self.ulog_params = {}
            
    def _calculate_intersection(self):
        """Calculate the intersection between static parameters CSV and ULog parameters."""
        if not self.static_params or not self.ulog_params:
            print("Cannot calculate intersection: missing CSV or ULog parameters")
            self.intersected_params = {}
            return
            
        self.intersected_params = {}
        
        # Create a mapping from parameter name to parameter ID for easier lookup
        static_param_name_to_id = {}
        for param_id, param_data in self.static_params.items():
            param_name = param_data['name']
            static_param_name_to_id[param_name] = param_id
        
        # Find parameters that exist in both CSV and ULog
        for ulog_param_name, ulog_param_data in self.ulog_params.items():
            # Check if this parameter exists in the static parameters
            if ulog_param_name in static_param_name_to_id:
                # Get the static parameter data
                param_id = static_param_name_to_id[ulog_param_name]
                static_param_data = self.static_params[param_id]
                
                # Combine data from both sources
                combined_data = static_param_data.copy()
                # Override the value with the actual value from ULog
                combined_data['value'] = ulog_param_data['value']
                # Add ULog access information
                combined_data['access_info'] = ulog_param_data['access_info']
                combined_data['source'] = 'intersection'
                
                # Add to intersected parameters
                self.intersected_params[param_id] = combined_data
                
        print(f"Found {len(self.intersected_params)} parameters in both CSV and ULog")
    
    def _determine_parameter_category(self, param_row):
        """
        Determine the category of a parameter based on its name, group, and description.
        Used for more targeted retrieval.
        """
        # Get relevant fields for categorization
        name = str(param_row['name'] if 'name' in param_row else '').lower()
        group = str(param_row['group'] if 'group' in param_row else '').lower()
        short_desc = str(param_row['shortDesc'] if 'shortDesc' in param_row else '').lower()
        long_desc = str(param_row['longDesc'] if 'longDesc' in param_row else '').lower()
        
        # Combine all text for category detection
        all_text = f"{name} {group} {short_desc} {long_desc}"
        
        # Define category keywords
        categories = {
            'speed': ['speed', 'velocity', 'how fast', 'pace', 'rate', 'knots', 'mph', 'kph',
                'meters per second', 'm/s', 'ft/s', 'acceleration', 'deceleration', 'thrust'
            ],
            'altitude': ['altitude', 'height', 'elevation', 'how high', 'ceiling', 'agl', 'amsl', 
                'flight level', 'vertical'
            ],
            'position': ['position', 'location', 'coordinates', 'where', 'place', 'lat', 'latitude', 
                'lon', 'longitude', 'gps', 'coordinate'
            ],
            'attitude': ['attitude', 'orientation', 'rotation', 'angle', 'tilt', 'pitch', 'roll', 'yaw',
                'heading', 'direction'
            ],
            'battery': ['battery', 'power', 'charge', 'energy', 'fuel', 'voltage', 'current', 'capacity',
                'drain', 'consumption', 'percentage', 'level'
            ],
            'motor': ['motor', 'propeller', 'rotor', 'rpm', 'thrust', 'esc', 'pwm'],
            'sensor': ['sensor', 'imu', 'gyro', 'accelerometer', 'magnetometer', 'barometer', 'camera'],
            'communication': ['communication', 'telemetry', 'radio', 'link', 'rc', 'protocol', 'mavlink'],
            'safety': ['safety', 'safe', 'protection', 'security', 'safeguard', 'emergency', 'failsafe',
                'abort', 'rescue', 'recover', 'protect', 'secure', 'precaution'
            ],
            'navigation': ['navigation', 'waypoint', 'mission', 'route', 'path', 'trajectory', 'autopilot'],
            'control': ['control', 'pid', 'tuning', 'gain', 'response', 'damping', 'filter', 'slew'],
            'performance': ['performance', 'efficiency', 'optimization', 'limit', 'constraint', 'maximum', 'minimum']
        }
        
        # Look for category keywords in all text
        matched_categories = []
        for category, keywords in categories.items():
            if any(keyword in all_text for keyword in keywords):
                matched_categories.append(category)
        
        # If multiple categories match, use the one with the most keyword matches
        if len(matched_categories) > 1:
            counts = {}
            for category in matched_categories:
                count = sum(1 for keyword in categories[category] if keyword in all_text)
                counts[category] = count
            
            # Return the category with the most matches
            return max(counts, key=counts.get)
        
        # Return the single matched category or 'general' if none matched
        return matched_categories[0] if matched_categories else 'general'
    
    def _extract_query_keywords(self, query: str) -> List[str]:
        """Extract keywords from user query to identify query intent."""
        query = query.lower()
        
        # Define comprehensive keyword categories with expanded terms
        keyword_categories = {
            'limit': [
                'limit', 'maximum', 'minimum', 'max', 'min', 'threshold', 'boundary', 'ceiling', 
                'floor', 'cap', 'restriction', 'constraint', 'bound', 'allowable', 'allowed', 
                'permissible', 'limited', 'highest', 'lowest', 'upper', 'lower'
            ],
            'config': [
                'config', 'configuration', 'setting', 'parameter', 'option', 'preference',
                'setup', 'arrangement', 'specification', 'value', 'default', 'preset'
            ],
            'speed': [
                'speed', 'velocity', 'how fast', 'pace', 'rate', 'knots', 'mph', 'kph',
                'meters per second', 'm/s', 'ft/s', 'acceleration', 'deceleration', 'thrust'
            ],
            'altitude': [
                'altitude', 'height', 'elevation', 'how high', 'ceiling', 'agl', 'amsl', 
                'flight level', 'vertical'
            ],
            'position': [
                'position', 'location', 'coordinates', 'where', 'place', 'lat', 'latitude', 
                'lon', 'longitude', 'gps', 'coordinate'
            ],
            'attitude': [
                'attitude', 'orientation', 'rotation', 'angle', 'tilt', 'pitch', 'roll', 'yaw',
                'heading', 'direction'
            ],
            'battery': [
                'battery', 'power', 'charge', 'energy', 'fuel', 'voltage', 'current', 'capacity',
                'drain', 'consumption', 'percentage', 'level'
            ],
            'safety': [
                'safety', 'safe', 'protection', 'security', 'safeguard', 'emergency', 'failsafe',
                'abort', 'rescue', 'recover', 'protect', 'secure', 'precaution'
            ],
            'requirement': [
                'require', 'requirement', 'needed', 'necessary', 'essential', 'mandatory',
                'obligatory', 'compulsory', 'prerequisite', 'required', 'must', 'should'
            ],
            'access': [
                'access', 'how to get', 'how to access', 'retrieve', 'get', 'find', 'locate',
                'read', 'obtain', 'use', 'using', 'way to', 'method to'
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
    
    def _retrieve_static_candidates(self, query: str, top_k: int = 5) -> List[Tuple[str, float]]:
        """Retrieve and rank static parameter candidates for a given query."""
        # Determine which parameter set to use
        param_set = self.intersected_params if self.intersected_params else self.static_params
        
        if not param_set:
            print("Parameters not loaded. Make sure parameters are loaded first.")
            return []
        
        # Encode the query
        query_embedding = self.model.encode(query.strip())
        
        # Extract query keywords for boosting
        query_keywords = self._extract_query_keywords(query.lower())
        
        # Calculate cosine similarity
        def cosine_similarity(vec1, vec2):
            return float(np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2) + 1e-10))
        
        # Score each parameter
        candidate_scores = []
        for param_id, param in param_set.items():
            # Combine relevant text for semantic matching
            param_name = str(param['name']) if 'name' in param else ''
            param_group = str(param['group']) if 'group' in param else ''
            param_short_desc = str(param['shortDesc']) if 'shortDesc' in param else ''
            param_long_desc = str(param['longDesc']) if 'longDesc' in param else ''
            
            param_text = f"{param_name} {param_group} {param_short_desc} {param_long_desc}"
            
            # Get semantic similarity
            param_embedding = self.model.encode(param_text)
            similarity = cosine_similarity(query_embedding, param_embedding)
            
            # Initialize boosted score
            boosted_score = similarity
            
            # Apply category-based boosting
            if 'speed' in query_keywords and param['category'] == 'speed':
                boosted_score += 0.2
            elif 'altitude' in query_keywords and param['category'] == 'altitude':
                boosted_score += 0.2
            elif 'position' in query_keywords and param['category'] == 'position':
                boosted_score += 0.2
            elif 'attitude' in query_keywords and param['category'] == 'attitude':
                boosted_score += 0.2
            elif 'battery' in query_keywords and param['category'] == 'battery':
                boosted_score += 0.2
            elif 'safety' in query_keywords and param['category'] == 'safety':
                boosted_score += 0.2
            
            # Boost parameters with limits for "maximum"/"minimum" queries
            if ('max' in param and param['max'] != '' and param['max'] is not None) and 'limit' in query_keywords and any(term in query.lower() for term in ['max', 'maximum', 'highest', 'upper']):
                boosted_score += 0.15
            elif ('min' in param and param['min'] != '' and param['min'] is not None) and 'limit' in query_keywords and any(term in query.lower() for term in ['min', 'minimum', 'lowest', 'lower']):
                boosted_score += 0.15
            
            # Apply field-level boosts for specific query types
            if 'config' in query_keywords and isinstance(param['shortDesc'], str) and param['shortDesc'] != '':
                boosted_score += 0.1
                
            # Boost parameters with access information for access-related queries
            if 'access' in query_keywords and 'access_info' in param:
                boosted_score += 0.25
            
            # Record the final score
            candidate_scores.append((param_id, boosted_score))
        
        # Sort by score descending and return top-k
        sorted_candidates = sorted(candidate_scores, key=lambda x: x[1], reverse=True)
        return sorted_candidates[:top_k]
    
    def query(self, query_text: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """
        Query the static parameters using natural language.
        
        Args:
            query_text: The natural language query (e.g., "What is the maximum allowed speed?")
            top_k: Number of top results to return
            
        Returns:
            List of dictionaries containing parameter information
        """
        # Determine which parameter set to use
        param_set = self.intersected_params if self.intersected_params else self.static_params
        
        if not param_set:
            if not self.static_params:
                self._load_static_parameters()
            param_set = self.static_params
        
        # Get top candidates
        candidates = self._retrieve_static_candidates(query_text, top_k)
        
        # Format results
        results = []
        for param_id, score in candidates:
            if param_id in param_set:
                param = param_set[param_id]
                
                # Ensure all values are properly converted to strings for safety
                safe_param = {}
                for key, value in param.items():
                    if value is None:
                        safe_param[key] = ''
                    elif isinstance(value, (int, float)):
                        safe_param[key] = str(value) if not pd.isna(value) else ''
                    else:
                        safe_param[key] = str(value) if value else ''
                
                # Use shortDesc as fallback for empty longDesc
                long_desc = safe_param['longDesc'] if safe_param['longDesc'] else safe_param['shortDesc']
                if not long_desc:
                    long_desc = "No description available"
                
                # Create a more usable result structure with consistent naming
                result = {
                    'id': param_id,
                    'name': safe_param['name'],
                    'group': safe_param['group'],
                    'value': param['value'],  # Keep original value type for calculations
                    'min': param['min'],      # Keep original value type for calculations
                    'max': param['max'],      # Keep original value type for calculations
                    'unit': safe_param['unit'],
                    'short_desc': safe_param['shortDesc'],
                    'long_desc': long_desc,
                    'category': safe_param['category'],
                    'score': score
                }
                
                # If this is a ULog parameter, add access info
                if 'access_info' in param:
                    result['access_info'] = param['access_info']
                    
                # Add to results
                results.append(result)
        
        return results

def SP(query: str) -> List[Dict[str, Any]]:
    """
    Utility function to search for static parameters based on a query.
    
    This function creates a StaticParameterTool instance that performs an 
    intersection between static parameters in CSV and those found in the ULog file.
    It returns parameters that exist in both sources, with descriptions from CSV
    and actual values from the ULog file.
    
    Args:
        query: The query text to search for static parameters
        
    Returns:
        A list of matching static parameters with their details
    """
    # Try various possible locations for the required files
    
    # Default relative paths
    default_csv_path = "../static_parameters.csv"
    default_ulog_file = CONFIG["files"].get("ulog_file", "flight_log.ulg") 
    
    # Current directory paths
    current_csv_path = "static_parameters.csv"
    
    # Environment paths if available
    env_csv_path = os.environ.get("STATIC_PARAMS_PATH")
    env_ulog_file = os.environ.get("ULOG_FILE_PATH")
    
    # Find the first existing CSV file
    csv_path = None
    for path in [env_csv_path, default_csv_path, current_csv_path]:
        if path and os.path.exists(path):
            csv_path = path
            break
    
    # Find the first existing ULog file
    ulog_file = None
    for path in [env_ulog_file, default_ulog_file]:
        if path and os.path.exists(path):
            ulog_file = path
            break
    
    if not csv_path:
        print("Warning: static_parameters.csv not found in any location")
        return []
    
    # Initialize the tool with the found paths
    print(f"Using static parameters file: {csv_path}")
    
    if ulog_file:
        print(f"Using ULog file for intersection: {ulog_file}")
        sp_tool = StaticParameterTool(csv_path=csv_path, ulog_file=ulog_file)
    else:
        print("Warning: No ULog file found. Falling back to CSV-only mode without intersection.")
        sp_tool = StaticParameterTool(csv_path=csv_path)
    
    # Query and return results
    return sp_tool.query(query)




def convert_to_python_types(data: dict) -> dict:
    """Convert NumPy types in a dict to plain Python types for JSON serialization."""
    converted = {}
    for k, v in data.items():
        if isinstance(v, (np.integer, np.int64)):
            converted[k] = int(v)
        elif isinstance(v, (np.floating, np.float32, np.float64)):
            converted[k] = float(v)
        elif isinstance(v, (np.bool_, bool)):
            converted[k] = bool(v)
        elif pd.isna(v):
            converted[k] = None
        else:
            converted[k] = v
    return converted

def get_param_by_name(param_name: str):
    """Get the full static parameter row by exact name from static_parameters.csv"""
    default_path = "../static_parameters.csv"
    current_dir_path = "static_parameters.csv"
    env_path = os.environ.get("STATIC_PARAMS_PATH")
    env_ulog_file = os.environ.get("ULOG_FILE_PATH")
    default_ulog_file = CONFIG["files"].get("ulog_file", "flight_log.ulg")

    # Resolve CSV path
    csv_path = None
    for path in [env_path, default_path, current_dir_path]:
        if path and os.path.exists(path):
            csv_path = path
            break

    # Resolve ULog file path
    ulog_file = None
    for path in [env_ulog_file, default_ulog_file]:
        if path and os.path.exists(path):
            ulog_file = path
            break

    if not csv_path:
        print("Warning: static_parameters.csv not found in any location")
        return None

    try:
        df = pd.read_csv(csv_path)
        param_row = df[df['name'].str.lower() == param_name.lower()]
        if param_row.empty:
            return None

        row = param_row.iloc[0]
        result = row.to_dict()

        # Try to get real-time value from ULog if available
        if ulog_file:
            try:
                ulog = ULog(ulog_file)
                if param_name in ulog.initial_parameters:
                    result['value'] = ulog.initial_parameters[param_name]
                    result['source'] = 'ulog'
                    result['access_info'] = f"ulog.initial_parameters['{param_name}']"
                else:
                    result['source'] = 'csv'
            except Exception as e:
                print(f"Error loading ULog file: {e}")
                result['source'] = 'csv'
        else:
            result['source'] = 'csv'

        return convert_to_python_types(result)

    except Exception as e:
        print(f"Error getting parameter by name: {e}")
        return None



if __name__ == "__main__":
    # Example query
    results = SP("What is the maximum allowed speed?")
    
    # Display results
    print("\nResults for: What is the maximum allowed speed?")
    print("=" * 60)
    
    for i, param in enumerate(results, 1):
        print(f"{i}. {param['name']} (Score: {param['score']:.4f})")
        print(f"   Group: {param['group']}")
        print(f"   Value: {param['value']} {param['unit']}")
        
        if param['min'] or param['max']:
            limits = []
            if param['min'] != '':
                limits.append(f"Min: {param['min']}")
            if param['max'] != '':
                limits.append(f"Max: {param['max']}")
            print(f"   Limits: {', '.join(limits)}")
        
        # Always display the description prominently
        print(f"   Description: {param['long_desc']}")
        
        if 'access_info' in param:
            print(f"   Access Info: {param['access_info']}")
        if 'source' in param:
            print(f"   Source: {param['source']}")
            
        print()