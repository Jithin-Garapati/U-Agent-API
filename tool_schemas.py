# Define tool schemas
TOOL_SCHEMAS = [
    {
        "type": "function",
        "function": {
            "name": "dynamic_param_search",
            "description": "Search for dynamic parameters from flight logs that match the query. Use this for finding data recorded during flight.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "The query to search for in the flight logs",
                    }
                },
                "required": ["query"],
            },
        }
    },
    {
        "type": "function",
        "function": {
            "name": "static_param_search",
            "description": (
                "Search for static parameters from configuration files that match the query. "
                "Returns the most relevant parameters with their name, description, value, min/max range, and unit if available."
            ),
            "parameters": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "The query to search for in the configuration files (e.g., 'battery failsafe', 'max speed')",
                }
            },
            "required": ["query"],
        },
    }
    },
    {
        "type": "function",
        "function": {
            "name": "static_param_lookup",
            "description": "Look up the full static parameter row by exact parameter name from static_parameters.csv.",
            "parameters": {
                "type": "object",
                "properties": {
                    "param_name": {
                        "type": "string",
                        "description": "The exact name of the static parameter to look up (e.g., 'MPC_XY_CRUISE')"
                    }
                },
                "required": ["param_name"]
            },
        }
    },
    {
        "type": "function",
        "function": {
            "name": "topic_fields",
            "description": "List all fields in a specific topic to see what data is available.",
            "parameters": {
                "type": "object",
                "properties": {
                    "topic_name": {
                        "type": "string",
                        "description": "The name of the topic to get fields from",
                    }
                },
                "required": ["topic_name"],
            },
        }
    },
    {
        "type": "function",
        "function": {
            "name": "topic_data",
            "description": "Retrieve data from a topic with minimal processing. ALWAYS specify the exact fields you need.",
            "parameters": {
                "type": "object",
                "properties": {
                    "topic_name": {
                        "type": "string",
                        "description": "The name of the topic to get data from",
                    },
                    "fields": {
                        "type": "array",
                        "items": {
                            "type": "string"
                        },
                        "description": "List of field names to retrieve (specify exact fields for better performance)",
                    },
                    "file_path": {
                        "type": "string",
                        "description": "Optional direct path to the CSV file if known (e.g., from dynamic_param_search results)",
                    }
                },
                "required": ["topic_name"],
            },
        }
    },
    {
        "type": "function",
        "function": {
            "name": "computation",
            "description": "Run arbitrary computation on data retrieved from topic_data.",
            "parameters": {
                "type": "object",
                "properties": {
                    "data_id": {
                        "type": "string",
                        "description": "ID of the data to run computation on (returned by topic_data)"
                    },
                    "computation_expr": {
                        "type": "string",
                        "description": "Python expression to evaluate on the data (e.g., 'max(df[\"altitude\"])')"
                    },
                    "comment": {
                        "type": "string",
                        "description": "Brief comment explaining what you're calculating"
                    },
                    "data": {
                        "type": "object",
                        "description": "DEPRECATED - Use data_id instead. Data to run the computation on (usually from topic_data)",
                        "properties": {
                            "dummy": {
                                "type": "string",
                                "description": "Deprecated parameter. Use data_id instead."
                            }
                        }
                    }
                },
                "required": ["computation_expr"],
            },
        }
    }
] 