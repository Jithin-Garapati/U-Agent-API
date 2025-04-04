import json
from constants import RED, YELLOW, BLUE, BOLD, RESET, GREEN

def format_tool_result_summary(tool_name, result):
    """Create a formatted summary of tool results for display."""
    summary = []
    
    if tool_name == 'submit_final_answer':
        # For final answer submissions
        summary.append(f"{BOLD}FINAL ANSWER:{RESET}")
        if "final_answer" in result:
            summary.append(result["final_answer"])
        
        if "explanation" in result:
            summary.append(f"\n{BOLD}How I arrived at this answer:{RESET}")
            summary.append(result["explanation"])
            
        return "\n".join(summary)
    
    elif tool_name in ['dynamic_param_search', 'static_param_search']:
        # For parameter search tools
        if not result:
            summary.append("No matching parameters found.")
            return "\n".join(summary)
            
        summary.append(f"Found {len(result)} matches.")
        summary.append("Top matches:")
        
        # Process and display the results
        for i, item in enumerate(result):
            param_name = item.get("name", "Unknown")
            
            # Different field names for different parameter search tools
            if tool_name == 'static_param_search':
                # For static parameters, use long_desc or short_desc
                description = item.get("long_desc", item.get("short_desc", "No description"))
                
                # Add value, min, max information if available
                value_info = []
                if 'value' in item and item['value'] not in ('', None):
                    value_info.append(f"Value: {item['value']}")
                if 'min' in item and item['min'] not in ('', None):
                    value_info.append(f"Min: {item['min']}")
                if 'max' in item and item['max'] not in ('', None):
                    value_info.append(f"Max: {item['max']}")
                if 'unit' in item and item['unit']:
                    value_info.append(f"Unit: {item['unit']}")
                
                value_str = ", ".join(value_info)
                
            else:  # dynamic_param_search
                # For dynamic parameters, use description field
                description = item.get("description", "No description")
            
            # Truncate description if it's too long to reduce display length
            if len(description) > 100:
                description = description[:100] + "..."
                
            # Display parameter with description
            summary.append(f"- {param_name}")
            summary.append(f"  Description: {description}")
            
            # Add value information for static parameters
            if tool_name == 'static_param_search' and value_str:
                summary.append(f"  {value_str}")
            
            # Show all field names as a comma-separated list
            if 'fields' in item and item['fields']:
                fields_str = ", ".join([str(f) for f in item['fields']])
                summary.append(f"  Fields: {fields_str}")
                
            # Show total field count if available
            if 'total_field_count' in item:
                total = item['total_field_count']
                if total > 0:
                    summary.append(f"  Total fields: {total}")
    
    elif tool_name == 'topic_fields':
        # For topic fields
        topic = result.get("topic", "Unknown")
        fields = result.get("fields", [])
        
        summary.append(f"Topic: {topic}")
        if len(fields) > 30:  # Truncate if there are too many fields
            display_fields = fields[:20]
            summary.append(f"Fields ({len(fields)}): {', '.join([str(f) for f in display_fields])} ... and {len(fields)-20} more")
        else:
            summary.append(f"Fields ({len(fields)}): {', '.join([str(f) for f in fields])}")
    
    elif tool_name == 'topic_data':
        # For topic data
        topic = result.get("topic", "Unknown")
        rows = result.get("rows", 0)
        columns = result.get("columns", [])
        data_id = result.get("data_id", "None")
        
        summary.append(f"Topic: {topic}")
        summary.append(f"Retrieved {rows} rows, {len(columns)} columns")
        
        # Make data_id very prominent for use with computation tool
        if data_id and data_id != "None":
            summary.append(f"\n{BOLD}DATA_ID: {GREEN}{data_id}{RESET} {BOLD}(use this ID for computation tool){RESET}")
        
        summary.append(f"Columns: {', '.join([str(col) for col in columns])}")
        
        # Data preview (first few rows)
        if "preview" in result and result["preview"]:
            summary.append("\nData preview (first 3 rows):")
            # Ensure preview is a string before appending
            if isinstance(result["preview"], list):
                summary.append("\n".join([str(item) for item in result["preview"]]))
            else:
                summary.append(str(result["preview"]))
        
        # Statistics (if available)
        if "statistics" in result and result["statistics"]:
            summary.append("\nBasic statistics:")
            # Ensure stats items are strings before appending
            for col_name, stats in list(result["statistics"].items())[:5]:  # Limit to first 5 columns
                if isinstance(stats, dict):
                    stat_str = ", ".join([f"{k}: {v}" for k, v in stats.items() if v is not None])
                    summary.append(f"- {col_name}: {stat_str}")
            
            if len(result["statistics"]) > 5:
                summary.append(f"... and {len(result['statistics'])-5} more columns")
        
        # Reminder to use data_id with computation tool
        if data_id and data_id != "None":
            summary.append(f"\n{BOLD}IMPORTANT:{RESET} To run computations on this data, use:")
            summary.append(f"{BOLD}computation{RESET} tool with {BOLD}data_id: \"{data_id}\"{RESET}")
    
    elif tool_name == 'computation':
        # For computation results
        if "success" in result and result["success"] == False:
            # Highlight errors in red
            summary.append(f"{RED}{BOLD}ERROR:{RESET} {result.get('error', 'Unknown error during computation')}")
        elif "result" in result:
            summary.append(f"Result: {result['result']}")
        
        # Remove detailed code information to prevent hallucination
        # Only mention that a computation was performed
        if "computation" in result:
            summary.append("Computation performed successfully.")
        
        # Add warning about empty or unexpected results
        if not result or (isinstance(result, dict) and not any(k in result for k in ["result", "success", "error"])):
            summary.append(f"{YELLOW}{BOLD}WARNING:{RESET} Computation returned no usable results. Verify data exists before computing.")
    else:
        # Generic summary for other tools
        # Ensure result is converted to string
        if isinstance(result, dict) and "error" in result:
            # Highlight errors in red
            summary.append(f"{RED}{BOLD}ERROR:{RESET} {result['error']}")
        else:
            summary.append(str(result))
    
    # Add validation warnings for empty results from key functions
    if (tool_name in ["topic_data", "dynamic_param_search"] and 
        (not result or (isinstance(result, dict) and 
                      ((tool_name == "topic_data" and result.get("rows", 0) == 0) or
                       (tool_name == "dynamic_param_search" and len(result) == 0))))):
        summary.append(f"{YELLOW}{BOLD}WARNING:{RESET} No results found. Try different search terms or check if the data exists.")
    
    # Ensure all items in summary are strings before joining
    summary = [str(item) if not isinstance(item, str) else item for item in summary]
    return "\n".join(summary)

def execute_tool_call(tool_call):
    """Execute a tool call and return the result."""
    # Import tool functions directly
    import importlib.util
    import sys
    
    # Load the tools.py module dynamically
    module_name = "tools_py"
    spec = importlib.util.spec_from_file_location(module_name, "tools.py")
    tools_module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = tools_module
    spec.loader.exec_module(tools_module)
    
    # Access the functions
    AVAILABLE_FUNCTIONS = {
        "dynamic_param_search": tools_module.dynamic_param_search,
        "static_param_search": tools_module.static_param_search,
        "static_param_lookup": tools_module.static_param_lookup,
        "topic_fields": tools_module.topic_fields,
        "topic_data": tools_module.topic_data,
        "computation": tools_module.computation,
        "submit_final_answer": tools_module.submit_final_answer
    }
    
    function_name = tool_call.function.name
    function_args = json.loads(tool_call.function.arguments)
    
    print(f"Executing tool: {function_name}")
    
    if function_name in AVAILABLE_FUNCTIONS:
        try:
            return AVAILABLE_FUNCTIONS[function_name](**function_args)
        except Exception as e:
            print(f"Error executing tool call: {str(e)}")
            return {"error": f"Error executing {function_name}: {str(e)}"}
    else:
        return {"error": f"Unknown function: {function_name}"} 