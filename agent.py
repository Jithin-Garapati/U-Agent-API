import time
import json
from groq import Groq
from typing import List, Dict, Any, Optional
import re

from constants import BLUE, RED, YELLOW, BOLD, GREY, RESET, GREEN
from token_management import truncate_message_history
from tool_schemas import TOOL_SCHEMAS
from helpers import format_tool_result_summary, execute_tool_call
from ulog_utils import preprocess_ulog
from config import CONFIG, GROQ_API_KEY, GEMINI_API_KEY
from gemini_client import GeminiClient

# Define the FunctionCall class for tool call execution
class FunctionCall:
    class Function:
        def __init__(self, name, arguments):
            self.name = name
            self.arguments = arguments
            
    def __init__(self, id, function_name, arguments):
        self.id = id
        self.function = self.Function(function_name, arguments)

class ULogReasoningAgent:
    """A simplified ULog agent that uses reasoning models with tool calling."""
    
    def __init__(self, api_key=None, model="qwen-qwq-32b", ulog_file=None):
        """Initialize the ULog agent."""
        self.model = model
        self.ulog_file = ulog_file
        
        # Get model-specific settings
        model_settings = CONFIG["agent"].get("model_settings", {}).get(self.model, {})
        
        # Determine the API provider based on model settings or default
        self.api_provider = model_settings.get("provider", CONFIG["agent"].get("api_provider", "groq"))
        
        # Initialize the appropriate client based on the provider
        if self.api_provider == "groq":
            self.api_key = api_key or GROQ_API_KEY
            self.client = Groq(api_key=self.api_key)
            print(f"{BLUE}Using Groq API with model: {self.model}{RESET}")
        elif self.api_provider == "gemini":
            self.api_key = api_key or GEMINI_API_KEY
            self.client = GeminiClient(api_key=self.api_key)
            print(f"{BLUE}Using Gemini API with model: {self.model}{RESET}")
        else:
            raise ValueError(f"Unsupported API provider: {self.api_provider}")
        
        # Track data retrieval to help with hallucination detection
        self.data_retrieval_history = {
            "topic_data_used": False,
            "data_id": None,
            "statistics": {}
        }
        
        # General system prompt with truly general data analysis principles
        self.system_prompt = """You are a helpful AI assistant that can analyze data and answer questions.

1. EXPLAIN what you're doing before using tools
2. USE TOOLS to find and retrieve data 
3. ALWAYS use computation tools for any calculations
4. NEVER present numerical results without using the computation tool
5. PROVIDE clear answers based on the data

Remember:
- Always wait for tool results before proceeding
- Never make up data or perform calculations yourself
- Provide accurate answers based only on the data you retrieve

GENERAL DATA ANALYSIS PRINCIPLES:
- Always consider three dimensions because this is about drone travelling in real world . 
- Begin with broad searches to understand what data is available in the system
- Use multiple search terms and synonyms to discover all relevant data sources
- For speed/velocity analysis, ALWAYS search for "position" first - speed data is derived from position information
- ALWAYS prioritize topics which provide accurate data over topics which rely on gps.
- For position and motion analysis, internal state estimation provides significantly higher accuracy than external sensing
- When no direct speed measurements are found, search for position-related metrics to derive speed
- Movement data is typically derived from position measurements - position topics are primary sources for all motion analysis
- when thinking of doing easy or hard stuff always choose to do hard stuff
- In flight data analysis, estimates from onboard computers typically provide higher accuracy than raw sensor readings
- Consider the measurement source and accuracy when selecting between similar data sources
- For time-series data, examine update frequencies to identify the most detailed sources
- Construct queries that focus on the nature of the question rather than assumed data structures
- If initial searches don't yield useful results, try reformulating with related concepts
- Contextual Knowledge Note (Orientation): When analyzing orientation data (e.g., from flight logs), interpret 'tilt angle' or 'total tilt' as the angle from the vertical, calculated using both roll (\\(\\phi\\)) and pitch (\\(\\theta\\)) via the formula \\( \\theta_{tilt} = \\arccos(\\cos(\\phi) \\cos(\\theta)) \\). Do not simply equate 'tilt' with 'pitch' unless explicitly requested.
"""
        
        # Automatically preprocess ULog file if enabled
        if CONFIG["agent"].get("auto_convert_ulog", True):
            print(f"\n{BLUE}{BOLD}Preprocessing ULog file...{RESET}")
            preprocess_ulog(ulog_file=self.ulog_file)
    
    def run(self, query, show_reasoning=True, stream=False):
        """Run the ULog agent to answer a query, letting the model naturally decide when it's done."""
        print(f"\n{BLUE}{BOLD}Running ULog agent with query: '{query}'{RESET}")
        
        # Set up the messages for the conversation
        messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": query}
        ]
        
        # Variables for API error handling
        retry_count = 0
        max_retries = 3
        wait_time = 2  # Initial wait time in seconds
        
        # For loop detection
        last_tool_calls = []
        tool_loop_count = 0
        max_iterations = 10
        iteration_count = 0
        
        # For hallucination detection
        calculation_tools_used = False
        topic_data_retrieved = False
        data_id_retrieved = None
        
        # For streaming responses
        response_chunks = []
        current_chunk = {"type": "thinking", "content": ""}
        
        # Main loop - continue until the model naturally stops calling tools
        while True:
            iteration_count += 1
            if iteration_count > max_iterations:
                print(f"{RED}Reached maximum iterations ({max_iterations}). Forcing final answer.{RESET}")
                # Force a final answer by adding a direct prompt
                messages.append({
                    "role": "user",
                    "content": "You've been working on this for a while. Based on the data you've gathered so far, please provide your final answer now."
                })
            
            try:
                print(f"\n{BLUE}{BOLD}Sending request to model (iteration {iteration_count}){RESET}")
                
                # Call the appropriate client based on API provider
                if self.api_provider == "groq":
                    response = self.client.chat.completions.create(
                        model=self.model,
                        messages=messages,
                        tools=TOOL_SCHEMAS,
                        tool_choice="auto",
                        temperature=CONFIG["agent"].get("temperature", 0.1),
                        max_tokens=CONFIG["agent"].get("max_tokens", 1024)
                    )
                    
                    # Process the response
                    response_content = response.choices[0].message.content or ""
                    tool_calls = response.choices[0].message.tool_calls or []
                    
                    # Add the response to messages
                    assistant_message = {"role": "assistant", "content": response_content}
                    if tool_calls:
                        assistant_message["tool_calls"] = tool_calls
                    
                    messages.append(assistant_message)
                    
                    # If streaming, add to chunks
                    if stream:
                        # Add thinking content first
                        if response_content.strip():
                            if current_chunk["content"]:
                                response_chunks.append(current_chunk)
                            current_chunk = {"type": "thinking", "content": response_content}
                    
                    # If no tool calls, the model is done - return the response
                    if not tool_calls:
                        print(f"{GREEN}Model completed its response with no additional tool calls.{RESET}")
                        
                        # If streaming, finalize response
                        if stream:
                            if current_chunk["content"]:
                                response_chunks.append(current_chunk)
                            return response_chunks
                        return response_content
                    
                    # Process tool calls
                    for tool_call in tool_calls:
                        result = self.process_tool_call(tool_call, messages, stream=stream, response_chunks=response_chunks)
                        
                        # If streaming, reset current chunk for next thinking
                        if stream and result:
                            current_chunk = {"type": "thinking", "content": ""}
                    
                elif self.api_provider == "gemini":
                    # For streaming response
                    print(f"\n{BLUE}Model response:{RESET}")
                    
                    # Initialize response tracking
                    response_content = ""
                    tool_calls = []
                    
                    # Get the response
                    api_stream = self.client.chat_completions_create(
                        model=self.model,
                        messages=messages,
                        tools=TOOL_SCHEMAS,
                        tool_choice="auto",
                        temperature=CONFIG["agent"].get("temperature", 0.1),
                        max_tokens=CONFIG["agent"].get("max_tokens", 1024),
                        stream=True
                    )
                    
                    # Process the stream
                    for chunk in api_stream:
                        delta = chunk.choices[0].delta
                        
                        # Handle content
                        if hasattr(delta, 'content') and delta.content:
                            response_content += delta.content
                            print(delta.content, end="", flush=True)
                            
                            # If streaming to client, add to current chunk
                            if stream:
                                current_chunk["content"] += delta.content
                                
                                # Check for embedded JSON in thinking and extract as proper tool results
                                self.extract_embedded_tool_results(current_chunk, response_chunks)
                        
                        # Handle tool calls
                        if hasattr(delta, 'tool_calls') and delta.tool_calls:
                            for tool_call_delta in delta.tool_calls:
                                # Get the tool call index
                                index = tool_call_delta.index
                                
                                # Ensure we have enough elements in our tool_calls list
                                while len(tool_calls) <= index:
                                    tool_calls.append({
                                        "id": None,
                                        "type": "function",
                                        "function": {"name": "", "arguments": ""}
                                    })
                                    
                                # Update id
                                if hasattr(tool_call_delta, 'id') and tool_call_delta.id:
                                    tool_calls[index]["id"] = tool_call_delta.id
                                    
                                # Update function name
                                if hasattr(tool_call_delta, 'function') and hasattr(tool_call_delta.function, 'name') and tool_call_delta.function.name:
                                    tool_calls[index]["function"]["name"] = tool_call_delta.function.name
                                    
                                # Update function arguments
                                if hasattr(tool_call_delta, 'function') and hasattr(tool_call_delta.function, 'arguments') and tool_call_delta.function.arguments:
                                    tool_calls[index]["function"]["arguments"] += tool_call_delta.function.arguments
                    
                    print("\n")
                    
                    # Check for empty response with tool calls
                    if not response_content.strip() and tool_calls:
                        # Add a generic explanation if the model didn't provide one
                        response_content = f"I'll use the {tool_calls[0]['function']['name']} tool to help answer your question."
                        print(f"{YELLOW}Adding default explanation for tool use: {response_content}{RESET}")
                        print(f"{YELLOW}DEBUG: Found {len(tool_calls)} tool calls in response{RESET}")
                        
                        # If this is the computation tool, log more detailed debug info
                        if tool_calls and tool_calls[0]['function']['name'] == 'computation':
                            print(f"{YELLOW}DEBUG: Computation tool call detected.{RESET}")
                    
                    # Add the response to messages
                    assistant_message = {"role": "assistant", "content": response_content}
                    if tool_calls:
                        assistant_message["tool_calls"] = tool_calls
                    
                    messages.append(assistant_message)
                    
                    # If streaming, handle final thinking chunk
                    if stream and response_content.strip():
                        # Check and extract any embedded JSON before adding the chunk
                        self.extract_embedded_tool_results(current_chunk, response_chunks)
                        if current_chunk["content"].strip():
                            response_chunks.append(current_chunk)
                        current_chunk = {"type": "thinking", "content": ""}
                    
                    # Check for calculation hallucinations in the response
                    calculation_patterns = [
                        r'calculat(?:e|ed|ing)',
                        r'convert(?:ed|ing)?',
                        r'(?:maximum|minimum|average|mean) (?:value|angle|speed|velocity|altitude|height|temperature) (?:is|was) (\d+\.?\d*)',
                        r'(?:pitch|roll|yaw|tilt|angle) (?:of|is|was) (\d+\.?\d*)',
                        r'(?:found|determined|calculated) (?:that|a) (?:the)? (?:maximum|minimum|mean) (?:to be|is|was) (\d+\.?\d*)'
                    ]
                    
                    has_calculation_claim = any(re.search(pattern, response_content, re.IGNORECASE) for pattern in calculation_patterns)
                    needs_computation = (
                        has_calculation_claim and 
                        topic_data_retrieved and 
                        not calculation_tools_used and
                        'computation' not in [tc['function']['name'] for tc in tool_calls]
                    )
                    
                    # If model claims to calculate but doesn't use computation tool
                    if needs_computation:
                        print(f"{RED}HALLUCINATION DETECTED: Model claims to calculate without using computation tool{RESET}")
                        # Check if we have a data_id from previous topic_data response
                        if data_id_retrieved:
                            print(f"{YELLOW}Enforcing use of computation tool with data_id: {data_id_retrieved}{RESET}")
                            
                            # Modify the last message to warn against hallucination
                            messages[-1] = {
                                "role": "system",
                                "content": """CRITICAL WARNING: You MUST use the computation tool to perform calculations on data. 
DO NOT calculate values manually or hallucinate results.

To perform a calculation, use the computation tool with the data_id from your previous topic_data call.
After you get the computation results, then you can provide your answer."""
                            }
                            
                            # Add an example of a proper computation tool call
                            messages.append({
                                "role": "user",
                                "content": f"Please use the computation tool with data_id: {data_id_retrieved} to calculate the values you need for your analysis. Do not try to calculate values without using the tool."
                            })
                            
                            continue  # Skip to next iteration to get a new response
                    
                    # If no tool calls, the model is done - return the response
                    if not tool_calls:
                        # Check if the response contains calculations but we never used computation tools
                        has_numbers = re.search(r'\b\d+\.?\d*\b\s*(?:degrees|meters|m/s|km/h)', response_content)
                        has_topic_data_stats = False
                        
                        # Check if the response is using statistics directly from topic_data
                        if topic_data_retrieved and data_id_retrieved:
                            # Common patterns for when model is referencing topic_data stats
                            topic_data_patterns = [
                                r'statistics show',
                                r'according to the data',
                                r'data shows',
                                r'retrieved data',
                                r'maximum value (is|was) \d',
                                r'minimum value (is|was) \d',
                                r'the (maximum|minimum|mean|average) (?:value |)?(?:is|was) \d'
                            ]
                            has_topic_data_stats = any(re.search(pattern, response_content, re.IGNORECASE) for pattern in topic_data_patterns)
                        
                        # Only enforce computation tool if using calculations NOT from topic_data stats
                        if has_numbers and topic_data_retrieved and not calculation_tools_used and not has_topic_data_stats:
                            print(f"{YELLOW}Numbers in response without computation tool, but might be using topic_data statistics{RESET}")
                            
                            # Add a warning and request proper tool use, but make it less strict
                            messages.append({
                                "role": "system",
                                "content": """NOTE: You appear to be providing numerical results without using the computation tool.
                                
If you're using statistics directly from the topic_data tool response, that's acceptable.
Otherwise, please use the computation tool to perform calculations on retrieved data before presenting results."""
                            })
                            
                            # Add a gentle nudge to use the computation tool
                            if data_id_retrieved:
                                messages.append({
                                    "role": "user",
                                    "content": f"If your values aren't directly from the topic_data statistics, please use the computation tool with data_id: {data_id_retrieved} to calculate the values. Otherwise, please clarify that you're using statistics from the topic_data response."
                                })
                            else:
                                messages.append({
                                    "role": "user",
                                    "content": "Please use the appropriate tools to retrieve and calculate the values you need for your answer. Do not fabricate numerical results."
                                })
                            
                            continue  # Skip to next iteration to get a new response
                        
                        # Check if we actually have content
                        if not response_content.strip():
                            # No content and no tool calls - this is likely an error
                            # Try one more time with a direct prompt
                            print(f"{YELLOW}Received empty response. Requesting final answer explicitly.{RESET}")
                            messages.append({
                                "role": "user",
                                "content": "Please provide your final answer based on the data you've gathered."
                            })
                            # Continue to next iteration to get a proper response
                            continue
                        else:
                            print(f"{GREEN}Model completed its response with no additional tool calls.{RESET}")
                            return response_content
                    
                    # Check for tool call loops
                    current_tool_calls = [f"{tc['function']['name']}:{tc['function']['arguments']}" for tc in tool_calls]
                    if current_tool_calls == last_tool_calls:
                        tool_loop_count += 1
                        if tool_loop_count >= 2:  # If we see the same tool calls three times in a row
                            print(f"{YELLOW}Detected tool call loop. Requesting final answer.{RESET}")
                            # Force a final answer
                            messages.append({
                                "role": "user",
                                "content": "I notice you're repeating the same tool calls. Based on the data you've gathered so far, please provide your final answer."
                            })
                            # Reset tool calls to avoid triggering this again
                            last_tool_calls = []
                            tool_loop_count = 0
                            continue
                    else:
                        # Update last tool calls and reset counter
                        last_tool_calls = current_tool_calls
                        tool_loop_count = 0
                    
                    # Process tool calls
                    for tool_call in tool_calls:
                        # Track tool usage for hallucination detection
                        if tool_call['function']['name'] == 'topic_data':
                            topic_data_retrieved = True
                            # Try to extract data_id from the arguments
                            try:
                                args = json.loads(tool_call['function']['arguments'])
                                if 'topic_name' in args:
                                    print(f"{BLUE}Topic data requested for: {args['topic_name']}{RESET}")
                            except Exception as e:
                                print(f"{RED}Error parsing topic_data arguments: {str(e)}{RESET}")
                        elif tool_call['function']['name'] == 'computation':
                            calculation_tools_used = True
                            # Try to extract data_id from the arguments
                            try:
                                args = json.loads(tool_call['function']['arguments'])
                                if 'data_id' in args:
                                    print(f"{BLUE}Computation requested with data_id: {args['data_id']}{RESET}")
                            except Exception as e:
                                print(f"{RED}Error parsing computation arguments: {str(e)}{RESET}")
                        
                        result = self.process_tool_call(tool_call, messages, stream=stream, response_chunks=response_chunks)
                        
                        # If streaming, reset current chunk for next thinking
                        if stream and result:
                            current_chunk = {"type": "thinking", "content": ""}
                        
                        # If this was a topic_data call, get the data_id for potential future computation
                        if tool_call['function']['name'] == 'topic_data' and result:
                            try:
                                result_data = json.loads(result.get('content', '{}'))
                                if 'data_id' in result_data:
                                    data_id_retrieved = result_data['data_id']
                                    print(f"{BLUE}Retrieved data_id: {data_id_retrieved}{RESET}")
                            except:
                                pass
                
                else:
                    raise ValueError(f"Unsupported API provider: {self.api_provider}")
                
                # Reset retry count on successful response
                retry_count = 0
                
            except Exception as e:
                retry_count += 1
                if retry_count >= max_retries:
                    print(f"{RED}API error: Failed after {retry_count} attempts. Error: {e}{RESET}")
                    
                    # If streaming, add error message to response
                    if stream:
                        response_chunks.append({"type": "error", "content": f"API error: {str(e)}"})
                        return response_chunks
                    return f"Error: {str(e)}"
                
                print(f"{YELLOW}API error: {e}. Retrying in {wait_time} seconds...{RESET}")
                time.sleep(wait_time)
                wait_time *= 2  # Exponential backoff
    
    def extract_embedded_tool_results(self, current_chunk, response_chunks):
        """Extract embedded JSON data from thinking text and convert to proper tool results."""
        if not current_chunk["content"]:
            return
            
        # Search for JSON-like patterns in the thinking text
        json_matches = re.finditer(r'(\{(?:[^{}]|(?:\{\})|(?:\[\]))*\})', current_chunk["content"])
        
        for match in json_matches:
            try:
                json_str = match.group(1)
                json_data = json.loads(json_str)
                
                # Check if this looks like a data result (has data_id and field_names)
                if isinstance(json_data, dict) and "data_id" in json_data and "field_names" in json_data:
                    # Create a proper tool result message
                    tool_result = {
                        "type": "tool_result",
                        "tool": "topic_data",
                        "content": json_str
                    }
                    
                    # Add to response chunks
                    response_chunks.append(tool_result)
                    
                    # Remove the JSON from the thinking text to avoid duplication
                    current_chunk["content"] = current_chunk["content"].replace(json_str, f"[Data with ID: {json_data['data_id']} retrieved]")
                    
                    print(f"{GREEN}Extracted embedded data result with ID: {json_data['data_id']}{RESET}")
            except json.JSONDecodeError:
                # Not valid JSON, ignore
                pass
                
    def process_tool_call(self, tool_call, messages, stream=False, response_chunks=None):
        """Process a tool call and return the result."""
        # Extract the function call
        function_name = tool_call["function"]["name"]
        function_id = tool_call["id"]
        
        print(f"\n{YELLOW}{BOLD}Executing tool: {function_name}{RESET}")
        
        try:
            # Create a FunctionCall object for execute_tool_call
            function_call_obj = FunctionCall(
                function_id,
                function_name,
                tool_call["function"]["arguments"]
            )
            
            # Simplified debug information without showing the structure
            print(f"{BLUE}Tool call: {function_name} (id: {function_id}){RESET}")
            
            # Execute the tool
            result = execute_tool_call(function_call_obj)
            
            # Format and display the result
            formatted_result = format_tool_result_summary(function_name, result)
            print(f"{GREEN}{formatted_result}{RESET}")
            
            # Add the tool result to the conversation
            tool_response = {
                "role": "tool",
                "tool_call_id": function_id,
                "name": function_name,
                "content": json.dumps(result) if result else " "
            }
            messages.append(tool_response)
            
            # If streaming, add a tool_result message
            if stream and response_chunks is not None:
                tool_result_msg = {
                    "type": "tool_result",
                    "tool": function_name,
                    "content": json.dumps(result) if result else " "
                }
                response_chunks.append(tool_result_msg)
            
            return tool_response
        
        except Exception as tool_error:
            print(f"{RED}Error executing tool: {tool_error}{RESET}")
            # Add error message to conversation
            error_response = {
                "role": "tool",
                "tool_call_id": function_id,
                "name": function_name,
                "content": json.dumps({"error": str(tool_error)})
            }
            messages.append(error_response)
            
            # If streaming, add an error message
            if stream and response_chunks is not None:
                error_msg = {
                    "type": "tool_error",
                    "tool": function_name,
                    "content": str(tool_error)
                }
                response_chunks.append(error_msg)
                
            return error_response