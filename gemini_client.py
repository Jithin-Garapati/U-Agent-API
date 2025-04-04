"""
Implementation of the Gemini API client for the ULog Reasoning Agent.
"""
import os
import json
import uuid
from google import genai
from google.genai import types
from typing import Dict, List, Any, Optional
import logging

from config import CONFIG, GEMINI_API_KEY
from tool_schemas import TOOL_SCHEMAS

class GeminiClient:
    """A client for the Google Gemini API."""
    
    def __init__(self, api_key=None):
        """Initialize the Gemini API client."""
        self.api_key = api_key or GEMINI_API_KEY
        self.client = genai.Client(api_key=self.api_key)
        # Keep track of the last returned tool responses for hallucination detection
        self.last_tool_responses = {}
        self.last_tool_mentioned = None
        
    def convert_tool_schema_to_gemini_format(self, tool_schemas):
        """Convert the tool schemas to the format expected by Gemini API."""
        function_declarations = []
        
        for tool in tool_schemas:
            # Convert each tool schema to a function declaration
            function_declaration = {
                "name": tool["function"]["name"],
                "description": tool["function"].get("description", ""),
                "parameters": self._process_parameters(tool["function"].get("parameters", {}))
            }
            function_declarations.append(function_declaration)
            
        return function_declarations
    
    def _process_parameters(self, parameters):
        """Process parameters to meet Gemini API requirements."""
        # Deep copy the parameters to avoid modifying the original
        processed = json.loads(json.dumps(parameters))
        
        # Process object properties
        if "properties" in processed:
            for prop_name, prop_value in processed["properties"].items():
                # Check if this is an object type with no properties
                if prop_value.get("type") == "object" and "properties" not in prop_value:
                    # For Gemini, add a dummy property to satisfy the requirement
                    # that object types must have at least one property
                    if prop_name == "data" and "description" in prop_value and "DEPRECATED" in prop_value["description"]:
                        # For the deprecated data parameter, we'll add a more appropriate dummy property
                        processed["properties"][prop_name]["properties"] = {
                            "dummy": {
                                "type": "string",
                                "description": "Deprecated parameter. Use data_id instead."
                            }
                        }
                    else:
                        # For other object types without properties
                        processed["properties"][prop_name]["properties"] = {
                            "value": {
                                "type": "string",
                                "description": "Generic value field for empty object type"
                            }
                        }
        
        return processed
        
    def chat_completions_create(self, model, messages, tools, tool_choice="auto", temperature=0.1, max_tokens=1024, stream=True):
        """Create a chat completion using the Gemini API."""
        logger = logging.getLogger(__name__)
        logger.info(f"Creating chat completion with {len(messages)} messages")
        
        # Get the model configuration
        model_config = CONFIG["agent"].get("model_settings", {}).get(model, {})
        
        # Check for valid Gemini messages:
        # 1. Ensure we have both user and model messages alternating correctly
        # 2. Handle the case where conversation has only a system message and a user message
        valid_gemini_messages = self._prepare_valid_gemini_messages(messages)
        logger.info(f"Prepared {len(valid_gemini_messages)} valid Gemini messages")
        
        # Convert messages to Gemini format
        gemini_messages = []
        for msg in valid_gemini_messages:
            role = msg["role"]
            content = msg["content"]
            
            # Extra validation to catch any empty content that might have slipped through
            if not content or content.strip() == "":
                logger.warning("Final check found empty content, replaced with space")
                content = " "  # Use space as minimum content
            
            # Map OpenAI roles to Gemini roles
            if role == "system":
                # Gemini doesn't have a system role, so we'll use user role for system messages
                role = "user"
            elif role == "assistant":
                role = "model"
            elif role == "tool":
                # Gemini doesn't have a tool role, so we'll convert tool responses to model responses
                # with the tool result as content
                role = "model"
            
            # Update the content for tool responses to include structured data for hallucination detection
            if msg.get("role") == "tool" and "tool_call_id" in msg:
                # Store the tool response for hallucination detection
                self.last_tool_responses[msg.get("name", "")] = msg.get("content", "{}")
                
                # Format tool response to make it more recognizable
                try:
                    # Try to parse the content as JSON
                    content_data = json.loads(msg.get("content", "{}"))
                    if msg.get("name") == "topic_data" and "statistics" in content_data:
                        # Emphasize statistics for hallucination detection
                        content = f"TOOL_RESPONSE_{msg.get('name')}:\n{content}\n\nDATA_RETRIEVAL_CONFIRMED: YES\nTOOL_DATA_ID: {content_data.get('data_id', '')}"
                except:
                    # If parsing fails, just use the content as is
                    pass
            
            # Final validation before creating the message
            if not content or content.strip() == "":
                logger.warning("Final check found empty content, replaced with space")
                content = " "
                
            gemini_message = types.Content(
                role=role,
                parts=[types.Part.from_text(text=content)]
            )
            
            gemini_messages.append(gemini_message)
        
        logger.info(f"Created {len(gemini_messages)} Gemini messages")
        
        # Convert tools to Gemini format
        function_declarations = []
        for tool in tools:
            function_declaration = types.FunctionDeclaration(
                name=tool["function"]["name"],
                description=tool["function"].get("description", ""),
                parameters=self._process_parameters(tool["function"].get("parameters", {}))
            )
            function_declarations.append(function_declaration)
            
        gemini_tools = [types.Tool(function_declarations=function_declarations)]
        
        # Create the generation config
        generate_content_config = types.GenerateContentConfig(
            temperature=temperature,
            top_p=model_config.get("top_p", 0.95),
            top_k=model_config.get("top_k", 40),
            max_output_tokens=max_tokens,
            tools=gemini_tools,
            response_mime_type="text/plain",
        )
        
        logger.info(f"Sending request to Gemini API with model {model}")
        logger.info(f"Streaming response started")
        
        # Create a generator that yields response chunks directly using Gemini's native format
        def response_generator():
            try:
                empty_response_detected = False
                has_tool_calls = False
                content_received = False
                
                streaming_response = self.client.models.generate_content_stream(
                    model=model,
                    contents=gemini_messages,
                    config=generate_content_config,
                )
                
                for chunk in streaming_response:
                    # Create a custom response object to maintain compatibility
                    response = type('Response', (), {})()
                    choice = type('Choice', (), {})()
                    delta = type('Delta', (), {})()
                    
                    # Handle text content
                    if hasattr(chunk, 'text') and chunk.text:
                        content_received = True
                        delta.content = chunk.text
                    else:
                        delta.content = ""
                    
                    # Handle function calls directly without complex conversion
                    if hasattr(chunk, 'function_calls') and chunk.function_calls:
                        has_tool_calls = True
                        logger.info(f"Received function call in chunk")
                        
                        # Store function calls directly
                        delta.tool_calls = []
                        for func_call in chunk.function_calls:
                            logger.info(f"Function call: {func_call.name}")
                            
                            # Store the tool call for future reference
                            self.last_tool_mentioned = func_call.name
                            
                            # Create a simple tool call object with necessary properties
                            tool_call = type('ToolCall', (), {})()
                            tool_call.id = str(uuid.uuid4())
                            tool_call.type = "function"
                            tool_call.index = 0
                            
                            # Create a simple function object
                            function_obj = type('Function', (), {})()
                            function_obj.name = func_call.name
                            function_obj.arguments = json.dumps(func_call.args)
                            
                            # Attach function to tool call
                            tool_call.function = function_obj
                            
                            # Add to delta.tool_calls
                            delta.tool_calls.append(tool_call)
                    
                    # Attach delta to choice and choice to response
                    choice.delta = delta
                    response.choices = [choice]
                    
                    yield response
                
                # After streaming completes, log if we had empty content
                if not content_received and has_tool_calls:
                    logger.warning("Empty text response with tool calls detected")
                elif not content_received and not has_tool_calls:
                    logger.warning("Completely empty response detected (no content, no tool calls)")
                
                logger.info(f"Streaming response completed")
                
            except Exception as e:
                logger.error(f"Error in Gemini API call: {str(e)}")
                raise e
        
        return response_generator()
    
    def _prepare_valid_gemini_messages(self, messages):
        """Ensure the message sequence is valid for Gemini API."""
        logger = logging.getLogger(__name__)
        
        # Validate each message to ensure none have empty content
        for i, msg in enumerate(messages):
            if "content" not in msg or msg["content"] is None or msg["content"].strip() == "":
                # Instead of just warning, actually fix the issue by adding a space
                logger.warning(f"Empty content in {msg.get('role', 'unknown')} message, added space placeholder")
                messages[i]["content"] = " "  # Add a space as minimum content
        
        # Extract system message if present
        system_msg = None
        other_msgs = []
        
        for msg in messages:
            if msg["role"] == "system":
                system_msg = msg
            else:
                other_msgs.append(msg)
        
        # If we only have system + user message, add a placeholder model response
        if len(other_msgs) == 1 and other_msgs[0]["role"] == "user":
            # Create a new message list with the system message content prepended to user message
            if system_msg:
                # Create a combined user message
                combined_user_msg = {
                    "role": "user",
                    "content": f"{system_msg['content']}\n\n{other_msgs[0]['content']}"
                }
                # Double-check the combined message isn't empty
                if not combined_user_msg["content"] or combined_user_msg["content"].strip() == "":
                    logger.warning("Final check found empty content, replaced with space")
                    combined_user_msg["content"] = " "
                return [combined_user_msg]
            else:
                # Double-check the message isn't empty
                if not other_msgs[0]["content"] or other_msgs[0]["content"].strip() == "":
                    logger.warning("Final check found empty content, replaced with space")
                    other_msgs[0]["content"] = " "
                return other_msgs
                
        # For history with tool responses, convert to a valid alternating sequence
        valid_sequence = []
        last_role = None
        
        for msg in messages:
            role = "user" if msg["role"] in ["system", "user"] else "model"
            
            # Ensure content is never empty
            content = msg.get("content", "")
            if not content or content.strip() == "":
                logger.warning(f"Empty content detected in {msg.get('role')} message, adding placeholder")
                content = " "  # Use space as minimal content
            
            # If we have consecutive messages with the same role, combine them
            if role == last_role and valid_sequence:
                valid_sequence[-1]["content"] += f"\n\n{content}"
            else:
                valid_sequence.append({"role": role, "content": content})
                last_role = role
        
        # Final check before returning - ensure no messages have empty content
        for i, msg in enumerate(valid_sequence):
            if not msg["content"] or msg["content"].strip() == "":
                logger.warning("Final check found empty content, replaced with space")
                valid_sequence[i]["content"] = " "
        
        return valid_sequence