# Helper functions for token estimation and truncation
def estimate_tokens(message):
    """Rough estimation of tokens in a message."""
    # Rough estimation: 1 token ≈ 4 characters for English text
    content = message.get("content", "")
    if isinstance(content, str):
        return len(content) // 4 + 20  # +20 for message structure overhead
    return 50  # Default estimate for structured messages

def truncate_content(message, max_tokens):
    """Truncate content in a message if needed."""
    if "content" not in message or not isinstance(message["content"], str):
        return message
    
    content = message["content"]
    # Keep the first part of the content up to approximately max_tokens
    if estimate_tokens(message) > max_tokens:
        truncated_content = content[:max_tokens * 4]  # 4 chars per token (estimate)
        message["content"] = truncated_content + "... [content truncated]"
    return message

def truncate_message_history(messages, max_tokens=3000):
    """Truncate the message history to keep it under the token limit while preserving context."""
    # Keep system prompt and the latest user message
    if len(messages) <= 2:
        return messages
    
    # Extract the original query from the first user message
    original_query = None
    for msg in messages:
        if msg.get("role") == "user" and not original_query:
            original_query = msg.get("content", "")
            break
            
    # CRITICAL: Always keep the full function definition / parameters section
    # This ensures the model retains knowledge about how to call functions
    system_messages = [msg for msg in messages if msg.get("role") == "system"]
    system_prompt = system_messages[0] if system_messages else {"role": "system", "content": ""}
    
    # Track what tools have been used so far and their results
    tool_history = []
    tool_results = {}
    for i, msg in enumerate(messages):
        if msg.get("role") == "assistant" and "tool_calls" in msg:
            for tool_call in msg.get("tool_calls", []):
                if "function" in tool_call and "name" in tool_call["function"]:
                    tool_name = tool_call["function"]["name"]
                    args = tool_call["function"].get("arguments", "{}")
                    tool_id = tool_call.get("id", f"tool_{len(tool_history)}")
                    tool_history.append(f"{tool_name}: {args}")
                    
        # Save tool results separately (they're more important than reasoning)
        if msg.get("role") == "tool":
            tool_id = msg.get("tool_call_id", "")
            if tool_id:
                name = ""
                # Find the corresponding tool call to get its name
                for prev_msg in messages[:i]:
                    if prev_msg.get("role") == "assistant" and "tool_calls" in prev_msg:
                        for tc in prev_msg.get("tool_calls", []):
                            if tc.get("id") == tool_id and "function" in tc:
                                name = tc["function"].get("name", "")
                                break
                if name:
                    # Prioritize important tool results (especially search results)
                    tool_results[tool_id] = {
                        "name": name,
                        "content": msg.get("content", ""),
                        "priority": 2 if name in ["dynamic_param_search", "static_param_search", "topic_data"] else 1
                    }
    
    # Start with system prompt
    truncated_messages = [system_prompt]
    
    # Create a context reminder message that includes the original query and tool history
    context_reminder = {
        "role": "system",
        "content": f"This is a continuation of analysis for query: '{original_query}'\n"
                   f"Tools used so far: {'; '.join(tool_history[:10])}"
    }
    truncated_messages.append(context_reminder)
    
    # Add the most recent k messages, keeping important tool results
    k = 10  # Number of most recent messages to keep
    recent_messages = messages[-k:] if len(messages) > k else messages[1:]
    
    # If we're at risk of exceeding the token limit, prioritize
    estimated_tokens = estimate_tokens(system_prompt) + estimate_tokens(context_reminder)
    
    # Prioritize including tool results based on their importance
    prioritized_results = sorted(tool_results.items(), 
                                 key=lambda x: (-x[1]["priority"],  # Higher priority first
                                                list(tool_results.keys()).index(x[0])))  # Earlier results first
    
    # Add as many tool results as we can fit within token limits
    tool_result_messages = []
    for tool_id, tool_info in prioritized_results:
        # Create the assistant message that contains the tool call
        assistant_msg = None
        tool_msg = None
        
        # Find the corresponding messages for this tool
        for i, msg in enumerate(messages):
            if msg.get("role") == "assistant" and "tool_calls" in msg:
                for tc in msg.get("tool_calls", []):
                    if tc.get("id") == tool_id:
                        assistant_msg = msg
                        # Find the tool message that follows
                        if i+1 < len(messages) and messages[i+1].get("role") == "tool" and messages[i+1].get("tool_call_id") == tool_id:
                            tool_msg = messages[i+1]
                        break
                if assistant_msg:
                    break
        
        # If we found both messages, add them (with potential truncation)
        if assistant_msg and tool_msg:
            # Truncate content if needed
            truncated_assistant = truncate_content(assistant_msg.copy(), 150)
            truncated_tool = truncate_content(tool_msg.copy(), 350)
            
            # Estimate tokens with these messages added
            assistant_tokens = estimate_tokens(truncated_assistant)
            tool_tokens = estimate_tokens(truncated_tool)
            
            if estimated_tokens + assistant_tokens + tool_tokens <= max_tokens:
                tool_result_messages.append(truncated_assistant)
                tool_result_messages.append(truncated_tool)
                estimated_tokens += assistant_tokens + tool_tokens
            else:
                # We're at risk of exceeding token limit
                break
    
    # Add the prioritized tool result messages to our truncated message history
    truncated_messages.extend(tool_result_messages)
    
    # Add the most recent messages
    for msg in recent_messages:
        # Skip system messages (already handled)
        if msg.get("role") == "system":
            continue
        
        # Skip tool messages already included in tool_result_messages
        if msg.get("role") == "tool" and msg.get("tool_call_id") in [tr.get("tool_call_id") for tr in tool_result_messages if tr.get("role") == "tool"]:
            continue
            
        msg_copy = msg.copy()
        msg_tokens = estimate_tokens(msg_copy)
        
        if estimated_tokens + msg_tokens <= max_tokens:
            truncated_messages.append(msg_copy)
            estimated_tokens += msg_tokens
        else:
            # Truncate content before adding if we're close to the limit
            if estimated_tokens + msg_tokens - 100 <= max_tokens and "content" in msg_copy:
                msg_copy = truncate_content(msg_copy, 150)  # Aggressive truncation
                truncated_messages.append(msg_copy)
            break
    
    # If we have performed truncation, add a notice
    if len(truncated_messages) < len(messages):
        truncation_notice = {
            "role": "system",
            "content": f"Note: Message history was truncated to stay within token limits. Key search results and data were preserved. REMEMBER: Be extremely concise in your reasoning (max 1-2 sentences)."
        }
        truncated_messages.insert(1, truncation_notice)
    
    return truncated_messages 