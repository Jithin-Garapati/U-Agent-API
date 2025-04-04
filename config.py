"""
Configuration settings for the Qwen ULog Reasoning Agent
"""
import os
import logging
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Configure logging - suppress specific warnings from google_genai.types
logging.getLogger('google_genai.types').setLevel(logging.ERROR)

# API key configuration
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

# Agent configuration
CONFIG = {
    "agent": {
        "api_provider": "gemini",    # API provider: "groq" or "gemini"
        "model": "gemini-2.0-flash",  # Model used by the agent
        "auto_convert_ulog": True,  # Whether to automatically preprocess ULog files
        "max_tokens": 8192,        # Maximum tokens for model responses
        "temperature": 0.1,        # Temperature for model responses
        "is_reasoning_model": False,  # Default to True for backward compatibility
        
        # Token management settings
        "token_management": {
            "groq": {"apply_reduction": True},
            "gemini": {"apply_reduction": False}
        },
        
        # Model-specific settings
        "model_settings": {
            # Groq models
            "qwen-qwq-32b": {
                "provider": "groq",
                "is_reasoning_model": True,
                "needs_function_extraction": True
            },
            "llama-3.3-70b-versatile": {
                "provider": "groq",
                "is_reasoning_model": False,
                "needs_function_extraction": False
            },
            # Gemini models
            "gemini-2.0-flash": {
                "provider": "gemini",
                "is_reasoning_model": False,  # Important for function calling to work
                "needs_function_extraction": False,
                "prevent_fake_json": True,
                "strict_function_calling": True,
                "function_calling_mode": "AUTO",  # Can be AUTO, ANY, or NONE
                "temperature": 0.7,  # Lower temperature for more predictable function calls
                "top_p": 0.95,
                "top_k": 40,
                "max_output_tokens": 8192
            },
            "gemini-2.0-pro": {
                "provider": "gemini",
                "is_reasoning_model": True,
                "needs_function_extraction": False,
                "prevent_fake_json": True,
                "strict_function_calling": True,
                "temperature": 0.4,
                "top_p": 0.95,
                "top_k": 40,
                "max_output_tokens": 8192
            },
            # Add other models as needed
        }
    },
    "files": {
        "ulog_file": "flight_log.ulg",  # Path to the ULog file to analyze
        "output_dir": "data",           # Directory to store processed data
    }
} 

# Agent prompt templates
AGENT_PROMPTS = {
    "hallucination_prevention": """
    IMPORTANT: Anti-Hallucination Rules
    1. NEVER make up flight data or statistics( this is very very important especially during tool calls you are doing this)
    2. ONLY base your answers on information retrieved from the flight log using tools
    3. If you're unsure about data, use tools to retrieve it - DON'T guess
    4. Clearly indicate when data is not available or if you cannot answer a question
    5. Verify your calculations with the calculate tool instead of performing them yourself
    """,
    
    "reasoning_prompt": """You are an expert flight log analyzer assistant that helps users analyze their drone flight logs. 
    Show your work step-by-step and explain your thought process clearly.
    
    When analyzing a log, ALWAYS:
    
    1. THINK CAREFULLY about what information you need and which tools to use
    2. After deciding on a tool, IMMEDIATELY call that tool with proper parameters
    3. NEVER mention a tool without calling it immediately afterward
    4. ALWAYS wait for tool results before proceeding
    5. ALWAYS use tool results to inform your next steps
    6. CONTINUE calling tools until you have ALL information needed
    7. Only provide a final answer when you've gathered ALL necessary information
    
    Your reasoning must be thorough and complete - NEVER stop halfway through a reasoning chain.
    When computing distance from position data, use cumulative displacement only if the data is smooth—otherwise, clean or resample first to avoid under/overestimation.
    Available Tools:
    - dynamic_param_search: Find parameters in the flight log matching specific criteria
    - topic_fields: Check what fields are available in a specific topic
    - topic_data: Get data from specific fields in a topic
    - calculate: Perform calculations on flight data 
    """
}