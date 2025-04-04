from agent import ULogReasoningAgent
from constants import BLUE, BOLD, RESET
from config import GROQ_API_KEY, CONFIG

def main():
    """Main function to run the ULog reasoning agent."""
    # Check if API key is set
    if not GROQ_API_KEY:
        print(f"{BOLD}Error: GROQ_API_KEY not found.{RESET}")
        print("Please set it in your .env file or environment variables.")
        return
    
    # Create the agent
    model = CONFIG["agent"]["model"]
    agent = ULogReasoningAgent(model=model)
    
    # Example queries
    example_queries = [
        "What is the maximum altitude difference ?",
        "What is the maximum tilt angle during the flight in degrees?",
        "What is the maximum speed during the flight in km/h?",
        "What is the total distance covered by the vehicle in km?",
        "What is the configured battery failsafe action?",
    ]
    
    # Let user select a query or enter their own
    print(f"{BLUE}{BOLD}Example queries:{RESET}")
    for i, query in enumerate(example_queries):
        print(f"{i+1}. {query}")
    print(f"0. Enter your own query")
    
    choice = input(f"\n{BOLD}Enter your choice (0-{len(example_queries)}): {RESET}")
    
    if choice == "0":
        query = input(f"\n{BOLD}Enter your query about the flight log: {RESET}")
    else:
        try:
            query = example_queries[int(choice)-1]
        except:
            print(f"{BOLD}Invalid choice, using the first query.{RESET}")
            query = example_queries[0]
    
    # Check if it's a reasoning model to determine default
    model_settings = CONFIG["agent"].get("model_settings", {}).get(model, {})
    is_reasoning_model = model_settings.get(
        "is_reasoning_model", 
        CONFIG["agent"].get("is_reasoning_model", True)
    )
    
    # If it's a reasoning model, ask if the user wants to see the reasoning
    if is_reasoning_model:
        show_reasoning = input(f"\n{BOLD}Show reasoning process? (y/n): {RESET}").lower() == "y"
    else:
        # For non-reasoning models, we'll still show messages but not formal reasoning
        show_reasoning = False  # This controls the formal reasoning display
        print(f"\n{BLUE}Using non-reasoning model '{model}'. Explanation messages will be shown, but detailed reasoning is disabled.{RESET}")
    
    # Run the agent
    agent.run(query, show_reasoning=show_reasoning)

if __name__ == "__main__":
    main() 