"""
Parameter Extraction Tools Test Script

This script demonstrates the usage of all three parameter extraction tools:
1. Dynamic Parameter Tool (DP) - For querying flight log data
2. Static Parameter Tool (SP) - For querying configuration parameters 
3. Combined Parameter Tool (CP) - For intelligently querying both sources

Usage:
    python test_tools.py
"""

from extract_dynamic_param_tool import DP
from extract_static_param_tool import SP
from combined_param_tool import CP

def test_dynamic_param_tool():
    """Test the dynamic parameter extraction tool."""
    print("\n" + "=" * 80)
    print("TESTING DYNAMIC PARAMETER TOOL (DP)")
    print("=" * 80)
    
    # Example queries
    queries = [
        "What was the maximum speed during the flight?",
        "What was the highest altitude reached?",
        "What was the battery voltage at landing?",
        "What was the aircraft attitude during takeoff?",
        "What was the GPS position at the beginning of the flight?"
    ]
    
    for query in queries:
        print(f"\nQuery: {query}")
        print("-" * 50)
        
        results = DP(query)
        
        print(f"Top {len(results)} results:")
        for i, param in enumerate(results, 1):
            print(f"{i}. {param['name']} (Score: {param['score']:.4f})")
            print(f"   Description: {param['description']}")
            print(f"   Key: {param['key']}")
            print(f"   Fields: {len(param['fields'])} fields available")
            print()

def test_static_param_tool():
    """Test the static parameter extraction tool."""
    print("\n" + "=" * 80)
    print("TESTING STATIC PARAMETER TOOL (SP)")
    print("=" * 80)
    
    # Example queries
    queries = [
        "What is the maximum allowed speed?",
        "What is the minimum battery voltage?",
        "What are the altitude limits?",
        "What is the hover throttle setting?",
        "What are the safety parameters for battery?"
    ]
    
    for query in queries:
        print(f"\nQuery: {query}")
        print("-" * 50)
        
        results = SP(query)
        
        print(f"Top {len(results)} results:")
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
                
            print(f"   Description: {param['long_desc']}")
            print()

def test_combined_param_tool():
    """Test the combined parameter extraction tool."""
    print("\n" + "=" * 80)
    print("TESTING COMBINED PARAMETER TOOL (CP)")
    print("=" * 80)
    
    # Example queries - mix of dynamic and static
    queries = [
        "What was the maximum speed during the flight?",  # Dynamic
        "What is the maximum allowed speed?",             # Static
        "What was the battery voltage at landing?",       # Dynamic
        "What is the minimum battery voltage allowed?",   # Static
        "Tell me about the altitude parameters",          # Ambiguous - could be either
    ]
    
    for query in queries:
        print(f"\nQuery: {query}")
        print("-" * 50)
        
        results = CP(query)
        
        print(f"Top {len(results)} results:")
        for i, result in enumerate(results, 1):
            param_type = result['param_type']
            print(f"{i}. [{param_type.upper()}] {result.get('name', 'Unknown')}")
            print(f"   Score: {result['score']:.4f} (Original: {result['original_score']:.4f})")
            
            if param_type == 'dynamic':
                print(f"   Description: {result.get('description', 'None')}")
                print(f"   Key: {result.get('key', 'None')}")
                print(f"   Fields: {len(result.get('fields', {}))} fields available")
            else:  # static
                print(f"   Group: {result.get('group', 'None')}")
                print(f"   Value: {result.get('value', 'None')} {result.get('unit', '')}")
                print(f"   Description: {result.get('short_desc', 'None')}")
                
                if result.get('min', '') or result.get('max', ''):
                    limits = []
                    if result.get('min', '') != '':
                        limits.append(f"Min: {result['min']}")
                    if result.get('max', '') != '':
                        limits.append(f"Max: {result['max']}")
                    print(f"   Limits: {', '.join(limits)}")
            print()

if __name__ == "__main__":
    print("PARAMETER EXTRACTION TOOLS DEMONSTRATION")
    print("This script demonstrates the three parameter extraction tools for flight analysis")
    
    # Run all tests
    test_dynamic_param_tool()
    test_static_param_tool()
    test_combined_param_tool() 