# Parameter Extraction Tools

This directory contains tools for extracting and querying parameters from flight logs and configuration files.

## Overview

Three main tools are provided:

1. **Dynamic Parameter Tool** (`extract_dynamic_param_tool.py`)
   - Extracts parameters from flight logs (CSV topics)
   - Answers questions about measurements during flight
   - Uses semantic search with keyword boosting

2. **Static Parameter Tool** (`extract_static_param_tool.py`)
   - Extracts parameters from configuration files (CSV format)
   - Answers questions about system limits, constraints, and settings
   - Uses semantic search with category boosting

3. **Combined Parameter Tool** (`combined_param_tool.py`)
   - Intelligently determines whether a query is about dynamic or static parameters
   - Routes queries to the appropriate tool and ranks results
   - Provides a unified interface for all parameter types

## Usage

Each tool provides a simple function interface for easy integration:

```python
# For dynamic parameters from flight logs
from extract_dynamic_param_tool import DP
results = DP("What was the maximum speed during the flight?")

# For static parameters from configuration
from extract_static_param_tool import SP
results = SP("What is the maximum allowed speed?")

# For automatic determination and combined results
from combined_param_tool import CP
results = CP("Tell me about the altitude parameters")
```

## Example Usage

See `test_tools.py` for comprehensive examples of using all three tools.

## File Requirements

- **Dynamic Parameter Tool**:
  - Directory of CSV topics (flight log data)
  - Knowledge base JSON file (parameter metadata)

- **Static Parameter Tool**:
  - Static parameters CSV file (configuration parameters)

- **Combined Parameter Tool**:
  - Requires all files from both tools above

## Dependencies

- sentence-transformers
- numpy
- pandas
- re
- os
- json
- csv 