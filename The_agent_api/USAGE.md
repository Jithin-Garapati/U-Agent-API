# ULog Analysis Agent API Usage

## Introduction

This document describes how to use the ULog Analysis Agent API. This API provides endpoints to upload ULog files (`.ulg`), analyze them using specialized tools, and manage analysis sessions. It's designed to be called by a frontend application (like one built with Vercel AI SDK) or other services.

## Base URL

When running locally for development, the API is typically available at:

```
http://127.0.0.1:8000
```

## Authentication

Currently, this API **does not implement any authentication or authorization**. Access is unrestricted when running. Ensure appropriate network security measures are in place if deploying.

## Core Workflow

The typical interaction flow is:

1.  **Upload ULog:** Send the `.ulg` file via a `POST` request to the `/upload` endpoint.
2.  **Get Session ID:** If successful, the response will contain a unique `session_id`.
3.  **Call Tools:** Use the obtained `session_id` in the path for all subsequent requests to the `/tools/...` endpoints to perform analysis on the uploaded file's data.
4.  **End Session (Optional):** When finished, send a `DELETE` request to `/session/{session_id}` to clean up the temporary data associated with that session.

## Endpoints

---

### 1. Upload ULog File

*   **Endpoint:** `/upload`
*   **Method:** `POST`
*   **Description:** Uploads a `.ulg` file, triggers its conversion to CSV format on the server, and creates a new analysis session.
*   **Request:**
    *   `Content-Type`: `multipart/form-data`
    *   Body: Requires a form field named `file` containing the `.ulg` file data.
*   **Response (Success - 200 OK):**
    ```json
    {
      "session_id": "a1b2c3d4-e5f6-7890-abcd-ef1234567890",
      "message": "File 'your_log_file.ulg' uploaded and processed successfully. Session created."
    }
    ```
*   **Response (Error):**
    *   `400 Bad Request`: If the uploaded file is not a `.ulg` file.
    *   `500 Internal Server Error`: If the file saving or CSV conversion fails on the server. Error details might be in the response body.

---

### 2. End Analysis Session

*   **Endpoint:** `/session/{session_id}`
*   **Method:** `DELETE`
*   **Description:** Ends the specified analysis session and deletes all associated temporary data (original ULog, converted CSVs, cached data) from the server.
*   **Request:**
    *   Path Parameter: `session_id` (string, required) - The ID obtained from the `/upload` response.
*   **Response (Success - 200 OK):**
    ```json
    {
      "message": "Session a1b2c3d4-e5f6-7890-abcd-ef1234567890 ended successfully and data cleaned up."
    }
    ```
    *   *Note: Also returns 200 OK if the session ID doesn't exist or was already deleted.*

---

### 3. Tool: Dynamic Parameter Search

*   **Endpoint:** `/tools/dynamic_search/{session_id}`
*   **Method:** `POST`
*   **Description:** Performs a semantic search over the available *dynamic* data topics (extracted from the converted CSV files) based on a natural language query. Useful for finding relevant time-series data streams (e.g., "what was the altitude?", "find velocity data").
*   **Request:**
    *   Path Parameter: `session_id` (string, required).
    *   Body (`application/json`):
        ```json
        {
          "query": "What was the drone's altitude during the flight?"
        }
        ```
*   **Response (Success - 200 OK):**
    *   Returns a list of matching topics, ranked by relevance. The exact structure depends on the underlying tool implementation.
    ```json
    {
      "success": true,
      "error": null,
      "results": [
        {
          "name": "vehicle_local_position",
          "description": "Vehicle's local position estimate...",
          "fields": ["timestamp", "x", "y", "z", "vx", "vy", "vz", "..."],
          "file_path": "/path/to/temp_data/session_id/csv_topics/vehicle_local_position_0.csv",
          "score": 0.85,
          "source": "combined",
          "all_file_paths": ["..."] 
        },
        {
          "name": "vehicle_global_position", 
          "description": "Vehicle's global position estimate...",
          "fields": ["timestamp", "lat", "lon", "alt", "..."],
          "file_path": "/path/to/temp_data/session_id/csv_topics/vehicle_global_position_0.csv",
          "score": 0.78,
          "source": "combined",
          "all_file_paths": ["..."] 
        }
        // ... more results
      ]
    }
    ```
*   **Response (Error):**
    *   `404 Not Found`: If the `session_id` is invalid or required files (like CSVs or KB) are missing.
    *   `500 Internal Server Error`: If an unexpected error occurs during the search.

---

### 4. Tool: Static Parameter Search

*   **Endpoint:** `/tools/static_search/{session_id}`
*   **Method:** `POST`
*   **Description:** Performs a semantic search over the vehicle's *static configuration parameters* (e.g., PID gains, battery failsafe voltage, maximum speed limits). It uses a base definition file and intersects it with the actual parameter values found in the specific uploaded ULog file for that session.
*   **Request:**
    *   Path Parameter: `session_id` (string, required).
    *   Body (`application/json`):
        ```json
        {
          "query": "What is the battery failsafe voltage?"
        }
        ```
*   **Response (Success - 200 OK):**
    *   Returns a list of matching static parameters, ranked by relevance. The exact structure depends on the underlying tool implementation.
    ```json
    {
      "success": true,
      "error": null,
      "results": [
        {
          "name": "BAT_LOW_THR",
          "longDesc": "Threshold for the low battery failsafe action...",
          "value": 10.5, // Value might be from ULog if intersected
          "unit": "V",
          "min": 9.0,
          "max": 12.0,
          "score": 0.92,
          "source": "intersection" // Indicates value came from ULog
        },
        {
          "name": "COM_LOW_BAT_ACT",
          "longDesc": "Action to take when low battery threshold is breached...",
          "value": 1, // Corresponds to an enum (e.g., RTL)
          "unit": "",
          "score": 0.81,
          "source": "intersection" 
        }
        // ... more results
      ]
    }
    ```
*   **Response (Error):**
    *   `404 Not Found`: If the `session_id` is invalid or required files (like ULog or static definitions) are missing.
    *   `500 Internal Server Error`: If an unexpected error occurs during the search.

---

### 5. Tool: Get Topic Fields

*   **Endpoint:** `/tools/topic_fields/{session_id}`
*   **Method:** `POST`
*   **Description:** Lists the available data fields (column names) within a specific topic's CSV file. Useful for knowing what data can be requested via the `topic_data` tool.
*   **Request:**
    *   Path Parameter: `session_id` (string, required).
    *   Body (`application/json`): Requires *at least one* identifier for the topic file.
        ```json
        // Option 1: By logical topic name (preferred if unique)
        { "topic_name": "vehicle_local_position" }
        
        // Option 2: By specific filename (if known, e.g., from dynamic search)
        { "topic_filename": "vehicle_local_position_0.csv" }
        
        // Option 3: By full path hint (if known, e.g., from dynamic search)
        { "file_path_hint": "/path/to/temp_data/session_id/csv_topics/vehicle_local_position_0.csv" } 
        ```
*   **Response (Success - 200 OK):**
    ```json
    {
      "success": true,
      "error": null,
      "topic": "vehicle_local_position",
      "fields": ["timestamp", "x", "y", "z", "vx", "vy", "vz", "ax", "ay", "az", "..."],
      "file": "/path/to/temp_data/session_id/csv_topics/vehicle_local_position_0.csv"
    }
    ```
*   **Response (Error):**
    *   `400 Bad Request`: If no identifier (`topic_name`, `topic_filename`, `file_path_hint`) is provided.
    *   `404 Not Found`: If the `session_id` is invalid or the specified topic/file cannot be found in the session's CSV directory.
    *   `500 Internal Server Error`: If an unexpected error occurs.

---

### 6. Tool: Get Topic Data (and Cache)

*   **Endpoint:** `/tools/topic_data/{session_id}`
*   **Method:** `POST`
*   **Description:** Retrieves data for specified fields from a topic's CSV file and caches the resulting DataFrame on the server for potential use with the `compute` tool. Returns a `data_id` referencing the cached data.
*   **Request:**
    *   Path Parameter: `session_id` (string, required).
    *   Body (`application/json`): Requires *at least one* identifier for the topic file and optionally specific columns.
        ```json
        {
          // Identifier (use one):
          "topic_name": "vehicle_local_position", 
          // "topic_filename": "vehicle_local_position_0.csv",
          // "file_path_hint": "/path/to/temp_data/session_id/csv_topics/vehicle_local_position_0.csv",
          
          // Optional: Specify columns (highly recommended for performance)
          "columns": ["timestamp", "x", "y", "z"] 
        }
        ```
        * If `columns` is omitted, all columns are loaded (can be slow for large files).
*   **Response (Success - 200 OK):**
    ```json
    {
      "data_id": "data_f1g2h3j4-k5l6-m7n8-opqr-st12345uvwxy",
      "rows_cached": 5873,
      "columns_cached": ["timestamp", "x", "y", "z"],
      "message": "Data cached successfully."
    }
    ```
*   **Response (Error):**
    *   `400 Bad Request`: If no identifier is provided.
    *   `404 Not Found`: If the `session_id` is invalid, the specified topic/file cannot be found, or requested `columns` do not exist in the file.
    *   `500 Internal Server Error`: If data reading or caching fails.

---

### 7. Tool: Run Computation

*   **Endpoint:** `/tools/compute/{session_id}`
*   **Method:** `POST`
*   **Description:** Executes a Python code snippet against a previously cached DataFrame (identified by `data_id`).
*   **⚠️ SECURITY WARNING ⚠️:** By default, this endpoint is **DISABLED** (`config.COMPUTATION_SANDBOXED = False`). Executing arbitrary code provided by a client is a major security risk. Enabling this requires implementing proper code sandboxing on the server and setting the config flag to `True`. **Do not enable without understanding and mitigating the risks.**
*   **Request:**
    *   Path Parameter: `session_id` (string, required).
    *   Body (`application/json`):
        ```json
        {
          "data_id": "data_f1g2h3j4-k5l6-m7n8-opqr-st12345uvwxy", // ID from topic_data response
          "code": "df['alt_m'] = df['z'] * -1; df[['timestamp', 'alt_m']].mean()", // Python code using 'df'
          "comment": "Calculate mean altitude" // Optional
        }
        ```
*   **Response (Success - 200 OK - *Only if sandboxing enabled*):**
    *   The structure depends heavily on what the `run_computation` tool returns.
    ```json
    {
      "success": true,
      "error": null,
      "result_summary": { // Example: result might be a dictionary or scalar
          "timestamp": 1678886400000.0, 
          "alt_m": 123.45 
      } 
    }
    ```
*   **Response (Security Disabled - Default):**
    ```json
    // Status Code: 200 OK (or potentially 403 Forbidden depending on implementation choice)
    {
      "success": false,
      "error": "Execution of arbitrary code is disabled for security reasons. Enable sandboxing and set COMPUTATION_SANDBOXED=True in config to proceed.",
      "result_summary": null
    }
    ```
*   **Response (Error - *If sandboxing enabled*):**
    *   `404 Not Found`: If `session_id` or `data_id` is invalid.
    *   `500 Internal Server Error`: If the computation fails during execution.

## Error Handling

The API uses standard HTTP status codes:

*   `200 OK`: Request successful.
*   `400 Bad Request`: Invalid request format (e.g., missing required fields, wrong file type).
*   `403 Forbidden`: Access denied (e.g., trying to use `compute` endpoint without sandboxing enabled).
*   `404 Not Found`: Resource not found (e.g., invalid `session_id`, `data_id`, or file path).
*   `405 Method Not Allowed`: Using the wrong HTTP method for an endpoint (e.g., GET instead of POST).
*   `500 Internal Server Error`: An unexpected error occurred on the server side.

Error responses typically include a JSON body with a `detail` or `error` field containing more information.

## Running Locally

1.  Ensure you have Python installed.
2.  Navigate to the `The_agent_api` directory in your terminal.
3.  Install dependencies: `pip install -r requirements.txt`
4.  Navigate **back up** to the parent directory (`Agent 1.0 - Copy (2)`).
5.  Run the server: `python -m The_agent_api.main`
6.  The API will be available at `http://127.0.0.1:8000`. 