import os
import shutil
import subprocess
from pyulog import ULog

def convert_ulog_to_csv(ulog_filepath, output_folder):
    """
    Runs 'ulog2csv' to convert a ULog file into CSV files in output_folder.
    
    Args:
        ulog_filepath: Path to the ULog file
        output_folder: Folder to store the CSV files
        
    Returns:
        bool: True if conversion successful, False otherwise
    """
    # Create the output folder if it doesn't exist
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    
    # Check if the file exists
    if not os.path.exists(ulog_filepath):
        print(f"ULog file '{ulog_filepath}' not found")
        return False
    
    # Check file size - a valid ULog file should have a minimum size
    file_size = os.path.getsize(ulog_filepath)
    if file_size < 1024:  # Less than 1KB is likely invalid
        print(f"ULog file is too small ({file_size} bytes). It may not be a valid ULog file.")
        return False
    
    # Validate file header before conversion
    try:
        # Attempt to read just the header to validate
        with open(ulog_filepath, 'rb') as f:
            header = f.read(16)  # Read the first 16 bytes
            if not header.startswith(b'ULog\x01'):
                print(f"Invalid ULog file format. The file does not have a valid ULog header.")
                return False
    except Exception as e:
        print(f"Error validating ULog file: {e}")
        return False
            
    print(f"Converting ULog file '{ulog_filepath}' to CSV in '{output_folder}'...")
    try:
        result = subprocess.run(["ulog2csv", ulog_filepath, "-o", output_folder],
                                capture_output=True, text=True)
        if result.returncode != 0:
            print("Error converting ULog to CSV:", result.stderr)
            return False
        else:
            print("ULog -> CSV conversion complete.")
            return True
    except Exception as e:
        print("Exception while running ulog2csv:", e)
        return False

def preprocess_ulog(ulog_file=None):
    """
    Preprocess the ULog file by:
    1. Deleting existing csv_topics folder (if it exists)
    2. Converting ULog file to CSV files
    (No longer deletes the temporary ULog file)
    """
    ulog_file = ulog_file or "flight_log.ulg"
    csv_topics_folder = CONFIG["files"].get("dynamic_folder", "csv_topics")
    
    print(f"\n===== ULog Preprocessing =====")
    print(f"ULog file: {ulog_file}")
    print(f"CSV topics folder: {csv_topics_folder}")
    
    if not os.path.exists(ulog_file):
        print(f"Warning: ULog file '{ulog_file}' not found. Skipping conversion.")
        return False
    
    if os.path.exists(csv_topics_folder):
        print(f"Removing existing '{csv_topics_folder}' folder...")
        try:
            shutil.rmtree(csv_topics_folder)
            print(f"Successfully removed '{csv_topics_folder}' folder.")
        except Exception as e:
            print(f"Error removing '{csv_topics_folder}' folder: {e}")
            return False
    
    success = convert_ulog_to_csv(ulog_file, csv_topics_folder)
    if success:
        print("ULog preprocessing completed successfully.")
    else:
        print("ULog preprocessing failed.")
    print("==============================\n")
    
    return success

