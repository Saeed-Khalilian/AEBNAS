"""
=====================
Given an iter_1.stats file, it copies it and removes the last 8 architectures, efficiently creating an iter_0.stats. 
=====================
"""
import ast
import json
import os
import sys

def process_stats_file(file_path):
    if not os.path.exists(file_path):
        print(f"Error: File not found at {file_path}")
        return

    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read().strip()
    except Exception as e:
        print(f"Error reading file: {e}")
        return

    if not content:
        print("Error: The file is empty.")
        return

    data = None
    # Try parsing as a Python literal first (supports single-quoted keys/strings)
    try:
        data = ast.literal_eval(content)
    except Exception:
        # Fallback to standard JSON parser
        try:
            data = json.loads(content)
        except Exception as json_err:
            print("Error: Could not parse file content as Python dictionary or JSON.")
            print(f"JSON Parse Error: {json_err}")
            return

    if not isinstance(data, dict):
        print(f"Error: Expected a dictionary/JSON object at root, but got {type(data).__name__}.")
        return

    if 'archive' not in data:
        print("Error: Key 'archive' not found in the dictionary.")
        print(f"Available keys: {list(data.keys())}")
        return

    archive = data['archive']
    if not isinstance(archive, list):
        print(f"Error: 'archive' should be a list, but got {type(archive).__name__}.")
        return

    orig_len = len(archive)
    print(f"Original length of 'archive': {orig_len}")

    if orig_len < 8:
        print("Error: Archive length is less than 8, cannot remove 8 samples.")
        return

    # Create a copy with 8 less (not in place)
    new_archive = archive[:-8]
    new_data = data.copy()
    new_data['archive'] = new_archive

    # Save to a new file (create a copy, leaving original intact)
    dir_name, file_name = os.path.split(file_path)
    base_name, ext = os.path.splitext(file_name)
    output_file_name = f"{base_name}_truncated{ext}"
    output_file_path = os.path.join(dir_name, output_file_name)

    try:
        with open(output_file_path, 'w', encoding='utf-8') as f:
            json.dump(new_data, f)
        print(f"Saved truncated archive copy to: {output_file_path}")
    except Exception as e:
        print(f"Error writing new file: {e}")
        return

    # Load the new file and check the length again to confirm it's fine
    try:
        with open(output_file_path, 'r', encoding='utf-8') as f:
            new_content = f.read().strip()
        new_data_loaded = json.loads(new_content)
        new_len = len(new_data_loaded['archive'])
        print(f"Verified new file length of 'archive': {new_len}")
        if new_len == orig_len - 8:
            print("Confirmation: Truncation check PASSED.")
        else:
            print("Confirmation check FAILED: length mismatch.")
    except Exception as e:
        print(f"Error verifying new file: {e}")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python print_archive_len.py <path_to_stats_file>")
        sys.exit(1)
    
    file_path = sys.argv[1]
    process_stats_file(file_path)

