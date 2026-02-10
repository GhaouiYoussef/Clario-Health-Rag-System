import os
import json
import re

def rename_and_index(base_dir, output_map_file):
    """
    Scans document directories, renames files to numbered indices (e.g. doc_001.pdf),
    and saves a mapping of ID -> Original Name.
    """
    
    # 1. Load existing mapping if available, to avoid re-renaming or losing data if run twice
    existing_map = {}
    if os.path.exists(output_map_file):
        try:
            with open(output_map_file, 'r', encoding='utf-8') as f:
                existing_map = json.load(f)
        except:
            pass
            
    # Find the next available ID in case we are adding files
    # ID format is 'doc_XXX'
    max_id = 0
    pattern = re.compile(r"doc_(\d+)")
    
    # Check keys in existing map to find max ID
    for key in existing_map:
        match = pattern.match(key)
        if match:
            val = int(match.group(1))
            if val > max_id:
                max_id = val
                
    current_index = max_id + 1
    
    subdirs = ['abrevs', 'curated', 'raw']
    new_mapping = existing_map.copy()
    
    files_processed = 0
    
    for category in subdirs:
        dir_path = os.path.join(base_dir, category)
        if not os.path.exists(dir_path):
            continue
            
        # Get list of files
        files = sorted([f for f in os.listdir(dir_path) if os.path.isfile(os.path.join(dir_path, f))])
        
        for filename in files:
            # Check if already renamed (matches doc_XXX.ext pattern)
            if re.match(r"doc_\d+\.", filename):
                # Ensure it's in the map. slightly edge case if map is deleted but files are renamed.
                print(f"Skipping {filename}, already looks like an index.")
                continue
                
            # It's a file that needs renaming
            name_part, ext = os.path.splitext(filename)
            
            # Generate new name
            doc_id = f"doc_{current_index:03d}"
            new_filename = f"{doc_id}{ext}"
            
            old_full_path = os.path.join(dir_path, filename)
            new_full_path = os.path.join(dir_path, new_filename)
            
            # Rename
            try:
                os.rename(old_full_path, new_full_path)
                
                # Update map
                rel_path = os.path.relpath(new_full_path, os.path.dirname(output_map_file)).replace("\\", "/")
                
                new_mapping[doc_id] = {
                    "original_name": filename,
                    "category": category,
                    "new_filename": new_filename,
                    "path": rel_path
                }
                
                print(f"Renamed: {filename} -> {new_filename}")
                current_index += 1
                files_processed += 1
                
            except OSError as e:
                print(f"Error renaming {filename}: {e}")

    # Save the updated map
    with open(output_map_file, 'w', encoding='utf-8') as f:
        json.dump(new_mapping, f, indent=4)
        print(f"Updated mapping file: {output_map_file}")


if __name__ == "__main__":
    workspace_root = os.getcwd()
    base_documents_dir = os.path.join(workspace_root, "data", "documents")
    output_mapping_file = os.path.join(workspace_root, "data", "document_mapping.json")
    
    rename_and_index(base_documents_dir, output_mapping_file)
