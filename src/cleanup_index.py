import os
import json
import re

def renumber_and_clean_index(base_dir, mapping_file):
    """
    Cleans up the mapping file by removing invalid entries and renumbers
    the files starting from doc_001.
    """
    with open(mapping_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
        
    valid_entries = []
    
    # 1. Identify valid entries
    for key, info in data.items():
        # current path relative to data/
        rel_path = info['path']
        full_path = os.path.join(os.path.dirname(mapping_file), rel_path)
        
        if os.path.exists(full_path):
            valid_entries.append({
                "old_key": key,
                "current_full_path": full_path,
                "info": info
            })
        else:
            print(f"Removing invalid entry {key}: {rel_path} not found.")

    # Sort by old key to maintain order
    valid_entries.sort(key=lambda x: x['old_key'])
    
    new_mapping = {}
    
    print("\nRenumbering files...")
    
    for idx, entry in enumerate(valid_entries):
        new_id = f"doc_{idx+1:03d}"
        
        old_path = entry['current_full_path']
        dir_name = os.path.dirname(old_path)
        ext = os.path.splitext(old_path)[1]
        
        new_filename = f"{new_id}{ext}"
        new_full_path = os.path.join(dir_name, new_filename)
        
        # Rename on disk if name changed
        if old_path != new_full_path:
            # Check if target exists (shouldn't if we are just compacting, unless we collide with existing unrenamed files, but we're only dealing with known files)
            # However, if we rename doc_013 -> doc_001, and doc_001 doesn't exist (it shouldn't), we are good.
            # But what if we rename doc_002 -> doc_001 (if 001 was deleted)?
            # Since valid_entries are sorted, we process 013 (first) -> 001.
            # 014 -> 002.
            # Collision is possible if we have doc_001.pdf and we want to rename doc_005.pdf to doc_001.pdf? 
            # In my specific case, doc_001..012 don't exist. doc_013..024 exist.
            # So I'm renaming 013 -> 001. Safe. 
            
            try:
                os.rename(old_path, new_full_path)
                print(f"Renamed {os.path.basename(old_path)} -> {new_filename}")
            except OSError as e:
                print(f"Error renaming {old_path}: {e}")
                # Keep old path references if fail?
                new_full_path = old_path 
                new_filename = os.path.basename(old_path)

        # Update info for new mapping
        info = entry['info']
        info['new_filename'] = new_filename
        info['path'] = os.path.relpath(new_full_path, os.path.dirname(mapping_file)).replace("\\", "/")
        
        new_mapping[new_id] = info

    with open(mapping_file, 'w', encoding='utf-8') as f:
        json.dump(new_mapping, f, indent=4)
        
    print(f"\nFinal mapping saved with {len(new_mapping)} entries.")

if __name__ == "__main__":
    workspace_root = os.getcwd()
    output_mapping_file = os.path.join(workspace_root, "data", "document_mapping.json")
    
    # Base dir is not strictly needed as we resolve from mapping file path
    base_documents_dir = os.path.join(workspace_root, "data", "documents")

    renumber_and_clean_index(base_documents_dir, output_mapping_file)
