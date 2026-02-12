import os
import json
import re
import shutil

def normalize_name(filename):
    """
    Removes extensions and '-CURATED' suffix to find the base document name.
    """
    name, _ = os.path.splitext(filename)
    # Remove -CURATED if it exists at the end
    # Note: re.sub with $ ensures end of string
    name = re.sub(r"-CURATED$", "", name, flags=re.IGNORECASE)
    return name

def group_and_rename(base_dir, mapping_file):
    # Load current mapping
    with open(mapping_file, 'r', encoding='utf-8') as f:
        current_map = json.load(f)
        
    # Group files by base name
    groups = {}
    
    # We need to look at the *files on disk* mapped by the json to get their current loc.
    # The 'path' in json points to current file. 'original_name' is what we group by.
    
    for key, info in current_map.items():
        original = info['original_name']
        base = normalize_name(original)
        
        category = info['category']
        rel_path = info['path']
        current_full_path = os.path.join(os.path.dirname(mapping_file), rel_path)
        
        if not os.path.exists(current_full_path):
            print(f"Warning: File not found {current_full_path}, skipping.")
            continue
            
        if base not in groups:
            groups[base] = []
            
        groups[base].append({
            "category": category,
            "current_path": current_full_path,
            "original_name": original,
            "ext": os.path.splitext(original)[1]
        })
        
    # Sort groups by base name to ensure deterministic ID assignment
    sorted_base_names = sorted(groups.keys())
    
    new_mapping = {}
    current_id = 1
    
    # To avoid collisions (e.g. renaming doc_005 to doc_001 while doc_001 still exists),
    # we will first rename EVERYTHING to a temporary UUID-like name, then rename to final.
    # Actually, simpler: rename to "tmp_doc_XXX..."
    
    # Step 1: Rename all involved files to temp names
    print("Step 1: Renaming to temporary paths...")
    temp_files = [] 
    
    for base in sorted_base_names:
        file_list = groups[base]
        # Sort files within group by category for consistency if needed, though they go to diff folders
        file_list.sort(key=lambda x: x['category'])
        
        for file_info in file_list:
            old_path = file_info['current_path']
            dir_name = os.path.dirname(old_path)
            temp_name = f"temp_{os.urandom(4).hex()}_{os.path.basename(old_path)}"
            temp_path = os.path.join(dir_name, temp_name)
            
            try:
                os.rename(old_path, temp_path)
                file_info['temp_path'] = temp_path
                temp_files.append(file_info)
            except OSError as e:
                print(f"Error moving to temp {old_path}: {e}")

    # Step 2: Assign new IDs and rename to final
    print("Step 2: assigning new IDs and renaming to final...")
    
    for base in sorted_base_names:
        doc_id = f"doc_{current_id:03d}"
        file_group = groups[base]
        
        mapping_entry = {
            "base_name": base,
            "files": {}
        }
        
        for file_info in file_group:
            category = file_info['category']
            ext = file_info['ext']
            
            # Construct new filename
            new_filename = f"{doc_id}{ext}"
            
            # Temp path is where it is now
            temp_path = file_info.get('temp_path')
            if not temp_path: 
                continue # Skip if failed previous step
                
            dir_name = os.path.dirname(temp_path)
            final_path = os.path.join(dir_name, new_filename)
            
            try:
                os.rename(temp_path, final_path)
                
                # relative path for mapping
                rel_path = os.path.relpath(final_path, os.path.dirname(mapping_file)).replace("\\", "/")
                
                mapping_entry["files"][category] = {
                    "path": rel_path,
                    "original_name": file_info['original_name']
                }
                
                print(f"Mapped {base} [{category}] -> {new_filename}")
                
            except OSError as e:
                print(f"Error renaming final {temp_path}: {e}")
                
        new_mapping[doc_id] = mapping_entry
        current_id += 1

    # Save new mapping
    with open(mapping_file, 'w', encoding='utf-8') as f:
        json.dump(new_mapping, f, indent=4)
        print(f"New grouped mapping saved to {mapping_file}")

if __name__ == "__main__":
    workspace_root = os.getcwd()
    mapping_file = os.path.join(workspace_root, "data", "document_mapping.json")
    base_documents_dir = os.path.join(workspace_root, "data", "documents")
    
    group_and_rename(base_documents_dir, mapping_file)
