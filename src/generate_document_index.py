import os
import json

def generate_file_index(base_dir, output_file):
    """
    Scans the documents directory and creates a mapping from a simple ID to the file information.
    """
    mapping = {}
    
    # Define the subdirectories to scan
    subdirs = ['abrevs', 'curated', 'raw']
    
    files_to_process = []

    for category in subdirs:
        dir_path = os.path.join(base_dir, category)
        if not os.path.exists(dir_path):
            print(f"Directory not found: {dir_path}")
            continue
            
        # Get all files, sorted to ensure deterministic order
        files = sorted([f for f in os.listdir(dir_path) if os.path.isfile(os.path.join(dir_path, f))])
        
        for filename in files:
             files_to_process.append({
                 "category": category,
                 "filename": filename,
                 "full_path": os.path.join(dir_path, filename).replace("\\", "/")
             })

    # Sort all files by category then filename to ensure absolute stability of IDs
    files_to_process.sort(key=lambda x: (x['category'], x['filename']))

    current_index = 1
    for file_info in files_to_process:
        # Generate a simple ID
        doc_id = f"doc_{current_index:03d}"
        
        # Calculate relative path from the output file's directory (data/)
        # output_file is in data/ so dirname is data/
        # file is in data/documents/category/filename
        
        # Actually, let's just store the path relative to the project root or the data folder
        # Storing relative to 'data' folder makes sense if json is in 'data' folder
        
        rel_path = os.path.relpath(file_info['full_path'], os.path.dirname(output_file)).replace("\\", "/")

        mapping[doc_id] = {
            "original_name": file_info['filename'],
            "category": file_info['category'],
            "path": rel_path
        }
        current_index += 1

    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(mapping, f, indent=4)
    
    print(f"Generated index for {len(mapping)} files.")
    print(f"Mapping saved to: {output_file}")

if __name__ == "__main__":
    # Base paths
    # Assuming the script is run from the project root
    workspace_root = os.getcwd() 
    base_documents_dir = os.path.join(workspace_root, "data", "documents")
    output_mapping_file = os.path.join(workspace_root, "data", "document_mapping.json")
    
    generate_file_index(base_documents_dir, output_mapping_file)
