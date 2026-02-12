import json
import os
import uuid
import chromadb
from sentence_transformers import SentenceTransformer
from src.data_processing.chunker import split_text_smart

# Configuration
# Assuming runs from root
JSON_PATH = os.path.join("data", "all_processed_chapters.json")
DB_PATH = os.path.join("data", "chroma_db_test_json")
COLLECTION_NAME = "chapter_knowledge_base"
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"

def ingest():
    if not os.path.exists(JSON_PATH):
        print(f"Error: JSON file not found at {JSON_PATH}")
        return

    # 1. Load Data
    print(f"Loading data from {JSON_PATH}...")
    with open(JSON_PATH, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # 2. Setup Chroma
    print(f"Initializing ChromaDB at {DB_PATH}...")
    client = chromadb.PersistentClient(path=DB_PATH)
    embedding_model = SentenceTransformer(EMBEDDING_MODEL_NAME)
    
    # Define Embedding Function
    class LocalHuggingFaceEmbedding(chromadb.EmbeddingFunction):
        def __init__(self, model):
            self.model = model
        def __call__(self, input: chromadb.Documents) -> chromadb.Embeddings:
            return self.model.encode(input, convert_to_tensor=False).tolist()

    collection = client.get_or_create_collection(
        name=COLLECTION_NAME,
        embedding_function=LocalHuggingFaceEmbedding(embedding_model)
    )

    ids = []
    documents = []
    metadatas = []

    # 3. Process
    print("Chunking documents...")
    # Supports both list (old structure) or dict (grouped structure)
    if isinstance(data, list):
        # normalize to dict
        new_data = {}
        for item in data:
            doc = item.get("doc_name", "unknown")
            if doc not in new_data: new_data[doc] = []
            new_data[doc].append(item)
        data = new_data
        
    for doc_name, chapters in data.items():
        print(f"  Processing {doc_name} ({len(chapters)} chapters)...")
        for chapter in chapters:
            # Reconstruct text from pages
            pages = chapter.get("pages", {})
            if not pages:
                continue
                
            # Sort pages keys (which are strings in JSON)
            try:
                sorted_keys = sorted(pages.keys(), key=lambda x: int(x) if x.isdigit() else float('inf'))
            except:
                sorted_keys = sorted(pages.keys())
            
            full_text = "\n\n".join([str(pages[k]) for k in sorted_keys])
            
            # Use our new module!
            chunks = split_text_smart(full_text)
            
            for i, chunk in enumerate(chunks):
                if not chunk.strip(): continue
                
                chunk_id = str(uuid.uuid4())
                ids.append(chunk_id)
                documents.append(chunk)
                
                # title might be string or something else, handle safely
                title = chapter.get("title", "Unknown")
                if not isinstance(title, str): title = str(title)

                meta = {
                    "doc_name": doc_name,
                    "title": title,
                    "page_range": f"{sorted_keys[0]}-{sorted_keys[-1]}" if sorted_keys else "unknown",
                    "chunk_index": i,
                    "source": "json_loader_test"
                }
                metadatas.append(meta)

    # 4. Upsert
    if not ids:
        print("No chunks generated.")
        return

    BATCH_SIZE = 256
    print(f"Upserting {len(ids)} chunks to ChromaDB...")
    for i in range(0, len(ids), BATCH_SIZE):
        end = min(i + BATCH_SIZE, len(ids))
        print(f"  Batch {i} to {end}...")
        collection.upsert(
            ids=ids[i:end],
            documents=documents[i:end],
            metadatas=metadatas[i:end]
        )
    
    print(f"✅ Success! Database created at {DB_PATH}")
    print(f"Total Chunks: {len(ids)}")

if __name__ == "__main__":
    ingest()
