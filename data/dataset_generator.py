import requests
import json
import time
import numpy as np
# Import your specific libraries for the retrieval system
# (Assuming chromadb, sentence_transformers, rank_bm25 are already imported and initialized)
import uuid
import chromadb
from sentence_transformers import SentenceTransformer
import re

# --- 2. Setup ChromaDB & Embedding ---
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"
DB_PATH = "./chroma_db_VI"

print("Initializing ChromaDB and Embeddings...")
client = chromadb.PersistentClient(path=DB_PATH)
embedding_model = SentenceTransformer(EMBEDDING_MODEL_NAME)

class LocalHuggingFaceEmbedding(chromadb.EmbeddingFunction):
    def __init__(self, model):
        self.model = model
    def __call__(self, input: chromadb.Documents) -> chromadb.Embeddings:
        return self.model.encode(input, convert_to_tensor=False).tolist()

collection = client.get_or_create_collection(
    name="chapter_knowledge_base",
    embedding_function=LocalHuggingFaceEmbedding(embedding_model)
)

# ==========================================
# 1.1 SETUP: MISSING ENTITIES (BM25, Reranker, IDs)
# ==========================================
from rank_bm25 import BM25Okapi
from sentence_transformers import CrossEncoder

print("Loading Reranker Model...")
rerank_model_name = 'cross-encoder/ms-marco-MiniLM-L-6-v2'
reranker = CrossEncoder(rerank_model_name)

print("Fetching all documents from ChromaDB to build Sparse Index...")
all_data = collection.get() # Fetches everything
ids = all_data['ids']
documents = all_data['documents']

if not ids:
    raise ValueError("ChromaDB collection is empty! Please run your ingestion notebook 'custom_chunking.ipynb' first.")

print(f"Building BM25 Index for {len(documents)} chunks...")
tokenized_corpus = [doc.lower().split() for doc in documents]
bm25 = BM25Okapi(tokenized_corpus)
print("✅ Initialization Complete.")

# ==========================================
# 1. SETUP: API & RETRIEVAL CONFIGURATION
# ==========================================

NVIDIA_API_KEY = "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" # Your Key
INVOKE_URL = "https://integrate.api.nvidia.com/v1/chat/completions"

# List of seed queries to "probe" your database and get diverse contexts.
# Adjust these based on the actual documents you ingested (e.g., if you only have heart documents, only ask heart questions).
SEED_QUERIES = [
    "standard CPR procedure steps",
    "symptoms of type 2 diabetes",
    "managing high blood pressure hypertension",
    "vaccination schedule for infants",
    "treatment for seasonal influenza",
    "signs of stroke and immediate action",
    "asthma attack management",
    "dietary guidelines for obesity",
    "mental health depression signs",
    "antibiotic resistance prevention"
]

# ==========================================
# 2. YOUR RETRIEVAL FUNCTION
# ==========================================
# Note: This function assumes 'collection', 'bm25', 'reranker', 'embedding_model', and 'ids' 
# are globally defined in your environment as per your previous setup.

def hybrid_rerank_retrieval(query: str, n_results=5, diversity_ratio=0.2, title_weight=0.1):
    """
    Your provided hybrid retrieval function.
    """
    # --- Step 1: Hybrid Retrieval (Candidate Generation) ---
    k_candidates = n_results * 4 
    
    # A. Dense Search (Chroma)
    chroma_res = collection.query(
        query_texts=[query], 
        n_results=k_candidates,
    )
    
    # B. Sparse Search (BM25)
    tokenized_query = query.lower().split()
    bm25_scores = bm25.get_scores(tokenized_query)
    top_bm25_indices = np.argsort(bm25_scores)[::-1][:k_candidates]
    
    # C. Reciprocal Rank Fusion (RRF)
    rrf_scores = {}
    rrf_k = 60
    
    # Process Dense Ranks
    if chroma_res['ids']:
        for rank, doc_id in enumerate(chroma_res['ids'][0]):
            rrf_scores[doc_id] = rrf_scores.get(doc_id, 0) + (1 / (rrf_k + rank + 1))
            
    # Process Sparse Ranks
    for rank, idx in enumerate(top_bm25_indices):
        if idx < len(ids): 
            doc_id = ids[idx]
            rrf_scores[doc_id] = rrf_scores.get(doc_id, 0) + (1 / (rrf_k + rank + 1))
            
    sorted_candidates = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)[:k_candidates]
    candidate_ids = [x[0] for x in sorted_candidates]
    
    if not candidate_ids: return []
    
    # --- Step 2: Fetch Data ---
    docs_data = collection.get(ids=candidate_ids, include=['documents', 'metadatas'])
    
    data_map = {}
    for i, doc_id in enumerate(docs_data['ids']):
        data_map[doc_id] = {
            'content': docs_data['documents'][i],
            'metadata': docs_data['metadatas'][i]
        }
    
    rerank_pairs = []
    final_items = []
    
    for doc_id in candidate_ids:
        if doc_id in data_map:
            item = data_map[doc_id]
            rerank_pairs.append([query, item['content']])
            item['id'] = doc_id
            final_items.append(item)
            
    # --- Step 3: Reranking ---
    if final_items:
        rerank_scores = reranker.predict(rerank_pairs)
        
        q_emb = embedding_model.encode(query, convert_to_tensor=False)
        q_norm = np.linalg.norm(q_emb)

        for i, item in enumerate(final_items):
            raw_score = rerank_scores[i]
            normalized_rerank = 1 / (1 + np.exp(-raw_score))
            
            title = item['metadata'].get('title', '')
            t_emb = embedding_model.encode(title, convert_to_tensor=False)
            t_norm = np.linalg.norm(t_emb)
            
            if q_norm > 0 and t_norm > 0:
                title_sim = np.dot(q_emb, t_emb) / (q_norm * t_norm)
            else:
                title_sim = 0
            
            title_sim = max(0.0, min(1.0, title_sim))
            item['final_score'] = normalized_rerank + (title_sim * title_weight)
            item['title_score'] = title_sim
            item['title_txt'] = title
            
        final_items.sort(key=lambda x: x['final_score'], reverse=True)
        
    # --- Step 4: Diversity Filtering ---
    limit_per_title = max(1, int(n_results * diversity_ratio))
    selection = []
    title_counts = {}
    
    for item in final_items:
        t = item['title_txt']
        if title_counts.get(t, 0) < limit_per_title:
            selection.append(item)
            title_counts[t] = title_counts.get(t, 0) + 1
            
        if len(selection) >= n_results:
            break
            
    return selection

# ==========================================
# 3. GENERATION AGENT (Nvidia Kimi)
# ==========================================

def call_kimi_generator(context_text):
    """
    Calls Nvidia Kimi to generate QA pairs from the provided context.
    """
    
    prompt = f"""
    You are an expert medical educational assistant. 
    I will provide you with a set of retrieved document chunks.
    
    Your task:
    1. Read the context carefully.
    2. Generate exactly 3 Question and Answer pairs based *strictly* on the information in the context.
    3. If the context contains specific medical advice, ensure the answer cites the context implicitly.
    4. Format the output as a valid JSON list of objects.
    
    Example Format:
    [
        {{"question": "What is the compression rate for CPR?", "answer": "According to the text, compressions should be performed at a rate of 100-120 per minute.", "source_snippet": "compressions at 100-120 bpm"}},
        ...
    ]

    CONTEXT:
    {context_text}
    
    Generate the JSON now:
    """

    headers = {
      "Authorization": f"Bearer {NVIDIA_API_KEY}",
      "Accept": "application/json", # Prefer non-stream for easier JSON parsing
      "Content-Type": "application/json"
    }

    payload = {
      "model": "moonshotai/kimi-k2.5",
      "messages": [{"role":"user", "content": prompt}],
      "max_tokens": 4096,
      "temperature": 0.7,
      "top_p": 1.00,
      "stream": False # Set to False to get the full JSON at once for parsing
    }

    try:
        response = requests.post(INVOKE_URL, headers=headers, json=payload)
        response.raise_for_status()
        result = response.json()
        content = result['choices'][0]['message']['content']
        
        # Clean up markdown formatting if the model adds it (e.g. ```json ... ```)
        if "```json" in content:
            content = content.split("```json")[1].split("```")[0]
        elif "```" in content:
            content = content.split("```")[1].split("```")[0]
            
        return json.loads(content)
        
    except Exception as e:
        print(f"Error calling Kimi: {e}")
        return []

# ==========================================
# 4. MAIN LOOP: BUILD DATASET
# ==========================================

def generate_golden_dataset(target_size=25):
    qa_dataset = []
    
    print(f"--- Starting Data Generation (Target: {target_size} pairs) ---")
    
    for seed_query in SEED_QUERIES:
        if len(qa_dataset) >= target_size:
            break
            
        print(f"\nProcessing Seed Query: '{seed_query}'")
        
        # 1. Retrieve Context (Top 10)
        try:
            hits = hybrid_rerank_retrieval(seed_query, n_results=10)
        except Exception as e:
            print(f"Skipping query '{seed_query}' due to retrieval error: {e}")
            continue
            
        if not hits:
            print("No documents found.")
            continue
            
        # 2. Prepare Context Block
        # We combine the text of the top 10 hits to give the LLM material to work with
        context_text = ""
        for i, hit in enumerate(hits):
            context_text += f"\n[Document {i+1} Title: {hit['title_txt']}]\n{hit['content']}\n"
            
        # 3. Generate QA with Kimi
        generated_pairs = call_kimi_generator(context_text)
        
        if generated_pairs:
            print(f" > Generated {len(generated_pairs)} pairs.")
            
            # Add metadata to the pairs
            for pair in generated_pairs:
                pair['retrieval_seed_query'] = seed_query
                # We can store the top retrieval ID as a reference
                pair['primary_source_doc'] = hits[0]['metadata'].get('doc_name', 'Unknown')
                qa_dataset.append(pair)
        else:
            print(" > Failed to generate pairs.")
            
        # Rate limit protection
        time.sleep(1)

    # ==========================================
    # 5. EXPORT
    # ==========================================
    
    # Trim to exact target if went over
    final_dataset = qa_dataset[:target_size]
    
    output_file = "healthcare_golden_dataset.json"
    with open(output_file, "w") as f:
        json.dump(final_dataset, f, indent=2)
        
    print(f"\nSUCCESS: Generated {len(final_dataset)} Q/A pairs.")
    print(f"Saved to {output_file}")
    
    # Preview
    print("\n--- Preview of First 2 Items ---")
    print(json.dumps(final_dataset[:2], indent=2))

# Run the generator
if __name__ == "__main__":

    generate_golden_dataset()
