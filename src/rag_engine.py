import os
import chromadb
import google.genai as genai
from typing import List, Dict, Any, Optional
from src.models import SourceDocument
from dotenv import load_dotenv
import json
from sentence_transformers import SentenceTransformer, CrossEncoder
from rank_bm25 import BM25Okapi
import numpy as np

# Configure GenAI
# .env is in a parent directory
load_dotenv(os.path.join(os.path.dirname(__file__), "..", ".env"))

CLIENT = genai.Client(api_key=os.environ.get("GOOGLE_API_KEY"))

class HuggingFaceEmbeddingFunction(chromadb.EmbeddingFunction):
    def __init__(self, model_name="all-MiniLM-L6-v2", device="cpu"):
        self.model = SentenceTransformer(model_name, device=device)

    def __call__(self, input: chromadb.Documents) -> chromadb.Embeddings:
        # SentenceTransformer encodes to numpy array, convert to list
        embeddings = self.model.encode(input, convert_to_tensor=False)
        return embeddings.tolist()

class HealthcareRAG:
    def __init__(self, persist_directory: str = "./data/chroma_db_test_json", model_name: str = "gemini-2.5-flash"):
        self.persist_directory = persist_directory
        self.client = chromadb.PersistentClient(path=persist_directory)
        self.embedding_function = HuggingFaceEmbeddingFunction()
        self.collection = self.client.get_or_create_collection(
            name="chapter_knowledge_base", # Matched to custom_chunking.ipynb definition
            embedding_function=self.embedding_function
        )
        self.client_gemini = CLIENT
        self.model_name = model_name

        # Initialize Hybrid Retrieval Components
        print("Initializing Reranker and Sparse Index...")
        self.reranker = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')
        
        # Build BM25 Index from existing documents
        try:
            # Fetch all documents to build in-memory index
            all_data = self.collection.get() 
            self.bm25_ids = all_data['ids']
            if self.bm25_ids:
                tokenized_corpus = [doc.lower().split() for doc in all_data['documents']]
                self.bm25 = BM25Okapi(tokenized_corpus)
                print(f"✅ BM25 Sparse Index built for {len(self.bm25_ids)} documents.")
            else:
                print("⚠️ Collection is empty. BM25 index could not be built.")
                self.bm25 = None
        except Exception as e:
            print(f"Error building BM25 index: {e}")
            self.bm25 = None

    def query_router(self, query: str) -> Dict[str, Any]:
        """
        Guardrail Router:
        1. Checks for safety/relevance.
        2. Refines the query for better retrieval if safe.
        3. Returns decision and potentially modified query.
        """
        prompt = f"""You are a query router and safety guardrail for a healthcare RAG system.
        
        Task 1: Safety Check
        - Is this query asking for medical advice, dangerous information, or is it out of scope (not health-related)?
        - Reject if: "I have chest pain, what do I do?" (Emergency), "How to make a bomb?" (Harmful), "Write python code" (Out of Scope).
        - Accept if: General health info like "What are flu symptoms?", "Explain CPR".
        
        Task 2: Query Refinement
        - If safe, rewrite the query to be more search-friendly for a vector database (e.g., adding keywords, removing conversational filler).
        
        Input Query: "{query}"
        
        Output JSON Format:
        {{
            "action": "proceed" or "reject",
            "reason": "Safe to process" or "Explanation for rejection",
            "refined_query": "The optimized search query" (only if proceed)
        }}
        """
        
        try:
            response = self.generate_content(prompt)
            # Basic cleanup to ensure JSON parsing
            clean_text = response.text.replace('```json', '').replace('```', '').strip()
            decision = json.loads(clean_text)
            return decision
        except Exception as e:
            print(f"Router Error: {e}")
            # Fail-safe: Allow to proceed with original query if router fails
            return {"action": "proceed", "reason": "Router bypass error", "refined_query": query}

    def generate_content(self, prompt: str) -> Any:
        response = self.client_gemini.models.generate_content(
            model=self.model_name,
            contents=prompt
        )
        return response

    def hybrid_retrieval(self, query: str, n_results: int = 5, k_candidates: int = 20, diversity_ratio: float = 0.2, doc_diversity_ratio: float = 0.5, title_weight: float = 0.1) -> List[Dict]:
        """
        Advanced Retrieval Function:
        1. Hybrid Search (Sparse BM25 + Dense ChromaDB) with RRF Fusion
        2. Cross-Encoder Reranking
        3. Explicit Title Semantic Boost
        4. Diversity Filtering (Per Title and Per Document)
        """
        if not self.bm25:
            # Fallback to simple dense retrieval if BM25 failed
            results = self.collection.query(query_texts=[query], n_results=k_candidates)
            hits = []
            if results['ids']:
                for i in range(len(results['ids'][0])):
                    hits.append({
                        'content': results['documents'][0][i],
                        'metadata': results['metadatas'][0][i],
                        'id': results['ids'][0][i]
                    })
            return hits[:n_results]

        # --- Step 1: Hybrid Retrieval (Candidate Generation) ---
        if k_candidates < n_results:
            k_candidates = n_results * 4
        
        # A. Dense Search (Chroma)
        chroma_res = self.collection.query(
            query_texts=[query], 
            n_results=k_candidates,
        )
        
        # B. Sparse Search (BM25)
        tokenized_query = query.lower().split()
        bm25_scores = self.bm25.get_scores(tokenized_query)
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
            if idx < len(self.bm25_ids): 
                doc_id = self.bm25_ids[idx]
                rrf_scores[doc_id] = rrf_scores.get(doc_id, 0) + (1 / (rrf_k + rank + 1))
                
        sorted_candidates = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)[:k_candidates]
        candidate_ids = [x[0] for x in sorted_candidates]
        
        if not candidate_ids: return []
        
        # --- Step 2: Fetch Data & Apply Diversity Filter (Pre-Rerank) ---
        docs_data = self.collection.get(ids=candidate_ids, include=['documents', 'metadatas'])
        
        data_map = {}
        for i, doc_id in enumerate(docs_data['ids']):
            data_map[doc_id] = {
                'content': docs_data['documents'][i],
                'metadata': docs_data['metadatas'][i]
            }
        
        rerank_pairs = []
        final_items = []
        
        # Diversity limits based on candidate pool size to ensure diversity before reranking
        limit_per_title = max(1, int(k_candidates * diversity_ratio))
        limit_per_doc = max(1, int(k_candidates * doc_diversity_ratio))
        
        title_counts = {}
        doc_counts = {}
        
        for doc_id in candidate_ids:
            if doc_id in data_map:
                item = data_map[doc_id]
                
                # Check diversity constraints
                t = item['metadata'].get('title', '')
                d = item['metadata'].get('doc_name', item['metadata'].get('source', ''))
                
                if (title_counts.get(t, 0) < limit_per_title and 
                    doc_counts.get(d, 0) < limit_per_doc):
                    
                    item['id'] = doc_id
                    rerank_pairs.append([query, item['content']])
                    final_items.append(item)
                    
                    title_counts[t] = title_counts.get(t, 0) + 1
                    doc_counts[d] = doc_counts.get(d, 0) + 1

        # --- Step 3: Reranking ---
        if final_items:
            rerank_scores = self.reranker.predict(rerank_pairs)
            
            # Use the internal model from EmbeddingFunction
            q_emb = self.embedding_function.model.encode(query, convert_to_tensor=False)
            q_norm = np.linalg.norm(q_emb)

            for i, item in enumerate(final_items):
                raw_score = rerank_scores[i]
                normalized_rerank = 1 / (1 + np.exp(-raw_score))
                
                title = item['metadata'].get('title', '')
                t_emb = self.embedding_function.model.encode(title, convert_to_tensor=False)
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
            
        # Return top n_results from the diverse, reranked set
        return final_items[:n_results]

    def split_text(self, text: str, chunk_size: int = 1000, overlap: int = 200) -> List[str]:
        """Simple recursive-like splitting strategy."""
        chunks = []
        if not text:
            return chunks
        
        start = 0
        while start < len(text):
            end = start + chunk_size
            chunk = text[start:end]
            chunks.append(chunk)
            start += (chunk_size - overlap)
        return chunks

    def ingest_documents(self, documents: List[SourceDocument]):
        """Splits documents and adds them to ChromaDB."""
        ids = []
        embeddings_texts = []
        metadatas = []

        for doc in documents:
            chunks = self.split_text(doc.page_content)
            for i, chunk in enumerate(chunks):
                # Unique ID
                chunk_id = f"{doc.metadata.get('source', 'doc')}_p{doc.metadata.get('page', 0)}_{i}"
                ids.append(chunk_id)
                embeddings_texts.append(chunk)
                # Helper to clean metadata for Chroma (no nested lists/dicts)
                clean_meta = {k: str(v) for k, v in doc.metadata.items()}
                metadatas.append(clean_meta)

        # Batch add to avoid limits if necessary, though Chroma handles reasonable batch sizes
        if ids:
            # Upsert ensures we update if ID exists
            self.collection.upsert(
                ids=ids,
                documents=embeddings_texts,
                metadatas=metadatas
            )
            print(f"Ingested {len(ids)} chunks into vector store.")
            
             # Rebuild BM25 after ingestion (Memory efficient way: just re-init specific doc if possible, but here we rebuild all for simplicity/correctness)
            try:
                # Update corpus list - Note: For Production, use add_documents on BM25 or similar
                # Here we just re-fetch for consistency in this prototype
                all_data = self.collection.get()
                self.bm25_ids = all_data['ids']
                tokenized_corpus = [doc.lower().split() for doc in all_data['documents']]
                self.bm25 = BM25Okapi(tokenized_corpus)
                print("BM25 Index updated.")
            except:
                pass
        else:
            print("No text to ingest.")

    def get_chunk(self, chunk_id: str) -> Dict[str, Any]:
        """Retrieves a single chunk by ID."""
        try:
            result = self.collection.get(ids=[chunk_id], include=["documents", "metadatas"])
            if result['ids']:
                return {
                    "id": result['ids'][0],
                    "content": result['documents'][0],
                    "metadata": result['metadatas'][0]
                }
        except Exception as e:
            print(f"Error fetching chunk {chunk_id}: {e}")
        return None

    def get_answer(self, query: str, n_results: int = 5, k_candidates: int = 20, doc_diversity: float = 0.5, status_callback=None) -> Dict[str, Any]:
        """
        Retrieves context and generates answer using Gemini directly.
        Supports status_callback(step_name, message) for UI updates.
        """
        
        # 0. Router / Guardrail Check
        if status_callback: status_callback("guardrails", "Checking query safety and intent...")
        
        router_decision = self.query_router(query)
        
        if router_decision.get('action') == 'reject':
            if status_callback: status_callback("rejected", "Query rejected by guardrails.")
            return {
                "result": f"⚠️ Query Refused: {router_decision.get('reason', 'Safety violation')}. Please consult a professional.",
                "source_documents": [],
                "token_usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
            }
            
        search_query = router_decision.get('refined_query', query)
        print(f"Router Refined Query: '{query}' -> '{search_query}'")
        
        # 1. Retrieve using Hybrid Reranking (using REFINED query)
        if status_callback: status_callback("retrieval", f"Searching usage hybrid retrieval for: '{search_query}'...")
        
        # Adjust diversity settings to ensure we get results
        # If strict limits yield too few results, we can fallback or relax them? 
        # For now, let's keep logic but ensure caller handles 'n_results'
        
        hits = self.hybrid_retrieval(search_query, n_results=n_results, k_candidates=k_candidates, doc_diversity_ratio=doc_diversity)
        
        if status_callback: status_callback("reranking", f"Reranking and filtering {len(hits)} candidates...")

        context_parts = []
        source_docs = [] # For return format compatibility
        
        for i, hit in enumerate(hits):
            doc_text = hit['content']
            meta = hit['metadata']
            score = hit.get('final_score', 0.0)
            chunk_id = hit.get('id', 'unknown')
            
            # Use preferred keys, fallback to legacy
            source = meta.get('doc_name', meta.get('source', 'Unknown'))
            page = meta.get('page_range', meta.get('page', '?'))
            title = meta.get('title', 'Unknown Title')
            
            print(f"Retrieved Doc {i+1} [Score: {score:.4f}]: {title} | Source: {source} | Page: {page}")
            
            source_info = f"Source: {source}, Page: {page}, Title: {title}"
            context_parts.append(f"[{source_info}]\n{doc_text}")
            
            # Ensure ID is included in source docs for UI
            doc_entry = {
                "page_content": doc_text, 
                "metadata": meta,
                "id": chunk_id,
                "score": score
            }
            source_docs.append(doc_entry)

        context_str = "\n\n".join(context_parts)

        # 2. Generate
        if status_callback: status_callback("generation", "Generating response with Gemini...")
        
        prompt = f"""You are an expert healthcare assistant. Answering the user's question using ONLY the provided context.

        Guidelines:
        1. **Accuracy**: Use only the information from the context. If the answer isn't there, state "I cannot find the answer in the provided documents."
        2. **Safety**: Do NOT provide medical diagnoses. If symptoms are severe, advise seeking professional help.
        3. **Clarity**: Structure your answer with clear headings or bullet points if appropriate.
        4. **Citations**: Cite sources inline using [Source: doc_name, Page: X]. Consolidate citations where possible (e.g., [Source: doc_002, Page: 5-11]).
        5. **Tone**: Professional, objective, and empathetic.

        Context:
        {context_str}
        
        Question: {query}
        
        Answer:"""

        try:
            response = self.generate_content(
                prompt=prompt,
                # model_name="gemini-2.5-flash-lite"
            )
            answer = response.text
            usage = response.usage_metadata
            token_usage = {
                "prompt_tokens": usage.prompt_token_count,
                "completion_tokens": usage.candidates_token_count,
                "total_tokens": usage.total_token_count
            }
        except Exception as e:
            answer = f"Error generating answer: {e}"
            token_usage = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
        
        if status_callback: status_callback("complete", "Response generated.")

        return {
            "result": answer,
            "source_documents": source_docs,
            "token_usage": token_usage
        }
