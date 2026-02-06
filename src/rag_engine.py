import os
import chromadb
import google.genai as genai
from typing import List, Dict, Any, Optional
from src.models import SourceDocument
from dotenv import load_dotenv
import json
from sentence_transformers import SentenceTransformer

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
    def __init__(self, persist_directory: str = "./data/chroma_db", model_name: str = "gemini-1.5-flash"):
        self.persist_directory = persist_directory
        self.client = chromadb.PersistentClient(path=persist_directory)
        self.embedding_function = HuggingFaceEmbeddingFunction()
        self.collection = self.client.get_or_create_collection(
            name="healthcare_docs",
            embedding_function=self.embedding_function
        )
        self.client_gemini = CLIENT
        self.model_name = model_name
    def generate_content(self, prompt: str) -> Any:
        response = self.client_gemini.models.generate_content(
            model=self.model_name,
            contents=prompt
        )
        return response
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
        else:
            print("No text to ingest.")

    def get_answer(self, query: str) -> Dict[str, Any]:
        """Retrieves context and generates answer using Gemini directly."""
        
        # 1. Retrieve
        results = self.collection.query(
            query_texts=[query],
            n_results=4
        )
        
        retrieved_docs = results['documents'][0] if results['documents'] else []
        retrieved_metas = results['metadatas'][0] if results['metadatas'] else []

        context_parts = []
        source_docs = [] # For return format compatibility
        
        for i, doc_text in enumerate(retrieved_docs):
            meta = retrieved_metas[i]
            print(f"Retrieved Doc {i+1} Metadata: {meta}")
            source_info = f"Source: {meta.get('source', 'Unknown')}, Page: {meta.get('page', '?')}"
            context_parts.append(f"[{source_info}]\n{doc_text}")
            source_docs.append({"page_content": doc_text, "metadata": meta})

        context_str = "\n\n".join(context_parts)

        # 2. Generate
        prompt = f"""Use the following pieces of context to answer the question at the end. 
        If you don't know the answer based on the context, just say that you don't know, don't try to make up an answer.
        
        IMPORTANT SAFETY GUIDELINES:
        - You are a helpful healthcare assistant, but you are NOT a doctor.
        - Do not provide medical diagnoses or prescribe treatments.
        - If the user describes severe symptoms, recommend seeking urgent professional care.
        - Always cite your sources from the context provided using the format [Source: ..., Page: ...].
        
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

        return {
            "result": answer,
            "source_documents": source_docs,
            "token_usage": token_usage
        }
