# Healthcare RAG System

A sophisticated Retrieval-Augmented Generation (RAG) system designed for healthcare queries, featuring a custom-built data pipeline, advanced hybrid retrieval with reranking, and a transparent user interface.

## 🏗️ Architecture & Data Pipeline

This system moves beyond standard RAG implementations by focusing heavily on data quality and contextual integrity.

### 1. Data Processing & Curation
*   **Manual Curation**: Raw documents were carefully curated to remove noisy elements such as headers, footers, bibliographies, and extensive appendices that could dilute retrieval quality.
*   **Structure-Aware Extraction**: We utilized a custom structuring approach to extract content while preserving the hierarchy of the documents.
*   **Contextual Chapter-Based Chunking**: 
    *   Instead of always splitting at arbitrary character limits, we implemented **"Smart Chunking"**.
    *   This logic respects sentence boundaries and paragraph breaks (checking for points and newlines) to ensure chunks are semantically complete and never cut halfway through a thought.
    *   **Context Preservation**: Meaningful chapter titles are prepended to every chunk within that section. This ensures that even small text fragments retain their broader context during retrieval.

### 2. The RAG Engine
*   **Guardrail & Query Router**: 
    *   The entry point is an intelligent Router Agent.
    *   **Safety Check**: Detects unsafe or medical-emergency queries (e.g., "I'm having a heart attack") and rejects them immediately.
    *   **Query Refinement**: Rewrites user queries to be more search-friendly (e.g., removing conversational filler) to improve vector matching.
    *   *Note: This step introduces a slight latency (~3 seconds) but significantly improves safety and retrieval accuracy.*
*   **Hybrid Retrieval**: Combines sparse keyword search (BM25) with dense vector search (ChromaDB) to capture both exact matches and semantic meaning.
*   **Reranking**: A Cross-Encoder model re-scores the retrieved candidates to promote the most relevant documents to the top before generation.
*   **Generation**: Uses **Google Gemini 2.5 (Flash/Pro)** to synthesize answers based *strictly* on the retrieved context.

### 3. User Interface (Streamlit)
*   **Observability**: The UI provides detailed insights into the "Black Box" of RAG. Users can see real-time status updates for every step (Guardrails -> Search -> Rerank -> Generate).
*   **Transparency**: Users can view the exact chunks used to generate the answer, including the **Chunk ID**, **Relevance Score**, and the **Full Content** of the chunk via expandable sections.
*   **Customization**: The sidebar allows users to swap between model variants (Flash vs. Pro) and tune retrieval parameters.

---

## 📊 Datasets & Evaluation

To ensure reliability, we generated custom evaluation datasets.

### Dataset Creation
*   **Synthetic Generation**: Leveraging the **NVIDIA API (Llama 3 / Kim)**, we created synthetic Q&A pairs.
*   **Methodology**: The generator randomly selected healthcare topics and retrieved documents to formulate ground-truth questions and answers.
*   **Datasets**:
    1.  **Safety/Adversarial Dataset**: Used to test the Router's ability to refuse out-of-scope or dangerous queries.
    2.  **Golden Dataset**: A high-quality set of questions used to benchmark the model's accuracy, faithfulness, and latency.

### Performance & Resource Report
*   **Baseline Evaluation**: Performance metrics were initially established *before* the introduction of the Router Agent to set a baseline for raw retrieval speed.
*   **Resource Usage**: The system is optimized to run efficiently on standard hardware.
    *   **RAM**: ~2 GB memory footprint.
    *   **CPU**: Moderate usage (CPU-bound for local embeddings/reranking) for roughly one hour of continuous operation during batch processing.
*   **Cost Efficiency**: The implementation using Gemini Flash provides an extremely low cost per 1,000 tokens compared to larger proprietary models.
*   **Latency Trade-off**: While the Router Agent adds ~3 seconds to the response time, it is a necessary trade-off for the implemented safety guardrails and query optimization.

## 🚀 Setup & Usage

1.  **Environment Setup (using `uv`)**:
    We use `uv` for fast, reproducible dependency management and environment setup.
    ```bash
    # Install uv if you haven't already
    pip install uv

    # Create a virtual environment and install dependencies
    uv venv venv-test --python 3.11.4
    
    # Activate the environment
    # Windows: venv-test\Scripts\activate
    # Mac/Linux: source venv-test/bin/activate

    # Install dependencies from locked requirements
    uv pip install -r requirements.txt
    ```

2.  **Configuration**:
    If no `.env` file exists, the application will prompt you to enter your `GOOGLE_API_KEY` directly in the sidebar on the first run and save it for you. Alternatively, you can create a `.env` file manually:
    ```bash
    GOOGLE_API_KEY=your_key_here
    ```

3.  **Run Application**:
    ```bash
    streamlit run app/main.py
    ```

## Disclaimer
**This tool does not provide medical advice.** It is a demonstration of RAG technology for informational lookup only. Always consult a qualified healthcare professional.
