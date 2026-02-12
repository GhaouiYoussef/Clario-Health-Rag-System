# Healthcare RAG Agent

A RAG-based Q&A agent for general healthcare questions, grounded in curated documents. This project includes a Streamlit UI, evaluation suite, and strict safety guardrails.

## Features

- **Healthcare Q&A**: Answers general health questions based *only* on provided documents.
- **Safety First**: Clear disclaimers, triage suggestions, and refusal to partial diagnosis.
- **Guardrail Router**: Semantic routing to detect unsafe queries and refine safe ones before retrieval.
- **Citations**: All answers include references to specific source documents and chunks.
- **Evaluation**: Tools to benchmark accuracy, latency, and cost.

## Setup

1.  **Environment**:
    ```bash
    python -m venv venv
    source venv/bin/activate  # Windows: venv\Scripts\activate
    pip install -r requirements.txt
    ```

2.  **Configuration**:
    Copy `.env.example` to `.env` and add your API keys (e.g., OPENAI_API_KEY).

3.  **Data**:
    Place your PDF documents in `data/documents/`.

4.  **Run UI**:
    ```bash
    streamlit run app/main.py
    ```

5.  **Run Evaluation**:
    ```bash
    python evaluation/evaluate_agent.py
    ```

## Architecture

- **Routing**: LLM-based Guardrail Router parses intent for safety and optimizes queries.
- **Retrieval**: Hybrid Search (BM25 + ChromaDB) with Cross-Encoder Reranking and RRF Fusion.
- **Generation**: Gemini 2.5 Flash/Pro with strict system prompts for safety.
- **UI**: Streamlit for easy interaction.

## Disclaimer

**This tool does not provide medical advice.** It is for informational purposes only. Always consult a qualified healthcare professional for medical concerns.
