# Presentation Outline: Healthcare RAG Agent

## Slide 1: Title & Project Goal
- **Title**: Healthcare RAG Q&A Agent
- **Goal**: Deliver a grounded, safe, and accurate Q&A system for general medical queries.
- **Key Constraints**: Safety (no medical advice), accuracy (no hallucinations), privacy (local RAG option).

## Slide 2: Problem Framing & Safety
- **Risk**: Hallucinations in healthcare are dangerous.
- **Solution**:
  - Strict System Prompts ("You are not a doctor").
  - Deterministic constraints (Temperature = 0).
  - Explicit Citation requirement.
  - Triage/Refusal logic for severe symptoms.

## Slide 3: Architecture
- **Inference**: GPT-4o (Reasoning & Generation).
- **Retrieval**: ChromaDB (Vector Store).
- **Embedding**: OpenAI Embeddings (ada-002 or v3-small).
- **Chunking**: RecursiveCharacterTextSplitter (1000 chunks, 200 overlap) to maintain context.

## Slide 4: Retrieval Strategy
- **Why RAG?**: LLMs have outdated or generalized knowledge. RAG grounds answers in *trusted* documents (CDC, WHO).
- **Process**:
  1. Load PDFs.
  2. Chunk text.
  3. Embed & Index.
  4. Retrieve top-k (k=4) matches.
  5. Synthesize answer with citations.

## Slide 5: Evaluation Setup
- **Dataset**: 15 custom questions (Flu, Diabetes, First Aid).
- **Metrics**:
  - **Latency**: Time to first token / total time.
  - **Cost**: Token usage per query.
  - **Accuracy**: Checked against expected ground truth key points.

## Slide 6: Results (Placeholder)
- *Latency*: ~2-3 seconds avg.
- *Cost*: ~$0.01 per 10 queries (estimated).
- *Quality*: High relevance for factual queries; good refusal behavior for "diagnose me" queries.

## Slide 7: Demo Walkthrough
- [Screen recording or Live Demo steps]
- 1. Ingest Data.
- 2. Ask "What are flu symptoms?"
- 3. Show safety disclaimer in action.
- 4. Show citations.

## Slide 8: Limitations & Next Steps
- **Limitations**:
  - Image/Table parsing in PDFs is basic.
  - No "history" in current MVP (stateless).
- **Next Steps**:
  - Hybrid Search (Keyword + Semantic).
  - "Reranking" step for better precision.
  - Multi-turn conversation memory.
