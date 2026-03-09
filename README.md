# RAG AI Demo (LangChain + Gemini + FAISS)

A simple Retrieval-Augmented Generation (RAG) project that:
- Loads a PDF knowledge source (`pdf document`)
- Splits text into chunks
- Creates embeddings with Google Gemini embeddings
- Stores vectors in FAISS
- Answers user questions using retrieved context and Gemini chat model

This repo includes:
- A script version: `RAGLANGC/RAGSCRIPT.py`
- A Streamlit UI: `RAGLANGC/streamlit_app.py`

## Tech Stack

- Python 3.11+
- LangChain
- Google Generative AI (Gemini)
- FAISS (local vector store)
- Streamlit

## Project Structure

```text
RAGLANG02_02/
  README.md
  RAGLANGC/
    RAGSCRIPT.py
    streamlit_app.py
    .env
    question related to pdf document
```

## Prerequisites

1. Python 3.11 or newer
2. A Google API key with Gemini access

## Setup

From repository root:

```powershell
cd RAGLANGC
python -m venv .venv
.\.venv\Scripts\activate
pip install -U pip
pip install streamlit python-dotenv langchain langchain-community langchain-google-genai langchain-text-splitters faiss-cpu pypdf
```

Create or edit `.env`:

```env
GOOGLE_API_KEY="your_google_api_key_here"
```

## Run Option 1: Python Script

```powershell
cd RAGLANGC
python RAGSCRIPT.py
```

It will run one sample question:
- `question about content in pdf document?`

## Run Option 2: Streamlit App (Recommended)

```powershell
cd RAGLANGC
streamlit run streamlit_app.py
```

In the sidebar:
- Set `GOOGLE_API_KEY` (or keep it in `.env`)
- Set PDF path (default: `what-is-cancer.pdf`)
- Optionally enable `Show retrieved chunks`

Then click **Run RAG**.

## How It Works

1. PDF is loaded with `PyPDFLoader`
2. Text is chunked (`chunk_size=500`, `chunk_overlap=50`)
3. Embeddings are created with `models/gemini-embedding-001`
4. Chunks are indexed in FAISS
5. Retriever uses MMR (`k=2`, `fetch_k=10`)
6. Prompt enforces context-only answers
7. Response is generated with `gemini-2.5-flash`

## Configuration Notes

- Change source PDFs in `RAGSCRIPT.py` (`files = [...]`) or update the Streamlit PDF path.
- To improve recall, tune:
  - `chunk_size`, `chunk_overlap`
  - retriever `k` / `fetch_k`
- Model settings used now:
  - Embeddings: `models/gemini-embedding-001`
  - Chat: `gemini-2.5-flash`

## Troubleshooting

- `GOOGLE_API_KEY is missing` or warning in UI:
  - Add key in `.env` or Streamlit sidebar.
- `PDF not found`:
  - Verify path in sidebar or place PDF inside `RAGLANGC/`.
- FAISS install issues on Windows:
  - Ensure you are on a supported Python version (3.11 recommended).

## Next improvements

1. Production ingestion pipeline
- Build async document ingestion with idempotent jobs, retries, dead-letter queue, and versioned chunks.
- Add OCR and layout-aware parsing (tables, headings, citations) with source provenance.

2. Hybrid retrieval architecture
- Combine dense vectors, BM25, metadata filtering, and reranking (cross-encoder).
- Add query rewriting, multi-query expansion, and dynamic top-k based on confidence.

3. Knowledge graph augmentation
- Extract entities and relations from chunks and store them in a graph database.
- Route questions to vector retrieval, graph traversal, or hybrid retrieval based on query intent.

4. Evaluation framework (offline and online)
- Build a gold dataset with question, expected answer, and citation labels.
- Track retrieval metrics (Recall@k, MRR, nDCG) and generation quality (faithfulness, groundedness).
- Add CI regression gates for prompt, retriever, and model changes.

5. Observability and tracing
- Instrument the full pipeline with tracing (request path, retrieval steps, generation stages).
- Monitor latency per stage, token and cost usage, cache hit rates, and failure modes.

6. Hallucination and safety controls
- Require citation-backed responses with source spans.
- Add unsupported-claim detection, confidence scoring, and safe fallback responses.

7. Performance and cost optimization
- Persist and reload FAISS index instead of rebuilding per run.
- Add embedding batching, semantic caching, and adaptive model routing for cost/latency tradeoffs.

8. Security and governance
- Move secrets to a proper secret manager and enforce key rotation.
- Add PII detection/redaction, role-based document access, and auditable query logs.

9. Service architecture
- Split system into ingestion, indexing, retrieval, and generation services.
- Add queue-based orchestration, backpressure handling, and horizontal scaling.

10. Release engineering
- Containerize with reproducible builds and pinned dependencies.
- Implement CI/CD with tests, eval gates, canary releases, and A/B testing for model/prompt updates.

11. Trust-centered UX
- Show citations, confidence indicators, and "why these chunks" explanations.
- Capture user feedback in-app and feed failures back into the eval dataset.

12. Model lifecycle management
- Maintain model and prompt version registry with rollback support.
- Use blue/green deployment strategy for model changes with quality thresholds.
