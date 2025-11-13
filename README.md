# README.md


## Project Overview

RAG-based insurance policy Q&A system using hybrid retrieval (BM25 + vector search) with strict extractive answer synthesis. The system is designed to provide zero-hallucination responses by enforcing that all answers must be verbatim from retrieved document chunks.

**Key Technology Stack:**
- **LLM**: Qwen2.5-7B-Instruct Q6_K (5.9GB) via llama.cpp server
- **Embeddings**: sentence-transformers/all-MiniLM-L6-v2 (384-dim)
- **Database**: PostgreSQL with pgvector extension
- **Retrieval**: Hybrid search (BM25 0.7 + Vector 0.3)
- **Backend**: FastAPI (port 8000)
- **UI**: Streamlit (port 8501) or Flask (ui_simple)

## Common Development Commands

### Starting the System

**Full system with all components:**
```bash
python run_full_system.py
```
This orchestrates: llama-server (8080) → FastAPI (8000) → embedding watcher → auto-ingestion → Streamlit UI (8501)

**UI only (assumes backend already running):**
```bash
python run_full_ui.py
```

**Manual component startup:**
```bash
# 1. Start llama-server (adjust -ngl for GPU layers)
cd scripts/llama.cpp
./build/bin/llama-server -m /path/to/models/Qwen2.5-7B-Instruct-Q6_K.gguf -ngl 10 --port 8080

# 2. Start FastAPI backend
uvicorn api.server:app --host 0.0.0.0 --port 8000 --reload

# 3. Start Streamlit UI
streamlit run ui/streamlit_app.py --server.port 8501 --server.address 0.0.0.0
```

**Using the convenience script:**
```bash
./start_ragbot.sh
```
Note: Hardcoded model path; update before use.

### Database Setup

**Start PostgreSQL with pgvector:**
```bash
docker-compose up -d
```
Creates `rag_db` with schema from `init.sql`. Default credentials: `rag_user:rag_password@localhost:5432/rag_db`

### Document Ingestion Pipeline

**Full ingestion (PDF → Markdown → Chunks → Embeddings → BM25):**
```bash
# 1. Convert PDFs to markdown with semantic chunking
python scripts/pdf2md_and_semantic_chunks.py

# 2. Load chunks into PostgreSQL
python scripts/load_all_chunks.py

# 3. Generate embeddings and store in DB
python scripts/embed_all_chunks.py

# 4. Build BM25 index
python scripts/build_bm25_index.py
```

**Auto-ingestion (watches `data/new_uploads/` for new PDFs):**
```bash
python scripts/auto_ingest.py
```
Triggers full pipeline automatically when new PDFs are added.

**Embedding watcher (watches `data/markdown/` and `data/chunks/`):**
```bash
python scripts/embedding_watcher.py
```

### Testing & Evaluation

**Run evaluation on golden dataset:**
```bash
python scripts/eval_run.py
```
Computes P@1, MRR, R@5, and answer accuracy against `data/eval_queries.json`. Results saved to `eval/eval_results.json`.

**Test hybrid retrieval interactively:**
```bash
python scripts/hybrid_retrieval.py
```

**Test retrieval evaluation:**
```bash
python scripts/evaluate_retrieval.py
```

**Test query rewriting:**
```bash
python scripts/query_rewriter.py
```

### Building llama.cpp

If `llama.cpp` is not built:
```bash
cd scripts/llama.cpp  # or llama.cpp/
mkdir -p build && cd build
cmake ..
cmake --build .
```
Binary will be at `llama.cpp/build/bin/llama-server`.

## High-Level Architecture

### Core Components

**1. Document Processing Pipeline**
- `scripts/pdf2md_and_semantic_chunks.py`: Converts PDFs to markdown, then applies semantic chunking with sentence-transformer-based similarity boundaries. Chunks are 150-900 tokens with 60-token overlap.
- Output: `data/markdown/*.md` and `data/chunks/*.jsonl`

**2. Retrieval System**
- **BM25 Index**: Pickled in `data/bm25_index.pkl` (contains tokenized corpus)
- **Vector Store**: PostgreSQL `chunk_embeddings` table with pgvector cosine similarity index
- **Hybrid Search**: `scripts/answer_synthesis.py:hybrid_search()` combines BM25 and vector scores with configurable weights (default: 0.7 BM25, 0.3 vector)

**3. Answer Synthesis**
- `scripts/answer_synthesis.py:synthesize_answer()`: Main orchestration function
  1. Query rewriting (expand abbreviations like "NCB" → "No Claim Bonus")
  2. Hybrid retrieval (top-10 chunks)
  3. Select top-5 for LLM context
  4. Call llama-server with strict extractive prompt
  5. Validate answer is verbatim from retrieved chunks

**4. Configuration**
- `config/synthesis_config.py`: **DO NOT MODIFY** without running evals
  - Contains locked Week 2 settings proven to eliminate hallucinations
  - Defines `SYSTEM_PROMPT_TEMPLATE` enforcing extractive-only responses

**5. API Layer**
- `api/server.py`: FastAPI with single `/ask` endpoint
  - Accepts `{"question": "..."}` JSON
  - Returns `{"query", "answer", "sources": ["doc_id::chunk_id", ...]}`

**6. User Interfaces**
- `ui/streamlit_app.py`: Chat interface with message history
- `ui_simple/`: Flask alternative with templates/static files

### Data Flow

```
PDF → pymupdf4llm → Markdown → Semantic Chunker → PostgreSQL (chunks table)
                                                 ↓
                                         sentence-transformers → PostgreSQL (chunk_embeddings)
                                                 ↓
User Query → Query Rewriter → Hybrid Search (BM25 + Vector) → Top-5 Chunks
                                                 ↓
                           Build Context → LLM (llama-server) → Extract Answer
                                                 ↓
                                    Validate (substring check) → Return to User
```

### Critical Design Decisions

**Zero-Hallucination Strategy:**
1. Strict system prompt forbids external knowledge
2. Post-generation validation: answer must be substring of retrieved chunks (with normalization)
3. Format enforcement: `Answer: <text>\nSources: [doc|chunk], ...`
4. If answer not found, return "Not found in provided excerpts."

**Chunking Strategy:**
- Semantic boundaries detected via sentence embedding cosine similarity (threshold: 0.38)
- Hard token cap: 900 tokens (immediate flush)
- Soft target: 450 tokens with 60-token overlap
- Preserves markdown heading lineage for context

**Retrieval Tuning:**
- BM25 weight 0.7 (term-based recall for specific policy details)
- Vector weight 0.3 (semantic generalization)
- Top-10 retrieval, top-5 for synthesis (reduces context noise)

## Important File Locations

**Configuration:**
- `config/synthesis_config.py` - Locked synthesis parameters
- `docker-compose.yml` - PostgreSQL setup
- `init.sql` - Database schema

**Core Scripts:**
- `scripts/answer_synthesis.py` - Main synthesis logic
- `scripts/hybrid_retrieval.py` - Standalone retrieval tester
- `scripts/pdf2md_and_semantic_chunks.py` - Document ingestion
- `scripts/query_rewriter.py` - Query expansion
- `scripts/reranker.py` - Cross-encoder reranking (optional)

**Data:**
- `data/chunks/*.jsonl` - Chunked documents (doc_id, chunk_id, text, approx_tokens)
- `data/bm25_index.pkl` - BM25 index + chunk metadata
- `data/eval_queries.json` - Golden evaluation dataset

**Orchestration:**
- `run_full_system.py` - Full stack startup
- `run_full_ui.py` - UI-only startup
- `start_ragbot.sh` - Bash convenience script

## Environment Setup

**Python Environment:**
```bash
python -m venv .RAGBOT
source .RAGBOT/bin/activate  # Linux/Mac
```

**Environment Variables (optional):**
- `DB_HOST` (default: localhost)
- `DB_PORT` (default: 5432)
- `DB_NAME` (default: rag_db)
- `DB_USER` (default: rag_user)
- `DB_PASSWORD` (default: rag_password)

## Development Workflow

### Adding New Documents

1. Place PDF in `data/new_uploads/` (if auto-ingest running) OR `data/policies/`
2. If manual: run full ingestion pipeline (see commands above)
3. Verify chunks in PostgreSQL: `SELECT COUNT(*) FROM chunks;`

### Modifying Retrieval

1. Edit weights in `scripts/answer_synthesis.py:synthesize_answer()` call to `hybrid_search()`
2. Test with `python scripts/hybrid_retrieval.py`
3. Run eval: `python scripts/eval_run.py`
4. If metrics improve, update `config/synthesis_config.py:HYBRID_SEARCH_DEFAULTS`

### Modifying Synthesis Prompt

**WARNING:** Changes to `config/synthesis_config.py:SYSTEM_PROMPT_TEMPLATE` can reintroduce hallucinations.

1. Edit prompt template
2. Run eval: `python scripts/eval_run.py`
3. Manually inspect `eval/eval_results.json` for hallucinations
4. Only commit if answer accuracy >= previous baseline

### Testing Changes

1. Start llama-server: `cd scripts/llama.cpp && ./build/bin/llama-server -m <model> --port 8080`
2. Run eval: `python scripts/eval_run.py`
3. Check metrics in console output and `eval/eval_results.json`

## Known Issues

- `run_full_system.py` hardcodes model path (line 82) - update before use
- `embedding_watcher.py` references non-existent `scripts/pdf_to_markdown.py` and `scripts/semantic_chunker.py` (should use `pdf2md_and_semantic_chunks.py`)
- `auto_ingest.py` also references non-existent scripts
- llama.cpp submodule may need recursive clone if missing
- GPU layers (`-ngl`) need tuning based on VRAM (RTX 3060 = 10 layers works)

 Milestone

The system achieved zero-hallucination status with:
- Query: "What is the No Claim Bonus for 2 consecutive claim-free years?"
- Answer: 25%
- Sources: [Reliance_General_Motor | chunk-33], [SBI_General_Motor | chunk-44]
- Validation: ✅ Passed

See `docs/week2_summary.md` for details.
