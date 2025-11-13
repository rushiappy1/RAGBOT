import pickle
import psycopg2
from sentence_transformers import SentenceTransformer
import os
from pathlib import Path
import requests
import re
import time
from scripts.query_rewriter import rewrite_query

import warnings
warnings.filterwarnings('ignore', message='CUDA initialization')

os.environ['CUDA_VISIBLE_DEVICES'] = ''

from config.synthesis_config import (
    HYBRID_SEARCH_DEFAULTS,
    SYNTHESIS_DEFAULTS,
    SYSTEM_PROMPT_TEMPLATE,
    VALIDATION_ENABLED,
    REGENERATION_ON_INVALID_FORMAT
)

# Change to repo root
repo_root = Path(__file__).parent.parent
os.chdir(repo_root)
print(f"Working directory: {os.getcwd()}\n")

# Database config
DB_HOST = os.getenv("DB_HOST", "localhost")
DB_PORT = int(os.getenv("DB_PORT", "5432"))
DB_NAME = os.getenv("DB_NAME", "rag_db")
DB_USER = os.getenv("DB_USER", "rag_user")
DB_PASSWORD = os.getenv("DB_PASSWORD", "rag_password")

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

# Global variables for lazy initialization
_bm25 = None
_chunks = None
_model = None
_conn = None
_cursor = None

def _initialize_resources():
    global _bm25, _chunks, _model, _conn, _cursor
    
    if _bm25 is not None:
        return  # Already initialized
    
    print("Loading BM25 index...")
    with open("data/bm25_index.pkl", "rb") as f:
        index_data = pickle.load(f)
    _bm25 = index_data["bm25"]
    _chunks = index_data["chunks"]
    
    print("Loading sentence transformer model...")
    _model = SentenceTransformer(MODEL_NAME, device="cpu")
    
    print("Connecting to PostgreSQL...")
    _conn = psycopg2.connect(
        host=DB_HOST, port=DB_PORT, database=DB_NAME,
        user=DB_USER, password=DB_PASSWORD
    )
    _cursor = _conn.cursor()
    print("Setup complete.\n")

def hybrid_search(query, top_k=None, bm25_weight=None, vector_weight=None):
    _initialize_resources()
    
    top_k = top_k or HYBRID_SEARCH_DEFAULTS["top_k"]
    bm25_weight = bm25_weight or HYBRID_SEARCH_DEFAULTS["bm25_weight"]
    vector_weight = vector_weight or HYBRID_SEARCH_DEFAULTS["vector_weight"]

    tokenized_query = query.lower().split()
    bm25_scores = _bm25.get_scores(tokenized_query)
    bm25_scores = (bm25_scores - bm25_scores.min()) / (bm25_scores.max() - bm25_scores.min() + 1e-8)

    q_emb = _model.encode([query], normalize_embeddings=True)[0].tolist()
    _cursor.execute("""
        SELECT c.chunk_id, c.doc_id, 1 - (ce.embedding <=> %s::vector) as similarity
        FROM chunk_embeddings ce
        JOIN chunks c ON c.chunk_id = ce.chunk_id
    """, (q_emb,))
    vector_results = {row[0]: (row[1], row[2]) for row in _cursor.fetchall()}

    combined_scores = []
    for i, chunk in enumerate(_chunks):
        chunk_id = chunk["chunk_id"]
        vec_score = vector_results.get(chunk_id, (None, 0.0))[1]
        hybrid_score = (bm25_weight * bm25_scores[i]) + (vector_weight * vec_score)

        combined_scores.append({
            "chunk_id": chunk_id,
            "doc_id": chunk["doc_id"],
            "text": chunk["text"],
            "score": hybrid_score
        })

    combined_scores.sort(key=lambda x: x["score"], reverse=True)
    return combined_scores[:top_k]

def convert_table_to_text(chunk_text):
    lines = chunk_text.split('\n')
    result = []

    for line in lines:
        if line.startswith('|') and line.endswith('|'):
            cells = [c.strip() for c in line.split('|')[1:-1]]
            if cells and not all(c.startswith('-') for c in cells):
                result.append(' → '.join(cells))
        else:
            result.append(line)

    return '\n'.join(result)

def build_llm_prompt(query, retrieved_chunks):
    context = "\n\n".join([
        f"SOURCE {i+1}: [{chunk['doc_id']} | {chunk['chunk_id']}]\n{convert_table_to_text(chunk['text'])}"
        for i, chunk in enumerate(retrieved_chunks)
    ])
    return SYSTEM_PROMPT_TEMPLATE.format(context=context, query=query)

def call_llm(prompt, temperature=0.0, max_tokens=160):
    for attempt in range(3):
        try:
            response = requests.post(
                "http://localhost:8080/completion",
                json={
                    "prompt": prompt,
                    "n_predict": max_tokens,
                    "temperature": temperature,
                    "stop": ["<|im_end|>", "<|im_start|>"]
                },
                timeout=180
            )
            response.raise_for_status()
            try:
                result = response.json()
                content = result.get("content") or result.get("choices", [{}])[0].get("text", "")
                return content.strip() if content else "Not found in provided excerpts."
            except Exception as e:
                print(f"[LLM ERROR] Invalid response: {e}")
                print(response.text)
                return "Not found in provided excerpts." 
            
        except requests.exceptions.RequestException as e:
            print(f"LLM request attempt {attempt+1} failed: {e}")
            time.sleep(5)
    raise RuntimeError("Failed to call llama-server after 3 attempts.")

def extract_answer_from_response(resp_text):
    m = re.search(r"Answer:\s*(.+?)\s*\n\s*Sources:\s*(.+)", resp_text, flags=re.S)
    if not m:
        if "not found" in resp_text.lower():
            return None, []
        return None, None

    ans = m.group(1).strip()
    sources_raw = m.group(2).strip()
    sources = [s.strip() for s in re.split(r"[,\n]+", sources_raw) if s.strip() and '[' in s]
    return ans, sources

def validate_answer(ans, retrieved_chunks):
    if ans is None:
        return False

    for ch in retrieved_chunks:
        if ans in ch['text']:
            return True

    ans_normalized = ans.lower().replace(" ", "")
    for ch in retrieved_chunks:
        ch_normalized = ch['text'].lower().replace(" ", "")
        if ans_normalized in ch_normalized:
            return True

    return False

def synthesize_answer(query, debug=False, use_reranker=True):
    _initialize_resources()
    
    expanded_query = rewrite_query(query)
    if debug:
        print(f"Original query: {query}")
        print(f"Expanded query: {expanded_query}")

    retrieved = hybrid_search(expanded_query, top_k=10, bm25_weight=0.7, vector_weight=0.3)
    if not any("25%" in c["text"] for c in retrieved):
        if debug:
            print("No match after rewrite, falling back to raw query.")
        retrieved = hybrid_search(query, top_k=10, bm25_weight=0.7, vector_weight=0.3)

    top_chunks = retrieved[:5]
    prompt = build_llm_prompt(query, top_chunks)
    response_text = call_llm(prompt)

    ans, sources = extract_answer_from_response(response_text)
    return {"answer": ans, "retrieved_chunks": top_chunks, "sources": sources}

def retrieval_debug(query):
    _initialize_resources()
    top_chunks = hybrid_search(query, top_k=10, bm25_weight=0.7, vector_weight=0.3)
    print(f"Top {len(top_chunks)} chunks for query: {query}\n")
    for i, chunk in enumerate(top_chunks, 1):
        print(f"[{i}] doc_id: {chunk['doc_id']} | chunk_id: {chunk['chunk_id']} | score: {chunk['score']:.4f}")
        print(f"Text snippet: {chunk['text'][:200]}...\n")
        if "25%" in chunk['text']:
            print(" --> Contains '25%' indicator\n")
        if "2 consecutive" in chunk['text'].lower() or "2 years" in chunk['text'].lower():
            print(" --> Contains '2 consecutive/years' indicator\n")

if __name__ == "__main__":
    test_query = "What is the No Claim Bonus for 2 consecutive claim-free years?"
    result = synthesize_answer(test_query, debug=True)
    
    if _cursor:
        _cursor.close()
    if _conn:
        _conn.close()
