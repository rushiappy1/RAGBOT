import pickle
import psycopg2
from sentence_transformers import SentenceTransformer
import os
from pathlib import Path
import requests
import re
from scripts.reranker import ChunkReranker

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


# Load BM25 index
print("Loading BM25 index...")
with open("data/bm25_index.pkl", "rb") as f:
    index_data = pickle.load(f)
bm25 = index_data["bm25"]
chunks = index_data["chunks"]

# Load embedding model
print("Loading sentence transformer model...")
model = SentenceTransformer(MODEL_NAME, device="cpu")

# Connect to database
print("Connecting to PostgreSQL...")
conn = psycopg2.connect(
    host=DB_HOST, port=DB_PORT, database=DB_NAME,
    user=DB_USER, password=DB_PASSWORD
)
cursor = conn.cursor()
print("Setup complete.\n")


def hybrid_search(query, top_k=None, bm25_weight=None, vector_weight=None):
    """Hybrid retrieval with config defaults"""
    top_k = top_k or HYBRID_SEARCH_DEFAULTS["top_k"]
    bm25_weight = bm25_weight or HYBRID_SEARCH_DEFAULTS["bm25_weight"]
    vector_weight = vector_weight or HYBRID_SEARCH_DEFAULTS["vector_weight"]
    """Hybrid retrieval with keyword-heavy weighting"""
    tokenized_query = query.lower().split()
    bm25_scores = bm25.get_scores(tokenized_query)
    bm25_scores = (bm25_scores - bm25_scores.min()) / (bm25_scores.max() - bm25_scores.min() + 1e-8)
    
    q_emb = model.encode([query], normalize_embeddings=True)[0].tolist()
    cursor.execute("""
        SELECT c.chunk_id, c.doc_id, 1 - (ce.embedding <=> %s::vector) as similarity
        FROM chunk_embeddings ce
        JOIN chunks c ON c.chunk_id = ce.chunk_id
    """, (q_emb,))
    vector_results = {row[0]: (row[1], row[2]) for row in cursor.fetchall()}
    
    combined_scores = []
    for i, chunk in enumerate(chunks):
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


def build_llm_prompt(query, retrieved_chunks):
    """Build prompt from locked template"""
    context = "\n\n".join([
        f"SOURCE {i+1}: [{chunk['doc_id']} | {chunk['chunk_id']}]\n{chunk['text']}"
        for i, chunk in enumerate(retrieved_chunks)
    ])
    return SYSTEM_PROMPT_TEMPLATE.format(context=context, query=query)


def call_llm(prompt, temperature=0.0, max_tokens=160):
    """Call llama-server with strict parameters"""
    
    formatted_prompt = f"""<|im_start|>system
You are an expert insurance policy analyst.<|im_end|>
<|im_start|>user
{prompt}<|im_end|>
<|im_start|>assistant
"""
    
    try:
        response = requests.post(
            "http://localhost:8080/completion",
            json={
                "prompt": formatted_prompt,
                "n_predict": max_tokens,
                "temperature": temperature,
                "stop": ["<|im_end|>", "<|im_start|>"]
            },
            timeout=120
        )
        response.raise_for_status()
    except requests.exceptions.ConnectionError:
        raise RuntimeError("Cannot connect to llama-server on port 8080. Is it running?")
    except requests.exceptions.Timeout:
        raise RuntimeError("llama-server request timed out (>120s)")
    except requests.exceptions.HTTPError as e:
        raise RuntimeError(f"llama-server HTTP error: {e}")
    
    result = response.json()
    return result["content"].strip()


def extract_answer_from_response(resp_text):
    """Parse structured Answer: ... Sources: ... format"""
    # Try exact format first
    m = re.search(r"Answer:\s*(.+?)\s*\n\s*Sources:\s*(.+)", resp_text, flags=re.S)
    if not m:
        # Fallback: check if answer contains "Not found"
        if "not found" in resp_text.lower():
            return None, []
        # Otherwise invalid format
        return None, None
    
    ans = m.group(1).strip()
    sources_raw = m.group(2).strip()
    sources = [s.strip() for s in re.split(r"[,\n]+", sources_raw) if s.strip() and '[' in s]
    return ans, sources


def validate_answer(ans, retrieved_chunks):
    """Verify answer appears verbatim in at least one chunk"""
    if ans is None:
        return False
    
    # Check if answer substring exists in any chunk
    for ch in retrieved_chunks:
        if ans in ch['text']:
            return True
    
    # More lenient: check if key numeric/phrase is present
    # e.g., "25%" should match "25%"
    ans_normalized = ans.lower().replace(" ", "")
    for ch in retrieved_chunks:
        ch_normalized = ch['text'].lower().replace(" ", "")
        if ans_normalized in ch_normalized:
            return True
    
    return False
reranker = ChunkReranker()

def synthesize_answer(query, debug=False, use_reranker=True):
    """End-to-end with optional reranking"""
    # ... existing retrieval code ...
    
    retrieved = hybrid_search(query, top_k=10, bm25_weight=0.7, vector_weight=0.3)
    
    # Rerank top-10 → top-5
    if use_reranker:
        top_chunks = reranker.rerank(query, retrieved, top_k=5)
        if debug:
            print("\n✓ Reranked with cross-encoder\n")
    else:
        top_chunks = retrieved[:5]
    """End-to-end with validation"""
    print(f"\n{'='*80}")
    print(f"QUERY: {query}")
    print(f"{'='*80}\n")
    
    # Retrieve top-10 with BM25-heavy weighting
    retrieved = hybrid_search(query, top_k=10, bm25_weight=0.7, vector_weight=0.3)
    
    if debug:
        print("TOP 10 RETRIEVED CHUNKS:")
        for i, chunk in enumerate(retrieved, 1):
            print(f"\n[{i}] {chunk['doc_id']} | {chunk['chunk_id']} (score: {chunk['score']:.3f})")
            # Check if NCB data is present
            if "25%" in chunk['text']:
                print("  ✓ Contains '25%'")
            if "2 consecutive" in chunk['text'].lower() or "2 years" in chunk['text'].lower():
                print("  ✓ Contains '2 consecutive/years'")
            print(f"Text: {chunk['text'][:200]}...\n")
    else:
        print(f"Retrieved {len(retrieved)} chunks from:")
        for chunk in retrieved[:5]:
            print(f"  - {chunk['doc_id']}")
    
    # Use top-5 for synthesis (context window constraint)
    top_chunks = retrieved[:5]
    
    # Build prompt
    prompt = build_llm_prompt(query, top_chunks)
    
    if debug:
        print(f"\n{'='*80}")
        print("PROMPT (first 1000 chars):")
        print(f"{'='*80}")
        print(prompt[:1000])
        print("...")
        print(f"{'='*80}\n")
    
    # Generate
    print("\nGenerating answer with llama-server...")
    response_text = call_llm(prompt, temperature=0.0, max_tokens=160)
    
    if debug:
        print(f"\nRaw LLM response:\n{response_text}\n")
    
    # Extract and validate
    ans, sources = extract_answer_from_response(response_text)
    
    if ans is None and sources is None:
        # Invalid format - regenerate with stricter directive
        print("⚠️  Invalid format, regenerating with stricter prompt...")
        regen_prompt = prompt + "\n\nEXTRA: The previous response did not follow format. You MUST respond with:\nAnswer: <exact text>\nSources: [doc_id | chunk_id]\nor:\nNot found in provided excerpts."
        response_text = call_llm(regen_prompt, temperature=0.0, max_tokens=160)
        ans, sources = extract_answer_from_response(response_text)
    
    # Validate answer is grounded
    if ans and validate_answer(ans, top_chunks):
        final = f"{ans}\nSources: {', '.join(sources) if sources else 'None cited'}"
    elif ans is None and sources == []:
        # "Not found" response
        final = "Not found in provided excerpts."
    else:
        # Hallucination detected
        print("⚠️  Validation failed: answer not found verbatim in chunks")
        final = "Not found in provided excerpts. (validation rejected response)"
    
    print(f"\n{'='*80}")
    print("FINAL ANSWER:")
    print(f"{'='*80}")
    print(final)
    print(f"{'='*80}\n")
    
    return {
        "query": query,
        "retrieved_chunks": [{"chunk_id": c["chunk_id"], "doc_id": c["doc_id"]} for c in top_chunks],
        "answer": final,
        "raw_response": response_text
    }


if __name__ == "__main__":
    # Test with debug mode to see retrieval details
    test_query = "What is the No Claim Bonus for 2 consecutive claim-free years?"
    
    result = synthesize_answer(test_query, debug=True)
    
    cursor.close()
    conn.close()
