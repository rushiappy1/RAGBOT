import json
import re

def extract_citations(answer):
    """Extract [doc_id | chunk_id] citations from answer"""
    pattern = r'\[([^\]]+)\|([^\]]+)\]'
    return re.findall(pattern, answer)

def check_hallucination(answer, retrieved_chunks):
    """Check if all citations are in retrieved chunks"""
    citations = extract_citations(answer)
    retrieved_ids = {(c['doc_id'], c['chunk_id']) for c in retrieved_chunks}
    
    hallucinated = []
    for doc_id, chunk_id in citations:
        if (doc_id.strip(), chunk_id.strip()) not in retrieved_ids:
            hallucinated.append(f"[{doc_id}|{chunk_id}]")
    
    return hallucinated

def evaluate_answer(question, answer, retrieved_chunks, expected):
    """Evaluate a single answer"""
    results = {
        "question": question,
        "has_citations": len(extract_citations(answer)) > 0,
        "hallucinations": check_hallucination(answer, retrieved_chunks),
        "citation_count": len(extract_citations(answer)),
    }
    
    # Check expected docs are cited
    if expected.get("expected_docs"):
        cited_docs = {doc for doc, _ in extract_citations(answer)}
        results["cited_expected_docs"] = any(
            exp in cited_docs for exp in expected["expected_docs"]
        )
    
    return results

# Will integrate with answer_synthesis.py later
