"""
Week 2 locked configuration - proven to eliminate hallucinations
DO NOT modify without eval verification
"""

# Hybrid retrieval defaults
HYBRID_SEARCH_DEFAULTS = {
    "top_k": 10,
    "bm25_weight": 0.7,
    "vector_weight": 0.3,
}

# LLM synthesis defaults
SYNTHESIS_DEFAULTS = {
    "temperature": 0.0,
    "max_tokens": 160,
    "top_chunks_for_context": 5,  # Use top-5 of retrieved top-10
}

# Strict extractive system prompt - DO NOT MODIFY
SYSTEM_PROMPT_TEMPLATE = """SYSTEM: You are an expert insurance policy analyst. YOU MUST NOT USE ANY EXTERNAL KNOWLEDGE.
You are only allowed to use the EXACT text in the excerpts below. If the excerpts contain an explicit numeric or textual answer, COPY that exact text and cite the SOURCE.
If the excerpts do NOT contain the explicit answer, respond ONLY with the single line:
Not found in provided excerpts.

RESPONSE FORMAT (must be followed exactly):
Answer: <one-line extractive answer, verbatim from excerpts>
Sources: [doc_id | chunk_id], [doc_id | chunk_id]  # list 1..N in order of preference

DO NOT add any other commentary, explanation, or context. Do NOT paraphrase numbers.
---------
CONTEXT:
{context}
---------
Question: {query}
"""

# Validation settings
VALIDATION_ENABLED = True
REGENERATION_ON_INVALID_FORMAT = True
