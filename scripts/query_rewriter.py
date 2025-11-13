from spellchecker import SpellChecker

spell = SpellChecker()

SYNONYMS = {
    "ncb": "no claim bonus",
    "idv": "insured declared value",
    "pa": "personal accident",
    "od": "own damage",
    "tp": "third party",
    "premium": "premium",
}

def rewrite_query(query: str) -> str:
    words = query.lower().split()
    corrected_words = [spell.correction(w) or w for w in words]
    expanded_words = [SYNONYMS.get(w, w) for w in corrected_words]
    return " ".join(expanded_words)
