"""Query preprocessing for insurance domain"""
import re

def expand_abbreviations(query):
    """Expand common insurance abbreviations to full terms"""
    
    # Map abbreviations to full terms (keep abbreviation for BM25 match)
    expansions = {
        r'\bNCB\b': 'No Claim Bonus NCB',
        r'\bIDV\b': 'Insured Declared Value IDV',
        r'\bOD\b': 'Own Damage OD',
        r'\bTP\b': 'Third Party TP',
        r'\bPA\b': 'Personal Accident PA',
        r'\bCTL\b': 'Constructive Total Loss CTL',
    }
    
    expanded = query
    for abbr, full in expansions.items():
        expanded = re.sub(abbr, full, expanded, flags=re.IGNORECASE)
    
    return expanded


if __name__ == "__main__":
    # Test cases
    test_queries = [
        "What is the NCB for 2 consecutive years?",
        "How is IDV calculated?",
        "What does OD cover?",
        "What is the TP liability limit?",
    ]
    
    print("Query Rewriting Tests:")
    print("="*80)
    for q in test_queries:
        expanded = expand_abbreviations(q)
        print(f"Original: {q}")
        print(f"Expanded: {expanded}")
        print()
