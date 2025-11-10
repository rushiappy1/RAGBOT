def rewrite_query(query):
    """
    Expand user queries with insurance domain terminology
    """
    query_lower = query.lower()
    
    # Domain-specific expansions
    expansions = []
    
    # NCB/Bonus related
    if "ncb" in query_lower or "no claim bonus" in query_lower or "claim bonus" in query_lower:
        expansions.extend(["no claim bonus", "NCB", "discount", "claim-free", "bonus percentage"])
    
    # Add-ons / Endorsements
    if "add-on" in query_lower or "addon" in query_lower or "add on" in query_lower:
        expansions.extend(["add-on covers", "optional covers", "endorsements", "riders"])
    
    # Third party
    if "third party" in query_lower or "tp" in query_lower:
        expansions.extend(["third party", "TP", "liability only", "third-party liability"])
    
    # Eligibility / Age
    if "age" in query_lower or "eligibility" in query_lower or "requirement" in query_lower:
        expansions.extend(["eligibility", "age criteria", "minimum age", "driver age", "requirements"])
    
    # Commercial use / Carpooling
    if "carpool" in query_lower or "commercial" in query_lower or "paid" in query_lower:
        expansions.extend(["commercial use", "paid passengers", "hire", "reward", "private car exclusion"])
    
    # Legal heir / Death / Succession
    if "legal heir" in query_lower or "dies" in query_lower or "death" in query_lower:
        expansions.extend(["legal heir", "succession", "nominee", "death", "policy transfer", "claim settlement"])
    
    # Salvage
    if "salvage" in query_lower:
        expansions.extend(["salvage value", "wreckage", "total loss settlement", "IDV"])
    
    # Constructive total loss
    if "constructive" in query_lower or "ctl" in query_lower or "total loss" in query_lower:
        expansions.extend(["constructive total loss", "CTL", "repair estimate", "IDV", "total loss"])
    
    # Water damage / Engine
    if "water" in query_lower or "engine damage" in query_lower or "monsoon" in query_lower:
        expansions.extend(["water ingress", "hydrostatic lock", "consequential damage", "engine damage exclusion"])
    
    # Depreciation
    if "depreciation" in query_lower or "zero dep" in query_lower:
        expansions.extend(["depreciation", "zero depreciation", "rubber plastic parts", "material depreciation"])
    
    # Combine original query with expansions
    if expansions:
        rewritten = query + " " + " ".join(set(expansions))
        return rewritten
    
    return query
