def get_trending_keywords(category: str):
    # In production, connect to APIs or cached trend data
    mock = {
        "retail": ["holiday deals", "eco packaging", "influencer marketing"],
        "tech": ["AI tools", "SaaS growth", "automation"]
    }
    return {"category": category, "trending_keywords": mock.get(category.lower(), ["innovation", "marketing"])}
