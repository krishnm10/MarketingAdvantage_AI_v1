def generate_ad_campaign(data: dict):
    business = data.get("business", "Unknown Business")
    platform = data.get("platform", "Meta Ads")
    objective = data.get("objective", "Brand Awareness")
    headline = f"{business} — Boost your {objective.lower()} today!"
    body = f"Engage your audience on {platform} with AI-optimized campaigns."
    return {"platform": platform, "headline": headline, "body": body, "objective": objective}
