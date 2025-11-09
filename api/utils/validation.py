def validate_request(data: dict, required_fields: list):
    missing = [f for f in required_fields if f not in data]
    if missing:
        return {"valid": False, "missing": missing}
    return {"valid": True}
