from fastapi import APIRouter
from api.services.llm_connector import call_marketingadvantage_chain

router = APIRouter(prefix="/engagement", tags=["Engagement"])

@router.post("/generate")
def generate_playbook(data: dict):
    prompt = f"Create customer engagement script for {data.get('industry','generic business')}"
    return {"playbook": call_marketingadvantage(prompt)}
