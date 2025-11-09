from fastapi import APIRouter
from api.services.ad_service import generate_ad_campaign
from api.services import llm_connector
from api.services import ad_service
from api.services import seo_service



router = APIRouter(prefix="/campaigns", tags=["Campaigns"])

@router.post("/create")
def create_campaign(data: dict):
    return generate_ad_campaign(data)
