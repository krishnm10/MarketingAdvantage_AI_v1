from fastapi import APIRouter, HTTPException
from api.db import crud
from api.services import llm_connector
from api.services import ad_service
from api.services import seo_service


router = APIRouter(prefix="/business", tags=["Business"])

@router.get("/{business_id}")
def get_business(business_id: int):
    business = crud.get_business_by_id(business_id)
    if not business:
        raise HTTPException(status_code=404, detail="Business not found")
    return business
