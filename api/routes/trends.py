from fastapi import APIRouter
from api.services.seo_service import get_trending_keywords

router = APIRouter(prefix="/trends", tags=["Trends"])

@router.get("/")
def latest_trends(category: str):
    return get_trending_keywords(category)
