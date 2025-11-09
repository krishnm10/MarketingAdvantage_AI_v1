from fastapi import APIRouter
from api.services.report_service import generate_kpi_report

router = APIRouter(prefix="/kpi", tags=["KPI"])

@router.get("/{business_id}")
def get_kpi(business_id: int):
    return generate_kpi_report(business_id)
