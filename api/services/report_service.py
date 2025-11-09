from datetime import date

def generate_kpi_report(business_id: int):
    # Example static data
    return {
        "business_id": business_id,
        "report_date": str(date.today()),
        "ROAS": 4.9,
        "CTR": 0.086,
        "CRR": 0.28,
        "summary": "Strong engagement; optimize creative for 25-34 age group."
    }
