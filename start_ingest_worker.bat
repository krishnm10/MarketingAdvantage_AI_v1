@echo off
cd C:\ProjectK\MarketingAdvantage_AI_v1
set PYTHONPATH=%cd%
python -m workers.ingest_service
pause
