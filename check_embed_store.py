from app.services.ingestion.ingestion_service_v2 import IngestionServiceV2
print(hasattr(IngestionServiceV2, 'embed_and_store'))   # must print True
print(hasattr(IngestionServiceV2, '_embed_and_store'))