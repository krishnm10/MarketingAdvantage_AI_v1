"""
E2E test for the structure_aware chunking strategy.

Tests with a document containing all structural elements:
headings, code blocks, tables, lists, and prose paragraphs.
"""
import asyncio
from app.core.chunking_stratagies.chunking_registry import get_chunker

STRUCTURED_DOC = """
# API Reference Guide

This document describes the authentication and usage of our REST API.
All endpoints require valid credentials and follow standard HTTP conventions.
The API is designed for enterprise-grade integrations with financial systems,
marketing platforms, and operational dashboards.

## Authentication

All requests must include a bearer token in the Authorization header.
Tokens are obtained via the /oauth/token endpoint using client credentials.
Invalid or expired tokens will result in a 401 Unauthorized response.
The token has a default TTL of 3600 seconds and must be refreshed before expiry.

```python
import requests

def get_token(client_id, client_secret):
    resp = requests.post(
        "https://api.example.com/oauth/token",
        data={
            "grant_type": "client_credentials",
            "client_id": client_id,
            "client_secret": client_secret,
        },
    )
    resp.raise_for_status()
    return resp.json()["access_token"]
```

## Rate Limits

| Tier       | Requests/min | Burst | Monthly Cap |
|------------|-------------|-------|-------------|
| Free       | 60          | 10    | 10,000      |
| Pro        | 600         | 50    | 500,000     |
| Enterprise | 6,000       | 200   | Unlimited   |

## Features

- Real-time data ingestion from multiple sources
- Automatic deduplication with 3-layer hash verification
- Semantic chunking with quality scoring
- Multi-vector database support (Milvus, Qdrant, Weaviate, Redis)
- Role-based access control with SSO integration
- Audit logging for compliance (GDPR, HIPAA, SOX)
- Configurable retention policies per business unit

## Error Handling

When the API encounters an error, it returns a structured JSON response.
The response includes a machine-readable error code, a human-readable message,
and optional diagnostic details for debugging.

```json
{
    "error": {
        "code": "RATE_LIMIT_EXCEEDED",
        "message": "You have exceeded the rate limit for your tier.",
        "retry_after": 30,
        "request_id": "req_abc123"
    }
}
```

### Common Error Codes

- `400 Bad Request` — Malformed request body or missing required fields
- `401 Unauthorized` — Invalid or expired authentication token
- `403 Forbidden` — Insufficient permissions for the requested resource
- `404 Not Found` — The requested resource does not exist
- `429 Too Many Requests` — Rate limit exceeded, check retry_after header
- `500 Internal Server Error` — Unexpected server failure, contact support

## Conclusion

This API provides a robust foundation for enterprise data operations.
For additional support, contact the engineering team or consult the
internal knowledge base. Keep your client credentials secure and rotate
them quarterly as per security policy.
"""


async def main():
    chunker = get_chunker("structure_aware")
    chunks = await chunker.chunk(
        STRUCTURED_DOC,
        source_type="pdf",
        embedding_model="test-model",
    )

    print(f"Total chunks: {len(chunks)}")
    print("=" * 80)

    for i, ch in enumerate(chunks):
        ri = ch.get("reasoning_ingestion", {})
        content_type = ri.get("content_type", "?")
        section = ri.get("section_title", "")
        depth = ri.get("section_depth", 0)
        quality = ri.get("chunk_quality_score", -1)
        lang = ri.get("code_language", "")
        tokens = ch.get("tokens", 0)
        text_preview = ch.get("text", "")[:120].replace("\n", " ")

        lang_str = f" lang={lang}" if lang else ""
        print(
            f"Chunk {i}: type={content_type:6s} | "
            f"section={'#'*depth} {section:30s} | "
            f"tokens={tokens:4d} | quality={quality:.4f}{lang_str}"
        )
        print(f"  text: {text_preview}...")
        print()


if __name__ == "__main__":
    asyncio.run(main())
