# Marketing Advantage AI

FastAPI application: ingestion, RAG, admin APIs, and pipeline configuration.

## Documentation for developers

| Topic | Where |
|--------|--------|
| **Config: `app/config` vs `core/config` vs `core/configs` vs `settings`** | **[`docs/configuration_layout.md`](docs/configuration_layout.md)** — start here |
| **Package index: legacy ingestion / multimodal toggles** | **[`app/config/README.md`](app/config/README.md)** |
| **Package index: tenant pipeline schema & resolution** | **[`app/core/config/README.md`](app/core/config/README.md)** |
| **Admin UI (Next.js)** | [`app/frontend-admin/README.md`](app/frontend-admin/README.md) |

## Common entrypoints

- **API**: `uvicorn app.main:app` (see `app/main.py`)
- **Environment**: copy from `.env.example` if present; never commit real `.env` secrets

## Contributing

When adding configuration, use the map in [`docs/configuration_layout.md`](docs/configuration_layout.md) to choose **platform** (`settings`), **tenant pipeline** (`core/config` + `core/configs`), or **legacy globals** (`app/config`).
