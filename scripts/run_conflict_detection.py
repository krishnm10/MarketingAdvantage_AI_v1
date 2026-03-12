import asyncio
from app.services.validation.semantic_conflict_engine import run_semantic_conflict_detection
from app.utils.env_flags import get_env_bool

if __name__ == "__main__":
    if not get_env_bool(
        "ENABLE_CONFLICT_ANALYSIS",
        default=True,
        aliases=("ENABLE_CONFLICT",),
    ):
        print("Conflict analysis is disabled via .env toggle. Skipping run.")
    else:
        asyncio.run(run_semantic_conflict_detection(batch_size=50))
