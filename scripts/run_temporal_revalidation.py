import asyncio
from app.services.validation.temporal_revalidation_engine import (
    run_temporal_revalidation
)
from app.utils.env_flags import get_env_bool

if __name__ == "__main__":
    if not get_env_bool(
        "ENABLE_TEMPORAL_REVALIDATION",
        default=True,
        aliases=("ENABLE_TEMPORAL",),
    ):
        print("Temporal revalidation is disabled via .env toggle. Skipping run.")
    else:
        asyncio.run(run_temporal_revalidation(batch_size=50))
