import asyncio
from app.services.validation.agentic_validation_worker import run_agentic_validation
from app.utils.env_flags import get_env_bool

if __name__ == "__main__":
    if not get_env_bool(
        "ENABLE_AGENTIC_VALIDATION",
        default=True,
        aliases=("ENABLE_VALIDATION",),
    ):
        print("Agentic validation is disabled via .env toggle. Skipping run.")
    else:
        asyncio.run(run_agentic_validation(batch_size=50))
