from logging.config import fileConfig
import os
import sys
from sqlalchemy import engine_from_config, pool
from sqlalchemy import create_engine
from alembic import context

ALEMBIC_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(ALEMBIC_DIR, ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from app.db.session_v2 import DATABASE_URL
from app.db.base import Base  # Ensure Base imports all models
from app.db.models import *   # Import all models (GlobalContentIndex, IngestedFile, etc.)

# -----------------------------------------------------
# Alembic Config
# -----------------------------------------------------
config = context.config
fileConfig(config.config_file_name)
target_metadata = Base.metadata

# -----------------------------------------------------
# Helper: Convert async → sync connection URL
# -----------------------------------------------------
def make_sync_url(async_url: str) -> str:
    """
    Converts asyncpg connection string to psycopg2 for Alembic migrations.
    Example:
        postgresql+asyncpg:// → postgresql+psycopg2://
    """
    return async_url.replace("+asyncpg", "+psycopg2")

# -----------------------------------------------------
# OFFLINE mode (no DB connection, generates SQL)
# -----------------------------------------------------
def run_migrations_offline():
    """Run migrations in 'offline' mode."""
    url = make_sync_url(DATABASE_URL)
    context.configure(
        url=url,
        target_metadata=target_metadata,
        literal_binds=True,
        compare_type=True,
    )

    with context.begin_transaction():
        context.run_migrations()

# -----------------------------------------------------
# ONLINE mode (connects to DB and executes)
# -----------------------------------------------------
def run_migrations_online():
    """Run migrations in 'online' mode."""
    connectable = create_engine(
        make_sync_url(DATABASE_URL),
        poolclass=pool.NullPool,
    )

    with connectable.connect() as connection:
        context.configure(connection=connection, target_metadata=target_metadata)

        with context.begin_transaction():
            context.run_migrations()

# -----------------------------------------------------
# Execute
# -----------------------------------------------------
if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()
