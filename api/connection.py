# ===============================================
# 🚀 connection.py — Unified DB Connection Layer
# ===============================================
import os
import logging
from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker, declarative_base
from contextlib import contextmanager

# ===============================================
# ⚙️ Configuration
# ===============================================

# Environment variable or fallback to local SQLite for dev
DATABASE_URL = os.getenv(
    "DATABASE_URL",
    "postgresql://postgres:Mahadeva%40123@localhost:5432/marketing_advantage"
)

# ✅ Fallback for local testing if Postgres not available
if DATABASE_URL.startswith("sqlite"):
    connect_args = {"check_same_thread": False}
else:
    connect_args = {}

# ===============================================
# 🧠 Engine & Session Factory
# ===============================================
engine = create_engine(
    DATABASE_URL,
    pool_pre_ping=True,        # Detect dropped connections automatically
    pool_size=10,              # Adjust for concurrency (embedding-heavy ops)
    max_overflow=20,           # Allow temporary bursts
    pool_timeout=30,           # Wait time before raising pool errors
    connect_args=connect_args,
    echo=False                 # Turn True for SQL debug logs
)

SessionLocal = sessionmaker(
    autocommit=False,
    autoflush=False,
    bind=engine
)

Base = declarative_base()

# ===============================================
# 🧩 FastAPI Dependency
# ===============================================
def get_db():
    """
    FastAPI dependency that yields a SQLAlchemy session.
    Usage:
        @router.get("/items")
        def list_items(db: Session = Depends(get_db)):
            ...
    """
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

# ===============================================
# 🧰 Context Manager (for scripts, CLI, or background jobs)
# ===============================================
@contextmanager
def get_db_connection():
    """
    Backward-compatible context manager.
    Example:
        with get_db_connection() as db:
            db.execute(text("SELECT * FROM contents"))
    """
    db = SessionLocal()
    try:
        yield db
        db.commit()
    except Exception as e:
        db.rollback()
        logging.error(f"[DB] Transaction rolled back due to: {e}")
        raise
    finally:
        db.close()

# ===============================================
# 🧱 Schema Initialization
# ===============================================
def init_db():
    """
    Initialize or migrate schema (for CLI or one-time setup).
    Automatically imports all models so Base.metadata is aware.
    """
    import api.models as models
    Base.metadata.create_all(bind=engine)
    print("✅ Database schema initialized successfully.")

# ===============================================
# 🩺 Healthcheck Utility
# ===============================================
def test_connection():
    """Verify DB connection for startup health checks."""
    try:
        with engine.connect() as conn:
            conn.execute(text("SELECT 1"))
        print("✅ [DB] Connection healthy.")
        return True
    except Exception as e:
        print(f"❌ [DB] Connection failed: {e}")
        return False
# Backward-compatible alias for older imports
get_db_session = get_db_connection
