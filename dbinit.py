# ==============================================================
# 🧩 init_database.py — One-time database initializer
# ==============================================================

from api.connection import Base, engine
from api.db import models

def main():
    print("🚀 Initializing database schema...")
    Base.metadata.create_all(bind=engine)
    print("✅ Database tables created successfully!")

if __name__ == "__main__":
    main()
