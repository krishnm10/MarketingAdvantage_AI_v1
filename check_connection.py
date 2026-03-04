from sqlalchemy import create_engine
engine = create_engine("postgresql+psycopg2://postgres:Mahadeva%40123@localhost:5432/marketing_advantage")
engine.connect()
print("Connection successful!")
