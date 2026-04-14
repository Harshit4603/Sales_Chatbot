from sqlalchemy import create_engine
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from dotenv import load_dotenv
import os

load_dotenv()

DATABASE_URL = os.getenv("DATABASE_URL")

if not DATABASE_URL:
    print("[!] Warning: DATABASE_URL not found in environment variables.")

# Create the engine
# Note: For Heroku/Railway Postgres, you might need to handle 'postgres://' vs 'postgresql://'
if DATABASE_URL and DATABASE_URL.startswith("postgres://"):
    DATABASE_URL = DATABASE_URL.replace("postgres://", "postgresql://", 1)

if DATABASE_URL:
    engine = create_engine(DATABASE_URL)
    SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
else:
    engine = None
    SessionLocal = None

Base = declarative_base()

# Dependency — used in every endpoint
def get_db():
    if SessionLocal is None:
        return
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
