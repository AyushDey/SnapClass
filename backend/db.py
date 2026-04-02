import os
from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker, DeclarativeBase
from sqlalchemy import URL
from dotenv import load_dotenv

load_dotenv()

db_pass = os.getenv('DB_PASSWORD')
db_user = os.getenv('DB_USER')
db_name = os.getenv('DB_NAME')
db_port = int(os.getenv('DB_PORT'))
db_host = os.getenv('DB_HOST')

# ✅ SSL controlled ONLY by env
sslmode = os.getenv("DB_SSLMODE", "disable")

DB_URL = URL.create(
    "postgresql+psycopg",
    username=db_user,
    password=db_pass,
    host=db_host,
    port=db_port,
    database=db_name
)

# SSL handling
connect_args = {"sslmode": sslmode}

engine = create_engine(
    DB_URL,
    connect_args=connect_args,
    pool_size=10,
    max_overflow=10,
    pool_timeout=30,
    pool_recycle=1800
)

# Safe DB init
try:
    with engine.connect() as conn:
        conn.execute(text('CREATE EXTENSION IF NOT EXISTS vector'))
        conn.commit()
except Exception as e:
    print("⚠️ DB connection failed during startup:", e)

class Base(DeclarativeBase):
    pass

sessionLocal = sessionmaker(bind=engine, autoflush=False, autocommit=False)