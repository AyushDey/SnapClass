import os
from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker, DeclarativeBase
from sqlalchemy import URL
from dotenv import load_dotenv
from pathlib import Path
load_dotenv()

db_pass = os.getenv('DB_PASSWORD')
db_user = os.getenv('DB_USER')
db_name = os.getenv('DB_NAME')
db_port = int(os.getenv('DB_PORT'))
db_host = os.getenv('DB_HOST')
BASE_DIR = Path(__file__).parent
server_ca = BASE_DIR / "certs" / "server-ca.pem"
client_key = BASE_DIR / "certs" / "client-key.pk8"
client_cert = BASE_DIR / "certs" / "client-cert.pem"

DB_URL = URL.create(
    "postgresql+psycopg",
    username=db_user,
    password=db_pass,
    host=db_host,
    port=db_port,
    database=db_name
)

engine = create_engine(
    DB_URL,
    pool_size=5,
    pool_timeout=30,
    pool_recycle=1800,
    connect_args={
        "sslmode": "verify-ca",
        "sslrootcert": str(server_ca),
        "sslcert": str(client_cert),
        "sslkey": str(client_key)
    }
)

# Enable pgvector extension
with engine.connect() as conn:
    conn.execute(text('CREATE EXTENSION IF NOT EXISTS vector'))
    conn.commit()

class Base(DeclarativeBase):
    pass

sessionLocal = sessionmaker(bind=engine, autoflush=False, autocommit=False)
