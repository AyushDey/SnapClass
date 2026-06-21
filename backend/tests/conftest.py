"""
conftest.py – Test configuration and fixtures.

db.py runs module-level code that opens a real PostgreSQL connection, so it
must be replaced in sys.modules BEFORE any of the app modules (models,
classifier, main) are imported.  We do this with a lightweight mock "db"
module that exposes the same public names the rest of the code uses.
"""

import sys
import types
from contextlib import asynccontextmanager
from unittest.mock import MagicMock, patch
import pytest
from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker, DeclarativeBase
from sqlalchemy.pool import StaticPool


# ---------------------------------------------------------------------------
# 1.  Build a fake "db" module and inject it into sys.modules BEFORE anything
#     else is imported.  This prevents db.py from ever trying to reach Postgres.
# ---------------------------------------------------------------------------

class _FakeBase(DeclarativeBase):
    pass


_fake_engine = create_engine(
    "sqlite:///:memory:",
    connect_args={"check_same_thread": False},
    poolclass=StaticPool,
)
_FakeSessionLocal = sessionmaker(bind=_fake_engine, autoflush=False, autocommit=False)

fake_db = types.ModuleType("db")
fake_db.Base = _FakeBase
fake_db.engine = _fake_engine
fake_db.sessionLocal = _FakeSessionLocal
sys.modules["db"] = fake_db

# ---------------------------------------------------------------------------
# 2.  Patch pgvector's Vector type so models.py can be imported without the
#     real Postgres extension.  We replace it with a plain column type.
# ---------------------------------------------------------------------------

from sqlalchemy import PickleType

_fake_pgvector = types.ModuleType("pgvector")
_fake_pgvector_sqlalchemy = types.ModuleType("pgvector.sqlalchemy")
_fake_pgvector_sqlalchemy.Vector = lambda dim: PickleType()
sys.modules["pgvector"] = _fake_pgvector
sys.modules["pgvector.sqlalchemy"] = _fake_pgvector_sqlalchemy

# Also stub sqlalchemy.dialects.postgresql.insert so db_actions.py can import it
import sqlalchemy.dialects.postgresql as _pg_dialect
if not hasattr(_pg_dialect, "insert"):
    from sqlalchemy.dialects.sqlite import insert as _sqlite_insert
    _pg_dialect.insert = _sqlite_insert  # type: ignore[attr-defined]

# ---------------------------------------------------------------------------
# 3.  Now it is safe to import app modules.
# ---------------------------------------------------------------------------

from models import BookletItem, BookletCategory  # noqa: E402 – intentional late import


def pytest_sessionfinish(session, exitstatus):
    """Dispose the module-level fake SQLite engine created during test bootstrap."""
    _fake_engine.dispose()


# ---------------------------------------------------------------------------
# 4.  Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="function")
def db_engine():
    """Creates a fresh in-memory SQLite engine per test function."""
    engine = create_engine(
        "sqlite:///:memory:",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    _FakeBase.metadata.create_all(engine)
    yield engine
    _FakeBase.metadata.drop_all(engine)
    engine.dispose()


@pytest.fixture(scope="function")
def db_session(db_engine):
    """Provides a SQLAlchemy session bound to the in-memory test DB."""
    Session = sessionmaker(bind=db_engine, autoflush=False, autocommit=False)
    session = Session()
    yield session
    session.close()


@pytest.fixture(scope="function")
def session_factory(db_engine):
    """Returns a sessionmaker factory for the in-memory test DB."""
    _FakeBase.metadata.create_all(db_engine)
    return sessionmaker(bind=db_engine, autoflush=False, autocommit=False)


@pytest.fixture(autouse=True)
def cleanup_fake_db_engine():
    yield
    _fake_engine.dispose()


@pytest.fixture
def mock_classifier():
    """
    A MagicMock that mimics ImageClassifier's public API.
    Used for testing FastAPI routes without loading ResNet.
    """
    clf = MagicMock()
    clf.references_dir = MagicMock()
    clf.reference_embeddings = {"chair": []}
    clf.classify.return_value = {"class": "chair", "confidence": 0.95, "category_name": "Furniture", "matches": []}
    clf.load_references.return_value = None
    clf._compute_hash.return_value = "deadbeef"
    return clf


@pytest.fixture
def client(mock_classifier):
    """
    FastAPI TestClient with the global `classifier` replaced by mock_classifier
    and the lifespan disabled.
    """
    from fastapi.testclient import TestClient
    import main

    @asynccontextmanager
    async def noop_lifespan(app):
        yield

    # Patch the module-level `classifier` and skip the lifespan so we control startup
    with patch.object(main, "classifier", mock_classifier):
        with patch.object(main.app.router, "lifespan_context", noop_lifespan):
            with TestClient(main.app, raise_server_exceptions=False) as c:
                yield c
