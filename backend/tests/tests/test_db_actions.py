"""Tests for db_actions.py – DBActions class and schema migration."""

from sqlalchemy import Column, Integer, MetaData, PickleType, String, Table, create_engine, inspect, select
from sqlalchemy.orm import Session

from db_actions import DBActions
from models import BookletCategory, BookletEmbedding, BookletItem
from schema_migrations import initialize_database


def _create_embedding(db_session, item_name: str, category_id: int, image_hash: str, vector: list[float]):
    item = BookletItem(item_name=item_name)
    db_session.add(item)
    db_session.flush()

    embedding = BookletEmbedding(
        booklet_item_id=item.id,
        booklet_category_id=category_id,
        image_hash=image_hash,
        embedding=vector,
    )
    db_session.add(embedding)
    db_session.flush()
    return embedding


# ---------------------------------------------------------------------------
# get_existing_hashes
# ---------------------------------------------------------------------------

def test_get_existing_hashes_empty(db_session):
    actions = DBActions(db_session)
    assert actions.get_existing_hashes() == set()


def test_get_existing_hashes_with_data(db_session):
    cat = BookletCategory(category_name="Fruits")
    db_session.add(cat)
    db_session.flush()
    _create_embedding(db_session, "apple", cat.id, "hash1", [0.1] * 512)
    _create_embedding(db_session, "banana", cat.id, "hash2_0", [0.2] * 512)

    actions = DBActions(db_session)
    existing = actions.get_existing_hashes()
    assert "hash1" in existing
    assert "hash2" in existing


# ---------------------------------------------------------------------------
# get_or_create_item / get_or_create_category
# ---------------------------------------------------------------------------

def test_get_or_create_item_returns_existing_id(db_session):
    actions = DBActions(db_session)
    item_id = actions.get_or_create_item("hammer")
    same_item_id = actions.get_or_create_item("hammer")
    assert item_id == same_item_id


def test_get_or_create_category_none_input(db_session):
    """None input defaults to 'Uncategorized'."""
    actions = DBActions(db_session)
    cat_id = actions.get_or_create_category(None)
    assert isinstance(cat_id, int)
    assert cat_id == actions.get_or_create_category(None)


def test_get_or_create_category_empty_string(db_session):
    """Empty string also defaults to 'Uncategorized'."""
    actions = DBActions(db_session)
    cat_id = actions.get_or_create_category("")
    assert isinstance(cat_id, int)


def test_get_or_create_category_returns_existing(db_session):
    actions = DBActions(db_session)
    cat_id1 = actions.get_or_create_category("Furniture")
    cat_id2 = actions.get_or_create_category("Furniture")
    assert cat_id1 == cat_id2


# ---------------------------------------------------------------------------
# insert_embeddings
# ---------------------------------------------------------------------------

def test_insert_embeddings_empty_list(db_session):
    actions = DBActions(db_session)
    actions.insert_embeddings([])


def test_insert_embeddings_inserts_rows(db_session):
    actions = DBActions(db_session)
    cat_id = actions.get_or_create_category("Tools")
    hammer_id = actions.get_or_create_item("hammer")
    wrench_id = actions.get_or_create_item("wrench")

    embeddings = [
        {
            "image_hash": "h_0",
            "booklet_item_id": hammer_id,
            "booklet_category_id": cat_id,
            "embedding": [0.5] * 512,
        },
        {
            "image_hash": "h_1",
            "booklet_item_id": wrench_id,
            "booklet_category_id": cat_id,
            "embedding": [0.3] * 512,
        },
    ]
    actions.insert_embeddings(embeddings)
    db_session.commit()
    assert len(actions.get_all_embeddings()) == 2


def test_insert_embeddings_ignores_duplicates(db_session):
    actions = DBActions(db_session)
    cat_id = actions.get_or_create_category("Tools")
    item_id = actions.get_or_create_item("drill")
    embedding = {
        "image_hash": "dup_hash",
        "booklet_item_id": item_id,
        "booklet_category_id": cat_id,
        "embedding": [0.1] * 512,
    }

    actions.insert_embeddings([embedding])
    db_session.commit()
    actions.insert_embeddings([embedding])
    db_session.commit()

    all_embeddings = actions.get_all_embeddings()
    assert sum(1 for row in all_embeddings if row.image_hash == "dup_hash") == 1


# ---------------------------------------------------------------------------
# get_all_embeddings
# ---------------------------------------------------------------------------

def test_get_all_embeddings_empty(db_session):
    actions = DBActions(db_session)
    assert actions.get_all_embeddings() == []


def test_get_all_embeddings_returns_joined_rows(db_session):
    actions = DBActions(db_session)
    cat_id = actions.get_or_create_category("Plants")
    _create_embedding(db_session, "plant_0", cat_id, "ph_0", [0.0] * 512)
    _create_embedding(db_session, "plant_1", cat_id, "ph_1", [1.0] * 512)

    rows = actions.get_all_embeddings()
    assert len(rows) == 2
    assert {row.item.item_name for row in rows} == {"plant_0", "plant_1"}
    assert all(row.booklet_category_id == cat_id for row in rows)


# ---------------------------------------------------------------------------
# delete_embeddings
# ---------------------------------------------------------------------------

def test_delete_embeddings_empty_list(db_session):
    actions = DBActions(db_session)
    actions.delete_embeddings([])


def test_delete_embeddings_removes_by_id(db_session):
    actions = DBActions(db_session)
    cat_id = actions.get_or_create_category("Animals")
    embedding = _create_embedding(db_session, "cat", cat_id, "cat_hash", [0.2] * 512)

    actions.delete_embeddings([embedding.id])
    db_session.commit()
    assert actions.get_all_embeddings() == []


# ---------------------------------------------------------------------------
# commit
# ---------------------------------------------------------------------------

def test_commit(db_session):
    actions = DBActions(db_session)
    actions.commit()


# ---------------------------------------------------------------------------
# get_category_by_id
# ---------------------------------------------------------------------------

def test_get_category_by_id_found(db_session):
    cat = BookletCategory(category_name="Vehicles")
    db_session.add(cat)
    db_session.flush()
    actions = DBActions(db_session)
    assert actions.get_category_by_id(cat.id) == "Vehicles"


def test_get_category_by_id_not_found(db_session):
    actions = DBActions(db_session)
    assert actions.get_category_by_id(99999) == "Unkown"


# ---------------------------------------------------------------------------
# schema migration
# ---------------------------------------------------------------------------

def test_initialize_database_migrates_legacy_booklet_items():
    engine = create_engine("sqlite:///:memory:", connect_args={"check_same_thread": False})
    try:
        metadata = MetaData()

        category_table = Table(
            "booklet_category",
            metadata,
            Column("id", Integer, primary_key=True),
            Column("category_name", String),
        )
        legacy_items = Table(
            "booklet_items",
            metadata,
            Column("id", Integer, primary_key=True),
            Column("item_name", String),
            Column("category_id", Integer),
            Column("image_hash", String),
            Column("embedding", PickleType()),
        )
        metadata.create_all(engine)

        with engine.begin() as conn:
            conn.execute(category_table.insert(), [{"id": 1, "category_name": "Tools"}])
            conn.execute(
                legacy_items.insert(),
                [
                    {
                        "id": 1,
                        "item_name": "hammer",
                        "category_id": 1,
                        "image_hash": "hammer_0",
                        "embedding": [0.1] * 512,
                    },
                    {
                        "id": 2,
                        "item_name": "hammer",
                        "category_id": 1,
                        "image_hash": "hammer_1",
                        "embedding": [0.2] * 512,
                    },
                    {
                        "id": 3,
                        "item_name": "wrench",
                        "category_id": 1,
                        "image_hash": "wrench_0",
                        "embedding": [0.3] * 512,
                    },
                ],
            )

        initialize_database(engine)
        initialize_database(engine)

        inspector = inspect(engine)
        assert "booklet_embeddings" in inspector.get_table_names()
        assert {column["name"] for column in inspector.get_columns("booklet_items")} == {"id", "item_name"}

        with Session(engine) as session:
            items = session.scalars(select(BookletItem).order_by(BookletItem.item_name)).all()
            embeddings = session.scalars(select(BookletEmbedding).order_by(BookletEmbedding.image_hash)).all()
            embedding_item_names = {embedding.item.item_name for embedding in embeddings}

        assert [item.item_name for item in items] == ["hammer", "wrench"]
        assert [embedding.image_hash for embedding in embeddings] == ["hammer_0", "hammer_1", "wrench_0"]
        assert {embedding.booklet_category_id for embedding in embeddings} == {1}
        assert embedding_item_names == {"hammer", "wrench"}
    finally:
        engine.dispose()
