"""Tests for db_actions.py – DBActions class."""

import pytest
from sqlalchemy import insert
from db_actions import DBActions
from models import BookletItem, BookletCategory


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
    db_session.add(BookletItem(
        item_name="apple", category_id=cat.id, image_hash="hash1", embedding=[0.1] * 512
    ))
    db_session.flush()
    actions = DBActions(db_session)
    assert "hash1" in actions.get_existing_hashes()


# ---------------------------------------------------------------------------
# get_or_create_category
# ---------------------------------------------------------------------------

def test_get_or_create_category_none_input(db_session):
    """None input defaults to 'Uncategorized'."""
    actions = DBActions(db_session)
    cat_id = actions.get_or_create_category(None)
    assert isinstance(cat_id, int)
    # Second call must return the same ID (already exists)
    cat_id2 = actions.get_or_create_category(None)
    assert cat_id == cat_id2


def test_get_or_create_category_empty_string(db_session):
    """Empty string also defaults to 'Uncategorized'."""
    actions = DBActions(db_session)
    cat_id = actions.get_or_create_category("")
    assert isinstance(cat_id, int)


def test_get_or_create_category_creates_new(db_session):
    actions = DBActions(db_session)
    cat_id = actions.get_or_create_category("Electronics")
    assert isinstance(cat_id, int)


def test_get_or_create_category_returns_existing(db_session):
    actions = DBActions(db_session)
    cat_id1 = actions.get_or_create_category("Furniture")
    cat_id2 = actions.get_or_create_category("Furniture")
    assert cat_id1 == cat_id2


# ---------------------------------------------------------------------------
# insert_items
# ---------------------------------------------------------------------------

def test_insert_items_empty_list(db_session):
    """Empty list causes early return without DB interaction."""
    actions = DBActions(db_session)
    actions.insert_items([])  # Should not raise


def test_insert_items_inserts_rows(db_session):
    actions = DBActions(db_session)
    cat_id = actions.get_or_create_category("Tools")
    items = [
        {"image_hash": "h_0", "item_name": "hammer", "category_id": cat_id, "embedding": [0.5] * 512},
        {"image_hash": "h_1", "item_name": "wrench", "category_id": cat_id, "embedding": [0.3] * 512},
    ]
    actions.insert_items(items)
    db_session.commit()
    assert len(actions.get_all_items()) == 2


def test_insert_items_ignores_duplicates(db_session):
    actions = DBActions(db_session)
    cat_id = actions.get_or_create_category("Tools")
    item = {"image_hash": "dup_hash", "item_name": "drill", "category_id": cat_id, "embedding": [0.1] * 512}
    actions.insert_items([item])
    db_session.commit()
    # Insert again – should silently ignore the duplicate
    actions.insert_items([item])
    db_session.commit()
    all_items = actions.get_all_items()
    assert sum(1 for i in all_items if i.image_hash == "dup_hash") == 1


# ---------------------------------------------------------------------------
# get_all_items
# ---------------------------------------------------------------------------

def test_get_all_items_empty(db_session):
    actions = DBActions(db_session)
    assert actions.get_all_items() == []


def test_get_all_items_returns_all(db_session):
    actions = DBActions(db_session)
    cat_id = actions.get_or_create_category("Plants")
    for i in range(3):
        db_session.add(BookletItem(
            item_name=f"plant_{i}", category_id=cat_id,
            image_hash=f"ph_{i}", embedding=[float(i)] * 512
        ))
    db_session.flush()
    assert len(actions.get_all_items()) == 3


# ---------------------------------------------------------------------------
# delete_items
# ---------------------------------------------------------------------------

def test_delete_items_empty_list(db_session):
    """Empty list causes early return without error."""
    actions = DBActions(db_session)
    actions.delete_items([])  # Should not raise


def test_delete_items_removes_by_id(db_session):
    actions = DBActions(db_session)
    cat_id = actions.get_or_create_category("Animals")
    item = BookletItem(item_name="cat", category_id=cat_id, image_hash="cat_hash", embedding=[0.2] * 512)
    db_session.add(item)
    db_session.flush()
    item_id = item.id
    actions.delete_items([item_id])
    db_session.commit()
    assert actions.get_all_items() == []


# ---------------------------------------------------------------------------
# commit
# ---------------------------------------------------------------------------

def test_commit(db_session):
    """commit() delegates to session.commit() without raising."""
    actions = DBActions(db_session)
    actions.commit()  # Should not raise


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
    """Missing ID returns the fallback string 'Unkown' (note: deliberate typo in source)."""
    actions = DBActions(db_session)
    result = actions.get_category_by_id(99999)
    assert result == "Unkown"
