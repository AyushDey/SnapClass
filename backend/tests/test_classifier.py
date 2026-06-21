"""Tests for classifier.py – ImageClassifier class."""

import hashlib
import io
import os
import threading
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch, PropertyMock, call
import pytest
import torch
from PIL import Image

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_rgb_image(width=64, height=64):
    """Creates a minimal PIL RGB image."""
    img = Image.new("RGB", (width, height), color=(100, 150, 200))
    return img


def _make_embedding_record(label: str, category_id: int, image_hash: str, embedding_id: int = 1):
    return SimpleNamespace(
        id=embedding_id,
        image_hash=image_hash,
        booklet_category_id=category_id,
        embedding=[0.1] * 512,
        item=SimpleNamespace(item_name=label),
    )


def _make_classifier(session_factory, references_dir):
    """Builds an ImageClassifier with a mocked ResNet model."""
    with patch("classifier.resnet18") as mock_resnet:
        mock_model = MagicMock()
        mock_model.return_value = torch.zeros(1, 512)  # fake embedding
        mock_resnet.return_value = mock_model

        from classifier import ImageClassifier
        clf = ImageClassifier(
            session_factory=session_factory,
            references_dir=references_dir,
        )
    return clf


# ---------------------------------------------------------------------------
# _init_model
# ---------------------------------------------------------------------------

def test_init_model_success(session_factory, tmp_path):
    clf = _make_classifier(session_factory, tmp_path)
    assert clf.model is not None


def test_init_model_failure(session_factory, tmp_path):
    """If model loading raises, the exception propagates."""
    with patch("classifier.resnet18", side_effect=RuntimeError("GPU error")):
        from classifier import ImageClassifier
        with pytest.raises(RuntimeError, match="GPU error"):
            ImageClassifier(session_factory=session_factory, references_dir=tmp_path)


# ---------------------------------------------------------------------------
# get_embeddings / get_embedding
# ---------------------------------------------------------------------------

def test_get_embeddings_empty_list(session_factory, tmp_path):
    clf = _make_classifier(session_factory, tmp_path)
    assert clf.get_embeddings([]) == []


def test_get_embeddings_returns_list(session_factory, tmp_path):
    clf = _make_classifier(session_factory, tmp_path)

    # Return a tensor whose batch dim matches the actual input batch size
    def _fake_forward(tensors):
        return torch.zeros(tensors.shape[0], 512)

    with patch.object(clf, "model", side_effect=_fake_forward):
        result = clf.get_embeddings([_make_rgb_image(), _make_rgb_image()])
    assert len(result) == 2
    assert isinstance(result[0], list)


def test_get_embeddings_propagates_exception(session_factory, tmp_path):
    clf = _make_classifier(session_factory, tmp_path)

    with patch.object(clf, "model", side_effect=ValueError("bad tensor")):
        with pytest.raises(ValueError, match="bad tensor"):
            clf.get_embeddings([_make_rgb_image()])


def test_get_embedding_single(session_factory, tmp_path):
    clf = _make_classifier(session_factory, tmp_path)

    def _fake_forward(tensors):
        return torch.zeros(tensors.shape[0], 512)

    with patch.object(clf, "model", side_effect=_fake_forward):
        result = clf.get_embedding(_make_rgb_image())
    assert isinstance(result, list)
    assert len(result) == 512


# ---------------------------------------------------------------------------
# classify
# ---------------------------------------------------------------------------

def test_classify_no_search_matrix(session_factory, tmp_path):
    clf = _make_classifier(session_factory, tmp_path)
    clf._clear_memory()
    result = clf.classify(_make_rgb_image())
    assert result["class"] == "Unknown"
    assert result["confidence"] == 0.0


def test_classify_above_threshold(session_factory, tmp_path, db_session):
    clf = _make_classifier(session_factory, tmp_path)

    # Build a trivial search matrix with 1 item
    emb = torch.zeros(1, 512)
    emb[0, 0] = 1.0
    clf.search_matrix = emb
    clf.search_labels = ["chair"]
    clf.search_categories = [1]

    from db_actions import DBActions
    mock_db_actions = MagicMock()
    mock_db_actions.get_category_by_id.return_value = "Furniture"

    with patch("classifier.DBActions", return_value=mock_db_actions):
        with patch.object(clf, "get_embeddings", return_value=[[1.0] + [0.0] * 511] * 3):
            result = clf.classify(_make_rgb_image(), threshold=0.5)

    assert result["class"] == "chair"
    assert result["category_name"] == "Furniture"


def test_classify_below_threshold(session_factory, tmp_path):
    clf = _make_classifier(session_factory, tmp_path)

    emb = torch.zeros(1, 512)
    clf.search_matrix = emb
    clf.search_labels = ["chair"]
    clf.search_categories = [1]

    mock_db_actions = MagicMock()
    mock_db_actions.get_category_by_id.return_value = "Furniture"

    with patch("classifier.DBActions", return_value=mock_db_actions):
        with patch.object(clf, "get_embeddings", return_value=[[0.0] * 512] * 3):
            result = clf.classify(_make_rgb_image(), threshold=0.99)

    assert result["class"] == "Unknown"
    assert result["confidence"] == 0.0


def test_classify_propagates_exception(session_factory, tmp_path):
    clf = _make_classifier(session_factory, tmp_path)

    emb = torch.zeros(1, 512)
    clf.search_matrix = emb
    clf.search_labels = ["chair"]
    clf.search_categories = [1]

    with patch.object(clf, "_compute_multi_scale_scores", side_effect=RuntimeError("oops")):
        with pytest.raises(RuntimeError, match="oops"):
            clf.classify(_make_rgb_image())


def test_classify_empty_sorted_scores(session_factory, tmp_path):
    clf = _make_classifier(session_factory, tmp_path)

    emb = torch.zeros(1, 512)
    clf.search_matrix = emb
    clf.search_labels = ["chair"]
    clf.search_categories = [1]

    with patch.object(clf, "_compute_multi_scale_scores", return_value={}):
        result = clf.classify(_make_rgb_image())

    assert result["class"] == "Unknown Image"


def test_classify_matches_limit_break(session_factory, tmp_path):
    clf = _make_classifier(session_factory, tmp_path)
    clf.search_matrix = torch.zeros(1, 512)
    clf.search_labels = [f"label{i}" for i in range(10)]
    clf.search_categories = [1 for _ in range(10)]

    mock_db_actions = MagicMock()
    mock_db_actions.get_category_by_id.return_value = "Furniture"

    # Return 7 results; only first 5 should be appended to matches and then break
    multi_scores = {f"label{i}": (0.9 - i * 0.01, 1) for i in range(7)}

    with patch("classifier.DBActions", return_value=mock_db_actions):
        with patch.object(clf, "_compute_multi_scale_scores", return_value=multi_scores):
            with patch.object(clf, "get_reference_image_path", return_value="/ref.png"):
                result = clf.classify(_make_rgb_image(), threshold=0.5)

    assert result["class"] == "label0"
    assert len(result["matches"]) <= 5


def test_sync_new_references_item_cache_hit(session_factory, tmp_path):
    clf = _make_classifier(session_factory, tmp_path)

    img1 = tmp_path / "a.jpg"
    img2 = tmp_path / "b.jpg"
    _make_rgb_image().save(str(img1))
    _make_rgb_image().save(str(img2))

    active_files = {
        "h1": {"path": str(img1), "label": "shared", "category": "CatA"},
        "h2": {"path": str(img2), "label": "shared", "category": "CatB"},
    }

    mock_db = MagicMock()
    mock_db.get_existing_hashes.return_value = set()
    mock_db.get_or_create_item.return_value = 1
    mock_db.get_or_create_category.side_effect = [1, 2]

    fake_emb = [0.1] * 512
    with patch.object(clf, "get_embeddings", return_value=[fake_emb] * 14):
        clf._sync_new_references(mock_db, active_files, {"h1", "h2"})

    mock_db.get_or_create_item.assert_called_once_with("shared")


def test_get_reference_image_path_none(session_factory, tmp_path):
    clf = _make_classifier(session_factory, tmp_path)
    clf._cached_active_files = {"h1": {"path": "x", "label": "a", "category": "Cat"}}
    assert clf.get_reference_image_path("b", "Other") is None


def test_get_reference_image_path_valid(session_factory, tmp_path):
    clf = _make_classifier(session_factory, tmp_path)
    clf._cached_active_files = {
        "h1": {"path": "/references/Cat/a.jpg", "label": "a", "category": "Cat"}
    }
    assert clf.get_reference_image_path("a", "Cat") == "//references/Cat/a.jpg"


# ---------------------------------------------------------------------------
# _compute_multi_scale_scores
# ---------------------------------------------------------------------------

def test_compute_multi_scale_scores(session_factory, tmp_path):
    clf = _make_classifier(session_factory, tmp_path)

    emb = torch.eye(3, 512)  # 3 embeddings
    clf.search_matrix = emb
    clf.search_labels = ["a", "b", "c"]
    clf.search_categories = [1, 1, 2]

    fake_embs = [[1.0] + [0.0] * 511] * 3
    with patch.object(clf, "get_embeddings", return_value=fake_embs):
        scores = clf._compute_multi_scale_scores(_make_rgb_image(), emb, clf.search_labels, clf.search_categories)
    assert isinstance(scores, dict)


# ---------------------------------------------------------------------------
# _compute_hash
# ---------------------------------------------------------------------------

def test_compute_hash_success(session_factory, tmp_path):
    clf = _make_classifier(session_factory, tmp_path)

    test_file = tmp_path / "img.png"
    test_file.write_bytes(b"fake image data")
    h = clf._compute_hash(test_file)
    expected = hashlib.sha256(b"fake image data").hexdigest()
    assert h == expected


def test_compute_hash_missing_file(session_factory, tmp_path):
    clf = _make_classifier(session_factory, tmp_path)
    h = clf._compute_hash(tmp_path / "nonexistent.png")
    assert h == ""


# ---------------------------------------------------------------------------
# _scan_local_references
# ---------------------------------------------------------------------------

def test_scan_local_references_empty_dir(session_factory, tmp_path):
    clf = _make_classifier(session_factory, tmp_path)
    clf.references_dir = tmp_path
    result = clf._scan_local_references()
    assert result == {}


def test_scan_local_references_skips_top_level_files(session_factory, tmp_path):
    """A plain file (not a dir) at the top of references_dir hits the `continue` guard."""
    clf = _make_classifier(session_factory, tmp_path)
    clf.references_dir = tmp_path

    # Place a file directly in references_dir (not a subdir)
    (tmp_path / "stray_readme.txt").write_text("hello")

    result = clf._scan_local_references()
    assert result == {}


def test_scan_local_references_flat_layout(session_factory, tmp_path):
    """Label folder directly inside references_dir containing images."""
    clf = _make_classifier(session_factory, tmp_path)
    clf.references_dir = tmp_path

    label_dir = tmp_path / "hammer"
    label_dir.mkdir()
    img_file = label_dir / "img1.jpg"
    img = _make_rgb_image()
    img.save(str(img_file))

    result = clf._scan_local_references()
    assert any(v["label"] == "hammer" for v in result.values())


def test_scan_local_references_nested_layout(session_factory, tmp_path):
    """Category / Label nested structure."""
    clf = _make_classifier(session_factory, tmp_path)
    clf.references_dir = tmp_path

    cat_dir = tmp_path / "Tools"
    label_dir = cat_dir / "drill"
    label_dir.mkdir(parents=True)
    img_file = label_dir / "img1.jpg"
    _make_rgb_image().save(str(img_file))

    result = clf._scan_local_references()
    assert any(v["label"] == "drill" and v["category"] == "Tools" for v in result.values())


def test_scan_local_references_skips_non_image_files(session_factory, tmp_path):
    clf = _make_classifier(session_factory, tmp_path)
    clf.references_dir = tmp_path

    label_dir = tmp_path / "notes"
    label_dir.mkdir()
    (label_dir / "readme.txt").write_text("not an image")

    result = clf._scan_local_references()
    assert result == {}


def test_is_valid_image_file_rejects_hidden_image(session_factory, tmp_path):
    clf = _make_classifier(session_factory, tmp_path)

    hidden_file = tmp_path / ".hidden.jpg"
    hidden_file.write_bytes(b"fake image bytes")

    assert clf._is_valid_image_file(hidden_file) is False


def test_scan_local_references_manual_updates_merge(session_factory, tmp_path):
    """manual_updates for new hash gets merged in."""
    clf = _make_classifier(session_factory, tmp_path)
    clf.references_dir = tmp_path

    manual = {"new_hash_abc": {"path": "/some/path.jpg", "label": "vase", "category": "Decor"}}
    result = clf._scan_local_references(manual_updates=manual)
    assert "new_hash_abc" in result


def test_scan_local_references_manual_updates_existing_hash(session_factory, tmp_path):
    """manual_updates for a hash already on disk updates the entry."""
    clf = _make_classifier(session_factory, tmp_path)
    clf.references_dir = tmp_path

    label_dir = tmp_path / "chair"
    label_dir.mkdir()
    img_file = label_dir / "img1.jpg"
    _make_rgb_image().save(str(img_file))

    with patch.object(clf, "_compute_hash", return_value="known_hash"):
        manual = {"known_hash": {"label": "updated_chair", "category": "Furniture", "path": str(img_file)}}
        result = clf._scan_local_references(manual_updates=manual)

    assert result["known_hash"]["label"] == "updated_chair"


def test_scan_skips_hash_failure(session_factory, tmp_path):
    """If _compute_hash returns empty string, the file is skipped."""
    clf = _make_classifier(session_factory, tmp_path)
    clf.references_dir = tmp_path

    label_dir = tmp_path / "gadget"
    label_dir.mkdir()
    img_file = label_dir / "img1.jpg"
    _make_rgb_image().save(str(img_file))

    with patch.object(clf, "_compute_hash", return_value=""):
        result = clf._scan_local_references()
    assert result == {}


# ---------------------------------------------------------------------------
# load_references
# ---------------------------------------------------------------------------

def test_load_references_no_active_hashes(session_factory, tmp_path):
    clf = _make_classifier(session_factory, tmp_path)
    clf.references_dir = tmp_path

    with patch.object(clf, "_scan_local_references", return_value={}):
        clf.load_references()

    assert clf.search_matrix is None


def test_load_references_db_exception_triggers_rollback(session_factory, tmp_path):
    clf = _make_classifier(session_factory, tmp_path)
    clf.references_dir = tmp_path

    mock_session = MagicMock()
    mock_factory = MagicMock(return_value=mock_session)
    clf.session_factory = mock_factory

    with patch.object(clf, "_scan_local_references", return_value={"h1": {"path": "/x", "label": "a", "category": "b"}}):
        with patch.object(clf, "_sync_new_references", side_effect=RuntimeError("db crash")):
            clf.load_references()  # Should not propagate

    mock_session.rollback.assert_called_once()
    mock_session.close.assert_called_once()


def test_load_references_full_flow(session_factory, tmp_path):
    clf = _make_classifier(session_factory, tmp_path)
    clf.references_dir = tmp_path

    with patch.object(clf, "_scan_local_references", return_value={"h1": {"path": "/x", "label": "a", "category": "Cat"}}):
        with patch.object(clf, "_sync_new_references") as mock_sync:
            with patch.object(clf, "_prune_and_load_references") as mock_prune:
                clf.load_references()

    mock_sync.assert_called_once()
    mock_prune.assert_called_once()


# ---------------------------------------------------------------------------
# _sync_new_references
# ---------------------------------------------------------------------------

def test_sync_new_references_no_missing(session_factory, tmp_path):
    clf = _make_classifier(session_factory, tmp_path)

    mock_db = MagicMock()
    mock_db.get_existing_hashes.return_value = {"h1"}

    clf._sync_new_references(mock_db, {"h1": {"path": "/x", "label": "a", "category": "Cat"}}, {"h1"})
    mock_db.insert_embeddings.assert_not_called()


def test_sync_new_references_with_new_image(session_factory, tmp_path):
    clf = _make_classifier(session_factory, tmp_path)

    img_file = tmp_path / "test.jpg"
    _make_rgb_image().save(str(img_file))
    active_files = {"abc": {"path": str(img_file), "label": "thing", "category": "Stuff"}}

    mock_db = MagicMock()
    mock_db.get_existing_hashes.return_value = set()
    mock_db.get_or_create_item.return_value = 11
    mock_db.get_or_create_category.return_value = 1

    fake_emb = [0.1] * 512
    with patch.object(clf, "get_embeddings", return_value=[fake_emb] * 7):
        clf._sync_new_references(mock_db, active_files, {"abc"})

    mock_db.insert_embeddings.assert_called_once()


def test_sync_new_references_image_open_error(session_factory, tmp_path):
    clf = _make_classifier(session_factory, tmp_path)

    active_files = {"broken": {"path": str(tmp_path / "missing.jpg"), "label": "x", "category": "y"}}
    mock_db = MagicMock()
    mock_db.get_existing_hashes.return_value = set()
    mock_db.get_or_create_item.return_value = 5
    mock_db.get_or_create_category.return_value = 1

    # Missing file → Image.open raises, all_images stays empty → early return
    clf._sync_new_references(mock_db, active_files, {"broken"})
    mock_db.insert_embeddings.assert_not_called()


def test_sync_new_references_batch_exception(session_factory, tmp_path):
    clf = _make_classifier(session_factory, tmp_path)

    img_file = tmp_path / "test.jpg"
    _make_rgb_image().save(str(img_file))
    active_files = {"abc": {"path": str(img_file), "label": "thing", "category": "Stuff"}}

    mock_db = MagicMock()
    mock_db.get_existing_hashes.return_value = set()
    mock_db.get_or_create_item.return_value = 7
    mock_db.get_or_create_category.return_value = 1

    with patch.object(clf, "get_embeddings", side_effect=RuntimeError("embed fail")):
        clf._sync_new_references(mock_db, active_files, {"abc"})
    # Exception is caught internally; insert_embeddings should still be called (with empty list)
    mock_db.insert_embeddings.assert_called()


# ---------------------------------------------------------------------------
# _prune_and_load_references
# ---------------------------------------------------------------------------

def test_prune_keeps_valid_items(session_factory, tmp_path):
    clf = _make_classifier(session_factory, tmp_path)

    embedding = _make_embedding_record("tool", 1, "abc_0", embedding_id=10)

    mock_db = MagicMock()
    mock_db.get_all_embeddings.return_value = [embedding]
    mock_db.get_or_create_category.return_value = 1

    active_files = {"abc": {"category": "Tools"}}
    with patch.object(clf, "_build_search_index") as mock_build:
        clf._prune_and_load_references(mock_db, active_files, {"abc"})

    # No stale embeddings → delete_embeddings is guarded by `if del_ids:` so it's never called
    mock_db.delete_embeddings.assert_not_called()
    mock_build.assert_called_once()


def test_prune_deletes_stale_items(session_factory, tmp_path):
    clf = _make_classifier(session_factory, tmp_path)

    embedding = _make_embedding_record("tool", 1, "stale_0", embedding_id=99)

    mock_db = MagicMock()
    mock_db.get_all_embeddings.return_value = [embedding]

    with patch.object(clf, "_build_search_index"):
        clf._prune_and_load_references(mock_db, {}, set())

    mock_db.delete_embeddings.assert_called_with([99])


def test_prune_syncs_changed_category(session_factory, tmp_path):
    clf = _make_classifier(session_factory, tmp_path)

    embedding = _make_embedding_record("tool", 1, "abc_0", embedding_id=5)

    mock_db = MagicMock()
    mock_db.get_all_embeddings.return_value = [embedding]
    mock_db.get_or_create_category.return_value = 2  # different category

    active_files = {"abc": {"category": "NewCat"}}
    with patch.object(clf, "_build_search_index"):
        clf._prune_and_load_references(mock_db, active_files, {"abc"})

    assert embedding.booklet_category_id == 2


def test_prune_no_category_in_file_info(session_factory, tmp_path):
    """_sync_embedding_category returns early when file_info has no category."""
    clf = _make_classifier(session_factory, tmp_path)

    embedding = _make_embedding_record("tool", 1, "abc_0", embedding_id=7)

    mock_db = MagicMock()
    mock_db.get_all_embeddings.return_value = [embedding]

    # file_info has no category key → _sync_embedding_category must early-return
    active_files = {"abc": {}}
    with patch.object(clf, "_build_search_index") as mock_build:
        clf._prune_and_load_references(mock_db, active_files, {"abc"})

    # category id should remain unchanged, and get_or_create_category not called
    assert embedding.booklet_category_id == 1
    mock_db.get_or_create_category.assert_not_called()
    mock_build.assert_called_once()


# ---------------------------------------------------------------------------
# _build_search_index
# ---------------------------------------------------------------------------

def test_build_search_index_empty_items(session_factory, tmp_path):
    clf = _make_classifier(session_factory, tmp_path)
    clf._build_search_index([])
    assert clf.search_matrix is None


def test_build_search_index_builds_matrix(session_factory, tmp_path):
    clf = _make_classifier(session_factory, tmp_path)

    embedding = _make_embedding_record("pen", 1, "pen_0")

    clf._build_search_index([embedding])
    assert clf.search_matrix is not None
    assert clf.search_labels == ["pen"]
    assert "pen" in clf.reference_embeddings


# ---------------------------------------------------------------------------
# _clear_memory
# ---------------------------------------------------------------------------

def test_clear_memory(session_factory, tmp_path):
    clf = _make_classifier(session_factory, tmp_path)
    clf.search_matrix = torch.zeros(1, 512)
    clf.search_labels = ["something"]
    clf.search_categories = [1]
    clf.reference_embeddings = {"something": []}

    clf._clear_memory()
    assert clf.search_matrix is None
    assert clf.search_labels == []
    assert clf.search_categories == []
    assert clf.reference_embeddings == {}


# ---------------------------------------------------------------------------
# Branch coverage – _index_image_files with an empty directory
# ---------------------------------------------------------------------------

def test_index_image_files_skips_non_image_files(session_factory, tmp_path):
    """Files in directory that are not images must be skipped (branch 226->225: if condition is False)."""
    clf = _make_classifier(session_factory, tmp_path)
    # Place only a non-image file so the loop runs but never adds to active
    (tmp_path / "readme.txt").write_text("not an image")
    active = {}
    clf._index_image_files(tmp_path, "label", "category", active)
    assert active == {}


# ---------------------------------------------------------------------------
# Branch coverage – _sync_new_references: category ID cache hit
# ---------------------------------------------------------------------------

def test_sync_new_references_category_cache_hit(session_factory, tmp_path):
    """When two images share the same category, get_or_create_category is called only once."""
    clf = _make_classifier(session_factory, tmp_path)

    img1 = tmp_path / "a.jpg"
    img2 = tmp_path / "b.jpg"
    _make_rgb_image().save(str(img1))
    _make_rgb_image().save(str(img2))

    active_files = {
        "h1": {"path": str(img1), "label": "thing1", "category": "SharedCat"},
        "h2": {"path": str(img2), "label": "thing2", "category": "SharedCat"},
    }

    mock_db = MagicMock()
    mock_db.get_existing_hashes.return_value = set()
    mock_db.get_or_create_item.side_effect = [1, 2]
    mock_db.get_or_create_category.return_value = 5

    fake_emb = [0.1] * 512
    with patch.object(clf, "get_embeddings", return_value=[fake_emb] * 7):
        clf._sync_new_references(mock_db, active_files, {"h1", "h2"})

    # Category must have been looked up only once despite two images
    mock_db.get_or_create_category.assert_called_once_with("SharedCat")


# ---------------------------------------------------------------------------
# Branch coverage – _sync_embedding_category: cat_id already in cache
# ---------------------------------------------------------------------------

def test_prune_category_cache_hit(session_factory, tmp_path):
    """Two embedding rows with the same category name trigger a cache hit on the second row."""
    clf = _make_classifier(session_factory, tmp_path)

    item1 = _make_embedding_record("thing1", 1, "h1_0", embedding_id=1)
    item2 = _make_embedding_record("thing2", 1, "h2_0", embedding_id=2)

    mock_db = MagicMock()
    mock_db.get_all_embeddings.return_value = [item1, item2]
    mock_db.get_or_create_category.return_value = 1

    active_files = {
        "h1": {"category": "Tools"},
        "h2": {"category": "Tools"},  # same category → second call is a cache hit
    }
    with patch.object(clf, "_build_search_index"):
        clf._prune_and_load_references(mock_db, active_files, {"h1", "h2"})

    # Should be looked up only once, second item hits the cache
    mock_db.get_or_create_category.assert_called_once_with("Tools")


# ---------------------------------------------------------------------------
# Branch coverage – _build_search_index: duplicate label (385->387)
# ---------------------------------------------------------------------------

def test_build_search_index_duplicate_labels(session_factory, tmp_path):
    """Two embeddings with the same label exercise the `lbl in reference_embeddings` True branch."""
    clf = _make_classifier(session_factory, tmp_path)

    item1 = _make_embedding_record("pen", 1, "pen_0", embedding_id=1)
    item2 = _make_embedding_record("pen", 1, "pen_1", embedding_id=2)
    item2.embedding = [0.2] * 512

    clf._build_search_index([item1, item2])
    # Both embeddings should be stored under the same label key
    assert len(clf.reference_embeddings["pen"]) == 2


# ---------------------------------------------------------------------------
# Production classify path tests (No in-memory search matrix)
# ---------------------------------------------------------------------------

def test_classify_production_path_success(session_factory, tmp_path):
    clf = _make_classifier(session_factory, tmp_path)
    clf._clear_memory()

    # Populate DB with a few different items
    db_session = session_factory()
    try:
        from models import BookletCategory, BookletItem, BookletEmbedding
        cat = BookletCategory(category_name="Tools")
        db_session.add(cat)
        db_session.flush()

        # Item 1: hammer
        item_hammer = BookletItem(item_name="hammer")
        db_session.add(item_hammer)
        db_session.flush()
        emb_hammer = BookletEmbedding(
            booklet_item_id=item_hammer.id,
            booklet_category_id=cat.id,
            image_hash="hammer_0",
            embedding=[1.0] + [0.0] * 511
        )
        db_session.add(emb_hammer)

        # Item 2: wrench (same category)
        item_wrench = BookletItem(item_name="wrench")
        db_session.add(item_wrench)
        db_session.flush()
        emb_wrench = BookletEmbedding(
            booklet_item_id=item_wrench.id,
            booklet_category_id=cat.id,
            image_hash="wrench_0",
            embedding=[0.0, 1.0] + [0.0] * 510
        )
        db_session.add(emb_wrench)

        db_session.commit()
    finally:
        db_session.close()

    # Mock get_embeddings to return [1.0] + [0.0]*511 for the 3 scales
    fake_embs = [[1.0] + [0.0] * 511] * 3
    with patch.object(clf, "get_embeddings", return_value=fake_embs):
        result = clf.classify(_make_rgb_image(), threshold=0.1)

    assert result["class"] == "hammer"
    assert result["category_name"] == "Tools"
    assert len(result["matches"]) == 1
    assert result["matches"][0]["class"] == "wrench"


def test_classify_production_path_below_threshold(session_factory, tmp_path):
    clf = _make_classifier(session_factory, tmp_path)
    clf._clear_memory()

    db_session = session_factory()
    try:
        from models import BookletCategory, BookletItem, BookletEmbedding
        cat = BookletCategory(category_name="Tools")
        db_session.add(cat)
        db_session.flush()

        item = BookletItem(item_name="hammer")
        db_session.add(item)
        db_session.flush()
        emb = BookletEmbedding(
            booklet_item_id=item.id,
            booklet_category_id=cat.id,
            image_hash="hammer_0",
            embedding=[0.1] * 512
        )
        db_session.add(emb)
        db_session.commit()
    finally:
        db_session.close()

    # Low score query
    fake_embs = [[0.0] * 512] * 3
    with patch.object(clf, "get_embeddings", return_value=fake_embs):
        result = clf.classify(_make_rgb_image(), threshold=0.9)

    assert result["class"] == "Unknown"
    assert result["confidence"] == 0.0


def test_classify_production_path_empty_scores(session_factory, tmp_path):
    clf = _make_classifier(session_factory, tmp_path)
    clf._clear_memory()

    db_session = session_factory()
    try:
        from models import BookletCategory, BookletItem, BookletEmbedding
        cat = BookletCategory(category_name="Tools")
        db_session.add(cat)
        db_session.flush()

        item = BookletItem(item_name="hammer")
        db_session.add(item)
        db_session.flush()
        emb = BookletEmbedding(
            booklet_item_id=item.id,
            booklet_category_id=cat.id,
            image_hash="hammer_0",
            embedding=[0.5] * 512
        )
        db_session.add(emb)
        db_session.commit()
    finally:
        db_session.close()

    # If search_similar_embeddings returns empty list
    with patch.object(clf, "get_embeddings", return_value=[[0.5]*512]*3):
        from db_actions import DBActions
        with patch.object(DBActions, "search_similar_embeddings", return_value=[]):
            result = clf.classify(_make_rgb_image())
            assert result == {'class': 'Unknown Image'}


# ---------------------------------------------------------------------------
# reference_embeddings property tests
# ---------------------------------------------------------------------------

def test_reference_embeddings_property(session_factory, tmp_path):
    clf = _make_classifier(session_factory, tmp_path)

    # 1. Test read from DB dynamically
    db_session = session_factory()
    try:
        from models import BookletCategory, BookletItem, BookletEmbedding
        cat = BookletCategory(category_name="Tools")
        db_session.add(cat)
        db_session.flush()

        item = BookletItem(item_name="hammer")
        db_session.add(item)
        db_session.flush()

        emb1 = BookletEmbedding(
            booklet_item_id=item.id,
            booklet_category_id=cat.id,
            image_hash="hammer_0",
            embedding=[0.7] * 512
        )
        db_session.add(emb1)
        emb2 = BookletEmbedding(
            booklet_item_id=item.id,
            booklet_category_id=cat.id,
            image_hash="hammer_1",
            embedding=[0.8] * 512
        )
        db_session.add(emb2)
        db_session.commit()
    finally:
        db_session.close()

    # Clear any overrides/matrix and force DB lookup
    clf._clear_memory()
    clf._reference_embeddings_override = None
    ref_embs = clf.reference_embeddings
    assert "hammer" in ref_embs
    assert len(ref_embs["hammer"]) == 2
    assert ref_embs["hammer"][0] == [0.7] * 512
    assert ref_embs["hammer"][1] == [0.8] * 512

    # 2. Test setter override
    clf.reference_embeddings = {"custom": [[1.0] * 512]}
    assert clf.reference_embeddings == {"custom": [[1.0] * 512]}


# ---------------------------------------------------------------------------
# load_references postgresql branch test
# ---------------------------------------------------------------------------

def test_load_references_postgresql_dialect(session_factory, tmp_path):
    clf = _make_classifier(session_factory, tmp_path)
    clf.references_dir = tmp_path

    # Mock DB dialect name as postgresql
    from db_actions import DBActions

    # We mock DBActions session bind dialect name
    mock_embeddings = [_make_embedding_record("pen", 1, "pen_0")]

    with patch.object(clf, "_scan_local_references", return_value={"pen": {"path": "/x", "label": "pen", "category": "Tools"}}):
        with patch.object(DBActions, "get_all_embeddings", return_value=mock_embeddings):
            # We mock the bind.dialect.name
            with patch("sqlalchemy.engine.interfaces.Dialect") as mock_dialect:
                mock_dialect.name = "postgresql"
                
                # Mock session factory to return a session with this dialect name
                db_session = session_factory()
                db_session.bind.dialect = mock_dialect
                
                with patch.object(clf, "session_factory", return_value=db_session):
                    clf.load_references()
                    
                    # Since it is postgresql dialect, search_matrix should NOT be built in memory
                    assert clf.search_matrix is None


def test_classify_production_path_matches_limit_break(session_factory, tmp_path):
    clf = _make_classifier(session_factory, tmp_path)
    clf._clear_memory()

    db_session = session_factory()
    try:
        from models import BookletCategory, BookletItem, BookletEmbedding
        cat = BookletCategory(category_name="Tools")
        db_session.add(cat)
        db_session.flush()

        # Insert 8 items, each with 1 embedding
        # 1 best match (item0) + 7 other matches (item1 to item7)
        for i in range(8):
            item = BookletItem(item_name=f"item{i}")
            db_session.add(item)
            db_session.flush()
            
            # Use distinct one-hot embedding for each item
            vec = [0.0] * 512
            vec[i] = 1.0
            
            emb = BookletEmbedding(
                booklet_item_id=item.id,
                booklet_category_id=cat.id,
                image_hash=f"hash_{i}",
                embedding=vec
            )
            db_session.add(emb)
            
        db_session.commit()
    finally:
        db_session.close()

    # Query with a vector that is very close to item0, but has some overlap with all others
    # so they all get calculated with a positive score > threshold
    query = [1.0] + [0.1] * 7 + [0.0] * 504
    
    with patch.object(clf, "get_embeddings", return_value=[query] * 3):
        result = clf.classify(_make_rgb_image(), threshold=0.01)

    assert result["class"] == "item0"
    # Matches list should be capped at 5
    assert len(result["matches"]) == 5


def test_classify_production_path_propagates_exception(session_factory, tmp_path):
    clf = _make_classifier(session_factory, tmp_path)
    clf._clear_memory()

    # Make get_embeddings raise an exception, which should be caught and raised by classify()
    with patch.object(clf, "get_embeddings", side_effect=ValueError("Classification test error")):
        with pytest.raises(ValueError, match="Classification test error"):
            clf.classify(_make_rgb_image())


