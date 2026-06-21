"""Tests for main.py – FastAPI endpoints and helper functions."""

import io
import tarfile
import zipfile
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch, call
import pytest
from fastapi.testclient import TestClient
from PIL import Image, UnidentifiedImageError


# ---------------------------------------------------------------------------
# Helper: create a minimal in-memory PNG/ZIP/TAR
# ---------------------------------------------------------------------------

def _make_png_bytes(width=32, height=32) -> bytes:
    buf = io.BytesIO()
    Image.new("RGB", (width, height), (100, 100, 100)).save(buf, format="PNG")
    return buf.getvalue()


def _make_zip_bytes(file_tree: dict) -> bytes:
    """
    file_tree: {"path/inside/archive.png": b"contents", ...}
    """
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:
        for name, data in file_tree.items():
            zf.writestr(name, data)
    return buf.getvalue()


def _make_tar_bytes(file_tree: dict, mode="w") -> bytes:
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode=mode) as tf:
        for name, data in file_tree.items():
            info = tarfile.TarInfo(name=name)
            info.size = len(data)
            tf.addfile(info, io.BytesIO(data))
    return buf.getvalue()


def _completed_future(loop: object):
    future = loop.create_future()
    future.set_result(None)
    return future


def _close_coro_and_return_future(loop, coro):
    coro.close()
    return _completed_future(loop)


# ---------------------------------------------------------------------------
# Helper functions (unit tested separately from the ASGI app)
# ---------------------------------------------------------------------------

def test_is_valid_image_valid_extensions():
    from main import is_valid_image
    for ext in ["photo.jpg", "photo.jpeg", "photo.png", "photo.webp", "photo.bmp"]:
        assert is_valid_image(ext), f"Expected {ext} to be valid"


def test_is_valid_image_invalid_extension():
    from main import is_valid_image
    assert not is_valid_image("archive.zip")
    assert not is_valid_image("document.pdf")
    assert not is_valid_image("data.txt")


def test_sanitize_name_none():
    from main import sanitize_name
    assert sanitize_name(None) is None


def test_sanitize_name_with_spaces():
    from main import sanitize_name
    assert sanitize_name("hello world") == "hello_world"


def test_sanitize_name_strips_leading_trailing():
    from main import sanitize_name
    assert sanitize_name("  item  ") == "item"


def test_is_valid_reference_filename_rejects_hidden():
    from main import is_valid_reference_filename
    assert not is_valid_reference_filename("._image.png")
    assert not is_valid_reference_filename(".DS_Store")
    assert not is_valid_reference_filename(".hidden.jpg")
    assert not is_valid_reference_filename("DS_Store")


def test_is_valid_reference_filename_rejects_ds_store_without_dot():
    from main import is_valid_reference_filename
    assert not is_valid_reference_filename("DS_Store")


def test_is_valid_archive_file_filters_metadata(tmp_path):
    from main import _is_valid_archive_file
    assert _is_valid_archive_file(tmp_path / "image.png") is False

    file_path = tmp_path / "__MACOSX" / "image.png"
    file_path.parent.mkdir(parents=True, exist_ok=True)
    file_path.write_bytes(b"x")
    assert _is_valid_archive_file(file_path) is False

    ds_file = tmp_path / ".DS_Store"
    ds_file.write_bytes(b"x")
    assert _is_valid_archive_file(ds_file) is False

    ds_file2 = tmp_path / "DS_Store"
    ds_file2.write_bytes(b"x")
    assert _is_valid_archive_file(ds_file2) is False

    hidden_file = tmp_path / ".hidden.png"
    hidden_file.write_bytes(b"x")
    assert _is_valid_archive_file(hidden_file) is False

    txt_file = tmp_path / "image.txt"
    txt_file.write_bytes(b"x")
    assert _is_valid_archive_file(txt_file) is False


def test_process_archive_creates_dest_dirs(tmp_path, mock_classifier):
    from main import _process_archive
    zip_bytes = _make_zip_bytes({"Cat/Label/img.png": _make_png_bytes()})
    mock_classifier.references_dir = tmp_path
    mock_classifier._compute_hash.return_value = "h"
    updates, counts = _process_archive(zip_bytes, "upload.zip", mock_classifier)
    assert counts.get("Label") == 1
    assert (tmp_path / "Cat" / "Label" / "img.png").exists()


def test_process_archive_skips_metadata_entries(tmp_path, mock_classifier):
    from main import _process_archive
    zip_bytes = _make_zip_bytes({"__MACOSX/._file": b"x", "Cat/Label/img.png": _make_png_bytes()})
    mock_classifier.references_dir = tmp_path
    mock_classifier._compute_hash.return_value = "h"
    updates, counts = _process_archive(zip_bytes, "upload.zip", mock_classifier)
    assert counts.get("Label", 0) == 1


def test_process_archive_skipped_entries_logs(tmp_path, mock_classifier):
    from main import _process_archive
    zip_bytes = _make_zip_bytes({"chair1.png": _make_png_bytes()})
    mock_classifier.references_dir = tmp_path
    mock_classifier._compute_hash.return_value = "h"
    updates, counts = _process_archive(zip_bytes, "upload.zip", mock_classifier)
    assert updates == {}
    assert counts == {}


def test_process_archive_skipped_branch_is_executed(tmp_path, mock_classifier, monkeypatch):
    from main import _process_archive
    with monkeypatch.context() as m:
        m.setattr("main._extract_archive", lambda contents, filename, dest: None)
        m.setattr("main._parse_archive_entries", lambda temp_path, filename: ([], 2))
        mock_classifier.references_dir = tmp_path
        mock_classifier._compute_hash.return_value = "h"
        updates, counts = _process_archive(b"", "upload.zip", mock_classifier)
    assert updates == {}
    assert counts == {}


def test_initialize_db_failure(monkeypatch):
    import main
    from main import _initialize_db
    monkeypatch.setattr(main, "initialize_database", lambda engine: (_ for _ in ()).throw(RuntimeError("fail")))
    # Should not raise
    _initialize_db()


# ---------------------------------------------------------------------------
# _extract_archive
# ---------------------------------------------------------------------------

def test_extract_archive_zip(tmp_path):
    from main import _extract_archive
    png_data = _make_png_bytes()
    zip_bytes = _make_zip_bytes({"image.png": png_data})
    _extract_archive(zip_bytes, "upload.zip", tmp_path)
    assert (tmp_path / "image.png").exists()


def test_extract_archive_tar(tmp_path):
    from main import _extract_archive
    png_data = _make_png_bytes()
    tar_bytes = _make_tar_bytes({"image.png": png_data})
    _extract_archive(tar_bytes, "upload.tar", tmp_path)
    assert (tmp_path / "image.png").exists()


def test_extract_archive_tar_gz(tmp_path):
    from main import _extract_archive
    png_data = _make_png_bytes()
    tar_bytes = _make_tar_bytes({"image.png": png_data}, mode="w:gz")
    _extract_archive(tar_bytes, "upload.tar.gz", tmp_path)
    assert (tmp_path / "image.png").exists()


def test_extract_archive_tgz(tmp_path):
    from main import _extract_archive
    png_data = _make_png_bytes()
    tar_bytes = _make_tar_bytes({"image.png": png_data}, mode="w:gz")
    _extract_archive(tar_bytes, "upload.tgz", tmp_path)
    assert (tmp_path / "image.png").exists()


def test_extract_archive_tar_bz2(tmp_path):
    from main import _extract_archive
    png_data = _make_png_bytes()
    tar_bytes = _make_tar_bytes({"image.png": png_data}, mode="w:bz2")
    _extract_archive(tar_bytes, "upload.tar.bz2", tmp_path)
    assert (tmp_path / "image.png").exists()


def test_extract_archive_tar_uses_data_filter(tmp_path):
    import main

    fake_tf = MagicMock()
    fake_tf.__enter__.return_value = fake_tf
    fake_tf.__exit__.return_value = None

    with patch("main.tarfile.open", return_value=fake_tf):
        main._extract_archive(b"tar bytes", "upload.tar", tmp_path)

    fake_tf.extractall.assert_called_once_with(tmp_path, filter="data")


# ---------------------------------------------------------------------------
# _process_archive
# ---------------------------------------------------------------------------

def test_process_archive_nested_3_levels(tmp_path, mock_classifier):
    """Category/Item/image.png → 3-level path."""
    from main import _process_archive
    png_data = _make_png_bytes()
    zip_bytes = _make_zip_bytes({"Furniture/Chair/chair1.png": png_data})

    mock_classifier.references_dir = tmp_path
    mock_classifier._compute_hash.return_value = "abc123"

    updates, counts = _process_archive(zip_bytes, "upload.zip", mock_classifier)
    assert "Chair" in counts or "chair" in str(counts).lower() or counts  # at least 1 label


def test_process_archive_2_level_path(tmp_path, mock_classifier):
    """2-level path uses archive stem as category."""
    from main import _process_archive
    png_data = _make_png_bytes()
    zip_bytes = _make_zip_bytes({"Chair/chair1.png": png_data})

    mock_classifier.references_dir = tmp_path
    mock_classifier._compute_hash.return_value = "abc456"

    updates, counts = _process_archive(zip_bytes, "myarchive.zip", mock_classifier)
    assert counts  # at least 1 label found


def test_process_archive_root_image_skipped(tmp_path, mock_classifier):
    """Images at root level (no folder) are skipped."""
    from main import _process_archive
    png_data = _make_png_bytes()
    zip_bytes = _make_zip_bytes({"chair1.png": png_data})

    mock_classifier.references_dir = tmp_path
    mock_classifier._compute_hash.return_value = "xyz"

    updates, counts = _process_archive(zip_bytes, "upload.zip", mock_classifier)
    assert counts == {}


def test_process_archive_hash_failure(tmp_path, mock_classifier):
    """If _compute_hash returns empty string, skip that file."""
    from main import _process_archive
    png_data = _make_png_bytes()
    zip_bytes = _make_zip_bytes({"Cat/Label/img.png": png_data})

    mock_classifier.references_dir = tmp_path
    mock_classifier._compute_hash.return_value = ""

    updates, counts = _process_archive(zip_bytes, "upload.zip", mock_classifier)
    assert updates == {}


# ---------------------------------------------------------------------------
# GET /
# ---------------------------------------------------------------------------

def test_health_check(client):
    resp = client.get("/")
    assert resp.status_code == 200
    assert resp.json()["status"] == "active"


# ---------------------------------------------------------------------------
# POST /classify
# ---------------------------------------------------------------------------

def test_classify_invalid_file_type(client):
    resp = client.post(
        "/classify",
        files={"file": ("document.pdf", b"not an image", "application/pdf")},
    )
    assert resp.status_code == 400


def test_classify_valid_image(client, mock_classifier):
    import main
    with patch.object(main, "classifier", mock_classifier):
        resp = client.post(
            "/classify",
            files={"file": ("photo.png", _make_png_bytes(), "image/png")},
        )
    assert resp.status_code == 200
    assert resp.json()["class"] == "chair"


def test_classify_unidentified_image(client, mock_classifier):
    import main
    with patch("main.Image") as mock_pil, patch.object(main, "classifier", mock_classifier):
        mock_pil.open.side_effect = UnidentifiedImageError("bad image")
        resp = client.post(
            "/classify",
            files={"file": ("photo.png", b"garbage", "image/png")},
        )
    assert resp.status_code == 400


def test_classify_internal_error(client, mock_classifier):
    import main
    mock_classifier.classify.side_effect = RuntimeError("crash")
    with patch.object(main, "classifier", mock_classifier):
        resp = client.post(
            "/classify",
            files={"file": ("photo.png", _make_png_bytes(), "image/png")},
        )
    assert resp.status_code == 500


# ---------------------------------------------------------------------------
# POST /refresh
# ---------------------------------------------------------------------------

def test_refresh_success(client, mock_classifier):
    import main
    with patch.object(main, "classifier", mock_classifier):
        mock_classifier.load_references.return_value = None
        mock_classifier.reference_embeddings = {"chair": [], "table": []}
        with patch("main.run_in_threadpool", new_callable=AsyncMock) as mock_threadpool:
            mock_threadpool.side_effect = lambda fn: fn()
            resp = client.post("/refresh")
    assert resp.status_code == 200
    data = resp.json()
    assert "classes" in data
    mock_threadpool.assert_awaited_once_with(mock_classifier.load_references)


def test_refresh_internal_error(client, mock_classifier):
    import main
    mock_classifier.load_references.side_effect = RuntimeError("db gone")
    with patch.object(main, "classifier", mock_classifier):
        with patch("main.run_in_threadpool", new_callable=AsyncMock) as mock_threadpool:
            mock_threadpool.side_effect = lambda fn: fn()
            resp = client.post("/refresh")
    assert resp.status_code == 500
    mock_threadpool.assert_awaited_once_with(mock_classifier.load_references)


def test_refresh_classifier_unavailable(client):
    import main

    with patch.object(main, "classifier", None):
        with patch("main.run_in_threadpool", new_callable=AsyncMock) as mock_threadpool:
            resp = client.post("/refresh")

    assert resp.status_code == 500
    mock_threadpool.assert_not_awaited()


# ---------------------------------------------------------------------------
# POST /add_reference
# ---------------------------------------------------------------------------

def test_add_reference_invalid_file_type(client):
    resp = client.post(
        "/add_reference",
        data={"label": "chair"},
        files={"file": ("doc.pdf", b"data", "application/pdf")},
    )
    assert resp.status_code == 400


def test_add_reference_success(client, mock_classifier, tmp_path):
    import main
    mock_classifier.references_dir = tmp_path
    mock_classifier._compute_hash.return_value = "abc"
    mock_classifier.load_references.return_value = None

    with patch.object(main, "classifier", mock_classifier):
        resp = client.post(
            "/add_reference",
            data={"label": "chair", "category": "Furniture"},
            files={"file": ("chair.png", _make_png_bytes(), "image/png")},
        )
    assert resp.status_code == 200
    assert resp.json()["category"] == "Furniture"


def test_add_reference_no_category_defaults_uncategorized(client, mock_classifier, tmp_path):
    import main
    mock_classifier.references_dir = tmp_path
    mock_classifier._compute_hash.return_value = "abc"

    with patch.object(main, "classifier", mock_classifier):
        resp = client.post(
            "/add_reference",
            data={"label": "pen"},
            files={"file": ("pen.png", _make_png_bytes(), "image/png")},
        )
    assert resp.status_code == 200
    assert resp.json()["category"] == "Uncategorized"


def test_add_reference_internal_error(client, mock_classifier, tmp_path):
    import main
    mock_classifier.references_dir = tmp_path
    mock_classifier._compute_hash.side_effect = RuntimeError("io error")

    with patch.object(main, "classifier", mock_classifier):
        resp = client.post(
            "/add_reference",
            data={"label": "chair"},
            files={"file": ("chair.png", _make_png_bytes(), "image/png")},
        )
    assert resp.status_code == 500


# ---------------------------------------------------------------------------
# POST /bulk_upload
# ---------------------------------------------------------------------------

def test_bulk_upload_unsupported_format(client):
    resp = client.post(
        "/bulk_upload",
        files={"file": ("archive.7z", b"data", "application/octet-stream")},
    )
    assert resp.status_code == 400


def test_bulk_upload_success_with_updates(client, mock_classifier, tmp_path):
    import main
    png_data = _make_png_bytes()
    zip_bytes = _make_zip_bytes({"Cat/Item/img.png": png_data})

    mock_classifier.references_dir = tmp_path
    mock_classifier._compute_hash.return_value = "hash1"

    with patch.object(main, "classifier", mock_classifier):
        with patch("main._process_archive", return_value=({"h1": {}}, {"Item": 1})):
            resp = client.post(
                "/bulk_upload",
                files={"file": ("upload.zip", zip_bytes, "application/zip")},
            )
    assert resp.status_code == 200
    assert resp.json()["total_images"] == 1


def test_bulk_upload_no_manual_updates(client, mock_classifier, tmp_path):
    """When _process_archive returns empty updates, load_references is not called."""
    import main
    zip_bytes = _make_zip_bytes({"img.png": _make_png_bytes()})  # root-level → skipped

    mock_classifier.references_dir = tmp_path

    with patch.object(main, "classifier", mock_classifier):
        with patch("main._process_archive", return_value=({}, {})):
            resp = client.post(
                "/bulk_upload",
                files={"file": ("upload.zip", zip_bytes, "application/zip")},
            )
    mock_classifier.load_references.assert_not_called()
    assert resp.status_code == 200


def test_bulk_upload_internal_error(client, mock_classifier):
    import main
    with patch.object(main, "classifier", mock_classifier):
        with patch("main._process_archive", side_effect=RuntimeError("disk full")):
            resp = client.post(
                "/bulk_upload",
                files={"file": ("upload.zip", b"bad", "application/zip")},
            )
    assert resp.status_code == 500


# ---------------------------------------------------------------------------
# auto_refresh_task and lifespan
# ---------------------------------------------------------------------------

def test_auto_refresh_task_calls_load_references():
    """auto_refresh_task refreshes via the threadpool after sleep."""
    import asyncio
    import main

    mock_clf = MagicMock()
    mock_clf.reference_embeddings = {}

    async def run():
        with patch.object(main, "classifier", mock_clf):
            with patch("asyncio.sleep", new_callable=AsyncMock) as mock_sleep:
                with patch("main.run_in_threadpool", new_callable=AsyncMock) as mock_threadpool:
                    mock_threadpool.side_effect = lambda fn: fn()
                    # Make it run once then raise CancelledError to exit the loop
                    mock_sleep.side_effect = [None, asyncio.CancelledError()]
                    with pytest.raises(asyncio.CancelledError):
                        await main.auto_refresh_task()

        mock_threadpool.assert_awaited_once_with(mock_clf.load_references)
        mock_clf.load_references.assert_called_once()

    asyncio.run(run())


def test_auto_refresh_task_handles_exception():
    """auto_refresh_task swallows exceptions inside the loop."""
    import asyncio
    import main

    mock_clf = MagicMock()
    mock_clf.load_references.side_effect = [RuntimeError("fail"), None]
    mock_clf.reference_embeddings = {}

    async def run():
        with patch.object(main, "classifier", mock_clf):
            with patch("asyncio.sleep", new_callable=AsyncMock) as mock_sleep:
                with patch("main.run_in_threadpool", new_callable=AsyncMock) as mock_threadpool:
                    mock_threadpool.side_effect = lambda fn: fn()
                    mock_sleep.side_effect = [None, None, asyncio.CancelledError()]
                    with pytest.raises(asyncio.CancelledError):
                        await main.auto_refresh_task()

        assert mock_threadpool.await_count == 2

    asyncio.run(run())


def test_lifespan_startup_classifier_init_success():
    """Lifespan starts classifier and background task."""
    import asyncio
    import main

    mock_clf = MagicMock()

    async def run():
        loop = asyncio.get_event_loop()

        with patch("main.ImageClassifier", return_value=mock_clf):
            with patch("main.intercept_uvicorn_logs"):
                with patch(
                    "main.asyncio.create_task",
                    side_effect=lambda coro: _close_coro_and_return_future(loop, coro),
                ):
                    async with main.lifespan(main.app):
                        pass

    asyncio.run(run())


def test_lifespan_startup_classifier_init_failure():
    """Lifespan logs critical error but doesn't crash if classifier fails."""
    import asyncio
    import main

    async def run():
        loop = asyncio.get_event_loop()

        with patch("main.ImageClassifier", side_effect=RuntimeError("model fail")):
            with patch("main.intercept_uvicorn_logs"):
                with patch(
                    "main.asyncio.create_task",
                    side_effect=lambda coro: _close_coro_and_return_future(loop, coro),
                ):
                    async with main.lifespan(main.app):
                        pass  # Should not raise

    asyncio.run(run())


def test_lifespan_shutdown_cancels_task():
    """Lifespan cancels the background task on shutdown."""
    import asyncio
    import main

    cancelled = []

    async def run():
        loop = asyncio.get_event_loop()
        future = _completed_future(loop)
        # Wrap cancel to record call
        original_cancel = future.cancel
        future.cancel = lambda *a, **kw: cancelled.append(True) or original_cancel(*a, **kw)

        with patch("main.ImageClassifier", return_value=MagicMock()):
            with patch("main.intercept_uvicorn_logs"):
                with patch(
                    "main.asyncio.create_task",
                    side_effect=lambda coro: coro.close() or future,
                ):
                    async with main.lifespan(main.app):
                        pass

        assert cancelled, "Expected task.cancel() to be called"

    asyncio.run(run())


# ---------------------------------------------------------------------------
# Branch coverage – auto_refresh_task: classifier is None (50->46)
# ---------------------------------------------------------------------------

def test_auto_refresh_task_classifier_none():
    """When classifier is None the load_references branch is skipped (branch 50->46)."""
    import asyncio
    import main

    async def run():
        with patch.object(main, "classifier", None):
            with patch("asyncio.sleep", new_callable=AsyncMock) as mock_sleep:
                with patch("main.run_in_threadpool", new_callable=AsyncMock) as mock_threadpool:
                    mock_sleep.side_effect = [None, asyncio.CancelledError()]
                    with pytest.raises(asyncio.CancelledError):
                        await main.auto_refresh_task()
        mock_threadpool.assert_not_awaited()
        # No classifier → load_references should never be called (no AttributeError either)

    asyncio.run(run())


# ---------------------------------------------------------------------------
# Branch coverage – lifespan: no _refresh_task at shutdown (78->85)
# ---------------------------------------------------------------------------

def test_lifespan_shutdown_no_refresh_task():
    """If _refresh_task is None at shutdown, the cancel block is skipped (branch 78->85)."""
    import asyncio
    import main

    async def run():
        with patch("main.ImageClassifier", return_value=MagicMock()):
            with patch("main.intercept_uvicorn_logs"):
                # Return None so _refresh_task stays None after create_task
                with patch("main.asyncio.create_task", side_effect=lambda coro: coro.close() or None):
                    async with main.lifespan(main.app):
                        pass  # Should complete without error

    asyncio.run(run())


# ---------------------------------------------------------------------------
# Branch coverage – _extract_archive: neither .zip nor known tar (139->exit)
# ---------------------------------------------------------------------------

def test_extract_archive_unknown_format(tmp_path):
    """An unrecognised filename hits neither branch and returns silently (branch 139->exit)."""
    from main import _extract_archive
    # Pass a name that doesn't match .zip or any tar variant
    _extract_archive(b"some bytes", "file.unknown", tmp_path)
    # Nothing should be extracted and no exception raised
    assert list(tmp_path.iterdir()) == []


# ---------------------------------------------------------------------------
# Branch coverage – _process_archive: duplicate dest_dir key (187->185)
# ---------------------------------------------------------------------------

def test_process_archive_duplicate_dest_dir_key(tmp_path, mock_classifier):
    """Two images with the same category+label hit the dest_dirs cache (branch 187->185)."""
    from main import _process_archive
    png_data = _make_png_bytes()
    # Two images under the same Category/Label → same dest_dir key
    zip_bytes = _make_zip_bytes({
        "Cat/Label/img1.png": png_data,
        "Cat/Label/img2.png": png_data,
    })

    mock_classifier.references_dir = tmp_path
    mock_classifier._compute_hash.side_effect = ["hash_a", "hash_b"]

    updates, counts = _process_archive(zip_bytes, "upload.zip", mock_classifier)
    assert counts.get("Label", 0) == 2  # both images counted
