import io
import os
import shutil
import zipfile
import tarfile
import tempfile
import asyncio
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Annotated, Optional
from contextlib import asynccontextmanager

from fastapi import FastAPI, UploadFile, File, Form, HTTPException, Request
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.concurrency import run_in_threadpool
from fastapi.staticfiles import StaticFiles
from PIL import Image, UnidentifiedImageError

from classifier import ImageClassifier
from utils import setup_logger, intercept_uvicorn_logs
from db import engine, sessionLocal
from schema_migrations import initialize_database

# =========================================================================
# Setup & Initialization
# =========================================================================



logger = setup_logger("snapclass.api")


def _initialize_db():
    try:
        initialize_database(engine)
        logger.info("Database connected successfully.")
    except Exception as e:
        logger.error(f"Database initialization failed: {e}")

# Initialize database tables
_initialize_db()

ALLOWED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
SUPPORTED_ARCHIVES = {".zip", ".tar", ".tar.gz", ".tar.bz2", ".tgz"}

# Globals
classifier: ImageClassifier | None = None
_refresh_task: asyncio.Task | None = None


# =========================================================================
# Lifespan & Background Tasks
# =========================================================================

async def _reload_classifier_references(current_classifier: ImageClassifier) -> list[str]:
    """Reload references without blocking the event loop."""
    await run_in_threadpool(current_classifier.load_references)
    return list(current_classifier.reference_embeddings.keys())


async def auto_refresh_task():
    """Background task that refreshes references every 24 hours."""
    while True:
        await asyncio.sleep(24 * 60 * 60)  # 24 hours
        try:
            logger.info("Auto-refresh: Scanning for new folders and images...")
            current_classifier = classifier
            if current_classifier is None:
                logger.warning("Auto-refresh skipped because classifier is unavailable.")
                continue

            classes = await _reload_classifier_references(current_classifier)
            logger.info(f"Auto-refresh complete. Classes: {classes}")
        except Exception as e:
            logger.error(f"Auto-refresh failed: {e}")

@asynccontextmanager
async def lifespan(app: FastAPI):
    global classifier, _refresh_task
    
    # Unify Uvicorn and custom logging
    intercept_uvicorn_logs()
    logger.info("Application startup: Logging unified.")
    
    try:
        classifier = ImageClassifier(session_factory=sessionLocal)
        logger.info("Classifier initialized successfully.")
    except Exception as e:
        logger.critical(f"Failed to initialize classifier: {e}")
    
    # Start background auto-refresh task
    _refresh_task = asyncio.create_task(auto_refresh_task())
    logger.info("Background auto-refresh task started (runs every 24 hours).")
    
    yield
    
    # Shutdown logic
    if _refresh_task:
        _refresh_task.cancel()
        import contextlib
        with contextlib.suppress(asyncio.CancelledError):
            await _refresh_task
        logger.info("Background auto-refresh task stopped.")
    
    logger.info("Application shutdown.")


# =========================================================================
# FastAPI App Configuration
# =========================================================================


app = FastAPI(title="SnapClass: Offline Few-Shot Classifier", lifespan=lifespan)

Path("references").mkdir(exist_ok=True)
app.mount("/references", StaticFiles(directory="references"), name="references")

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://127.0.0.1:5173",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.middleware("http")
async def log_requests(request: Request, call_next):
    start_time = time.perf_counter()
    logger.info(f"Incoming request: {request.method} {request.url.path}")
    response = await call_next(request)
    process_time = time.perf_counter() - start_time
    logger.info(
        f"Method: {request.method} | "
        f"Path: {request.url.path} | "
        f"Status: {response.status_code} | "
        f"Time: {process_time:.4f}s"
    )
    return response


# =========================================================================
# Helper Functions
# =========================================================================

def is_valid_image(filename: str) -> bool:
    return Path(filename).suffix.lower() in ALLOWED_EXTENSIONS


def is_valid_reference_filename(filename: str) -> bool:
    """Reject hidden macOS metadata files before saving references."""
    name = Path(filename).name
    if name in {'.DS_Store'}:
        return False
    if name.startswith('._'):
        return False
    if name.startswith('.'):
        return False
    return Path(name).suffix.lower() in ALLOWED_EXTENSIONS


def sanitize_name(name: str | None) -> str | None:
    """Replaces spaces with underscores for safe directory/label names."""
    return name.strip().replace(" ", "_") if name else None

IGNORED_ARCHIVE_DIRS = {"__MACOSX"}
IGNORED_ARCHIVE_FILES = {".DS_Store"}


def _extract_archive(contents: bytes, filename: str, dest_dir: Path):
    """Extracts a supported archive into the destination directory."""
    lower_name = filename.lower()

    if lower_name.endswith(".zip"):
        with zipfile.ZipFile(io.BytesIO(contents), 'r') as zf:
            zf.extractall(dest_dir)

    elif lower_name.endswith(tuple(SUPPORTED_ARCHIVES - {".zip"})):
        mode = 'r:gz' if lower_name.endswith((".tar.gz", ".tgz")) else \
               'r:bz2' if lower_name.endswith(".tar.bz2") else 'r'
        with tarfile.open(fileobj=io.BytesIO(contents), mode=mode) as tf:
            try:
                tf.extractall(dest_dir, filter="data")
            except TypeError:  # pragma: no cover
                tf.extractall(dest_dir)


def _is_valid_archive_file(file_path: Path) -> bool:
    """Validates extracted archive files before creating tasks."""
    if not file_path.is_file():
        return False
    name = file_path.name
    if name in IGNORED_ARCHIVE_FILES:
        return False
    if name.startswith('._'):
        return False
    if name.startswith('.'):  # pragma: no cover
        return False
    if file_path.suffix.lower() not in ALLOWED_EXTENSIONS:
        return False
    if any(part in IGNORED_ARCHIVE_DIRS for part in file_path.parts):
        return False
    return True


def _parse_archive_entries(temp_path: Path, filename: str) -> tuple[list, int]:
    """Scans extracted archive tree and returns (file_path, safe_label, safe_category) tuples and skipped count."""
    tasks = []
    skipped_count = 0
    archive_stem = Path(filename).stem.replace('.tar', '')

    for file_path in temp_path.rglob('*'):
        if not _is_valid_archive_file(file_path):
            skipped_count += 1
            continue

        parts = file_path.relative_to(temp_path).parts

        if len(parts) >= 3:
            category_name, item_name = parts[-3], parts[-2]
        elif len(parts) == 2:
            category_name, item_name = archive_stem, parts[-2]
        else:
            skipped_count += 1
            continue  # Skip images in the root of the archive

        tasks.append((
            file_path,
            sanitize_name(item_name),
            sanitize_name(category_name) or "Uncategorized",
        ))

    return tasks, skipped_count


def _process_archive(contents: bytes, filename: str, classifier: ImageClassifier):
    """Extracts, processes, and copies bulk uploaded images to the nested reference dir."""
    labels_count = {}
    manual_updates = {}

    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        _extract_archive(contents, filename, temp_path)

        tasks, skipped = _parse_archive_entries(temp_path, filename)
        if skipped:  # pragma: no cover
            logger.info(f"Skipped {skipped} archive entries (metadata/invalid files) during bulk upload.")

        # Pre-create all destination directories (fast, sequential)  # pragma: no cover
        dest_dirs: dict[tuple, Path] = {}  # pragma: no cover
        for _, safe_label, safe_category in tasks:
            key = (safe_category, safe_label)
            if key not in dest_dirs:
                dest_label_dir = classifier.references_dir / safe_category / safe_label
                dest_label_dir.mkdir(parents=True, exist_ok=True)
                dest_dirs[key] = dest_label_dir

        def _copy_and_hash(task):
            """Copy a single file to its destination and compute its hash."""
            file_path, safe_label, safe_category = task
            dest_path = dest_dirs[(safe_category, safe_label)] / file_path.name
            shutil.copy2(file_path, dest_path)
            h = classifier._compute_hash(dest_path)
            return h, dest_path, safe_label, safe_category

        # Parallel copy + hash using I/O threads
        max_workers = min(16, len(tasks)) if tasks else 1
        with ThreadPoolExecutor(max_workers=max_workers) as pool:
            results = list(pool.map(_copy_and_hash, tasks))

        for h, dest_path, safe_label, safe_category in results:
            if h:
                manual_updates[h] = {
                    "path": str(dest_path),
                    "label": safe_label,
                    "category": safe_category,
                }
            labels_count[safe_label] = labels_count.get(safe_label, 0) + 1

    return manual_updates, labels_count


# =========================================================================
# API Endpoints
# =========================================================================

@app.get("/")
async def health_check():
    return {"message": "Image Classifier API is running. Use /classify to check images.", "status": "active"}


@app.post("/classify", responses={400: {"description": "Invalid file type or content."}})
async def classify_image(file: Annotated[UploadFile, File(...)]):
    if not is_valid_image(file.filename):
        logger.warning(f"Rejected classification request for file: {file.filename}")
        raise HTTPException(status_code=400, detail=f"Invalid file type. Allowed: {', '.join(ALLOWED_EXTENSIONS)}")

    try:
        contents = await file.read()
        image = Image.open(io.BytesIO(contents)).convert("RGB")
        result = classifier.classify(image)
        return result
    except UnidentifiedImageError:
        logger.error(f"Failed to identify image file: {file.filename}")
        raise HTTPException(status_code=400, detail="Invalid image file or content.")
    except Exception as e:
        logger.error(f"Error processing classification request: {e}")
        return JSONResponse(status_code=500, content={"error": "Internal server error processing image."})


@app.post("/refresh")
async def refresh_references():
    """Reloads the reference images from disk."""
    try:
        if classifier is None:
            raise RuntimeError("Classifier unavailable.")

        classes = await _reload_classifier_references(classifier)
        logger.info(f"References refreshed. Classes: {classes}")
        return {"message": "References reloaded", "classes": classes}
    except Exception as e:
        logger.error(f"Error refreshing references: {e}")
        return JSONResponse(status_code=500, content={"error": "Failed to refresh references."})


@app.post("/add_reference", responses={400: {"description": "Invalid file type."}})
async def add_reference(
    label: Annotated[str, Form(...)], 
    file: Annotated[UploadFile, File(...)],
    category: Annotated[Optional[str], Form()] = None
):
    """Uploads a new reference image for a specific label."""
    if not is_valid_reference_filename(file.filename):
        logger.warning(f"Rejected add_reference request for file: {file.filename}")
        raise HTTPException(status_code=400, detail=f"Invalid reference filename. Allowed image extensions: {', '.join(ALLOWED_EXTENSIONS)}")

    try:
        safe_label = sanitize_name(label)
        safe_category = sanitize_name(category) or "Uncategorized"

        # Create destination directory (Category/Label nested)
        label_dir = classifier.references_dir / safe_category / safe_label
        label_dir.mkdir(parents=True, exist_ok=True)
        file_path = label_dir / file.filename
        
        # Save file to disk
        def save_upload_file(src, dest):
            with open(dest, "wb") as buffer:
                shutil.copyfileobj(src, buffer)
        
        await run_in_threadpool(save_upload_file, file.file, file_path)
        logger.info(f"Added new reference: {file.filename} to class {safe_label} under category {safe_category}")
        
        # Update classifier memory & DB immediately
        h = classifier._compute_hash(file_path)
        manual_updates = {h: {"path": str(file_path), "label": safe_label, "category": safe_category}}
        classifier.load_references(manual_updates=manual_updates)
        
        return {"message": f"Added reference for '{safe_label}'", "filename": file.filename, "category": safe_category}
    except Exception as e:
        logger.error(f"Error adding reference: {e}")
        return JSONResponse(status_code=500, content={"error": "Failed to add reference."})


@app.post("/bulk_upload", responses={400: {"description": "Unsupported archive format."}})
async def bulk_upload(file: Annotated[UploadFile, File(...)]):
    """Upload an archive (.zip, .tar, .tar.gz) containing nested folders (Category -> Item -> Images)."""
    filename_lower = file.filename.lower()
    if not any(filename_lower.endswith(ext) for ext in SUPPORTED_ARCHIVES):
        raise HTTPException(status_code=400, detail=f"Unsupported format. Allowed: {', '.join(SUPPORTED_ARCHIVES)}")
    
    try:
        contents = await file.read()
        
        # Process archive in a threadpool to prevent blocking the async event loop
        manual_updates, labels_count = await run_in_threadpool(
            _process_archive, contents, file.filename, classifier
        )
        
        # Load the newly added files into classifier / db
        if manual_updates:
            classifier.load_references(manual_updates=manual_updates)
        
        total_images = sum(labels_count.values())
        logger.info(f"Bulk upload complete: {len(labels_count)} labels, {total_images} total images")
        
        return {
            "message": "Bulk upload successful", 
            "labels": labels_count, 
            "total_images": total_images,
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Bulk upload error: {e}")
        return JSONResponse(status_code=500, content={"error": "Bulk upload failed."})
