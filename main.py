from fastapi import FastAPI, UploadFile, File, Form, HTTPException, Request
from fastapi.responses import JSONResponse
from classifier import ImageClassifier
from PIL import Image, UnidentifiedImageError
import io
import os
import shutil
from utils import setup_logger, intercept_uvicorn_logs
from fastapi.concurrency import run_in_threadpool

from contextlib import asynccontextmanager

# Setup Logger
logger = setup_logger("snapclass.api")

# Initialize classifier global variable
classifier = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup logic
    intercept_uvicorn_logs()
    logger.info("Application startup: Logging unified.")
    
    global classifier
    try:
        classifier = ImageClassifier()
        logger.info("Classifier initialized successfully.")
    except Exception as e:
        logger.critical(f"Failed to initialize classifier: {e}")
        # We allow app startup but endpoints might fail, or we could exit.
        # Ideally, if core dependency fails, we should probably crash or health check should fail.
    
    yield
    
    # Shutdown logic (if any)
    logger.info("Application shutdown.")

app = FastAPI(title="SnapClass: Offline Few-Shot Classifier", lifespan=lifespan)

ALLOWED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}

def validate_image_file(filename: str):
    ext = os.path.splitext(filename)[1].lower()
    if ext not in ALLOWED_EXTENSIONS:
        return False
    return True

@app.middleware("http")
async def log_requests(request: Request, call_next):
    logger.info(f"Incoming request: {request.method} {request.url}")
    response = await call_next(request)
    logger.info(f"Response status: {response.status_code}")
    return response

@app.post("/classify")
async def classify_image(file: UploadFile = File(...)):
    if not validate_image_file(file.filename):
        logger.warning(f"Rejected classification request for file: {file.filename}")
        raise HTTPException(status_code=400, detail="Invalid file type. Allowed: jpg, jpeg, png, webp, bmp")

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
        classifier.load_references()
        classes = list(classifier.reference_embeddings.keys())
        logger.info(f"References refreshed. Classes: {classes}")
        return {"message": "References reloaded", "classes": classes}
    except Exception as e:
        logger.error(f"Error refreshing references: {e}")
        return JSONResponse(status_code=500, content={"error": "Failed to refresh references."})

@app.post("/add_reference")
async def add_reference(label: str = Form(...), file: UploadFile = File(...)):
    """Uploads a new reference image for a specific label."""
    if not validate_image_file(file.filename):
        logger.warning(f"Rejected add_reference request for file: {file.filename}")
        raise HTTPException(status_code=400, detail="Invalid file type. Allowed: jpg, jpeg, png, webp, bmp")

    try:
        # Sanitize label to prevent directory traversal or weird filenames
        safe_label = "".join([c for c in label if c.isalnum() or c in ('_', '-')])
        if not safe_label:
             raise HTTPException(status_code=400, detail="Invalid label format")

        label_dir = os.path.join(classifier.references_dir, safe_label)
        os.makedirs(label_dir, exist_ok=True)
        
        # Sanitize filename to prevent path traversal
        safe_filename = os.path.basename(file.filename)
        file_path = os.path.join(label_dir, safe_filename) # Potentially overwrite if exists
        
        def save_upload_file(src, dest):
            with open(dest, "wb") as buffer:
                shutil.copyfileobj(src, buffer)
        
        await run_in_threadpool(save_upload_file, file.file, file_path)
            
        logger.info(f"Added new reference: {file.filename} to class {safe_label}")
        
        # Reload to ensure consistency
        classifier.load_references()
        
        return {"message": f"Added reference for '{safe_label}'", "filename": file.filename}
    except Exception as e:
        logger.error(f"Error adding reference: {e}")
        return JSONResponse(status_code=500, content={"error": f"Failed to add reference: {str(e)}"})

@app.get("/")
async def root():
    return {"message": "Image Classifier API is running. Use /classify to check images.", "status": "active"}
