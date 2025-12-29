from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.responses import JSONResponse
from classifier import ImageClassifier
from PIL import Image
import io
import os
import shutil

app = FastAPI(title="SnapClass: Offline Few-Shot Classifier")

# Initialize classifier
classifier = ImageClassifier()

@app.post("/classify")
async def classify_image(file: UploadFile = File(...)):
    try:
        contents = await file.read()
        image = Image.open(io.BytesIO(contents)).convert("RGB")
        result = classifier.classify(image)
        return result
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})

@app.post("/refresh")
async def refresh_references():
    """Reloads the reference images from disk."""
    classifier.load_references()
    return {"message": "References reloaded", "classes": list(classifier.reference_embeddings.keys())}

@app.post("/add_reference")
async def add_reference(label: str = Form(...), file: UploadFile = File(...)):
    """Uploads a new reference image for a specific label."""
    try:
        label_dir = os.path.join(classifier.references_dir, label)
        os.makedirs(label_dir, exist_ok=True)
        
        file_path = os.path.join(label_dir, file.filename)
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
            
        # Reload to include new reference (or we could just update memory incrementally for efficiency, 
        # but reloading is safer for consistency for now)
        classifier.load_references()
        
        return {"message": f"Added reference for '{label}'", "filename": file.filename}
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})

@app.get("/")
async def root():
    return {"message": "Image Classifier API is running. Use /classify to check images."}
