# SnapClass: Offline Few-Shot Image Classifier

**SnapClass** is a lightweight, offline-first image classification API built with FastAPI and PyTorch. It allows you to recognize objects by learning from just a single reference image (few-shot learning), enabling you to update image classes dynamically without retraining.

## Key Features

- **Zero/Few-Shot Learning**: Classify images based on a set of reference images. Add new classes instantly by just adding a reference photo.
- **Offline Operation**: Uses a pre-trained **ResNet18** model to generate embeddings locally. No external APIs or heavy GPUs required.
- **Dynamic References**: Upload new reference images via the API to expand the classifier's knowledge base on the fly.
- **Unknown Detection**: Automatically categorizes images as "Unknown" if they don't sufficiently match any existing reference class.

## How It Works

1. **Embedding Generation**: The system uses `ResNet18` (pre-trained on ImageNet) to convert images into dense vector embeddings.
2. **Similarity Matching**: When an image is submitted for classification, its embedding is compared against the stored embeddings of reference images using Cosine Similarity.
3. **Classification**: The class with the best similarity matches is returned.

## Prerequisites

- Python 3.13+
- Dependencies as listed in `pyproject.toml` (managed by `uv` or `pip`).

## Installation

### Method 1: Docker (Recommended for Production)

Run the API using the included Dockerfile.

```bash
# Build the image
docker build -t snapclass .

# Run the container
docker run -p 8000:8000 snapclass
```
The API is now running at `http://localhost:8000`.

### Method 2: Local Setup

1. **Clone the repository**:
   ```bash
   git clone <repository-url>
   cd snapclass
   ```

2. **Install Dependencies**:
   
   Using `uv` (Recommended):
   ```bash
   uv sync
   ```

   Using standard `pip`:
   ```bash
   pip install fastapi uvicorn torch torchvision pillow requests numpy python-multipart
   ```

## Running the Application

Start the FastAPI server:

```bash
# Using uv
uv run fastapi dev main.py

# Or using uvicorn directly
uvicorn main:app --reload
```

The API will be accessible at `http://127.0.0.1:8000`.

## API Usage

### 1. Check Status
**GET /**
- Returns a welcome message confirming the API is running.

### 2. Add a Reference Image
**POST /add_reference**
- **Form Fields**:
  - `label` (string): The name of the class (e.g., "binoculars", "rope").
  - `file` (file): The reference image.

Example (Python):
```python
import requests
requests.post("http://127.0.0.1:8000/add_reference", 
              data={"label": "binoculars"}, 
              files={"file": open("binoculars_ref.jpg", "rb")})
```

### 3. Classify an Image
**POST /classify**
- **Form Fields**:
  - `file` (file): The image to classify.
- **Returns**: JSON object with the predicted class, confidence score, and top matches.

Example Response:
```json
{
    "class": "binoculars",
    "confidence": 0.92,
    "matches": [
        {"class": "rope", "score": 0.45},
        {"class": "flower_vase", "score": 0.12}
    ]
}
```

### 4. Refresh References
**POST /refresh**
- Forces the server to reload all reference images from the `references/` directory. Useful if you manually added files to the folder.

## Testing

This repository includes a Postman Collection for easy testing of all endpoints.

1. Install [Postman](https://www.postman.com/downloads/).
2. Import `postman_collection.json`.
3. Follow the instructions in `POSTMAN_README.md` to run the tests against your local server.

## Unit Testing

For developers, the project uses `pytest` for unit and integration testing.

```bash
# Install dev dependencies
uv sync --dev

# Run tests
uv run pytest
```

## Directory Structure

- `main.py`: The FastAPI application entry point.
- `classifier.py`: Contains the `ImageClassifier` logic using PyTorch and ResNet18.
- `references/`: Directory where reference images are stored (organized by class label).
- `postman_collection.json`: API test suite.
- `pyproject.toml` / `uv.lock`: Project dependency management.
