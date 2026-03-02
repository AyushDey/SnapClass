# SnapClass: Offline Few-Shot Image Classifier

**SnapClass** is a lightweight, offline-first image classification API built with FastAPI and PyTorch. It allows you to recognize objects by learning from just a single reference image (few-shot learning), enabling you to update image classes dynamically without retraining.

## Key Features

- **Zero/Few-Shot Learning**: Classify images based on a set of reference images. Add new classes instantly by just adding a reference photo.
- **Offline Operation**: Uses a pre-trained **ResNet18** model to generate embeddings locally. No external APIs or heavy GPUs required.
- **Dynamic References**: Upload new reference images via the API to expand the classifier's knowledge base on the fly.
- **Unknown Detection**: Automatically categorizes images as "Unknown" if they don't sufficiently match any existing reference class.

### How It Works

1. **Embedding Generation**: The system uses `ResNet18` (pre-trained on ImageNet) to convert images into dense vector embeddings.
2. **Persistent Storage (PostgreSQL + pgvector)**: Embeddings and hash metadata are stored in a PostgreSQL database using the `pgvector` extension to allow fast similarity matching and persistence. This means your training data survives restarts.
3. **Similarity Matching**: When an image is submitted for classification, its embedding is compared against the stored embeddings of reference images using Cosine Similarity.
4. **Classification**: The class with the best similarity matches is returned.

## Prerequisites

- Python 3.13+
- Dependencies as listed in `pyproject.toml` (managed by `uv` or `pip`).

## Installation

### Method 1: Docker (Recommended for Production)

Run the API using the included Dockerfile.

```bash
# Build the image
docker build -t snapclass .

# Run the container (mounting volumes for persistence)
docker run -p 8000:8000 \
  -v $(pwd)/references:/app/references \
  --env-file .env \
  snapclass
```
The API is now running at `http://localhost:8000`. Reference images and the vector database are persisted on the host.

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
   pip install "fastapi" "uvicorn" "torch" "torchvision" "pillow" "requests" "numpy" "python-multipart" "psycopg[binary]" "pgvector" "sqlalchemy"
   ```

3. **Configure Database**: Create a `.env` file in the root directory based on `.env.example` with your PostgreSQL connection details.

## Running the Application

Start the FastAPI server:

```bash
# Using uv (Recommended)
uv run uvicorn main:app --reload

# Or using uvicorn directly (if installed in system python)
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
  - `label` (string): The name of the class (e.g., "binoculars", "rope"). **Note:** Spaces in labels will be automatically replaced with underscores (e.g., "blue sky" -> "blue_sky").
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

### 4. Bulk Upload References
**POST /bulk_upload**
- Upload an archive file (.zip, .tar, .tar.gz, .tar.bz2, .7z) containing folders of images. Each folder name becomes a label.
- **Note:** Spaces in folder names will be automatically replaced with underscores (e.g., "golden retriever" -> "golden_retriever").
- **Form Fields**:
  - `file` (file): An archive with the structure below.

Expected ZIP structure:
```
references.zip
├── cat/
│   ├── cat1.jpg
│   └── cat2.png
└── dog/
    └── dog1.jpg
```

Example Response:
```json
{
    "message": "Bulk upload successful",
    "labels": {
        "cat": 2,
        "dog": 1
    },
    "total_images": 3
}
```

### 5. Refresh References
**POST /refresh**
- Forces the server to reload all reference images from the `references/` directory. Useful if you manually added files to the folder.

## Auto-Refresh

The API includes a background task that automatically scans the `references/` directory every 24 hours for new folders and images. Any new references added directly to the folder will be loaded automatically without needing to call `/refresh`.

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
