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
- PostgreSQL 14+ with the [`pgvector` extension](https://github.com/pgvector/pgvector) installed
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

3. **Configure Database**: Create a `.env` file in the root directory based on `.env.template` with your PostgreSQL connection details:
   ```
   DB_HOST=localhost
   DB_PORT=5432
   DB_USER=your_user
   DB_PASSWORD=your_password
   DB_NAME=snapclass
   ```
   > **SSL (Remote Hosts):** If `DB_HOST` is anything other than `localhost` / `127.0.0.1`, the engine automatically enables SSL (`sslmode=verify-ca`). Place your certificate files in a `certs/` directory in the project root:
   > ```
   > certs/
   > ├── server-ca.pem
   > ├── client-cert.pem
   > └── client-key.pk8
   > ```

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
  - `label` (string, required): The name of the class (e.g., "binoculars", "rope"). Spaces are automatically replaced with underscores (e.g., "blue sky" → "blue_sky").
  - `category` (string, optional): A grouping category for the label (e.g., "Tools", "Furniture"). Defaults to `"Uncategorized"` if not provided.
  - `file` (file): The reference image. Allowed formats: `.jpg`, `.jpeg`, `.png`, `.webp`, `.bmp`.

Example (Python):
```python
import requests
requests.post("http://127.0.0.1:8000/add_reference", 
              data={"label": "binoculars", "category": "Optics"}, 
              files={"file": open("binoculars_ref.jpg", "rb")})
```

### 3. Classify an Image
**POST /classify**
- **Form Fields**:
  - `file` (file): The image to classify. Allowed formats: `.jpg`, `.jpeg`, `.png`, `.webp`, `.bmp`.
- **Returns**: JSON object with the predicted class, category name, confidence score, and top matches within the same category.

Example Response (known class):
```json
{
    "class": "binoculars",
    "category_name": "Optics",
    "confidence": 0.92,
    "matches": [
        {"class": "telescope", "score": 0.45}
    ]
}
```

Example Response (no match):
```json
{
    "class": "Unknown",
    "confidence": 0.0,
    "message": "No references available"
}
```

### 4. Bulk Upload References
**POST /bulk_upload**
- Upload an archive file (`.zip`, `.tar`, `.tar.gz`, `.tar.bz2`, `.tgz`) containing a nested folder structure of images.
- Spaces in folder names are automatically replaced with underscores.
- **Form Fields**:
  - `file` (file): An archive with the `Category/Label/` structure below.

Expected archive structure (Category → Label → Images):
```
references.zip
└── Furniture/
    ├── Chair/
    │   ├── chair1.jpg
    │   └── chair2.png
    └── Table/
        └── table1.jpg
```

> **Note:** If the archive uses only one level of nesting (just `Label/images`), the archive's filename is used as the category name. Images placed directly at the root of the archive are skipped.

Example Response:
```json
{
    "message": "Bulk upload successful",
    "labels": {
        "Chair": 2,
        "Table": 1
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

### Postman
This repository includes a Postman Collection for easy manual testing of all endpoints.

1. Install [Postman](https://www.postman.com/downloads/).
2. Import `postman_collection.json`.
3. Follow the instructions in `POSTMAN_README.md` to run the tests against your local server.

### Unit Tests & Coverage

The project uses `pytest` with 100% code coverage enforced via CI.

> **Note on Test Isolation:** Tests use a fully mocked database (in-memory SQLite) and mocked ResNet model — **no real PostgreSQL or GPU is needed** to run the test suite.

```bash
# Install dev dependencies
uv sync --all-extras --dev

# Run tests with coverage report
uv run pytest --cov=. --cov-report=term-missing

# Run the CI check (enforces 100% coverage)
uv run pytest --cov=. --cov-fail-under=100
```

## CI/CD

The project uses **GitHub Actions** (`.github/workflows/test.yml`) to automatically run the full test suite with coverage checks on every push or pull request to `main` and `feature` branches.

## Directory Structure

```
├── main.py              # FastAPI application entry point and route handlers
├── classifier.py        # ImageClassifier logic (PyTorch + ResNet18)
├── db.py                # SQLAlchemy engine and session factory
├── db_actions.py        # Database CRUD operations (DBActions class)
├── models.py            # SQLAlchemy ORM models (BookletItem, BookletCategory)
├── utils.py             # Logging setup utilities
├── tests/               # Pytest test suite (88 tests, 100% coverage)
├── references/          # Reference images organized by Category/Label/
├── .coveragerc          # Coverage configuration
├── pyproject.toml       # Project metadata and dependencies
├── uv.lock              # Locked dependency versions
└── postman_collection.json  # Postman API test collection
```
