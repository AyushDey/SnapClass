# SnapClass

SnapClass is an offline-first few-shot image classifier. The backend uses FastAPI, PyTorch, PostgreSQL, and `pgvector` to store image embeddings and classify uploaded images by similarity against reference photos. The repo also includes a small React/Vite frontend for capturing or uploading images and sending them to the API.

## Repo Layout

```text
.
├── backend/                  # FastAPI API, classifier logic, DB integration, tests
│   ├── main.py               # FastAPI entry point
│   ├── classifier.py         # Embedding generation and similarity search
│   ├── db.py                 # SQLAlchemy engine/session setup
│   ├── db_actions.py         # Database CRUD helpers
│   ├── models.py             # ORM models
│   ├── schema_migrations.py  # DB initialization and legacy migration helpers
│   ├── tests/tests/          # Pytest suite
│   └── .env.template         # Backend environment template
├── frontend/                 # React/Vite UI
│   └── src/
├── examples/                 # Sample images for manual testing
├── postman_collection.json   # Postman collection for API testing
├── POSTMAN_README.md         # Postman usage guide
└── Dockerfile                # Backend container image
```

## Features

- Add new classes from reference images without retraining a model.
- Classify uploaded images with similarity search against stored embeddings.
- Store embeddings persistently in PostgreSQL with `pgvector`.
- Bulk import references from archives using `Category/Label/image` folder layouts.
- Use the included frontend for webcam capture and manual uploads.

## Prerequisites

- Python 3.13+
- Node.js 20+ and npm
- PostgreSQL 14+ with the `pgvector` extension available

## Backend Setup

1. Create a backend env file from the template:

   ```bash
   cp backend/.env.template backend/.env
   ```

2. Fill in the required database settings in `backend/.env`:

   ```dotenv
   DB_USER=your_user
   DB_PASSWORD=your_password
   DB_NAME=snapclass
   DB_PORT=5432
   DB_HOST=localhost
   ```

3. Install backend dependencies:

   ```bash
   cd backend
   uv sync
   ```

4. Start the API:

   ```bash
   uv run uvicorn main:app --reload
   ```

The API will be available at `http://127.0.0.1:8000`.

### Remote PostgreSQL with SSL

If `DB_HOST` is not `localhost` or `127.0.0.1`, the backend enables SSL and expects certificate files inside `backend/certs/`:

```text
backend/certs/
├── server-ca.pem
├── client-cert.pem
└── client-key.pk8
```

## Frontend Setup

The frontend is a separate Vite app in `frontend/`.

```bash
cd frontend
npm install
npm run dev
```

By default the UI talks to `http://localhost:8000`, so the backend should be running on that address while you use the frontend.

## Docker

The included `Dockerfile` builds the backend service only.

### Build

```bash
docker build -t snapclass .
```

### Run

```bash
docker run --rm -p 8000:8000 \
  -v "$(pwd)/backend/references:/app/backend/references" \
  --env-file backend/.env \
  snapclass
```

If you use SSL certificates for a remote database, also mount them into the backend container:

```bash
docker run --rm -p 8000:8000 \
  -v "$(pwd)/backend/references:/app/backend/references" \
  -v "$(pwd)/backend/certs:/app/backend/certs:ro" \
  --env-file backend/.env \
  snapclass
```

## API Endpoints

- `GET /` returns a basic health message.
- `POST /classify` classifies an uploaded image.
- `POST /add_reference` stores a new reference image under a label and optional category.
- `POST /bulk_upload` imports many references from an archive.
- `POST /refresh` reloads references from disk into memory.

### Reference Layout

Reference images are stored under `backend/references/` using this structure:

```text
backend/references/
└── Category/
    └── Label/
        ├── image1.jpg
        └── image2.png
```

For bulk uploads, SnapClass also accepts a two-level archive layout (`Label/image`) and uses the archive filename as the category.

## Testing

Backend tests live under `backend/tests/tests/`.

```bash
cd backend
uv sync --all-extras --dev
uv run pytest --cov=. --cov-report=term-missing --cov-fail-under=100
```

The test suite uses mocks for the model and database interactions, so you do not need a running PostgreSQL instance to execute it.

## Postman and Examples

- Import `postman_collection.json` into Postman for manual API testing.
- See `POSTMAN_README.md` for a step-by-step guide.
- Sample images for manual testing live in `examples/`.
