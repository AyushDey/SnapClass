FROM python:3.13-slim-bookworm

# The installer requires curl (and certificates) to download the release archive
RUN apt-get update && apt-get install -y --no-install-recommends curl ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# Download the latest installer
ADD https://astral.sh/uv/install.sh /uv-installer.sh

# Run the installer then remove it
RUN sh /uv-installer.sh && rm /uv-installer.sh

# Ensure the installed binary is on the `PATH`
ENV PATH="/root/.local/bin/:$PATH"
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

WORKDIR /app/backend

# Copy dependency definitions
COPY backend/pyproject.toml backend/uv.lock ./

# Install dependencies
# --frozen: Require uv.lock to act as the source of truth
# --no-dev: Exclude development dependencies (like pytest)
RUN uv sync --frozen --no-dev

# Place the virtualenv in the path
ENV PATH="/app/backend/.venv/bin:$PATH"

# Copy source code (explicit listing to avoid accidental inclusion of sensitive files)
COPY backend/main.py backend/classifier.py backend/db.py backend/db_actions.py backend/models.py backend/schema_migrations.py backend/utils.py ./

# Ensure the runtime reference directory exists inside the backend working tree
RUN mkdir -p /app/backend/references

EXPOSE 8000

# Expose volumes for persistence
VOLUME ["/app/backend/references"]

# Run application
# Host 0.0.0.0 is crucial for Docker networking
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
