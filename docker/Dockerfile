FROM python:3.11-slim

WORKDIR /app

# System deps for sentence-transformers + networkx + healthcheck
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential curl \
    && rm -rf /var/lib/apt/lists/*

# Install runtime deps first (better layer caching)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Install the package itself so `from memovex import ...` resolves cleanly
COPY pyproject.toml ./
COPY README.md ./
COPY memovex/ ./memovex/
RUN pip install --no-cache-dir -e .

# Bring in the rest (config, plugins, scripts, etc.)
COPY . .

ENV PYTHONUNBUFFERED=1
ENV PYTHONPATH=/app
# Default snapshot directory inside the container — mount a volume here to persist
ENV MEMOVEX_SNAPSHOT_DIR=/app/data/snapshots

EXPOSE 7914

CMD ["uvicorn", "memovex.api:app", "--host", "0.0.0.0", "--port", "7914", "--workers", "1"]
