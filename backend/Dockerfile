FROM python:3.12-slim

WORKDIR /app

# System deps for pypdf and other packages
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application source
COPY app/ ./app/
COPY scripts/ ./scripts/

# Copy HR policy documents + pre-built vector store (if they exist locally)
# Run `python -m scripts.index_documents` locally first to populate data/processed/
COPY data/ ./data/

# Startup script
COPY start.sh .
RUN chmod +x start.sh

EXPOSE 8000

CMD ["./start.sh"]
