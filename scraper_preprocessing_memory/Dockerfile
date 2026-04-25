FROM python:3.12-slim

WORKDIR /app

# System dependencies for spaCy and newspaper3k
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    gcc \
    && rm -rf /var/lib/apt/lists/*

# Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Download spaCy English model
RUN python -m spacy download en_core_web_sm

# Copy source code
COPY . .

# Keep container alive for interactive use (docker-compose exec app ...)
CMD ["tail", "-f", "/dev/null"]
