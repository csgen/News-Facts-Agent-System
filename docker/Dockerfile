FROM python:3.12-slim

WORKDIR /app

# System dependencies (build toolchain for spaCy / lxml + git for any VCS deps)
RUN apt-get update && apt-get install -y --no-install-recommends \
      build-essential \
      gcc \
      git \
      curl \
    && rm -rf /var/lib/apt/lists/*

# Install Python deps first (maximises Docker layer cache reuse)
COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r /app/requirements.txt \
    && python -m spacy download en_core_web_sm

# Copy the three subfolders + any top-level files
# (At runtime a volume mount typically overlays /app so edits don't need a rebuild.)
COPY scraper_preprocessing_memory /app/scraper_preprocessing_memory
COPY FakeNewsAgent                /app/FakeNewsAgent
COPY PredictionAgent              /app/PredictionAgent

# Make the three subfolders importable from any service without CLI gymnastics.
#   /app/scraper_preprocessing_memory       → lets `from src.*` resolve (scapper code)
#   /app/scraper_preprocessing_memory/src   → lets flat `from id_utils`, `from config`,
#                                             `from models.*` resolve (task3 code)
#   /app/FakeNewsAgent                      → lets `from fact_check_agent.*` resolve
#   /app/PredictionAgent                    → lets `from agents.*` resolve (adapters)
ENV PYTHONPATH=/app/scraper_preprocessing_memory:/app/scraper_preprocessing_memory/src:/app/FakeNewsAgent:/app/PredictionAgent
ENV PYTHONUNBUFFERED=1

# No CMD — each docker-compose service supplies its own `command:`.
