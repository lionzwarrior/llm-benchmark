FROM python:3.10-slim

WORKDIR /app

ENV PYTHONUNBUFFERED=1

# Install nano and curl
RUN apt-get update \
    && apt-get install -y --no-install-recommends nano curl dnsutils \
    && rm -rf /var/lib/apt/lists/*

RUN mkdir -p TruthfulQA/data/v1

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy benchmark scripts
COPY benchmark_ollama_csqa_parallel.py .
COPY benchmark_ollama_truthfulqa_parallel.py .
COPY dev_rand_split.jsonl .
COPY TruthfulQA/data/v1/TruthfulQA.csv TruthfulQA/data/v1/TruthfulQA.csv

CMD ["sleep", "infinity"]
