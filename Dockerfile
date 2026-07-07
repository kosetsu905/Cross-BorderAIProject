# syntax=docker/dockerfile:1.7
FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    HF_HOME=/models/huggingface \
    HF_HUB_CACHE=/models/huggingface/hub \
    SENTENCE_TRANSFORMERS_HOME=/models/sentence-transformers \
    NLTK_DATA=/models/nltk \
    GUARDRAILS_INSTALLER=pip

WORKDIR /app

RUN apt-get update \
    && apt-get install -y --no-install-recommends gcc git \
    && rm -rf /var/lib/apt/lists/*

RUN python -m pip install --no-cache-dir --upgrade pip setuptools wheel

COPY requirements.txt .
RUN python -m pip install --no-cache-dir -r requirements.txt
RUN python -m pip install --no-cache-dir --upgrade "guardrails-ai>=0.10,<0.11"

RUN python -m pip install --no-cache-dir --index-url https://download.pytorch.org/whl/cpu torch \
    && python -m pip install --no-cache-dir \
        detoxify \
        nltk \
        presidio-analyzer \
        presidio-anonymizer \
        sentence-transformers \
        spacy \
        transformers

COPY scripts/install_guardrails_hub_validators.py scripts/install_guardrails_hub_validators.py
COPY vendor/guardrails-hub-wheels vendor/guardrails-hub-wheels
RUN --mount=type=secret,id=guardrails_token,required=true \
    GUARDRAILS_TOKEN="$(cat /run/secrets/guardrails_token)" \
    python scripts/install_guardrails_hub_validators.py

COPY . .

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
