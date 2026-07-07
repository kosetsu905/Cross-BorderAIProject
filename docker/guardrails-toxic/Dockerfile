# syntax=docker/dockerfile:1.7
FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    HF_HOME=/models/huggingface \
    HF_HUB_CACHE=/models/huggingface/hub \
    NLTK_DATA=/models/nltk

WORKDIR /app

RUN apt-get update \
    && apt-get install -y --no-install-recommends gcc git \
    && rm -rf /var/lib/apt/lists/*

RUN python -m pip install --no-cache-dir --upgrade pip setuptools wheel
RUN python -m pip install --no-cache-dir --index-url https://download.pytorch.org/whl/cpu torch

COPY requirements.txt .
RUN python -m pip install --no-cache-dir -r requirements.txt

RUN --mount=type=secret,id=guardrails_token,required=true \
    mkdir -p "$HF_HOME" "$HF_HUB_CACHE" "$NLTK_DATA" \
    && guardrails configure \
        --token "$(cat /run/secrets/guardrails_token)" \
        --disable-metrics \
        --disable-remote-inferencing \
    && (guardrails hub install hub://guardrails/toxic_language \
        || python -m pip install --no-cache-dir git+https://github.com/guardrails-ai/toxic_language.git@v0.0.2) \
    && python -c "import nltk; nltk.download('punkt', download_dir='/models/nltk'); nltk.download('punkt_tab', download_dir='/models/nltk')" \
    && rm -f /root/.guardrailsrc \
    && rm -rf /root/.guardrails \
    && python -c "import importlib; hub = importlib.import_module('guardrails.hub'); klass = getattr(hub, 'ToxicLanguage', None); klass = klass or getattr(importlib.import_module('validator'), 'ToxicLanguage'); print(klass.__name__)"

COPY app.py .

EXPOSE 8011

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8011"]
