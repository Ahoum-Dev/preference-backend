# syntax=docker/dockerfile:1
FROM python:3.11-slim AS builder
WORKDIR /app
COPY pyproject.toml poetry.lock* /app/
RUN pip install poetry && poetry config virtualenvs.create false && poetry install --no-root --without dev && pip install transformers torch

FROM python:3.11-slim
WORKDIR /app
COPY --from=builder /usr/local/lib/python3.11/site-packages /usr/local/lib/python3.11/site-packages
COPY . /app
# Try to pre-download embedding model, but ignore failures (e.g. private model)
RUN python - <<EOF
import os
from transformers import AutoModel, AutoTokenizer
model_name = os.getenv("EMBEDDING_MODEL_NAME", "bge-m3")
try:
    AutoModel.from_pretrained(model_name)
    AutoTokenizer.from_pretrained(model_name)
except Exception as e:
    print(f"Warning: embedding model cache warmup failed: {e}")
EOF
EXPOSE 8001

CMD ["python", "-m", "uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8001"] 