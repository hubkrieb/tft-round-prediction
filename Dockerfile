# syntax=docker/dockerfile:1
# Inference image for AWS Lambda via the Lambda Web Adapter. The adapter is a
# Lambda extension that proxies invocations to the uvicorn server below, so the
# FastAPI app runs unmodified. Outside Lambda the extension is never started,
# which makes this same image runnable locally / on App Runner / on Fargate.

FROM ghcr.io/astral-sh/uv:python3.12-bookworm-slim AS builder
ENV UV_COMPILE_BYTECODE=1 UV_LINK_MODE=copy UV_PYTHON_DOWNLOADS=0
WORKDIR /app
COPY pyproject.toml uv.lock ./
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --locked --only-group serve

FROM public.ecr.aws/docker/library/python:3.12-slim-bookworm

COPY --from=public.ecr.aws/awsguru/aws-lambda-adapter:0.9.1 \
    /lambda-adapter /opt/extensions/lambda-adapter

WORKDIR /app
COPY --from=builder /app/.venv .venv

# API + frontend + the training transforms the API reuses for featurization.
COPY src/ src/
# Static game data loaded at import time, resolved relative to the CWD.
COPY data/set16/static/ data/set16/static/
# Serving models only — the Lightning .ckpt checkpoints stay out of the image.
COPY models/vit/vit.onnx models/vit/
COPY models/cnn/cnn.onnx models/cnn/
COPY models/baseline/xgboost.json models/baseline/xgboost_features.json models/baseline/

# PYTHONDONTWRITEBYTECODE: Lambda's filesystem is read-only outside /tmp.
# AWS_LWA_READINESS_CHECK_PATH: the adapter polls this until the app is up.
ENV PATH="/app/.venv/bin:$PATH" \
    PYTHONDONTWRITEBYTECODE=1 \
    PORT=8000 \
    AWS_LWA_READINESS_CHECK_PATH=/api/models

EXPOSE 8000
CMD ["uvicorn", "src.api.app:app", "--host", "0.0.0.0", "--port", "8000"]
