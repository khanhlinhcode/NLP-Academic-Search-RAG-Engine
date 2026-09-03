# syntax=docker/dockerfile:1.7
FROM python:3.11.16-slim-bookworm AS builder

WORKDIR /build
COPY pyproject.toml README.md LICENSE ./
COPY src ./src
RUN python -m pip install --no-cache-dir build==1.6.0 \
    && python -m build --wheel

FROM python:3.11.16-slim-bookworm AS runtime

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

RUN groupadd --system app && useradd --system --gid app --create-home app
COPY --from=builder /build/dist /tmp/dist
RUN set -e; WHEEL=$(ls /tmp/dist/*.whl) && python -m pip install "${WHEEL}[ui]" && rm -rf /tmp/dist
COPY scripts ./scripts
RUN mkdir -p /app/data && chown -R app:app /app

USER app
EXPOSE 8000 8501
HEALTHCHECK --interval=30s --timeout=5s --start-period=45s --retries=3 \
  CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/health/live', timeout=3)"

CMD exec uvicorn nlp_academic_search.api.main:app --host 0.0.0.0 --port ${PORT:-8000}
