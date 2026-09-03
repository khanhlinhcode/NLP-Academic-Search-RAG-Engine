#!/usr/bin/env sh
set -eu

docker compose config --quiet
docker compose build api ui
docker compose up -d ollama model-init api ui
docker compose exec -T api python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/health/live', timeout=5)"
docker compose exec -T api python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/search?q=retrieval&top_k=1', timeout=30)"
docker compose exec -T ui python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8501/_stcore/health', timeout=5)"
