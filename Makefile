.PHONY: help setup install download preprocess index api ui test test-unit test-integration test-security \
	coverage lint format format-check typecheck check eval-retrieval eval-rag load-smoke package \
	docker-config docker-smoke docker-build docker-up docker-down clean

UV ?= uv

help: ## Show available commands
	@awk 'BEGIN {FS = ":.*## "} /^[a-zA-Z_-]+:.*## / {printf "%-20s %s\n", $$1, $$2}' $(MAKEFILE_LIST)

setup: ## Create a Python 3.11 environment and install all dependency groups
	$(UV) sync --all-extras --python 3.11

install: ## Install runtime plus UI dependencies
	$(UV) sync --extra ui --python 3.11

download: ## Ingest real arXiv metadata (use ARXIV_MAX_RECORDS to limit)
	$(UV) run python -m scripts.download_data --max-records $${ARXIV_MAX_RECORDS:-15000}

preprocess: ## Validate, deduplicate, and write a versioned corpus
	$(UV) run python -m scripts.preprocess_data

index: ## Build and atomically activate a versioned FAISS index
	$(UV) run python -m scripts.build_index

api: ## Start FastAPI
	$(UV) run uvicorn nlp_academic_search.api.main:app --host $${API_HOST:-0.0.0.0} --port $${API_PORT:-8000}

ui: ## Start Streamlit
	$(UV) run streamlit run scripts/streamlit_app.py --server.port 8501

test: test-unit ## Run unit tests

test-unit: ## Run deterministic unit tests
	$(UV) run pytest tests/unit

test-integration: ## Run local service/model integration tests
	$(UV) run pytest -m integration

test-security: ## Run automated security test suite
	$(UV) run pytest tests/security

coverage: ## Run tests with the configured coverage gate
	$(UV) run pytest -m "not integration" --cov=nlp_academic_search --cov-report=term-missing

lint: ## Run Ruff lint checks
	$(UV) run ruff check .

format: ## Format Python sources
	$(UV) run ruff format .

format-check: ## Verify Python formatting
	$(UV) run ruff format --check .

typecheck: ## Run Pyright
	$(UV) run pyright

package: ## Build wheel and source distribution
	$(UV) run python -m build

check: lint format-check typecheck test-unit test-security coverage package ## Run the local quality gate
	$(UV) run python -m pip check

eval-retrieval: ## Run leakage-free retrieval evaluation
	$(UV) run python -m scripts.run_evaluation

eval-rag: ## Run deterministic RAG evaluation
	$(UV) run python -m scripts.evaluate_rag

load-smoke: ## Run a small smoke load test against Search endpoint
	$(UV) run python -m scripts.run_load_test --requests 10 --concurrency 2

docker-config: ## Validate Docker Compose configuration
	docker compose config --quiet

docker-smoke: ## Build and smoke-test API/UI containers
	./scripts/docker_smoke.sh

docker-build: ## Build service images
	docker compose build

docker-up: ## Start the stack
	docker compose up -d

docker-down: ## Stop the stack
	docker compose down

clean: ## Remove local caches and build outputs (keeps corpus/index data)
	rm -rf .pytest_cache .ruff_cache .mypy_cache htmlcov dist build nlp_academic_search.egg-info src/nlp_academic_search.egg-info
	find src scripts tests -type d -name __pycache__ -prune -exec rm -rf {} +
