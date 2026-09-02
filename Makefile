.PHONY: setup install download preprocess index api test eval clean help

help: ## Show this help message
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-20s\033[0m %s\n", $$1, $$2}'

setup: ## Create virtual environment and install dependencies
	python3 -m venv venv
	. venv/bin/activate && pip install --upgrade pip && pip install -r requirements.txt
	@echo "\n✅ Setup complete! Activate with: source venv/bin/activate"

install: ## Install dependencies (assumes venv is active)
	pip install -r requirements.txt

download: ## Download and prepare the dataset
	python -m scripts.download_data

preprocess: ## Preprocess the dataset
	python -m scripts.download_data

index: ## Build FAISS index and embeddings
	python -m scripts.build_index

api: ## Start the FastAPI server
	uvicorn src.api.main:app --host 0.0.0.0 --port 8000 --reload

test: ## Run all tests
	pytest tests/ -v

eval: ## Run evaluation benchmarks
	python -m scripts.run_evaluation

clean: ## Remove generated files
	rm -rf data/processed/* data/embeddings/*
	rm -rf __pycache__ .pytest_cache
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true

docker-build: ## Build Docker image
	docker-compose build

docker-up: ## Start services with Docker
	docker-compose up -d

docker-down: ## Stop Docker services
	docker-compose down
