.PHONY: help install install-dev install-docs test test-fast test-cov lint format type-check clean docker-build docker-up docker-down docker-logs jupyter streamlit generate-data run-notebooks pre-commit-install pre-commit-run security-check

help:  ## Show this help message
	@echo '━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━'
	@echo '  ScipyMasterPro - Development Commands'
	@echo '━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━'
	@echo ''
	@echo 'Usage: make [target]'
	@echo ''
	@echo 'Available targets:'
	@awk 'BEGIN {FS = ":.*?## "} /^[a-zA-Z_-]+:.*?## / {printf "  \033[36m%-25s\033[0m %s\n", $$1, $$2}' $(MAKEFILE_LIST)
	@echo ''

# ============================================================================
# Installation
# ============================================================================

install:  ## Install production dependencies
	pip install --upgrade pip
	pip install -r requirements.txt

install-dev:  ## Install development dependencies and setup pre-commit
	pip install --upgrade pip
	pip install -e ".[dev]"
	pre-commit install

install-docs:  ## Install documentation dependencies
	pip install -e ".[docs]"

install-all:  ## Install all dependencies (prod + dev + docs)
	pip install --upgrade pip
	pip install -e ".[dev,docs,notebook]"
	pre-commit install

# ============================================================================
# Testing
# ============================================================================

test:  ## Run all tests with coverage report
	pytest --cov=utils --cov=streamlit_app --cov=synthetic_data \
	       --cov-report=html --cov-report=term-missing --cov-report=xml \
	       -v

test-fast:  ## Run tests without coverage (faster)
	pytest -v

test-cov:  ## Run tests and open HTML coverage report
	pytest --cov=utils --cov=streamlit_app --cov=synthetic_data \
	       --cov-report=html --cov-report=term-missing
	@echo "Opening coverage report..."
	@open htmlcov/index.html || xdg-open htmlcov/index.html || echo "Please open htmlcov/index.html manually"

test-unit:  ## Run only unit tests
	pytest -v -m unit

test-integration:  ## Run only integration tests
	pytest -v -m integration

test-watch:  ## Run tests in watch mode
	pytest-watch

# ============================================================================
# Code Quality
# ============================================================================

lint:  ## Run all linting checks (pylint, black, isort, mypy)
	@echo "Running pylint..."
	pylint utils/ streamlit_app/ synthetic_data/ || true
	@echo "\nChecking black formatting..."
	black --check . || true
	@echo "\nChecking isort..."
	isort --check . || true
	@echo "\nRunning mypy type checking..."
	mypy utils/ streamlit_app/ synthetic_data/ || true

format:  ## Format code with black and isort
	@echo "Formatting with black..."
	black .
	@echo "Sorting imports with isort..."
	isort .
	@echo "✓ Code formatted successfully!"

type-check:  ## Run mypy type checking
	mypy utils/ streamlit_app/ synthetic_data/

pylint:  ## Run pylint only
	pylint utils/ streamlit_app/ synthetic_data/

black-check:  ## Check black formatting without making changes
	black --check .

isort-check:  ## Check isort without making changes
	isort --check .

# ============================================================================
# Pre-commit
# ============================================================================

pre-commit-install:  ## Install pre-commit hooks
	pre-commit install

pre-commit-run:  ## Run pre-commit on all files
	pre-commit run --all-files

pre-commit-update:  ## Update pre-commit hooks
	pre-commit autoupdate

# ============================================================================
# Security
# ============================================================================

security-check:  ## Run security checks (bandit + safety)
	@echo "Running bandit security scanner..."
	bandit -r utils/ streamlit_app/ synthetic_data/ -f screen || true
	@echo "\nChecking for vulnerable dependencies with safety..."
	safety check || true

# ============================================================================
# Cleaning
# ============================================================================

clean:  ## Clean up generated files
	@echo "Cleaning up Python cache files..."
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete 2>/dev/null || true
	find . -type f -name "*.pyo" -delete 2>/dev/null || true
	find . -type f -name "*.pyd" -delete 2>/dev/null || true
	find . -type d -name "*.egg-info" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name ".pytest_cache" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name ".mypy_cache" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name "htmlcov" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name ".ipynb_checkpoints" -exec rm -rf {} + 2>/dev/null || true
	rm -f .coverage coverage.xml 2>/dev/null || true
	@echo "✓ Clean complete!"

clean-exports:  ## Clean exported plots and tables
	rm -rf exports/plots/*/*.png exports/plots/*/*.pdf 2>/dev/null || true
	rm -rf exports/tables/*/*.csv 2>/dev/null || true
	@echo "✓ Exports cleaned!"

clean-all: clean clean-exports  ## Clean everything

# ============================================================================
# Docker
# ============================================================================

docker-build:  ## Build Docker image
	docker build -t scipymasterpro:latest .

docker-build-no-cache:  ## Build Docker image without cache
	docker build --no-cache -t scipymasterpro:latest .

docker-up:  ## Start all Docker containers (Jupyter + Streamlit)
	docker-compose up -d

docker-down:  ## Stop all Docker containers
	docker-compose down

docker-restart:  ## Restart all Docker containers
	docker-compose restart

docker-logs:  ## View Docker container logs
	docker-compose logs -f

docker-logs-jupyter:  ## View Jupyter container logs
	docker-compose logs -f jupyter

docker-logs-streamlit:  ## View Streamlit container logs
	docker-compose logs -f streamlit

docker-shell:  ## Open shell in running container
	docker-compose exec jupyter /bin/bash

docker-clean:  ## Remove all containers and images
	docker-compose down -v
	docker rmi scipymasterpro:latest || true

# ============================================================================
# Local Development
# ============================================================================

jupyter:  ## Start Jupyter Lab locally
	jupyter lab --allow-root --ip=0.0.0.0 --no-browser --port=8888

streamlit:  ## Start Streamlit app locally
	streamlit run streamlit_app/app.py --server.port=8501

generate-data:  ## Generate synthetic datasets
	python synthetic_data/generate_synthetic_data.py

run-notebooks:  ## Execute all notebooks (requires jupyter nbconvert)
	@echo "Executing all notebooks..."
	jupyter nbconvert --to notebook --execute notebooks/*.ipynb --output-dir=notebooks/
	@echo "✓ All notebooks executed!"

validate-notebooks:  ## Validate notebook execution without saving
	@for notebook in notebooks/*.ipynb; do \
		echo "Validating $$notebook..."; \
		jupyter nbconvert --to notebook --execute "$$notebook" --stdout > /dev/null || exit 1; \
	done
	@echo "✓ All notebooks validated successfully!"

# ============================================================================
# Package Management
# ============================================================================

build:  ## Build package distribution
	python -m build

build-wheel:  ## Build wheel only
	python -m build --wheel

build-sdist:  ## Build source distribution only
	python -m build --sdist

install-local:  ## Install package locally in editable mode
	pip install -e .

uninstall:  ## Uninstall package
	pip uninstall scipymasterpro -y

publish-test:  ## Publish to Test PyPI
	python -m twine upload --repository testpypi dist/*

publish:  ## Publish to PyPI (use with caution!)
	python -m twine upload dist/*

# ============================================================================
# Documentation
# ============================================================================

docs-build:  ## Build documentation
	cd docs && make html

docs-serve:  ## Serve documentation locally
	cd docs && python -m http.server 8000 --directory _build/html

docs-clean:  ## Clean documentation build
	cd docs && make clean

mkdocs-serve:  ## Serve MkDocs documentation
	mkdocs serve

mkdocs-build:  ## Build MkDocs documentation
	mkdocs build

mkdocs-deploy:  ## Deploy MkDocs to GitHub Pages
	mkdocs gh-deploy

# ============================================================================
# Development Utilities
# ============================================================================

update-deps:  ## Update all dependencies
	pip install --upgrade pip
	pip list --outdated
	@echo "\nTo update all packages, run: pip install --upgrade -r requirements.txt"

freeze-deps:  ## Freeze current dependencies
	pip freeze > requirements_frozen.txt
	@echo "✓ Dependencies frozen to requirements_frozen.txt"

check-deps:  ## Check for outdated dependencies
	pip list --outdated

count-lines:  ## Count lines of code
	@echo "Lines of code by category:"
	@echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
	@echo "Utilities:"
	@find utils -name "*.py" | xargs wc -l | tail -1
	@echo "Streamlit App:"
	@find streamlit_app -name "*.py" | xargs wc -l | tail -1
	@echo "Synthetic Data:"
	@find synthetic_data -name "*.py" | xargs wc -l | tail -1
	@echo "Tests:"
	@find tests -name "*.py" 2>/dev/null | xargs wc -l 2>/dev/null | tail -1 || echo "  0 total"
	@echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

tree:  ## Show project tree structure
	tree -I 'venv|__pycache__|*.pyc|.git|.pytest_cache|.mypy_cache|htmlcov|*.egg-info|exports|.ipynb_checkpoints' -L 3

# ============================================================================
# Quick Commands
# ============================================================================

dev:  ## Quick dev setup (install-dev + generate-data)
	make install-dev
	make generate-data

start: docker-up  ## Quick start with Docker

stop: docker-down  ## Quick stop Docker containers

check: lint test  ## Quick quality check (lint + test)

all: install-dev generate-data test lint  ## Do everything (install, generate, test, lint)

.DEFAULT_GOAL := help
