# Makefile for Trader Risk Management System

.PHONY: help install dev test lint format clean build deploy

# Default target
help:
	@echo "Available commands:"
	@echo "  install     - Install dependencies and setup environment"
	@echo "  dev         - Start development environment"
	@echo "  test        - Run test suite"
	@echo "  lint        - Run linting checks"
	@echo "  format      - Format code with black"
	@echo "  clean       - Clean up generated files"
	@echo "  build       - Build Docker images"
	@echo "  deploy      - Deploy to production"
	@echo "  api         - Start API server"
	@echo "  pipeline    - Run the complete ML pipeline"
	@echo "  dashboard   - Start trader dashboard"

# Installation and setup
install:
	@echo "Installing dependencies..."
	pip install -r requirements.txt
	pip install -e .
	@echo "Setting up pre-commit hooks..."
	pre-commit install
	@echo "Creating output directories..."
	mkdir -p outputs/{models,reports,logs,signals}
	@echo "Installation complete!"

# Development environment
dev:
	@echo "Starting development environment..."
	docker-compose --profile dev up --build

dev-api:
	@echo "Starting API in development mode..."
	python -m uvicorn src.api:app --host 0.0.0.0 --port 8000 --reload

# Testing
test:
	@echo "Running test suite..."
	python -m pytest tests/ -v --cov=src --cov-report=html --cov-report=term

test-docker:
	@echo "Running tests in Docker..."
	docker-compose --profile test up --build --abort-on-container-exit

# Code quality
lint:
	@echo "Running linting checks..."
	flake8 src/ tests/
	mypy src/
	black --check src/ tests/

format:
	@echo "Formatting code..."
	black src/ tests/
	isort src/ tests/

# Cleanup
clean:
	@echo "Cleaning up..."
	find . -type f -name "*.pyc" -delete
	find . -type d -name "__pycache__" -delete
	find . -type d -name "*.egg-info" -exec rm -rf {} +
	rm -rf .coverage htmlcov/ .pytest_cache/
	docker system prune -f

# Docker operations
build:
	@echo "Building Docker images..."
	docker-compose build

build-prod:
	@echo "Building production Docker image..."
	docker build --target production -t risk-management:latest .

# Deployment
deploy-staging:
	@echo "Deploying to staging..."
	docker-compose -f docker-compose.yml up -d

deploy-prod:
	@echo "Deploying to production..."
	docker-compose up -d

deploy-monitoring:
	@echo "Starting monitoring stack..."
	docker-compose --profile logging up -d prometheus grafana

# Application commands
api:
	@echo "Starting API server..."
	python -m uvicorn src.api:app --host 0.0.0.0 --port 8000

pipeline:
	@echo "Running ML pipeline..."
	python src/main_pipeline.py

dashboard:
	@echo "Starting trader dashboard..."
	streamlit run scripts/trader_dashboard.py

# Database operations
db-backup:
	@echo "Backing up database..."
	mkdir -p data/backups
	cp data/risk_tool.db data/backups/risk_tool_$(shell date +%Y%m%d_%H%M%S).db

db-restore:
	@echo "Restoring database from backup..."
	@read -p "Enter backup filename: " backup; \
	cp data/backups/$$backup data/risk_tool.db

# Monitoring and logs
logs:
	@echo "Viewing application logs..."
	docker-compose logs -f risk-api

logs-api:
	@echo "Viewing API logs..."
	tail -f outputs/logs/risk_management.log

monitoring:
	@echo "Opening monitoring dashboard..."
	@echo "Grafana: http://localhost:3000 (admin/admin)"
	@echo "Prometheus: http://localhost:9090"

# Data operations
download-data:
	@echo "Downloading latest trading data..."
	python scripts/update_database.py

validate-data:
	@echo "Validating data quality..."
	python -c "from src.data.data_validator import DataValidator; dv = DataValidator(); dv.load_and_validate_data(); dv.validate_data_quality()"

# Model operations
train-models:
	@echo "Training models..."
	python -c "from src.models.trader_models import TraderModelTraining; tm = TraderModelTraining(); tm.train_all_models()"

generate-signals:
	@echo "Generating current signals..."
	python -c "from src.models.signal_generator import DeploymentReadySignals; sg = DeploymentReadySignals(); sg.generate_real_time_signals()"

# Configuration
config-check:
	@echo "Checking configuration..."
	python -c "from src.config import get_config; config = get_config(); print(f'Environment: {config.app.environment}'); print(f'Log level: {config.logging.level}')"

config-create-local:
	@echo "Creating local configuration override..."
	@echo "# Local configuration overrides" > config/config.local.yaml
	@echo "app:" >> config/config.local.yaml
	@echo "  environment: development" >> config/config.local.yaml
	@echo "logging:" >> config/config.local.yaml
	@echo "  level: DEBUG" >> config/config.local.yaml

# Environment setup
setup-conda:
	@echo "Setting up conda environment..."
	conda env create -f environment.yml
	@echo "Activate with: conda activate risk-tool"

setup-venv:
	@echo "Setting up virtual environment..."
	python -m venv venv
	@echo "Activate with: source venv/bin/activate (Linux/Mac) or venv\\Scripts\\activate (Windows)"

# Security
security-scan:
	@echo "Running security scan..."
	pip-audit
	bandit -r src/

# Documentation
docs-serve:
	@echo "Serving documentation..."
	@echo "API docs available at: http://localhost:8000/docs"
	@echo "README: http://localhost:8000/redoc"

# Health checks
health:
	@echo "Checking system health..."
	curl -f http://localhost:8000/health || echo "API not responding"

health-detailed:
	@echo "Detailed health check..."
	python -c "
import requests
import json
try:
    response = requests.get('http://localhost:8000/health')
    print('API Status:', response.status_code)
    print('Response:', json.dumps(response.json(), indent=2))
except Exception as e:
    print('Error:', e)
"

# Performance testing
perf-test:
	@echo "Running performance tests..."
	@echo "Install locust first: pip install locust"
	locust -f tests/performance/locustfile.py --host=http://localhost:8000

# Git hooks
pre-commit:
	@echo "Running pre-commit checks..."
	pre-commit run --all-files
