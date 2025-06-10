# Makefile for Trading Risk Management System

.PHONY: help setup download train predict test clean dashboard report

help:
	@echo "Trading Risk Management System - Available Commands:"
	@echo ""
	@echo "Setup & Installation:"
	@echo "  make setup          - Create environment and install dependencies"
	@echo "  make download       - Download all historical data"
	@echo ""
	@echo "Model Operations:"
	@echo "  make train          - Train risk prediction models"
	@echo "  make predict        - Generate daily predictions"
	@echo "  make backtest       - Run backtesting analysis"
	@echo ""
	@echo "Monitoring & Reports:"
	@echo "  make dashboard      - Generate monitoring dashboard"
	@echo "  make report         - Create performance report"
	@echo "  make monitor        - Check system health and alerts"
	@echo ""
	@echo "Development:"
	@echo "  make test           - Run unit tests"
	@echo "  make format         - Format code with black"
	@echo "  make lint           - Run code linting"
	@echo "  make clean          - Clean generated files"
	@echo ""
	@echo "Quick Start:"
	@echo "  make all            - Setup, download, and train"

# Setup environment
setup:
	conda env create -f environment.yml || conda env update -f environment.yml
	@echo "Environment created/updated. Run: conda activate risk-tool"
	@echo "Don't forget to copy .env.template to .env and add your credentials"

# Download historical data
download:
	python scripts/setup_database.py

# Train models
train:
	python scripts/train_models.py

# Daily predictions
predict:
	python scripts/daily_predict.py

# Run backtesting
backtest:
	@echo "Running backtesting analysis..."
	python -c "from scripts.run_backtest import main; main()"

# Generate dashboard
dashboard:
	@echo "Generating dashboard..."
	python -c "from scripts.generate_dashboard import main; main()"
	@echo "Dashboard saved to reports/"

# Create report
report:
	@echo "Creating performance report..."
	python scripts/daily_predict.py --report-only
	@echo "Report saved to reports/"

# Monitor system
monitor:
	@echo "Checking system health..."
	python -c "from scripts.system_monitor import main; main()"

# Run tests
test:
	pytest tests/ -v --cov=src --cov-report=html

# Format code
format:
	black src/ scripts/ tests/
	isort src/ scripts/ tests/

# Lint code
lint:
	ruff check src/ scripts/ tests/

# Clean generated files
clean:
	find . -type f -name "*.pyc" -delete
	find . -type d -name "__pycache__" -delete
	find . -type d -name ".pytest_cache" -exec rm -rf {} +
	rm -rf htmlcov/
	rm -rf .coverage
	@echo "Cleaned temporary files"

# Clean all (including data and models)
clean-all: clean
	rm -rf data/models/*
	rm -rf data/predictions/*
	rm -rf reports/*
	rm -rf logs/*
	@echo "Warning: Cleaned all generated data, models, and reports"

# Complete setup and training
all: setup download train
	@echo "System setup complete! Run 'make predict' for daily predictions"

# Development setup
dev-setup: setup
	pre-commit install
	@echo "Development environment ready"

# Check system status
status:
	@echo "=== System Status ==="
	@echo "Database:"
	@python -c "from src.database import Database; db = Database(); stats = db.get_database_stats(); print(f'  Records: {stats}')"
	@echo ""
	@echo "Models:"
	@ls -la data/models/*.pkl 2>/dev/null || echo "  No models found"
	@echo ""
	@echo "Recent Predictions:"
	@ls -la data/predictions/*.csv 2>/dev/null | head -5 || echo "  No predictions found"
	@echo ""
	@echo "Alerts:"
	@tail -5 logs/alerts/*.jsonl 2>/dev/null || echo "  No recent alerts"

# Run daily workflow
daily: download predict dashboard
	@echo "Daily workflow complete"

# Run weekly workflow
weekly: download train predict report
	@echo "Weekly workflow complete"
