# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

# Trading Risk Tool MVP - Development Guide

This is a sophisticated ML-based risk management system for proprietary trading that analyzes day traders' behavior from PropreReports data. The system predicts high-risk trading days and provides behavioral analytics.

## Essential Commands

### Daily Development Commands
```bash
make test          # Run pytest with coverage
make format        # Format code with black and isort
make lint          # Run linting with ruff
make clean         # Clean generated files and caches
```

### Data & Model Commands
```bash
make download      # Download new trading data from PropreReports API
make train         # Train risk prediction models
make predict       # Generate daily risk predictions
make backtest      # Run backtesting analysis
```

### Workflow Commands
```bash
make daily         # Complete daily workflow: download → predict → dashboard
make weekly        # Weekly workflow: download → train → predict → report
make monitor       # Check system health and generate alerts
```

## High-Level Architecture

### Data Flow Pipeline
1. **Data Ingestion**: PropreReports API → PropreReportsParser → SQLite Database
2. **Feature Engineering**: DatabaseManager → Feature Pipelines → Feature Store (planned)
3. **Model Training**: FeaturePipeline → RiskModel (LightGBM) → Model Storage
4. **Prediction**: Latest features → Trained model → Risk predictions
5. **Monitoring**: Predictions → DriftDetector → AlertSystem → Dashboard

### Key Architectural Components

**Data Layer** (`src/data/`)
- `DatabaseManager`: SQLite interface with 3 tables (accounts, account_daily_summary, fills)
- `DataDownloader`: API integration with retry logic and CSV backups
- `PropreReportsParser`: Auto-detects report types, handles Equities/Options accounts

**Feature Engineering** (`src/features/`)
- Abstract `BaseFeatures` class ensures temporal consistency
- `TechnicalFeatures`: Price/volume indicators (20+ features)
- `BehavioralFeatures`: Trading psychology patterns (30+ features)
- `MarketRegimeFeatures`: Market condition indicators (15+ features)
- `FeaturePipeline`: Orchestrates feature generation with point-in-time guarantees

**Model Layer** (`src/models/`)
- `RiskModel`: LightGBM-based risk predictor with SHAP explanations
- `ModelPipeline`: Training orchestration with validation
- Planned: EnsembleModel, RegimeModel for advanced predictions

**Monitoring** (`src/monitoring/`)
- `ModelMonitor`: Tracks prediction accuracy and model drift
- `DriftDetector`: Detects data distribution shifts
- `AlertSystem`: Email alerts for high-risk predictions
- `DashboardGenerator`: HTML reports with Jinja2 templates

### Database Schema
- **accounts**: Trader metadata (account_id, name, type)
- **account_daily_summary**: Daily aggregated metrics (P&L, fees, portfolio)
- **fills**: Individual trade executions with fees

## Critical Implementation Details

### Temporal Data Handling
- All feature calculations use point-in-time data to prevent look-ahead bias
- Features generated with specific date cutoffs for backtesting validity
- Database queries always filter by date to ensure temporal consistency

### Multi-Account Support
- Handles both Equities and Options accounts with different fee structures
- Account type detection built into parser
- Separate feature handling for account-specific metrics

### Current Limitations
- No implemented backtesting engine (WalkForwardValidator planned)
- Limited test coverage (only RiskModel tested)
- No feature caching (recalculated each run)
- Single model only (no ensemble yet)

## Configuration

### Environment Setup
- Python 3.9 with conda environment
- API credentials in `.env` (use `.env.template`)
- 9 active traders in `config/traders.yaml` (60+ available)

### Key Dependencies
- ML: lightgbm, scikit-learn, numpy, pandas
- Database: sqlite3 with WAL mode
- Monitoring: jinja2, schedule, smtplib2
- Config: pyyaml, python-dotenv

## Development Workflow

### Adding New Features
1. Extend `BaseFeatures` class in `src/features/`
2. Implement `calculate_features()` with temporal consistency
3. Add to `FeaturePipeline` configuration
4. Update feature count in documentation

### Model Development
1. Use `src/experiments/` for tracking experiments
2. Implement new models extending base interfaces
3. Validate with proper train/test splits
4. Check for data leakage with temporal validation

### Testing Strategy
- Unit tests in `tests/` mirror `src/` structure
- Use `make test` for full test suite with coverage
- Integration tests for data pipeline components
- Mock PropreReports API for consistent testing

## Current Development State

### Active Work Areas
- Feature engineering improvements in `base_features.py`
- Notebook development for data validation
- Backtesting framework implementation

### Priority Next Steps
1. Complete `train_models.py` implementation
2. Add comprehensive test coverage
3. Implement feature caching for performance
4. Build backtesting engine for validation

## Common Issues & Solutions

### Data Issues
- Missing data: Parser handles gracefully, features have fallbacks
- API timeouts: Automatic retry with exponential backoff
- Database locks: WAL mode prevents most locking issues

### Model Issues
- Feature drift: Monitor with `DriftDetector`, retrain weekly
- Overfitting: Use walk-forward validation (when implemented)
- Class imbalance: RiskModel uses class weights

### Performance
- Slow feature calculation: Implement caching (planned)
- Large database: CSV backups allow selective data loading
- Memory issues: Process traders in batches if needed
