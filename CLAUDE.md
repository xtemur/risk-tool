# Trading Risk Manager v2 - System Design Guide

## Overview

This document outlines the architecture and implementation strategy for Trading Risk Manager v2, a comprehensive system for analyzing day traders' behavior and predicting risk using machine learning on PropreReports data.

## Core Design Principles

- **Temporal Integrity**: Strict separation of training, validation, and live data
- **Modularity**: Each component has a single responsibility
- **Auditability**: All predictions, decisions, and performance metrics are logged
- **Extensibility**: Easy to add new models, features, or strategies
- **MVP Focus**: Start simple, iterate based on results

## System Architecture

### Folder Structure

```
risk-tool-v2/
├── config/
│   ├── traders.yaml
│   ├── models.yaml
│   ├── features.yaml
│   └── backtest.yaml
├── src/
│   ├── core/
│   │   ├── __init__.py
│   │   ├── database.py (existing)
│   │   ├── data_downloader.py (existing)
│   │   └── constants.py
│   ├── pipeline/
│   │   ├── data_validator.py
│   │   ├── feature_pipeline.py
│   │   └── model_pipeline.py
│   ├── features/
│   │   ├── base_features.py
│   │   ├── technical_features.py
│   │   ├── behavioral_features.py
│   │   ├── market_regime_features.py
│   │   └── feature_store.py
│   ├── models/
│   │   ├── base_model.py
│   │   ├── ensemble_model.py
│   │   ├── regime_model.py
│   │   └── risk_model.py
│   ├── backtesting/
│   │   ├── backtest_engine.py
│   │   ├── walk_forward_validator.py
│   │   ├── portfolio_simulator.py
│   │   └── performance_metrics.py
│   ├── monitoring/
│   │   ├── model_monitor.py
│   │   ├── drift_detector.py
│   │   ├── alert_system.py
│   │   └── dashboard_generator.py
│   └── utils/
│       ├── time_series_cv.py
│       ├── data_quality.py
│       └── logger.py
├── notebooks/
│   ├── 01_eda/
│   ├── 02_feature_analysis/
│   ├── 03_model_experiments/
│   └── 04_backtest_analysis/
├── scripts/
│   ├── download_data.py
│   ├── generate_features.py
│   ├── train_models.py
│   ├── run_backtest.py
│   ├── daily_predict.py
│   └── generate_report.py
├── tests/
├── data/
│   ├── raw/
│   ├── features/
│   ├── models/
│   ├── predictions/
│   └── reports/
└── logs/
```

## Key Components

### 1. Data Layer

- **Database** (existing) - Central data storage with SQLite
- **DataDownloader** (existing) - PropreReports API integration
- **DataValidator** - Ensures data quality and completeness
  - Checks for missing values, outliers, data gaps
  - Validates temporal consistency
  - Logs data quality metrics

### 2. Feature Engineering Layer

- **BaseFeatures** - Abstract base class for all feature generators
- **TechnicalFeatures** - Price-based indicators
  - Returns (1d, 3d, 5d, 20d)
  - Volatility measures
  - Momentum indicators
- **BehavioralFeatures** - Trading behavior patterns
  - Time of day analysis
  - Trading frequency patterns
  - Win/loss streaks
- **MarketRegimeFeatures** - Market conditions
  - Volatility regimes
  - Trend strength
  - Market microstructure
- **FeatureStore** - Manages feature versioning and metadata
  - Prevents data leakage with proper time indexing
  - Tracks feature importance over time

### 3. Model Layer

- **BaseModel** - Abstract class defining model interface
- **RiskModel** - Primary risk prediction model using LightGBM
- **RegimeModel** - Market regime classification
- **EnsembleModel** - Combines multiple models with dynamic weighting
  - Uses out-of-sample performance for weight calculation
  - Implements model diversity metrics

### 4. Backtesting Framework

- **BacktestEngine** - Core simulation engine
  - Point-in-time data access
  - Realistic execution assumptions
  - Transaction cost modeling
- **WalkForwardValidator** - Implements walk-forward analysis
  - Expanding window training
  - Fixed test periods
  - Purged cross-validation
- **PortfolioSimulator** - Simulates trading decisions
  - Position sizing based on risk predictions
  - Capital allocation strategies
- **PerformanceMetrics** - Comprehensive metric calculation
  - Sharpe, Sortino, Calmar ratios
  - Maximum drawdown analysis
  - Risk-adjusted returns

### 5. Monitoring & Analysis

- **ModelMonitor** - Tracks model performance in production
  - Prediction accuracy over time
  - Feature importance stability
- **DriftDetector** - Detects distribution shifts
  - PSI (Population Stability Index)
  - Kolmogorov-Smirnov tests
  - Feature drift monitoring
- **AlertSystem** - Automated alerting for anomalies
- **DashboardGenerator** - Creates HTML reports with Plotly

## Data Flow Architecture

```
1. Raw Data → DataValidator → Database
                    ↓
2. Database → FeaturePipeline → FeatureStore
                    ↓
3. FeatureStore → ModelPipeline → TrainedModels
                    ↓
4. TrainedModels + NewData → PredictionEngine → Predictions
                    ↓
5. Predictions → BacktestEngine → PerformanceMetrics
                    ↓
6. LivePredictions → MonitoringSystem → Alerts/Reports
```

## Temporal Data Management

### Training/Validation/Test Split Strategy

- **Training**: Historical data up to T-60 days
- **Validation**: T-60 to T-30 days (for hyperparameter tuning)
- **Test**: T-30 to T-0 days (out-of-sample testing)
- **Production**: T+1 onwards

### Walk-Forward Scheme

1. Initial training: 365 days
2. Validation window: 30 days
3. Test window: 30 days
4. Retrain frequency: Weekly
5. Feature recalculation: Daily

## Tech Stack Recommendations

### Core Dependencies (Keep Existing)

```yaml
- Python 3.9+
- pandas (1.5+) - Data manipulation
- numpy - Numerical computing
- scikit-learn - ML algorithms
- SQLite - Simple, file-based database
- conda - Environment management
```

### Essential Additions for MVP

```yaml
dependencies:
  # Machine Learning
  - lightgbm>=3.3.5      # Better than basic sklearn models
  - optuna>=3.0          # Hyperparameter tuning

  # Time Series Analysis
  - statsmodels>=0.14    # Time series analysis
  - arch>=5.3            # Volatility modeling (optional)

  # Feature Engineering
  - feature-engine>=1.6  # Feature pipelines

  # Visualization & Monitoring
  - plotly>=5.0          # Interactive visualizations
  - evidently>=0.4       # ML monitoring

  # Development
  - pytest>=7.0          # Testing
  - loguru               # Better logging
  - python-dotenv>=1.0   # Config management
  - pre-commit           # Code quality
```

### Optional Advanced Tools

```yaml
# Backtesting (choose one)
- vectorbt>=0.25       # Fast, vectorized backtesting
- backtesting.py       # Simpler alternative

# Technical Analysis
- ta-lib               # Pre-built indicators

# Data Validation
- great-expectations   # Data quality testing

# Web Dashboard
- dash>=2.0            # Interactive web apps
```

### Why These Choices?

1. **LightGBM**: Fastest gradient boosting, excellent for tabular financial data
2. **Statsmodels**: Essential for proper time series analysis and statistical tests
3. **Plotly**: Interactive charts crucial for financial data exploration
4. **Evidently**: Purpose-built for ML model monitoring
5. **SQLite**: Perfect for single-user MVP, no server needed

## Best Practices Implementation

### 1. Preventing Data Leakage

- All features computed using only past data
- Strict timestamp-based data access
- Separate feature computation for train/test
- No future information in features

### 2. Avoiding Look-Ahead Bias

- Realistic execution delays
- End-of-day predictions for next day
- Proper handling of corporate actions
- No use of same-day closing prices for decisions

### 3. Managing Survivorship Bias

- Include all traders (active and inactive)
- Track trader lifecycle
- Adjust for selection bias in analysis
- Don't train only on "winners"

### 4. Ensemble Strategy

- Multiple uncorrelated models
- Dynamic weighting based on recent performance
- Regime-specific model selection
- Diversity in model types and features

## Workflow Pipelines

### Daily Operations

```
6:00 AM: Download Previous Day Data
    ↓
6:30 AM: Validate Data Quality
    ↓
7:00 AM: Generate Features
    ↓
7:30 AM: Generate Predictions
    ↓
8:00 AM: Monitor Performance
    ↓
8:30 AM: Send Reports
```

### Weekly Retraining

```
Sunday Night:
1. Collect Week's Data
2. Run Full Backtests
3. Retrain All Models
4. Validate Performance
5. Deploy Best Models
6. Generate Weekly Report
```

### Monthly Analysis

```
1. Full Historical Backtest
2. Feature Importance Analysis
3. Model Performance Review
4. Strategy Adjustment
5. Risk Parameter Tuning
```

## Implementation Roadmap

### Phase 1: Foundation (Week 1-2)
- [ ] Set up project structure
- [ ] Implement DataValidator
- [ ] Create BaseFeatures and TechnicalFeatures
- [ ] Build basic RiskModel with LightGBM
- [ ] Simple backtesting framework

### Phase 2: Enhancement (Week 3-4)
- [ ] Add BehavioralFeatures
- [ ] Implement WalkForwardValidator
- [ ] Create ModelMonitor
- [ ] Build performance reporting
- [ ] Add basic drift detection

### Phase 3: Production (Week 5-6)
- [ ] Implement full pipeline automation
- [ ] Add EnsembleModel
- [ ] Create dashboard generator
- [ ] Set up alerting system
- [ ] Comprehensive testing

### Phase 4: Optimization (Ongoing)
- [ ] Feature selection optimization
- [ ] Hyperparameter tuning
- [ ] Add market regime features
- [ ] Enhance monitoring
- [ ] Performance optimization

## Data Schema Reference

### Totals Table
- `date`: Trading date
- `account_id`: Trader identifier
- `orders_count`: Number of orders
- `fills_count`: Number of fills
- `quantity`: Total quantity traded
- `gross_pnl`: Gross P&L
- `net_pnl`: Net P&L (after fees)
- `total_fees`: Trading fees
- `unrealized_delta`: Unrealized P&L
- `total_delta`: Total P&L

### Fills Table
- `id`: Unique identifier
- `datetime`: Transaction timestamp
- `account_id`: Trader identifier
- `symbol`: Trading symbol
- `price`: Execution price
- `quantity`: Trade quantity
- `order_id`: Order identifier
- `total_fees`: Transaction fees

## Development Guidelines

1. **Start Simple**: Begin with basic features and models
2. **Test Everything**: Unit tests for all components
3. **Document Decisions**: Keep decision log in notebooks
4. **Version Control**: Tag releases, branch for features
5. **Monitor Everything**: Log all predictions and decisions
6. **Iterate Based on Data**: Let performance guide development

## Common Pitfalls to Avoid

- ❌ Using future data in features
- ❌ Training on test data
- ❌ Ignoring transaction costs
- ❌ Overfitting to recent data
- ❌ Not accounting for regime changes
- ❌ Assuming stationarity
- ❌ Ignoring survivorship bias

## Success Metrics

- **Model Performance**: Sharpe > 1.5, Max Drawdown < 10%
- **Prediction Accuracy**: Direction accuracy > 55%
- **System Reliability**: < 1% failed predictions
- **Risk Management**: Correct identification of high-risk periods
- **Business Value**: Improved trader P&L through better risk management

---

*This document is a living guide and should be updated as the system evolves.*
