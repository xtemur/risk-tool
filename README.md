# Risk Tool - Combined Feature Trader Risk Prediction

## Overview

The Risk Tool implements an advanced trader risk prediction system using **combined LightGBM models** that leverage both engineered features and raw sequential data. This approach achieved **AUC 0.836** on test data, representing a **5.1% improvement** over traditional engineered-features-only models.

## Quick Start

### Training Models
```bash
python main.py --mode train
```

### Running Backtests
```bash
python main.py --mode backtest
```

### Generating Signals
```bash
python main.py --mode signals
```

## Architecture

### Combined Feature Approach
- **18 Engineered Features**: EWMA, rolling volatility, Sortino ratio, profit factor, drawdown, behavioral metrics
- **49 Sequential Features**: 7 raw features × 7 lag days, flattened into single vector
- **Total**: 67 features per sample

### Model Details
- **Algorithm**: LightGBM Classifier
- **Training**: Per-trader models with expanding window validation
- **Sequence Length**: 7 days of historical data
- **Raw Features**: daily_pnl, daily_gross, daily_fees, daily_volume, n_trades, gross_profit, gross_loss

## Performance Results

### Final Model Performance
- **Test AUC**: 0.836
- **Test AP**: 0.156
- **Improvement over baseline**: +5.1%

### Feature Importance
- **Sequential features**: 66.8% of total importance
- **Engineered features**: 33.2% of total importance
- **Most important lag days**: 3-6 days ago
- **Most important sequential features**: daily_volume, daily_pnl, daily_gross

## Project Structure

```
risk-tool/
├── src/
│   ├── data_processing.py      # Data loading and panel creation
│   ├── feature_engineering.py  # Engineered feature creation
│   ├── model_training.py       # Combined LightGBM training
│   ├── backtesting.py         # Performance analysis
│   └── utils.py               # Utility functions
├── models/
│   └── expanding_window/      # Trained models and predictions
├── configs/
│   └── main_config.yaml       # Configuration file
├── main.py                    # Entry point
└── README.md                  # This file
```

## Implementation Details

### Data Processing
- Creates trader-day panel from raw trade data
- Handles missing data and inactive days
- Applies temporal validation to prevent data leakage

### Feature Engineering
- **Engineered Features**: Domain-specific risk metrics with 1-day lag
- **Sequential Features**: Raw data from 7 previous days, flattened
- **Target Variable**: Binary flag for large loss days (bottom 15th percentile)

### Model Training
- **Per-trader models**: Individual LightGBM model for each trader
- **Expanding window**: Uses all historical data up to prediction point
- **Temporal validation**: 80% train, 20% test split by time
- **Early stopping**: Prevents overfitting with validation monitoring

### Key Features
- ✅ **No data leakage**: Strict temporal validation
- ✅ **Combined features**: Engineered + raw sequential data
- ✅ **Per-trader personalization**: Individual model per trader
- ✅ **Production ready**: Signal generation for live trading
- ✅ **Comprehensive validation**: Backtesting and performance monitoring

## Configuration

Edit `configs/main_config.yaml` to modify:
- Data paths and date ranges
- Feature engineering parameters
- Model hyperparameters
- Target variable definitions

## Requirements

```bash
pip install pandas numpy scikit-learn lightgbm pyyaml pathlib
```

## Results Summary

The combined approach successfully demonstrates that:
1. **Domain knowledge + raw data = optimal performance**
2. **LightGBM handles mixed feature types effectively**
3. **Sequential patterns from 3-6 days ago are most predictive**
4. **Longer sequences (7 days) capture more temporal complexity**
5. **5.1% AUC improvement justifies additional complexity**

This represents a significant advancement in trader risk prediction, combining traditional feature engineering with modern sequential modeling techniques within a proven LightGBM framework.
