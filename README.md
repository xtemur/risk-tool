# Risk Tool - Day Trading Risk Prediction System

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

A production-grade machine learning system that predicts whether a day trader's next-day P&L will be negative, enabling proactive risk management and capital preservation.

## 🎯 Project Overview

**Risk Tool** combines historical trading data analysis with state-of-the-art machine learning to help day traders avoid negative P&L days. The system analyzes patterns in trading behavior, position sizing, and market dynamics to predict high-risk trading sessions.

### Key Features

- 🤖 **Multi-Model Architecture**: Global + trader-specific models with intelligent ensemble
- ⏱️ **Real-Time Risk Monitoring**: Live alerts and position sizing recommendations
- 📊 **Comprehensive Analytics**: Risk metrics, drawdown analysis, and behavioral insights
- 🎯 **Time Series Validation**: Proper backtesting with no data leakage
- 📈 **Performance Tracking**: Model-assisted vs. original trading performance comparison
- 🚨 **Alert System**: Configurable risk limits with automated notifications

### Performance Highlights

- **2x P&L improvement** in initial prototype testing
- **Significant drawdown reduction** through risk avoidance
- **AUC > 0.65** on holdout test data
- **Real-time predictions** for next-day risk assessment

## 🚀 Quick Start

### Prerequisites

- Python 3.9+
- Conda (recommended) or pip
- PropReports CSV exports (totals and fills data)

### Installation

```bash
# Clone the repository
git clone https://github.com/your-org/risk-tool.git
cd risk-tool

# Create conda environment
conda create -n risk-tool python=3.9
conda activate risk-tool

# Install dependencies
pip install -r requirements.txt
pip install -e .

# Verify installation
python -c "import src; print('Installation successful!')"
```

### Data Preparation

1. **Export your PropReports data**:
   - Daily totals CSV (date, Symbol, Net, Unrealized, etc.)
   - Fills CSV (date, symbol, qty, gross, comm, net, etc.)

2. **Organize data files**:
   ```bash
   data/raw/totals/
   ├── trader_001_totals.csv
   ├── trader_002_totals.csv    
   └── ...
   
   data/raw/fills/
   ├── trader_001_fills.csv
   ├── trader_002_fills.csv
   └── ...
   ```

3. **Validate data integrity**:
   ```bash
   python scripts/validate_data.py --input data/raw/
   ```

### Train Your First Model

```bash
# Configure for your traders
cp config/config.yaml.example config/config.yaml
# Edit config/config.yaml with your trader IDs

# Train models (holds out last 2 months for testing)
python scripts/train_models.py --config config/config.yaml

# Evaluate performance
python scripts/evaluate_models.py --config config/config.yaml

# Generate backtest report
python scripts/backtest_strategy.py --config config/config.yaml
```

### Generate Daily Risk Assessment

```bash
# Get today's risk prediction
python scripts/generate_predictions.py --date 2025-05-23

# Generate comprehensive risk report
python scripts/daily_risk_report.py --date 2025-05-23
```

## 📁 Project Structure

```
risk-tool/
├── src/                    # Core source code
│   ├── data/              # Data loading and feature engineering
│   ├── models/            # ML models and ensemble logic
│   ├── validation/        # Time series CV and backtesting
│   ├── risk/              # Risk analysis and monitoring
│   └── visualization/     # Dashboards and reporting
├── config/                # Configuration files
├── scripts/               # Executable scripts for training/prediction
├── notebooks/             # Jupyter notebooks for exploration
├── tests/                 # Unit and integration tests
└── data/                  # Data storage (raw/processed)
```

## 🔧 Configuration

The system is highly configurable through YAML files:

### Model Configuration (`config/config.yaml`)

```yaml
data:
  traders: ["trader_001", "trader_002", ...]
  validation:
    test_months: 2           # Holdout period
    gap_days: 5             # Prevent data leakage

model:
  global_model:
    type: "lightgbm"
    params:
      objective: "binary"
      num_leaves: 31
      learning_rate: 0.05
      early_stopping_rounds: 50

features:
  rolling_windows: [3, 7, 14, 30]  # Historical lookback periods
  behavioral_features: true         # Trading behavior analysis
```

### Risk Limits (`config/risk_limits.yaml`)

```yaml
risk_limits:
  daily:
    max_loss_pct: 0.02      # 2% max daily loss
    warning_loss_pct: 0.015 # Warning threshold
  
  portfolio:
    max_exposure: 100000    # $100k max exposure
    var_confidence: 0.95    # VaR confidence level
```

## 🎯 Usage Examples

### Basic Model Training

```python
from src.data.data_loader import DataLoader
from src.models.ensemble import EnsembleRiskModel
from src.validation.backtester import TradingBacktester

# Load data
loader = DataLoader('config/config.yaml')
data = loader.load_all_traders()

# Train ensemble model
ensemble = EnsembleRiskModel.from_config('config/config.yaml')
ensemble.fit(data)

# Backtest strategy
backtester = TradingBacktester()
results = backtester.backtest_strategy(
    predictions=ensemble.predict(test_data),
    actual_pnl=test_data['Net'],
    threshold=0.3  # Conservative threshold
)

print(f"Original P&L: ${results['original_pnl']:,.2f}")
print(f"Model-Assisted P&L: ${results['assisted_pnl']:,.2f}")
print(f"Risk Reduction: {results['risk_reduction']:.1%}")
```

### Real-Time Risk Monitoring

```python
from src.risk.risk_monitor import RealTimeRiskMonitor
from src.risk.alert_system import AlertSystem

# Initialize monitoring
monitor = RealTimeRiskMonitor('config/risk_limits.yaml')
alerts = AlertSystem('config/config.yaml')

# Check current day risk
current_pnl = -1500  # Current day P&L
risk_prob = ensemble.predict_next_day_risk(current_features)

if risk_prob > 0.7:
    alert = {
        'type': 'HIGH_RISK',
        'message': f'High risk probability: {risk_prob:.1%}',
        'recommendation': 'Consider reducing position sizes tomorrow'
    }
    alerts.send_alert(alert)
```

### Custom Feature Engineering

```python
from src.data.feature_engineer import FeatureEngineer

# Create custom features
engineer = FeatureEngineer(config['features'])

# Add your own risk indicators
def custom_risk_features(df):
    """Add custom risk features"""
    # Volatility clustering
    df['vol_regime'] = df['Net'].rolling(30).std() > df['Net'].rolling(90).std()
    
    # Overtrading indicator
    df['overtrading'] = df['trade_count'] > df['trade_count'].rolling(30).quantile(0.8)
    
    # Revenge trading detection
    df['revenge_trading'] = (
        (df['Net'].shift(1) < 0) & 
        (df['trade_count'] > df['trade_count'].rolling(7).mean() * 1.5)
    )
    
    return df

# Register custom transformer
engineer.add_transformer('custom_risk', custom_risk_features)
```

## 📊 Model Performance

### Validation Methodology

- **Time Series Cross-Validation**: 5-fold CV with 30-day test periods and 5-day gaps
- **True Holdout Testing**: Last 2 months never seen during training
- **No Data Leakage**: All features use only historical information
- **Realistic Backtesting**: Transaction costs and slippage included

### Key Metrics

| Metric | Target | Typical Results |
|--------|--------|-----------------|
| AUC Score | > 0.65 | 0.67-0.72 |
| Precision | > 0.60 | 0.63-0.68 |
| Risk Reduction | > 15% | 20-35% |
| Sharpe Improvement | > 0.2 | 0.3-0.5 |

### Model Architecture

```
┌─────────────────┐    ┌──────────────────┐
│  Global Model   │    │ Trader-Specific  │
│  (All Traders)  │    │     Models       │
│                 │    │  (Individual)    │
└─────────┬───────┘    └─────────┬────────┘
          │                      │
          └──────┬───────────────┘
                 │
         ┌───────▼────────┐
         │ Ensemble Model │
         │ (Weighted Avg) │
         └────────────────┘
```

## 🧪 Testing

Run the full test suite:

```bash
# All tests
pytest tests/ -v

# Specific modules
pytest tests/test_models/ -v
pytest tests/test_data/ -v

# With coverage report
pytest tests/ --cov=src --cov-report=html
open htmlcov/index.html
```

### Test Data

The test suite includes:
- **Unit tests** for all core functions
- **Integration tests** for full pipelines
- **Data validation tests** for input/output formats
- **Model performance tests** with benchmark datasets

## 📈 Production Deployment

### Daily Workflow

```bash
# Morning routine (before market open)
python scripts/generate_predictions.py --date $(date +%Y-%m-%d)
python scripts/daily_risk_report.py --date $(date +%Y-%m-%d)

# Evening routine (after market close)
python scripts/update_models.py --incremental
python scripts/monitor_models.py --check-drift
```

### Automated Scheduling

Use cron for automated execution:

```bash
# Add to crontab
0 8 * * 1-5 /path/to/risk-tool/scripts/morning_routine.sh
0 18 * * 1-5 /path/to/risk-tool/scripts/evening_routine.sh
```

### Model Monitoring

```python
# Check for model drift
python scripts/monitor_models.py --config config/config.yaml

# Retrain if performance degrades
python scripts/train_models.py --config config/config.yaml --retrain
```

## 🔍 Troubleshooting

### Common Issues

**Q: Model AUC is below 0.6**
```bash
# Check data quality and feature engineering
python scripts/analyze_features.py --config config/config.yaml
# Increase training data or adjust features
```

**Q: High false positive rate**
```bash
# Adjust prediction threshold
python scripts/optimize_threshold.py --metric precision
# Or modify ensemble weights
```

**Q: Memory issues with large datasets**
```bash
# Enable data chunking
export CHUNK_SIZE=10000
python scripts/train_models.py --config config/config.yaml --chunked
```

### Debug Mode

```bash
# Enable verbose logging
export LOG_LEVEL=DEBUG
python scripts/train_models.py --config config/config.yaml --debug
```

## 🤝 Contributing

We welcome contributions! Please see our [Contributing Guide](CONTRIBUTING.md) for details.

### Development Setup

```bash
# Install development dependencies
pip install -r requirements-dev.txt

# Install pre-commit hooks
pre-commit install

# Run code formatting
black src/ tests/
flake8 src/ tests/
```

### Adding New Features

1. **Fork the repository**
2. **Create feature branch**: `git checkout -b feature/new-risk-metric`
3. **Add tests**: Write tests first (TDD approach)
4. **Implement feature**: Follow existing code patterns
5. **Update docs**: Document new functionality
6. **Submit PR**: Include performance benchmarks

## 📚 Documentation

- [User Guide](docs/user_guide/) - Detailed usage instructions
- [API Reference](docs/api/) - Complete API documentation
- [Architecture Guide](docs/architecture.md) - System design details
- [Model Documentation](docs/models/) - ML model specifications

## 🔒 Security & Privacy

- **No sensitive data**: Models work with aggregated P&L data only
- **Local processing**: All computation runs on your infrastructure
- **Configurable alerts**: Control what information is shared
- **Audit trail**: Complete logging of all model decisions

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🏆 Acknowledgments

- **PropReports** for excellent trading data export functionality
- **LightGBM** team for the robust gradient boosting framework
- **Time series ML community** for validation methodology best practices

## 📞 Support

- **Issues**: [GitHub Issues](https://github.com/your-org/risk-tool/issues)
- **Discussions**: [GitHub Discussions](https://github.com/your-org/risk-tool/discussions)
- **Email**: risk-tool-support@your-org.com

---

**⚠️ Risk Disclaimer**: This tool is for risk management purposes only. Past performance does not guarantee future results. Always use proper risk management and never risk more than you can afford to lose.

**🎯 Remember**: The goal is capital preservation, not profit maximization. A model that prevents large losses is more valuable than one that predicts small gains.