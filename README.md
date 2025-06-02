# Risk Management MVP for Prop Trading

A machine learning system that analyzes prop trader behavior and predicts daily P&L to identify high-risk trading days before they happen.

## 🎯 Objective

Maximize profitability by predicting which traders are likely to have losing days and recommending position size reductions or trading restrictions.

## 🏗️ Architecture

```
┌─────────────────┐     ┌──────────────────┐     ┌─────────────────┐
│  PropreReports  │────▶│  Data Pipeline   │────▶│   SQLite DB     │
│      API        │     │  (Download/ETL)  │     │ (traders,fills) │
└─────────────────┘     └──────────────────┘     └────────┬────────┘
                                                           │
                        ┌──────────────────┐               │
                        │ Feature Engineer │◀──────────────┘
                        │ (Behavioral/Tech)│
                        └────────┬─────────┘
                                 │
                        ┌────────▼─────────┐
                        │  Model Training  │
                        │   (LightGBM)     │
                        └────────┬─────────┘
                                 │
                        ┌────────▼─────────┐     ┌─────────────────┐
                        │ Risk Prediction  │────▶│  Email Reports  │
                        │  (Daily/Batch)   │     │   (HTML/Text)   │
                        └──────────────────┘     └─────────────────┘
```

## 🚀 Quick Start

### Prerequisites
- Python 3.9+
- Conda/Miniconda
- PropreReports API token

### Installation

```bash
# Clone repository
git clone <repo-url>
cd risk-management-mvp

# Create conda environment
conda env create -f environment.yml
conda activate risk-tool

# Set up environment variables
cp .env.template .env
# Edit .env with your credentials:
# - API_TOKEN (PropreReports)
# - EMAIL_FROM, EMAIL_PASSWORD (Gmail app password)
# - EMAIL_TO (comma-separated recipient list)

# Initialize database and download historical data
python scripts/setup_database.py

# Train models
python scripts/train_models.py

# Verify setup
python verify_model_training.py
```

### Daily Usage

```bash
# Run daily prediction pipeline (downloads data, retrains, predicts, emails)
python scripts/daily_prediction_enhanced.py

# Or run individual components:
python scripts/daily_predict.py  # Just predictions
```

## 📊 Features

### Behavioral Signals
- **Trading frequency**: Orders/fills per day patterns
- **Time concentration**: Morning vs afternoon bias
- **Symbol diversity**: Concentration risk metrics
- **Size consistency**: Position sizing patterns

### Technical Indicators
- **P&L momentum**: 3/5/10/20-day rolling metrics
- **Win/loss streaks**: Consecutive performance patterns
- **Volatility measures**: Standard deviation of returns
- **Fee efficiency**: Fees as % of gross P&L

### Risk Levels
- **High Risk**: Predicted loss > $1,000 → Reduce positions by 50%
- **Medium Risk**: Predicted loss $0-1,000 → Monitor closely
- **Low Risk**: Predicted profit → Normal trading
- **Unknown**: Insufficient data → Manual review

## 🧪 Model Approach

### Current Implementation
- **Individual LightGBM models** per trader (when >30 days of data)
- **Time series splits**: 65% train, 15% validation, 20% test
- **Hyperparameter tuning** on validation set
- **Daily retraining** with latest data

### Key Metrics
- **RMSE**: Root Mean Square Error of P&L predictions
- **Directional Accuracy**: % of correct profit/loss predictions
- **Trade Reduction**: % of days flagged as no-trade
- **P&L Improvement**: Actual improvement from following signals

## 📁 Project Structure

```
risk-management-mvp/
├── src/                    # Core modules
│   ├── database.py         # SQLite database handler
│   ├── data_downloader.py  # PropreReports API client
│   ├── feature_engineer.py # Feature creation
│   ├── model_trainer.py    # LightGBM training
│   ├── predictor.py        # Risk predictions
│   └── email_service.py    # Report generation
├── scripts/                # Automation scripts
│   ├── setup_database.py   # Initial setup
│   ├── train_models.py     # Model training
│   ├── daily_prediction_enhanced.py  # Full daily pipeline
│   └── weekly_retrain.py   # Weekly retraining
├── config/                 # Configuration files
│   ├── traders.yaml        # Trader list
│   └── email_template.html # Report template
├── notebooks/              # Analysis notebooks
│   └── analysis.ipynb      # Model analysis
└── data/                   # Data directory (git-ignored)
    ├── trading.db          # SQLite database
    └── models/             # Trained models
```

## 🔧 Configuration

### traders.yaml
```yaml
traders:
  - account_id: 3946
    name: "NET005_OLD"
    strategy: "Day Trading"
    active: true
```

### Risk Thresholds (Adjustable)
- High risk threshold: -$1,000
- Medium risk threshold: $0
- Minimum training days: 30

## 📈 Performance Monitoring

### Database Statistics
```bash
sqlite3 data/trading.db "SELECT COUNT(*) as traders FROM traders;"
sqlite3 data/trading.db "SELECT COUNT(*) as days FROM daily_totals;"
```

### Model Performance
```python
# Check model quality
python verify_model_training.py

# Analyze predictions
python -c "from src.database import Database; db = Database(); print(db.get_latest_predictions())"
```

## 🐛 Troubleshooting

### Common Issues

1. **No predictions generated**
   - Check data availability: `python quick_diagnostics.py`
   - Verify API connection: Check `.env` credentials
   - Ensure >30 days of data per trader

2. **Models predicting zeros**
   - Run: `python debug_feature_engineering.py`
   - Check for data quality issues
   - Verify target variable creation

3. **Email not sending**
   - Verify Gmail app password (not regular password)
   - Check recipient email format in `.env`
   - Test with: `python -c "from src.email_service import EmailService; EmailService().send_test_email()"`

## 🚧 Known Limitations

1. **Cold start problem**: Need 30+ days of data per trader
2. **Individual models**: High maintenance for many traders
3. **Static thresholds**: Risk levels not adaptive
4. **No real-time updates**: Daily batch processing only

## 🔮 Future Enhancements

- [ ] Global model with trader embeddings
- [ ] Real-time risk monitoring
- [ ] Adaptive risk thresholds
- [ ] Multi-factor risk models
- [ ] Integration with trading systems
- [ ] Advanced backtesting framework
- [ ] A/B testing for strategies

## 📊 Sample Results

Based on historical testing:
- Average trade reduction: 25-35%
- Avoided losses: $500-2000/trader/month
- Directional accuracy: 55-65%
- False positive rate: 15-20%

## 🤝 Contributing

1. Follow existing code style
2. Add tests for new features
3. Update documentation
4. Submit PR with clear description

## 📄 License

Proprietary - Internal Use Only

## 🆘 Support

For issues or questions:
1. Check troubleshooting guide above
2. Run diagnostic scripts
3. Review logs in `logs/` directory
4. Contact: [your-email@company.com]
