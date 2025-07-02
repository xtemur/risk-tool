# Quantitative Risk Modeling System

A sophisticated Python-based system for modeling and monitoring trader risk using machine learning techniques. This system implements advanced time-series analysis, walk-forward backtesting, and real-time monitoring capabilities for quantitative risk management.

## 🎯 Overview

This project provides a complete end-to-end solution for:
- **Risk Prediction**: Predicting Value at Risk (VaR) and large loss probabilities for traders
- **Behavioral Analysis**: Detecting patterns like revenge trading and risk clustering
- **Model Monitoring**: Tracking feature drift and model stability in production
- **Causal Impact Analysis**: Measuring the economic impact of risk predictions

The system is designed with the realities of financial data in mind:
- Handles non-stationary market regimes
- Prevents lookahead bias through careful feature engineering
- Manages irregular trading patterns (traders not trading every day)
- Implements rigorous time-series cross-validation

## 🚀 Quick Start

### Prerequisites

- Conda (Miniconda or Anaconda)
- Python 3.9+
- SQLite database with trading data

### Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd risk-tool
```

2. Create and activate the conda environment:
```bash
conda env create -f environment.yml
```

3. Ensure your SQLite database is in place at `data/risk_tool.db`

### Basic Usage

```bash
# Run full backtest and train production models
python main.py --mode backtest

# Monitor existing models
python main.py --mode monitor
```

## 📁 Project Structure

```
risk-tool/
├── configs/
│   └── main_config.yaml         # Central configuration file
├── data/
│   ├── risk_tool.db            # SQLite database with trading data
│   └── processed/              # Processed feature sets
├── models/
│   └── production_model_artifacts/  # Trained models and results
├── reports/                    # Monitoring and analysis reports
├── src/
│   ├── data_processing.py      # Data cleaning and panel creation
│   ├── feature_engineering.py  # Feature generation pipeline
│   ├── modeling.py            # Model training and backtesting
│   ├── monitoring.py          # Drift detection and stability checks
│   ├── causal_impact.py       # Economic impact analysis
│   └── utils.py               # Utility functions
├── notebooks/                  # Exploration notebooks
├── main.py                    # Main orchestrator
├── environment.yml            # Conda environment specification
└── CLAUDE.md                  # Detailed project instructions
```

## 🔧 Configuration

The system is controlled via `configs/main_config.yaml`:

```yaml
# Key configuration sections:
paths:                    # File paths for data, models, reports
active_traders:          # List of trader IDs to analyze
feature_params:          # Feature engineering parameters
  ewma_spans: [3, 7, 21]
  rolling_vol_windows: [7, 21]
  sortino_window: 21
backtesting:             # Walk-forward validation settings
  n_splits: 5
  train_days: 126        # ~6 months
  test_days: 21          # ~1 month
production_model:        # LightGBM parameters
  var_model:            # Quantile regression for VaR
  loss_model:           # Binary classification for large losses
```

## 📊 Features

### Data Processing
- **Panel Data Creation**: Builds complete time-series panels for each trader
- **Business Day Calendar**: Handles weekends and holidays properly
- **Missing Data Handling**: Intelligent forward-filling for inactive trading days

### Feature Engineering
- **Base Features**:
  - Exponentially weighted moving averages (EWMA)
  - Rolling volatility measures
  - Trading frequency metrics

- **Advanced Risk Features**:
  - **Sortino Ratio**: Risk-adjusted returns focusing on downside
  - **Profit Factor**: Ratio of gross profits to gross losses
  - **Maximum Drawdown**: Worst peak-to-trough decline

- **Behavioral Features**:
  - **Revenge Trading Proxy**: Detects increased trading after losses
  - **Risk Clustering**: Identifies consecutive high-risk periods

### Modeling Approach

The system uses two complementary models:

1. **VaR Model** (LightGBM Quantile Regression)
   - Predicts the 5th percentile of next-day PnL
   - Helps set position limits and risk budgets

2. **Loss Model** (LightGBM Binary Classifier)
   - Predicts probability of large losses (bottom 15%)
   - Triggers additional risk reviews and interventions

### Backtesting Framework
- **Walk-Forward Validation**: Proper time-series cross-validation
- **Gap Days**: Prevents short-term information leakage
- **Out-of-Sample Metrics**: All reported metrics are from held-out test periods

### Monitoring Capabilities

1. **Feature Drift Detection**:
   - Statistical comparison of feature distributions
   - Identifies features that have changed significantly
   - Generates detailed drift reports

2. **Model Stability Analysis**:
   - SHAP-based feature importance tracking
   - Visual comparison of model behavior over time
   - Correlation analysis of feature importances

3. **Causal Impact Analysis**:
   - Economic impact metrics (avoided losses, missed opportunities)
   - Per-trader impact assessment
   - Temporal stability testing
   - Counterfactual analysis with position sizing

## 📈 Performance Metrics

The system tracks multiple performance indicators:

- **VaR Model**: Violation rate, economic capital efficiency
- **Loss Model**: AUC, precision-recall metrics, risk concentration
- **Economic Impact**: Total avoided losses, opportunity costs
- **Stability Metrics**: Feature importance correlation, drift scores

## 🛠️ Development

### Running Tests
```bash
# Run backtesting suite
python main.py --mode backtest

# Generate monitoring reports
python main.py --mode monitor
```

### Adding New Features

1. Define feature parameters in `configs/main_config.yaml`
2. Implement feature calculation in `src/feature_engineering.py`
3. Ensure proper shifting to prevent lookahead bias
4. Update target variable logic if needed

### Extending the Models

The system is designed for modularity. To add new models:
1. Implement training logic in `src/modeling.py`
2. Add model configuration to `main_config.yaml`
3. Update monitoring logic in `src/monitoring.py`
4. Extend causal impact analysis in `src/causal_impact.py`

## 📚 Key Concepts

### Lookahead Bias Prevention
All features are shifted by one day to ensure we only use information available at prediction time. This is critical for valid backtesting results.

### Time Series Cross-Validation
The system uses sklearn's `TimeSeriesSplit` with:
- Fixed training window (126 days)
- Fixed test window (21 days)
- Gap between train/test (1 day)
- No shuffling of data

### Handling Irregular Trading
The system creates complete panels with all business days, then:
- Forward-fills risk metrics (assumption: risk profile persists)
- Zero-fills PnL for inactive days
- Tracks trading frequency as a feature

## 📧 Daily Risk Signals

The system includes an automated daily risk signal service that generates and emails professional risk reports to stakeholders.

### Signal Generation
```bash
# Generate and save signals (no email)
python send_daily_signals.py --save-only

# Send to default recipients
python send_daily_signals.py

# Send to specific recipients
python send_daily_signals.py --email recipient1@example.com --email recipient2@example.com
```

### Interpreting the Daily Signals

#### **Risk Levels**
- **🔴 HIGH**: Loss probability > 70% or extreme volatility conditions
  - *Action Required*: Immediate position review, consider reducing exposure
- **🟡 MEDIUM**: Loss probability 40-70% or elevated volatility
  - *Monitoring*: Increased attention, prepare contingency plans
- **🟢 LOW**: Normal trading conditions
  - *Standard*: Continue regular monitoring

#### **Value at Risk (VaR)**
- **Definition**: 5th percentile of predicted next-day P&L
- **Example**: VaR of -$5,000 = 95% confidence of not losing more than $5,000
- **Usage**: Sets position limits and risk budgets per trader

#### **Loss Probability**
- **Definition**: Probability of experiencing a "large loss" (bottom 15% historically)
- **Trader-Specific**: Thresholds calibrated to each trader's historical performance
- **Range**: 0-100%, higher values indicate greater risk

#### **Warning Signals**
- **REVENGE_TRADING**: Excessive trading volume following recent losses
- **HIGH_VOLATILITY**: 7-day rolling volatility exceeds historical norms
- **LOW_WIN_RATE**: Win rate below 30% over past 21 trading days
- **LARGE_DRAWDOWN**: Significant peak-to-trough decline in cumulative P&L
- **ELEVATED_RISK**: Loss probability above 40% threshold

#### **Critical Alerts**
Automated flags for extreme situations requiring immediate intervention:
- Multiple consecutive high-risk days
- VaR violations exceeding tolerance
- Behavioral patterns indicating distressed trading

### Email Configuration

1. **Setup credentials**:
```bash
cp .env.example .env
# Edit .env with Gmail credentials and recipients
```

2. **Schedule daily reports** (optional):
```bash
# Add to crontab for 6 AM weekday delivery
0 6 * * 1-5 cd /path/to/risk-tool && python send_daily_signals.py --email team@company.com
```

### Report Components
The daily email includes:
- **Executive Summary**: Overall risk landscape and trader count by risk level
- **Trader Risk Table**: Individual risk metrics, VaR, probabilities, and warnings
- **Critical Alerts**: Items requiring immediate attention
- **Historical Context**: Comparison to recent performance trends

## 🤖 Production Automation

The system includes complete automation for production deployment with monitoring, backup, and alerting capabilities.

### Automated Pipeline

The master automation script orchestrates the entire daily workflow:

```bash
# Full automated pipeline
python scripts/daily_automation.py --email risk-team@company.com

# Test run without executing changes
python scripts/daily_automation.py --dry-run --email test@company.com

# Skip database update (for testing)
python scripts/daily_automation.py --skip-db --email test@company.com
```

### Key Automation Features

1. **Database Updates**: Automatically fetches and processes latest trading data
2. **Signal Generation**: Creates and sends daily risk reports
3. **Health Monitoring**: Validates database integrity and system performance
4. **Error Handling**: Comprehensive logging and failure notifications
5. **Backup Management**: Automated database backups with compression and retention

### Deployment Options

#### **Option 1: VPS Deployment (Recommended)**
- **Cost**: $5-20/month for professional setup
- **Reliability**: Always-on server with monitoring
- **Setup**: Complete deployment guide in `DEPLOYMENT.md`

```bash
# Cron job for daily automation (6 AM weekdays)
0 6 * * 1-5 /path/to/python scripts/daily_automation.py --email team@company.com

# Weekly backup (Sunday 3 AM)
0 3 * * 0 /path/to/python scripts/backup_database.py --remote-sync
```

#### **Option 2: Local Automation**
- **Cost**: Free (uses your machine)
- **Setup**: Simple cron job on development machine
- **Limitation**: Requires machine to stay on 24/7

#### **Option 3: Cloud Functions (Advanced)**
- **Platforms**: AWS Lambda, Google Cloud Functions, Azure Functions
- **Benefits**: Serverless, pay-per-execution
- **Considerations**: 15-minute execution limits, cold starts

### Monitoring & Alerting

The automation system provides comprehensive monitoring:

#### **Success Notifications**
- Daily execution summaries
- Performance metrics and timing
- Database health statistics
- Risk signal delivery confirmation

#### **Failure Alerts**
- Immediate email notifications on errors
- Detailed error logs with stack traces
- System health check failures
- Database integrity issues

#### **Health Checks**
- Database connectivity and integrity
- Active trader validation
- Recent trading data verification
- Model file availability
- Email service connectivity

### Configuration

#### **Environment Setup**
```bash
# Copy and configure environment variables
cp .env.example .env
# Edit .env with your credentials and settings
```

#### **Key Configuration Variables**
```bash
# Email settings
EMAIL_FROM=alerts@yourcompany.com
EMAIL_PASSWORD=your-app-password
EMAIL_RECIPIENTS=team@company.com,management@company.com

# Database path
DATABASE_PATH=data/risk_tool.db

# Logging level
LOG_LEVEL=INFO

# Environment
ENVIRONMENT=production
```

### Backup Strategy

#### **Automated Backups**
- **Daily**: Compressed database backups with integrity verification
- **Weekly**: Long-term backup retention (4 weeks)
- **Monthly**: Archive backups (12 months)
- **Cleanup**: Automatic old backup removal

```bash
# Manual backup
python scripts/backup_database.py

# Backup with remote sync
python scripts/backup_database.py --remote-sync

# Verify existing backups
python scripts/backup_database.py --verify-only
```

#### **Backup Features**
- SQLite integrity verification before backup
- File compression (typically 70-80% size reduction)
- SHA256 checksums for verification
- Optional remote sync via rsync
- Automatic cleanup of old backups

### Security & Maintenance

#### **Security Measures**
- Environment variable management for credentials
- File permission restrictions
- Secure SMTP authentication
- API key rotation support
- Access logging and monitoring

#### **Maintenance Tasks**
- **Daily**: Monitor automation logs and email delivery
- **Weekly**: Review system performance and backup integrity
- **Monthly**: Update dependencies and security patches
- **Quarterly**: Review and update risk thresholds

### Troubleshooting

#### **Common Issues**
```bash
# Check automation logs
tail -f logs/daily_automation_*.log

# Test database connectivity
python -c "import sqlite3; print('DB OK' if sqlite3.connect('data/risk_tool.db') else 'DB ERROR')"

# Test email configuration
python send_daily_signals.py --save-only

# Verify cron jobs
crontab -l
```

#### **Log Analysis**
```bash
# Search for errors in logs
grep -i error logs/*.log

# Check execution times
grep "Total Duration" logs/daily_automation_*.log

# Monitor resource usage
grep "Database health" logs/daily_automation_*.log
```

For detailed deployment instructions, see `DEPLOYMENT.md`.

## ⚠️ Important Considerations

1. **Data Quality**: The system assumes clean trade data in the SQLite database
2. **Market Regime Changes**: Monitor feature drift reports regularly
3. **Model Retraining**: Consider retraining when drift scores exceed thresholds
4. **Risk Limits**: VaR predictions should inform, not replace, risk management judgment
5. **Signal Interpretation**: Daily signals are predictive tools - combine with market context and trader communication
6. **Automation Monitoring**: Regularly check automation logs and failure notifications
7. **Backup Verification**: Periodically verify backup integrity and recovery procedures
