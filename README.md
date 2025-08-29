# Risk Management Tool for Day Traders

**Daily loss limit predictions for 15 expensive day traders using pooled risk models.**

## Quick Start

```bash
# Set up environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt

# Configure database and API
cp .env.example .env
# Edit .env with your credentials

# Run morning pipeline (normally runs at 5 AM via cron)
python morning_pipeline.py

# Or test individual components
python -m src.minimal_risk_system
```

## What This Does

Every morning at 5 AM:
1. Fetches yesterday's trading data (~1M transactions)
2. Predicts optimal daily loss limit reductions (0-80%)
3. Sends email alerts to risk managers by 10 AM
4. Risk managers apply limits before market open

**Example Output:**
```
TRADER NET001: Reduce limit by 50% ($5,000 → $2,500)
Reason: 3-day loss streak, elevated volatility
Confidence: High
```

## System Architecture

```
Morning Pipeline (5 AM - 10 AM)
├── [5:00] Data ingestion from database
├── [6:00] Feature computation (20 risk indicators)
├── [7:00] Pooled model predictions
├── [8:00] Comparison with rules baseline
├── [9:00] Email report generation
└── [10:00] Delivery to risk managers
```

## Core Components

### 1. Pooled Risk Model (`src/pooled_risk_model.py`)
- **Why Pooled?** 15 traders × 500 days = 7,500 samples (too small for individual models)
- **Model:** Ridge regression with heavy regularization
- **Features:** ~20 scale-invariant risk indicators
- **Critical:** Maintains temporal alignment to prevent data leakage

### 2. Rules Baseline (`src/rules_baseline.py`)
- Simple rules any risk manager would apply
- Restricts after 3+ consecutive losses
- Triggers on drawdowns > 15%
- ML must beat this by >10% to be used

### 3. Daily Pipeline (`morning_pipeline.py`)
- Orchestrates the complete workflow
- Falls back to rules if ML fails
- Sends formatted email alerts

## Configuration

Edit `config.py` (no JSON/YAML complexity):

```python
class Config:
    DB_PATH = 'data/risk_tool.db'        # SQLite database
    DEFAULT_LIMIT = 5000                 # Standard daily loss limit
    MAX_REDUCTION_PCT = 80               # Never reduce more than 80%
    RETRAIN_FREQUENCY = 7                # Retrain weekly
    LARGE_LOSS_THRESHOLD = 2000          # Define "significant loss"
```

## Database Schema

```sql
-- Main table: fills
CREATE TABLE fills (
    fill_datetime TEXT,    -- "MM/DD/YYYY HH:MM:SS"
    account TEXT,          -- Trader ID (e.g., "NET001")
    side TEXT,            -- "B" or "S"
    qty INTEGER,
    symbol TEXT,
    price REAL,
    -- ... fees and other columns
);

-- Daily P&L calculation
SELECT
    date(fill_datetime) as date,
    account as trader_id,
    SUM(CASE
        WHEN side = 'B' THEN -qty * price
        WHEN side = 'S' THEN qty * price
    END) as pnl
FROM fills
GROUP BY date(fill_datetime), account;
```

## Model Training

The system uses a **pooled model** trained on all traders simultaneously:

```python
# Key insight: All traders use same historical cutoff per prediction date
for prediction_date in all_dates:
    historical = data[data['date'] < prediction_date]  # Same for all
    for trader in traders:
        features = compute_features(historical[trader])
        features['prediction_date'] = prediction_date  # Critical!
```

**Why this matters:** Prevents data leakage where one trader's outcome influences another's prediction on the same day.

## Performance Metrics

```
Current Performance (Last 60 Days):
- Hit Rate: 72% (caught major losses)
- False Positive Rate: 18% (over-restricted profitable days)
- Net Benefit: $45,000 prevented losses - $8,000 opportunity cost
- Model Type: Ridge (beats rules baseline by 15%)
```

## Scheduled Execution

```bash
# Add to crontab for daily 5 AM execution
./setup_cron.sh

# Or manually add:
0 5 * * 1-5 cd /path/to/risk-tool && python morning_pipeline.py >> logs/morning.log 2>&1
```

## Testing

```bash
# Run all tests
python -m pytest tests/

# Test specific components
python -m src.pooled_risk_model --test
python -m src.rules_baseline --test

# Backtest on historical data
python scripts/backtest.py --start 2024-01-01 --end 2024-12-31
```

## Common Issues & Solutions

### Issue: "Model predictions seem too aggressive"
**Solution:** Check temporal alignment in features - most likely seeing future data

### Issue: "Different results in backtest vs production"
**Solution:** Ensure using `data[data['date'] < prediction_date]` everywhere

### Issue: "Model not beating rules baseline"
**Solution:** This is normal with small data - rules often sufficient

### Issue: "Predictions all very similar"
**Solution:** Model is pooled, expecting some convergence. Add trader-specific bias terms if needed

## File Structure

```
risk-tool/
├── data/
│   └── risk_tool.db              # SQLite database (147 MB)
├── src/
│   ├── pooled_risk_model.py      # Core pooled model with temporal alignment
│   ├── rules_baseline.py         # Simple rules that often work
│   ├── minimal_risk_system.py    # Orchestrator
│   └── email_service.py          # Email formatting and sending
├── scripts/
│   ├── save_accounts.py          # Fetch trader accounts
│   ├── save_trades.py            # Fetch daily trades
│   └── update_database.py        # Update all data
├── config.py                      # Single configuration file
├── morning_pipeline.py            # Main daily workflow
└── README.md                      # This file
```

## Key Design Decisions

1. **Pooled Model:** With only 15 traders, individual models would overfit
2. **Ridge Regression:** Simple, robust, interpretable (complexity not justified)
3. **Temporal Alignment:** Critical for preventing data leakage
4. **Rules Fallback:** Often good enough, always available
5. **Conservative Limits:** Better to over-restrict than miss a blowup

## Maintenance

### Daily
- Check morning pipeline logs for errors
- Verify email delivery
- Monitor hit rate on large losses

### Weekly
- Model retrains automatically
- Review false positive rate
- Adjust thresholds if needed

### Monthly
- Full backtest on recent month
- Compare model vs rules performance
- Update trader list if needed

## API Credentials

Required in `.env`:
```bash
# Database
DB_PATH=data/risk_tool.db

# Email
EMAIL_FROM=risk-alerts@yourfirm.com
EMAIL_PASSWORD=app-specific-password
EMAIL_RECIPIENTS=risk-manager@yourfirm.com,backup@yourfirm.com

# Trading API (for data updates)
API_URL=https://api.tradingplatform.com
API_USER=your-username
API_PASS=your-password
```

## Support

- **Documentation:** See `claude.md` for development guidelines
- **Issues:** Check temporal alignment first (90% of problems)
- **Contact:** risk-tech@yourfirm.com

---

**Remember:** This system manages real money. When in doubt, be conservative.
