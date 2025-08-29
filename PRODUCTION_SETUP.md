# Production Setup Guide

## 🎯 What You Should Do Next

Based on the analysis, here's your action plan to get the system production-ready:

### 1. **Account Filtering Decision** ⚠️ IMPORTANT
Your system found 23 traders but 11 are legacy "_OLD" accounts:
- **Current active traders: 12**
- **Legacy "_OLD" accounts: 11**

**DECISION NEEDED**: The system is now configured to exclude legacy accounts by default.
- To include legacy accounts: Set `EXCLUDE_LEGACY_ACCOUNTS = False` in config.py
- To exclude legacy accounts: Keep `EXCLUDE_LEGACY_ACCOUNTS = True` (current setting)

### 2. **Set Up Email Notifications** 📧

#### Quick Test:
```bash
# Test the system with active traders only
python -c "
from src.minimal_risk_system import MinimalRiskSystem
system = MinimalRiskSystem()
predictions = system.run_daily()
print(f'Active traders: {len(predictions)}')
"
```

#### Email Configuration:
Create a `.env` file in the project root:
```env
# Email Configuration
SMTP_SERVER=smtp.gmail.com
SMTP_PORT=587
SMTP_USERNAME=your-email@gmail.com
SMTP_PASSWORD=your-app-password
```

For Gmail:
1. Enable 2FA on your Gmail account
2. Generate an App Password: https://myaccount.google.com/apppasswords
3. Use the App Password in the .env file

### 3. **Set Up Daily Automation** 🤖

#### Install and Run:
```bash
# Make setup script executable
chmod +x setup_cron.sh

# Set up daily cron job (runs at 7 AM Monday-Friday)
./setup_cron.sh

# View cron jobs
crontab -l

# View logs
tail -f logs/morning_pipeline.log
```

### 4. **Test the Complete System** 🧪

```bash
# Test with current settings
python morning_pipeline.py

# Should show something like:
# "Filtered to 12 active traders (excluded legacy accounts)"
# "Generated predictions for 12 traders"
```

### 5. **Monitor System Performance** 📊

#### Key Metrics to Watch:
- **Total traders**: Should be 12 (active) or 23 (if including legacy)
- **Data quality**: Ensure sufficient trading history
- **Model performance**: Watch Train MAE vs Test MAE
- **Alert thresholds**: High risk traders (>40% reduction)

#### Files to Monitor:
```bash
# System logs
tail -f logs/morning_pipeline.log

# Database size
ls -lh data/risk_tool.db

# Recent predictions
ls -lt results/ 2>/dev/null || echo "No results directory yet"
```

## 📈 Expected Production Behavior

### With Active Traders Only (Recommended):
- **12 active traders**
- **Better data quality** (more recent trading)
- **More relevant risk predictions**

### With All Traders:
- **23 total traders** (12 active + 11 legacy)
- **Mixed data quality** (some very old data)
- **May include irrelevant legacy predictions**

## 🔧 Configuration Options

### config.py Settings:
```python
# Risk parameters
DEFAULT_LIMIT = 5000          # $5,000 default trading limit
MAX_REDUCTION = 80            # Never reduce more than 80%
MIN_SAMPLES_FOR_ML = 5000     # Need 5000 samples for ML

# Email
RECIPIENTS = ['temurbekkhujaev@gmail.com', 'risk_manager@firm.com']

# Account filtering
EXCLUDE_LEGACY_ACCOUNTS = True  # True = 12 traders, False = 23 traders
```

## 🚨 Alert Thresholds

### Risk Levels:
- **High Risk (>40% reduction)**: Immediate action required
- **Moderate Risk (20-40% reduction)**: Monitor closely
- **Low Risk (<20% reduction)**: Normal operations

### Sample Email Alert:
```
🚨 RISK ALERT - 3 High Risk Traders - 2025-08-29

IMMEDIATE ACTION REQUIRED (>40% reduction):
Trader NET001: REDUCE LIMIT BY 45%
  New limit: $2,750 (was $5,000)
  Reasons: 19 day loss streak, Large loss yesterday
```

## 🔄 Daily Workflow

1. **7:00 AM**: Cron job triggers morning_pipeline.py
2. **7:01 AM**: System loads latest trading data
3. **7:02 AM**: ML model makes predictions (or falls back to rules)
4. **7:03 AM**: Email sent to recipients
5. **7:05 AM**: Risk managers review and adjust limits

## ⚙️ Next Actions for You:

1. **✅ DECIDE**: Keep legacy accounts or exclude them?
2. **✅ CONFIGURE**: Set up email credentials in .env file
3. **✅ TEST**: Run `python morning_pipeline.py` to verify
4. **✅ AUTOMATE**: Run `./setup_cron.sh` for daily scheduling
5. **✅ MONITOR**: Check logs daily for first week

## 📞 Troubleshooting

### Common Issues:
```bash
# Database connection issues
python -c "import sqlite3; print(sqlite3.connect('data/risk_tool.db'))"

# Email sending issues
python -c "from src.email_service import EmailService; EmailService().send_error_alert('Test')"

# Cron job not running
# Check: crontab -l
# Logs: tail -f logs/morning_pipeline.log
```

The system is now production-ready! 🎉
