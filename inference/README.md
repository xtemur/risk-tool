# Risk Signal Email Service

This module generates and sends daily risk signals for active traders via email.

## Features

- **Clean, professional HTML emails** with minimal, readable design
- **Automated signal generation** based on trained VaR and loss models
- **Risk classification** (High/Medium/Low) with subtle color indicators
- **Critical alerts** for immediate attention
- **Warning signals** including revenge trading, high volatility, etc.
- **Responsive design** that works on desktop and mobile
- **Minimal color palette** - easy on the eyes, professional appearance

## Setup

1. **Install dependencies** (if not already installed):
   ```bash
   pip install jinja2 python-dotenv
   ```

2. **Configure email credentials**:
   ```bash
   cp .env.example .env
   # Edit .env with your email credentials
   ```

   Required environment variables:
   - `EMAIL_FROM`: Your Gmail address
   - `EMAIL_PASSWORD`: Your Gmail app password
   - `EMAIL_RECIPIENTS`: Comma-separated list of default recipients

3. **For Gmail users**:
   - Enable 2-factor authentication
   - Generate app password at: https://myaccount.google.com/apppasswords
   - Use app password as EMAIL_PASSWORD

## Usage

### Generate and save signals (no email):
```bash
python send_daily_signals.py --save-only
```

### Send to default recipients (from .env):
```bash
python send_daily_signals.py
```

### Send to specific email recipients:
```bash
python send_daily_signals.py --email recipient1@example.com --email recipient2@example.com
```

### Generate for specific date:
```bash
python send_daily_signals.py --date 2025-06-20 --save-only
```

### Local file generation (no email):
```bash
# Generate HTML file only
python send_daily_signals_local.py

# Generate and open in browser
python send_daily_signals_local.py --open-browser
```

## Signal Interpretation

### Risk Levels
- **HIGH**: Loss probability > 70% or extreme volatility with medium loss probability
- **MEDIUM**: Loss probability 40-70% or elevated volatility
- **LOW**: Normal trading conditions

### Warning Signals
- **REVENGE_TRADING**: Increased trading after recent losses
- **HIGH_VOLATILITY**: Rolling 7-day volatility exceeds threshold
- **LOW_WIN_RATE**: Win rate below 30% over 21 days
- **LARGE_DRAWDOWN**: Significant peak-to-trough decline
- **ELEVATED_RISK**: Loss probability above 40%

### VaR (Value at Risk)
- 5th percentile of predicted PnL distribution
- Example: VaR of -$5,000 means 95% confidence of not losing more than $5,000

### Loss Probability
- Probability of experiencing a "large loss" (bottom 15% of historical losses)
- Trader-specific thresholds based on their historical performance

## Output

- HTML files saved to: `inference/outputs/`
- File naming: `risk_signals_[DATE]_[TIMESTAMP].html`

## Scheduling

For daily automated reports, add to crontab:
```bash
# Run at 6:00 AM every weekday
0 6 * * 1-5 cd /path/to/risk-tool && /path/to/python send_daily_signals.py --email team@example.com
```

## Customization

### Modify risk thresholds:
Edit `inference/signal_generator.py`:
- `high_loss_prob` threshold (default: 0.7)
- `medium_loss_prob` threshold (default: 0.4)
- Volatility thresholds

### Customize email template:
Edit `inference/templates/daily_signals.html` for styling and layout changes.
