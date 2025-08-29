# Risk Management Tool for Day Traders

**Simple, conservative system for managing risk limits for 15 day traders with real money.**

This system follows the principle: *"The best model is the one that runs reliably every morning and prevents blowups, not the one with the best backtest metrics."*

## System Overview

### Problem Context
- **15 day traders** with significant capital at risk
- **Small N problem**: Not enough data for individual trader models
- **Real money constraints**: Conservative approach required, no room for experimentation
- **Daily workflow**: 5-6 hour processing window each morning
- **Goal**: Prevent large losses while minimizing unnecessary restrictions

### Architecture Philosophy
Based on quantitative finance best practices for small datasets:

1. **Rules-based baseline** - Simple, interpretable system that works
2. **Pooled ML model** - Single model for all traders (more statistical power)
3. **Conservative predictions** - Better to over-restrict than miss big losses
4. **Statistical validation** - ML must beat rules by >10% to be used
5. **Robust fallbacks** - Always revert to rules if ML fails

## Core Components

### 📁 Configuration
- **`config.py`** - Single Python config file (no JSON/YAML complexity)

### 🔧 Rules-Based System
- **`rules_based_system.py`** - Baseline system that any ML model must beat
- Simple rules any risk manager would implement:
  - Loss streak detection (3+ consecutive losses)
  - Drawdown monitoring (>15% from recent peak)
  - Volatility spike detection (>2x normal)
  - Large single-day loss identification

### 🚀 Daily Pipeline
- **`morning_pipeline.py`** - Complete daily workflow that runs at 6 AM
- Data validation → Risk prediction → Email report → Logging
- Automatic fallback to rules-based system if ML fails

### 📧 Email Integration
- **`inference/email_service.py`** - Bloomberg-style email reports (preserved)
- Clear, actionable recommendations with confidence levels
- Escalation for high-risk situations requiring immediate attention

### 💾 Data Management
- **`scripts/`** - Data ingestion and authentication (preserved)
- **`data/risk_tool.db`** - SQLite database with trader transaction data

### 🤖 Models (Optional)
- **`models/trader_specific/`** - Individual trader models (fallback only)
- Future: Pooled model implementation following CLAUDE.md guidelines

## Daily Workflow

```
06:00 - Pipeline starts
06:15 - Data quality validation
06:45 - Risk predictions generated
07:00 - Email report creation
07:30 - Risk managers receive recommendations
08:00 - Trading day begins with updated limits
```

## Usage

### Daily Production Run
```bash
python morning_pipeline.py
```

### Test Rules System
```bash
python rules_based_system.py
```

### Manual Email Send
```bash
python send_daily_signals.py
```

## Key Features

### ✅ **Designed for Small N**
- Single pooled model approach (not 15 individual models)
- Robust statistics (median/IQR over mean/std)
- Heavy regularization to prevent overfitting
- Requires statistical significance before using ML

### ✅ **Conservative by Design**
- Rules-based fallback always available
- Maximum 80% limit reduction (never completely restrict)
- Multiple confirmation signals before major restrictions
- Clear explanations for every decision

### ✅ **Production Ready**
- Comprehensive error handling and notifications
- Daily logging for monitoring and evaluation
- Simple deployment (no Docker/Kubernetes complexity)
- Minimal dependencies

### ✅ **Maintainable**
- Single config file
- Clear separation of concerns
- Extensive documentation in code
- Easy to understand and modify

## Risk Metrics

The system evaluates performance using professional risk metrics:

- **Hit Rate**: Percentage of actual large losses caught
- **Precision**: Correct restrictions / total restrictions
- **False Positive Rate**: Over-restrictions on profitable days
- **Net Economic Benefit**: Prevented losses minus opportunity costs
- **Statistical Significance**: p-value for improvement over baseline

## Configuration

Key settings in `config.py`:

```python
DEFAULT_LIMIT = 5000          # Default daily loss limit per trader
MAX_REDUCTION = 80            # Never reduce more than 80%
MIN_SAMPLES_FOR_ML = 5000     # Trader-days needed before using ML
MIN_IMPROVEMENT_FOR_ML = 0.10 # ML must beat rules by 10%
```

## Email Report Format

```
DAILY RISK LIMITS - 2024-01-09
====================

IMMEDIATE ACTION REQUIRED (>40% reduction):

Trader 3942: REDUCE LIMIT BY 50%
  New limit: $2,500 (from $5,000)
  Reasons: Extended loss streak (5 days), Drawdown 18.2%
  Confidence: high

MODERATE ADJUSTMENTS (20-40%):
Trader 3951: Reduce by 30% ($3,500) - Loss streak (3 days)

SUMMARY:
- Total traders: 15
- Restrictions applied: 4
- High risk: 1
- System: Rules-based (baseline)
```

## Development Philosophy

This system prioritizes:

1. **Reliability** over sophistication
2. **Interpretability** over accuracy
3. **Conservative approach** over optimization
4. **Simple maintenance** over feature completeness

*"With 15 traders and limited data, simple rules often beat complex models."*

## Next Steps

1. **Validate rules system** with historical data
2. **Implement pooled ML model** following CLAUDE.md guidelines
3. **A/B test** ML vs rules on subset of traders
4. **Monitor daily performance** and adjust thresholds as needed

---

*For technical details on the pooled model approach and small-N statistics, see `CLAUDE.md`.*
