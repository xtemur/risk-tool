# Risk Tool - Development & Maintenance Workflow Guide

## 🎯 Core Principles (NEVER VIOLATE)

1. **Temporal Alignment is EVERYTHING** - Check this first when debugging
2. **Real Money = No Experiments** - Conservative decisions only
3. **Small Data Reality** - 15 traders, not 15,000
4. **Simple > Complex** - Ridge regression might be all you need

## 📋 Daily Operations

### Morning Pipeline Checklist (5:00 AM - 8:00 AM)

```bash
# 1. Activate environment
source .venv/bin/activate

# 2. Update database (5:00 AM)
python scripts/update_database.py

# 3. Run temporal alignment test (5:30 AM)
python test_temporal_alignment.py

# 4. Execute morning pipeline (6:00 AM)
python morning_pipeline.py

# 5. Verify email sent (7:30 AM)
# Check logs for confirmation
```

### Quick Health Check
```bash
# Run this anytime to verify system health
python -c "
from src.minimal_risk_system import MinimalRiskSystem
system = MinimalRiskSystem()
data = system._load_data()
print(f'Traders: {data.trader_id.nunique()}')
print(f'Records: {len(data)}')
print(f'Date range: {data.date.min()} to {data.date.max()}')
system.verify_no_leakage(data)
"
```

## 🔧 Development Workflow

### Before Making ANY Changes

1. **Always Check Temporal Alignment First**
```python
# Run this debug checklist when things go wrong
python test_temporal_alignment.py

# If test fails, check:
print("Unique prediction dates:", features['prediction_date'].nunique())
print("Any future leakage?", (features['date'] >= features['prediction_date']).any())
print("Train/test overlap?", len(set(train_dates) & set(test_dates)))
```

2. **Verify Data Quality**
```python
# Check for common issues
data = model.load_data()
print("Missing P&L:", data['pnl'].isna().sum())
print("Traders with <20 days:", (data.groupby('trader_id').size() < 20).sum())
print("Extreme values:", data['pnl'].describe())
```

### Adding New Features

⚠️ **STOP AND THINK**: With only ~7,500 trader-days total, each feature reduces statistical power.

```python
# GOOD: Simple, interpretable features
def add_feature_simple(trader_data):
    """Add only if it significantly improves model"""
    # 1. Test on rules baseline first
    baseline_score = evaluate_rules_baseline()

    # 2. Add feature
    features['new_feature'] = calculate_new_feature()

    # 3. Verify improvement > 10%
    new_score = evaluate_with_feature()
    if new_score > baseline_score * 1.1:
        print("Feature accepted")
    else:
        print("Feature rejected - insufficient improvement")
```

### Model Changes

**Decision Tree for Model Selection:**
```
Data points < 200? → Use rules only
Data points < 1000? → Use Ridge regression only
Data points < 5000? → Ridge or shallow Random Forest
Data points > 5000? → Consider RF with strict limits
NEVER use deep learning with this data size!
```

### Testing Changes

1. **Temporal Test (ALWAYS FIRST)**
```bash
python test_temporal_alignment.py
```

2. **Backtest Against Historical Data**
```python
from src.validation import backtest_simple
results = backtest_simple(predictions, actual_data)
print(f"Prevented losses: ${results['losses_prevented']:,.0f}")
print(f"False restrictions: {results['false_restrictions']}")
print(f"Missed blowups: {results['missed_blowups']}")
```

3. **Compare with Baseline**
```python
from src.validation import evaluate_model_statistical_significance
significance = evaluate_model_statistical_significance(model, X, y)
assert significance['meets_threshold'], "Model must beat baseline by 10%"
```

## 🐛 Common Issues & Solutions

### Issue: "Drawdown 2000%+"
```python
# FIX: Check drawdown calculation
def _calculate_drawdown_pct(self, trader_data):
    cumulative = trader_data['pnl'].cumsum()
    running_max = cumulative.expanding().max()

    # Use proper normalization
    if running_max.iloc[-1] > 1000:  # Meaningful peak
        drawdown = (running_max - cumulative) / running_max * 100
        return min(drawdown.iloc[-1], 100.0)  # Cap at 100%
    return 0.0
```

### Issue: "Model overfitting"
```python
# FIX: Increase regularization
ridge = RidgeCV(alphas=[1.0, 10.0, 100.0, 1000.0])  # Higher alphas

rf = RandomForestRegressor(
    max_depth=2,           # Even shallower
    min_samples_split=50,  # Need more samples
    min_samples_leaf=30    # Bigger leaves
)
```

### Issue: "Temporal leakage detected"
```python
# FIX: Ensure proper date ordering
data = data.sort_values(['trader_id', 'date'])

# Use same cutoff for ALL traders
for pred_date in unique_dates:
    historical = data[data['date'] < pred_date]  # SAME for all
    for trader_id in trader_ids:
        # Process each trader with same historical cutoff
```

## 📊 Production Monitoring

### Daily Metrics to Track
```python
# Add to morning_pipeline.py
metrics = {
    'date': datetime.now(),
    'traders_restricted': len([p for p in predictions if p > 0]),
    'high_risk_count': len([p for p in predictions if p > 40]),
    'model_used': 'ML' if system.is_trained else 'Rules',
    'total_traders': len(predictions)
}

# Log to file for tracking
with open('logs/daily_metrics.json', 'a') as f:
    json.dump(metrics, f)
    f.write('\n')
```

### Weekly Review Checklist
- [ ] Check false positive rate (over-restrictions)
- [ ] Review missed large losses
- [ ] Verify model performance vs baseline
- [ ] Check for data quality issues
- [ ] Review trader feedback on restrictions

## 🚀 Deployment Changes

### Before Deploying ANY Change

1. **Run Full Test Suite**
```bash
# Temporal alignment
python test_temporal_alignment.py

# Backtest on last 30 days
python -c "
from src.validation import backtest_simple
# Run backtest
"

# Compare with baseline
python -c "
from src.rules_baseline import RulesBasedRiskSystem
# Compare performance
"
```

2. **A/B Test Protocol**
```python
# Deploy to subset first
TEST_TRADERS = ['TRADER1', 'TRADER2', 'TRADER3']  # ~20% of traders

if trader_id in TEST_TRADERS:
    use_new_model()
else:
    use_existing_model()

# Monitor for 1 week before full rollout
```

3. **Rollback Plan**
```bash
# Always maintain previous version
cp src/pooled_risk_model.py src/pooled_risk_model.py.backup

# Quick rollback if needed
cp src/pooled_risk_model.py.backup src/pooled_risk_model.py
```

## 📝 Code Review Checklist

Before merging any PR:

- [ ] Temporal alignment test passes
- [ ] No features using future information
- [ ] Model complexity appropriate for data size
- [ ] Improvement > 10% over baseline
- [ ] Backtest shows positive results
- [ ] Code follows CLAUDE.md principles
- [ ] No individual trader models (pooled only)
- [ ] Conservative parameters used

## 🔍 Debugging Workflow

```python
# 1. Check temporal alignment
print("Dates ordered?", data.groupby('trader_id')['date'].apply(lambda x: x.is_monotonic_increasing).all())

# 2. Check feature generation
features = model.prepare_features(data)
print(f"Features shape: {features.shape}")
print(f"Unique prediction dates: {features['prediction_date'].nunique()}")

# 3. Check train/test split
train_dates = features[train_mask]['prediction_date'].unique()
test_dates = features[test_mask]['prediction_date'].unique()
print(f"Train: {min(train_dates)} to {max(train_dates)}")
print(f"Test: {min(test_dates)} to {max(test_dates)}")
print(f"Overlap: {set(train_dates) & set(test_dates)}")  # Should be empty!

# 4. Check model scores
print(f"Ridge score: {ridge_score}")
print(f"RF score: {rf_score}")
print(f"Baseline score: {baseline_score}")

# 5. Check predictions
predictions = model.predict_for_tomorrow(data)
print(f"Traders with restrictions: {sum(1 for p in predictions.values() if p > 0)}")
print(f"Average reduction: {np.mean(list(predictions.values())):.1f}%")
```

## 💡 Best Practices

### DO ✅
- Start with rules, add ML only if proven better
- Use pooled model across all traders
- Maintain temporal alignment religiously
- Test on historical data before deploying
- Keep models simple (Ridge/shallow RF)
- Document why each change improves the system
- A/B test significant changes

### DON'T ❌
- Create individual trader models (not enough data)
- Use deep learning (will overfit)
- Random train/test splits (must be temporal)
- Add complex features without validation
- Trust backtest results blindly
- Deploy without temporal alignment test
- Optimize for perfection (good enough > perfect)

## 🆘 Emergency Procedures

### If Morning Pipeline Fails
```bash
# 1. Run rules-based fallback immediately
python -c "
from src.rules_baseline import RulesBasedRiskSystem
from src.pooled_risk_model import PooledRiskModel
model = PooledRiskModel()
data = model.load_data()
rules = RulesBasedRiskSystem()
predictions = rules.get_all_predictions(data)
# Send email with predictions
"

# 2. Investigate failure
tail -n 100 logs/morning_pipeline.log

# 3. Notify team
echo "Pipeline failed - using rules baseline" | mail -s "URGENT: Risk Pipeline Issue" risk_manager@firm.com
```

### If Extreme Predictions (>70% reduction for many traders)
```python
# Sanity check predictions
if sum(1 for p in predictions.values() if p > 70) > len(predictions) * 0.3:
    print("WARNING: Extreme predictions detected")
    # Fall back to rules
    predictions = rules_system.get_all_predictions(data)
```

## 📚 Resources

- **CLAUDE.md**: Primary reference for all decisions
- **config.py**: All configuration in one place
- **test_temporal_alignment.py**: Run before any change
- **src/validation.py**: Model evaluation tools

## 🎯 Remember

> "The best model is the one that runs reliably every morning and prevents blowups, not the one with the best backtest metrics."

**When in doubt:**
1. Check temporal alignment
2. Use rules instead of ML
3. Be more conservative
4. Test on historical data
5. Ask: "Would I trust this with my own money?"
