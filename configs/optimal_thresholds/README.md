# Optimal Risk Management Thresholds

This directory contains the optimized risk management thresholds for each trader, generated using ML model predictions and differential evolution optimization.

## Files Overview

- **`optimal_thresholds.json`** - Complete threshold data with metadata in JSON format
- **`optimal_thresholds.csv`** - Threshold data in CSV format for Excel/analysis
- **`optimal_thresholds.yaml`** - Configuration file in YAML format
- **`thresholds_config.py`** - Ready-to-use Python module with helper functions

## How to Use

### Python Implementation
```python
from configs.optimal_thresholds.thresholds_config import should_intervene, get_trader_thresholds

# Check if intervention is needed
trader_id = '3942'
var_prediction = -4000.0  # Predicted VaR
loss_probability = 0.15   # Predicted loss probability

if should_intervene(trader_id, var_prediction, loss_probability):
    # Reduce position size by 50%
    print("Risk intervention triggered!")

# Get specific thresholds
thresholds = get_trader_thresholds(trader_id)
print(f"VaR threshold: {thresholds['var_threshold']}")
print(f"Loss prob threshold: {thresholds['loss_prob_threshold']}")
```

### YAML Configuration
```python
import yaml

with open('configs/optimal_thresholds/optimal_thresholds.yaml', 'r') as f:
    config = yaml.safe_load(f)

trader_thresholds = config['trader_thresholds']['3942']
```

## Threshold Interpretation

1. **VaR Threshold**: Trigger intervention when predicted VaR is **below** this value (more negative = higher risk)
2. **Loss Probability Threshold**: Trigger intervention when loss probability is **above** this value
3. **Intervention**: Reduce position size by 50% when **either** condition is met

## Optimization Results Summary

- **Total Traders**: 10
- **Positive Improvements**: 9/10 traders
- **Average Improvement**: $12,619.91 (65.8%)
- **Total Portfolio Improvement**: $126,199.05

## Performance by Trader

| Trader | VaR Threshold | Loss Prob Threshold | Test Improvement |
|--------|---------------|-------------------|------------------|
| 3942   | -$3,441       | 73.0%             | +$8,631 (337%)   |
| 3943   | -$30,973      | 0.96%             | +$4 (0%)         |
| 3946   | -$2,368       | 0.51%             | -$995 (-10%)     |
| 3950   | -$18,362      | 91.85%            | +$35,270 (44%)   |
| 3951   | -$3,698       | 17.33%            | +$29,880 (42%)   |
| 3956   | -$5,841       | 0.40%             | +$25,647 (22%)   |
| 4003   | -$2,044       | 0.41%             | +$1,714 (18%)    |
| 4004   | -$37,262      | 2.07%             | +$6,495 (113%)   |
| 5093   | -$147         | 13.27%            | +$897 (51%)      |
| 5580   | -$2,967       | 6.41%             | +$18,656 (41%)   |

## Model Information

- **Model Version**: trader_specific_80pct
- **Optimization Method**: Differential evolution
- **Validation Period**: 80% of training data
- **Test Period**: Completely unseen data (2025-01-17 to 2025-07-01)
- **Generated**: {current_date}

## Implementation Notes

1. **Monitor Performance**: Re-optimize monthly or when market conditions change significantly
2. **Intervention Rate**: Aim for 15-30% intervention rate for optimal balance
3. **Position Reduction**: Start with 50% reduction, adjust based on performance
4. **Risk Management**: Always combine with other risk controls and human oversight
