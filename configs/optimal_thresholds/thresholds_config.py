"""
Optimal Risk Management Thresholds
Generated: 2025-07-07T17:25:25.479717
Model: trader_specific_80pct
Validation method: Differential evolution optimization

Usage:
    from configs.optimal_thresholds.thresholds_config import TRADER_THRESHOLDS

    trader_id = '3942'
    var_threshold = TRADER_THRESHOLDS[trader_id]['var_threshold']
    loss_prob_threshold = TRADER_THRESHOLDS[trader_id]['loss_prob_threshold']
"""

TRADER_THRESHOLDS = {
    "3942": {
        "var_threshold": -3440.71,
        "loss_prob_threshold": 0.7300,
        "intervention_rate": 0.3830,
        "expected_improvement": 8630.70
    },
    "3943": {
        "var_threshold": -30972.57,
        "loss_prob_threshold": 0.0096,
        "intervention_rate": 0.0851,
        "expected_improvement": 4.30
    },
    "3946": {
        "var_threshold": -2367.76,
        "loss_prob_threshold": 0.0051,
        "intervention_rate": 0.4043,
        "expected_improvement": -995.35
    },
    "3950": {
        "var_threshold": -18362.15,
        "loss_prob_threshold": 0.9185,
        "intervention_rate": 0.2234,
        "expected_improvement": 35270.21
    },
    "3951": {
        "var_threshold": -3697.84,
        "loss_prob_threshold": 0.1733,
        "intervention_rate": 0.4787,
        "expected_improvement": 29880.11
    },
    "3956": {
        "var_threshold": -5840.88,
        "loss_prob_threshold": 0.0040,
        "intervention_rate": 0.3936,
        "expected_improvement": 25646.93
    },
    "4003": {
        "var_threshold": -2044.34,
        "loss_prob_threshold": 0.0041,
        "intervention_rate": 0.4362,
        "expected_improvement": 1713.72
    },
    "4004": {
        "var_threshold": -37262.03,
        "loss_prob_threshold": 0.0207,
        "intervention_rate": 0.2872,
        "expected_improvement": 6495.02
    },
    "5093": {
        "var_threshold": -146.67,
        "loss_prob_threshold": 0.1327,
        "intervention_rate": 0.1489,
        "expected_improvement": 897.48
    },
    "5580": {
        "var_threshold": -2967.22,
        "loss_prob_threshold": 0.0641,
        "intervention_rate": 0.2128,
        "expected_improvement": 18655.94
    }
}

# Summary statistics
OPTIMIZATION_SUMMARY = {
    'total_traders': 10,
    'positive_improvements': 9,
    'average_improvement': 12619.91,
    'total_improvement': 126199.05
}

def get_trader_thresholds(trader_id: str) -> dict:
    """Get thresholds for a specific trader"""
    return TRADER_THRESHOLDS.get(trader_id, None)

def should_intervene(trader_id: str, var_prediction: float, loss_probability: float) -> bool:
    """Check if intervention should be triggered based on thresholds"""
    thresholds = get_trader_thresholds(trader_id)
    if not thresholds:
        return False

    var_trigger = var_prediction < thresholds['var_threshold']
    prob_trigger = loss_probability > thresholds['loss_prob_threshold']

    return var_trigger or prob_trigger
