#!/bin/bash

# Initial Setup Script for Risk Tool
# This script sets up data and trains models for first-time deployment

echo "=== Risk Tool Initial Setup ==="
echo "Starting at $(date)"

# Step 1: Authenticate with API
echo ""
echo "Step 1: Authenticating with API..."
docker exec risk-tool python scripts/authenticate.py

# Step 2: Save accounts data
echo ""
echo "Step 2: Fetching accounts data..."
docker exec risk-tool python scripts/save_accounts.py

# Step 3: Save trades data (last 90 days for initial training)
echo ""
echo "Step 3: Fetching trades data (last 90 days)..."
docker exec risk-tool python scripts/save_trades.py --days 90

# Step 4: Run backtest mode to train models
echo ""
echo "Step 4: Training risk models..."
docker exec risk-tool python main.py --mode backtest

# Step 5: Run validation mode
echo ""
echo "Step 5: Validating models..."
docker exec risk-tool python main.py --mode validate

# Step 6: Test signal generation
echo ""
echo "Step 6: Testing signal generation..."
docker exec risk-tool python send_daily_signals.py

echo ""
echo "=== Setup Complete ==="
echo "Finished at $(date)"
echo ""
echo "Next steps:"
echo "- Signals will be sent automatically daily at 8:00 AM UTC"
echo "- Access dashboard at http://195.158.30.167:8501"
echo "- Check logs in ~/risk-tool/logs/"
