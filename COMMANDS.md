# Risk-Tool Development Workflow Commands

## Setup Commands
```bash
# Initial setup
conda env create -f environment.yml
conda activate risk-tool
python main.py --setup

# Create .env file
touch .env
# Add: EMAIL_FROM=your-gmail@gmail.com
# Add: EMAIL_PASSWORD=your-gmail-app-password
```

## Data Pipeline
```bash
# Download data
python download_totals.py

# Test data loading
python -c "from src.data_loader import DataLoader; dl = DataLoader(); data = dl.load_all_traders_data(); print(f'Loaded {len(data)} traders')"

# Test feature engineering
python -c "from src.data_loader import DataLoader; from src.feature_engineering import FeatureEngineer; import yaml; config = yaml.safe_load(open('config/config.yaml')); dl = DataLoader(); fe = FeatureEngineer(config); data = dl.load_all_traders_data(); master_df = dl.create_master_dataset(data); features_df = fe.engineer_features(master_df); print(f'Features: {features_df.shape}')"
```

## Model Training & Testing
```bash
# Train models
python main.py --train

# Check training results
ls data/models/
cat data/models/evaluation_results.yaml

# Test prediction
python main.py --predict

# Check predictions
ls data/predictions/
head data/predictions/predictions_*.csv
```

## Email Testing
```bash
# Test email service (no sending)
python -c "from src.email_service import EmailService; import yaml; config = yaml.safe_load(open('config/config.yaml')); email_service = EmailService(config); print('Email service OK')"

# Send test email
python main.py --predict
```

## Production
```bash
# Start scheduler (production)
python main.py --schedule

# Manual operations
python scripts/daily_prediction.py    # Manual daily run
python scripts/weekly_retrain.py     # Manual retrain
```

## Debugging
```bash
# Check logs
tail -f logs/daily_prediction.log
tail -f logs/weekly_retrain.log
tail -f logs/data_loader.log

# Quick data verification
python -c "import pandas as pd; df = pd.read_csv('data/raw/TRADER001_totals.csv'); print(f'Shape: {df.shape}'); print(df.head())"

# Test individual components
python -c "from src.data_loader import DataLoader; print('DataLoader: OK')"
python -c "from src.predictor import RiskPredictor; print('Predictor: OK')"
```

## File Structure Check
```bash
# Verify structure
tree data/
ls config/
ls src/
ls scripts/

# Check required files
ls data/raw/*_totals.csv
ls data/raw/*_fills.csv
ls data/models/*.pkl
```

## Development Shortcuts
```bash
# Full pipeline test
python download_totals.py && python main.py --train && python main.py --predict

# Quick model retrain
python scripts/weekly_retrain.py

# Data quality check
python -c "from download_totals import PropreportsDownloader; pd = PropreportsDownloader('YOUR_TOKEN'); pd.verify_data_quality()"
```

## Configuration Updates
```bash
# Edit trader accounts
nano config/trader_accounts.yaml

# Edit main config
nano config/config.yaml

# Check config loading
python -c "import yaml; print(yaml.safe_load(open('config/config.yaml')))"
```

## Emergency Commands
```bash
# Stop scheduler
pkill -f "python main.py --schedule"

# Clean and restart
rm -rf data/models/* data/predictions/*
python main.py --train

# Reset environment
conda deactivate
conda remove --name risk-tool --all
conda env create -f environment.yml
```