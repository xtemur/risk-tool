import argparse
import logging
import sys
from pathlib import Path
import yaml
from datetime import datetime

from src.model_training import run_expanding_window_training
from src.data_processing import create_trader_day_panel
from src.feature_engineering import build_features

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def train_models(config_path: str = 'configs/main_config.yaml'):
    """
    Main training function that implements expanding window methodology
    """
    logger.info("Starting model training with expanding window approach")

    try:
        results = run_expanding_window_training(sequence_length=7)

        if results is not None:
            logger.info("Training completed successfully")

            from sklearn.metrics import roc_auc_score, average_precision_score
            overall_auc = roc_auc_score(results['actual'], results['prediction'])
            overall_ap = average_precision_score(results['actual'], results['prediction'])

            logger.info(f"Overall AUC: {overall_auc:.4f}")
            logger.info(f"Overall AP: {overall_ap:.4f}")
            logger.info(f"Total predictions: {len(results)}")
        else:
            logger.error("Training failed - no results returned")

        return results

    except Exception as e:
        logger.error(f"Error during training: {str(e)}")
        raise


def backtest(config_path: str = 'configs/main_config.yaml'):
    """
    Run backtesting with expanding window validation
    """
    logger.info("Running backtesting mode")

    try:
        results = run_expanding_window_training(sequence_length=7)

        if results is not None:
            from sklearn.metrics import roc_auc_score, classification_report

            logger.info("\nBacktest Results:")
            for trader_id in results['trader_id'].unique():
                trader_preds = results[results['trader_id'] == trader_id]

                if len(trader_preds['actual'].unique()) > 1:
                    auc = roc_auc_score(trader_preds['actual'], trader_preds['prediction'])
                    logger.info(f"Trader {trader_id} - Test AUC: {auc:.4f}")
                else:
                    logger.info(f"Trader {trader_id} - Insufficient test data for AUC calculation")

            overall_auc = roc_auc_score(results['actual'], results['prediction'])
            logger.info(f"\nOverall Test AUC: {overall_auc:.4f}")
        else:
            logger.error("Backtesting failed - no results returned")

        return results

    except Exception as e:
        logger.error(f"Error during backtesting: {str(e)}")
        raise


def generate_signals(config_path: str = 'configs/main_config.yaml'):
    """
    Generate trading signals using trained models
    """
    logger.info("Generating signals with latest models")

    try:
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)

        df = create_trader_day_panel(config)
        df = df.rename(columns={'account_id': 'trader_id', 'trade_date': 'date'})

        # Need to rename back for feature engineering
        df_for_features = df.rename(columns={'trader_id': 'account_id', 'date': 'trade_date'})
        df_for_features = build_features(df_for_features, config)
        df = df_for_features.rename(columns={'account_id': 'trader_id', 'trade_date': 'date'})

        latest_date = df['date'].max()
        latest_data = df[df['date'] == latest_date]

        models_dir = Path('models/expanding_window')
        signals = []

        from src.model_training import FeaturePreparer
        preparer = FeaturePreparer(sequence_length=7)

        for trader_id in latest_data['trader_id'].unique():
            model_files = list(models_dir.glob(f"{trader_id}_final.pkl"))

            if not model_files:
                logger.warning(f"No model found for trader {trader_id}")
                continue

            import pickle
            with open(model_files[0], 'rb') as f:
                model_data = pickle.load(f)

            model = model_data['model']

            # Prepare features for this trader
            trader_data = df[df['trader_id'] == trader_id].sort_values('date')

            # Get latest features
            features, _ = preparer.prepare_features(trader_data, trader_id)
            if features is not None and len(features) > 0:
                latest_features = features.iloc[-1:] # Get last row
                prediction = model.predict_proba(latest_features)[0, 1] # Probability of positive class
            else:
                logger.warning(f"Could not prepare features for trader {trader_id}")
                prediction = 0.0

            signals.append({
                'trader_id': trader_id,
                'date': latest_date,
                'risk_score': prediction,
                'risk_flag': prediction > 0.5
            })

        import pandas as pd
        signals_df = pd.DataFrame(signals)

        output_path = Path('models/expanding_window/latest_signals.csv')
        signals_df.to_csv(output_path, index=False)

        logger.info(f"Signals generated and saved to: {output_path}")
        logger.info(f"High risk traders: {sum(signals_df['risk_flag'])}")

        return signals_df

    except Exception as e:
        logger.error(f"Error generating signals: {str(e)}")
        raise


def main():
    parser = argparse.ArgumentParser(description='Risk Tool - Expanding Window Model Training')
    parser.add_argument('--mode', type=str, choices=['train', 'backtest', 'signals'],
                       default='train', help='Execution mode')
    parser.add_argument('--config', type=str, default='configs/main_config.yaml',
                       help='Path to configuration file')

    args = parser.parse_args()

    logger.info(f"Running in {args.mode} mode")

    if args.mode == 'train':
        train_models(args.config)
    elif args.mode == 'backtest':
        backtest(args.config)
    elif args.mode == 'signals':
        generate_signals(args.config)
    else:
        logger.error(f"Unknown mode: {args.mode}")
        sys.exit(1)


if __name__ == "__main__":
    main()
