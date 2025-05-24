# src/data/data_loader.py
class DataLoader:
    """
    Centralized data loading with validation and caching
    """

    def __init__(self, config_path: str):
        self.config = load_config(config_path)
        self.cache = {}

    def load_trader_data(
        self, trader_id: str, start_date: str = None, end_date: str = None
    ):
        """Load totals and fills for specific trader"""
        pass

    def load_all_traders(self, start_date: str = None, end_date: str = None):
        """Load data for all available traders"""
        pass

    def validate_data_integrity(self, df: pd.DataFrame) -> bool:
        """Validate data quality and completeness"""
        pass
