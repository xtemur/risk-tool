# src/validation/time_series_cv.py
class TimeSeriesCV:
    """
    Time series cross-validation with proper gap handling
    """

    def __init__(self, n_splits: int = 5, test_size_days: int = 30, gap_days: int = 5):
        self.n_splits = n_splits
        self.test_size_days = test_size_days
        self.gap_days = gap_days

    def split(self, X: pd.DataFrame, y: pd.Series):
        """Generate time series splits"""
        pass

    def validate_model(
        self, model: BaseRiskModel, X: pd.DataFrame, y: pd.Series
    ) -> dict:
        """Perform cross-validation"""
        pass
