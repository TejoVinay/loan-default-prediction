# src/data/data_processor.py

import pandas as pd
import numpy as np
from typing import Tuple, Optional

class LoanDataProcessor:
    """
    Class to handle loan data loading and processing.
    """
    def __init__(self):
        # Feature groupings will be finalized after EDA
        self.numerical_features = []
        self.categorical_features = []
        self.percentage_features = []

    def load_data(self, filepath: str) -> pd.DataFrame:
        """
        Load the loan data without any preprocessing.
        
        Args:
            filepath: Path to the CSV file
            
        Returns:
            Raw DataFrame for initial analysis
        """
        return pd.read_csv(filepath)