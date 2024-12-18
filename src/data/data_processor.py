# src/data/data_processor.py

import pandas as pd
import numpy as np
from typing import Tuple, Optional

class LoanDataProcessor:
    """
    Class to handle loan data processing, cleaning, and transformation.
    """
    def __init__(self):
        self.numerical_features = [
            'loan_amnt', 'annual_inc', 'inq_last_6mths',
            'total_bc_limit', 'tot_hi_cred_lim', 'tot_cur_bal',
            'internal_score'
        ]
        
        self.percentage_features = [
            'int_rate', 'revol_util', 'bc_util', 
            'percent_bc_gt_75', 'dti'
        ]
        
        self.categorical_features = [
            'term', 'home_ownership', 'purpose'
        ]

    def _clean_percentage(self, value: str) -> float:
        """Convert percentage string to float."""
        if pd.isna(value):
            return np.nan
        if isinstance(value, (int, float)):
            return value / 100 if value > 1 else value
        return float(str(value).strip('%').strip()) / 100

    def _clean_emp_length(self, value: str) -> int:
        """Convert employment length to years (0-10)."""
        if pd.isna(value):
            return np.nan
        value = str(value).lower().strip()
        if '10+' in value:
            return 10
        elif '< 1' in value:
            return 0
        return int(value.split()[0])

    def preprocess_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Preprocess the loan data with all cleaning steps.
        """
        # Create copy to avoid modifying original
        data = df.copy()
        
        # Create binary flags for high-missing columns
        data['has_major_derog'] = data['mths_since_last_major_derog'].notna().astype(int)
        data['has_description'] = data['desc'].notna().astype(int)
        
        # Handle mths_since_last_major_derog
        # Fill with -1 to indicate no history of major derogatory marks
        data['mths_since_last_major_derog'] = data['mths_since_last_major_derog'].fillna(-1)
        
        # Handle percentage features
        for col in self.percentage_features:
            data[col] = data[col].apply(self._clean_percentage)
        
        # Convert emp_length to numerical
        data['emp_length'] = data['emp_length'].apply(self._clean_emp_length)
        
        # Convert numerical features
        for col in self.numerical_features:
            data[col] = pd.to_numeric(data[col], errors='coerce')
        
        # Handle missing values for mths_since_recent_inq
        # Fill with -1 to indicate no recent inquiries
        data['mths_since_recent_inq'] = data['mths_since_recent_inq'].fillna(-1)
        
        # Handle remaining missing values with median for numerical features
        numerical_cols = self.numerical_features + ['emp_length']
        data[numerical_cols] = data[numerical_cols].fillna(data[numerical_cols].median())
        
        # Handle missing values in percentage features with median
        data[self.percentage_features] = data[self.percentage_features].fillna(
            data[self.percentage_features].median()
        )
        
        # Fill categorical missing values with mode
        data[self.categorical_features] = data[self.categorical_features].fillna(
            data[self.categorical_features].mode().iloc[0]
        )
        
        # Drop unnecessary columns
        columns_to_drop = ['id', 'member_id', 'desc', 'application_approved_flag']
        data = data.drop(columns=columns_to_drop)
        
        return data

    def load_and_preprocess(self, filepath: str) -> pd.DataFrame:
        """Load and preprocess the data in one step."""
        df = pd.read_csv(filepath)
        return self.preprocess_data(df)