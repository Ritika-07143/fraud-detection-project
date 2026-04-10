"""
Fast Feature Engineering
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest
import warnings
warnings.filterwarnings('ignore')

class FastFeatureEngineer:
    """Lightweight feature engineering"""
    
    def __init__(self):
        pass
    
    def create_basic_features(self, df):
        """Create only essential features (fast)"""
        df_copy = df.copy()
        
        # 1. Amount features
        df_copy['amount_log'] = np.log1p(df_copy['Amount'])
        df_copy['amount_zscore'] = (df_copy['Amount'] - df_copy['Amount'].mean()) / df_copy['Amount'].std()
        
        # 2. Time features
        df_copy['time_hour'] = (df_copy['Time'] // 3600) % 24
        df_copy['is_night'] = ((df_copy['time_hour'] >= 22) | (df_copy['time_hour'] <= 6)).astype(int)
        
        # 3. V-feature stats (only mean & std)
        v_features = [col for col in df_copy.columns if col.startswith('V')]
        df_copy['v_mean'] = df_copy[v_features].mean(axis=1)
        df_copy['v_std'] = df_copy[v_features].std(axis=1)
        
        return df_copy
    
    def engineer_features(self, df):
        """Apply feature engineering"""
        return self.create_basic_features(df)