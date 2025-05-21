"""
Extensions to the ML model functionality for the AI Options Trading Agent.
This file adds P&L-based evaluation and optimal threshold finding.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
from typing import Dict, List, Optional, Tuple, Union
import logging
from sklearn.model_selection import TimeSeriesSplit

from ml_models import MLModel
from backtest import run_backtest

logger = logging.getLogger(__name__)

# Check if optuna is available
try:
    import optuna
    OPTUNA_AVAILABLE = True
except ImportError:
    OPTUNA_AVAILABLE = False
    logger.warning("Optuna not available. Using RandomizedSearchCV instead.")

def prepare_features(data):
    """
    Prepare features from market data for model training.
    
    Args:
        data: Market data DataFrame
        
    Returns:
        X, y tuple for model training
    """
    # Ensure column names are lowercase
    df = data.copy()
    df.columns = [col.lower() for col in df.columns]
    
    # Create a label column if it doesn't exist
    if 'label' not in df.columns:
        # Define prediction horizon (how many bars forward to predict)
        horizon = 3  # Look ahead 3 bars
        
        # Calculate percentage price move over the horizon
        future_price = df["close"].shift(-horizon)
        pct_change = (future_price - df["close"]) / df["close"]
        
        # Define significant move threshold (e.g., 1% move)
        threshold = 0.01  # 1% threshold
        
        # Binary classification: 
        # 1 if price increases by at least threshold% over horizon
        # 0 if price decreases by at least threshold% over horizon
        # NaN for small moves (filtered out later)
        df["label"] = np.nan
        df.loc[pct_change >= threshold, "label"] = 1  # Significant upward move
        df.loc[pct_change <= -threshold, "label"] = 0  # Significant downward move
        
    # Drop NaN values 
    df = df.dropna(subset=['label'])
    
    # Extract features and labels
    X = df.drop(columns=['label'])
    y = df['label']
    
    return X, y

logger = logging.getLogger(__name__)

# Pruned legacy backtest and walk-forward extensions.
# Use MLModel.perform_walk_forward_backtest defined in ml_models.py for backtesting.
# End of file.
