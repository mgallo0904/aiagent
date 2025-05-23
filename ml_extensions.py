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

def prepare_features(data, n_forward: int = 3, threshold: float = 0.01):
    """
    Prepare features and ternary labels from market data for model training.
    
    Args:
        data: Market data DataFrame. Expected to have a 'close' column.
        n_forward: Number of time steps forward to calculate future return for labeling.
        threshold: Percentage change threshold for defining up/down/neutral moves.
                   e.g., 0.01 means a 1% change.
        
    Returns:
        X, y tuple for model training. 'y' will have labels:
           1: Price increase > threshold
          -1: Price decrease < -threshold
           0: Price change between -threshold and threshold (inclusive)
    """
    # Ensure column names are lowercase
    df = data.copy()
    df.columns = [col.lower() for col in df.columns]
    
    # Create a label column
    if 'label' in df.columns:
        logger.warning("Existing 'label' column found and will be overwritten.")

    # Calculate percentage price move over the n_forward horizon
    future_price = df["close"].shift(-n_forward)
    pct_change = (future_price - df["close"]) / df["close"]
    
    # Ternary classification:
    #  1: Significant upward move (price increases by more than threshold)
    # -1: Significant downward move (price decreases by more than threshold)
    #  0: No significant change (price move is within +/- threshold)
    
    df["label"] = 0  # Default to 0 (no significant change)
    df.loc[pct_change > threshold, "label"] = 1    # Up move
    df.loc[pct_change < -threshold, "label"] = -1  # Down move
        
    # Drop rows where future_price is NaN (due to shift at the end of the DataFrame)
    # These rows cannot have a valid label.
    df = df.dropna(subset=['close', future_price.name]) # Ensure close and future price were not NaN to begin with for pct_change calc
    df = df.dropna(subset=[pct_change.name]) # Drop rows where pct_change itself is NaN (mostly affects end of series)
                                         # This also effectively removes rows where 'label' could not be determined.
    
    # Extract features and labels
    # Ensure 'label' is not in X. If other columns are purely for labeling (e.g. future_price, pct_change), drop them too.
    # For now, assuming all other columns in df (after lowercasing) are features.
    # If 'future_price' or 'pct_change' were added to df, they should be dropped from X.
    # Let's explicitly drop them if they exist to be safe.
    features_to_drop = ['label']
    if future_price.name in df.columns: # future_price.name is usually 'close' if not renamed
        # Let's be more specific if pct_change was added to df
        # pct_change.name will be 'close' if future_price.name was 'close'.
        # It's better to name the intermediate series explicitly.
        # df['future_price_temp'] = future_price
        # df['pct_change_temp'] = (df['future_price_temp'] - df["close"]) / df["close"]
        # Then drop 'future_price_temp' and 'pct_change_temp'.
        # For now, assuming no intermediate columns were added with those specific names.
        pass

    X = df.drop(columns=features_to_drop, errors='ignore')
    y = df['label']
    
    return X, y

logger = logging.getLogger(__name__)

# Pruned legacy backtest and walk-forward extensions.
# Use MLModel.perform_walk_forward_backtest defined in ml_models.py for backtesting.
# End of file.
