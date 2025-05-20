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

def perform_walk_forward_backtest(
    model: MLModel,
    data: pd.DataFrame,
    train_window: int = 252,      # ~1 year of trading days
    test_window: int = 21,        # ~1 month
    **bt_kwargs                   # passed to run_backtest()
):
    """
    Rolling‑window walk‑forward: train on [t : t+train_window),
    test on [t+train_window : t+train_window+test_window).
    """
    import logging

    # ─── Make sure we never step by zero ────────────────────────────────────
    if test_window < 1:
        logging.warning(f"test_window={test_window!r} invalid; defaulting to 1")
        test_window = 1

    # ─── Build the pipeline if it hasn't been built yet ───────────────────
    # (so model.fit(...) will actually exist)
    if isinstance(model, MLModel) and getattr(model, "model", None) is None:
        logging.info("Building model pipeline before walk‐forward backtest")
        model._build_pipeline()

    all_results = []
    
    # Ensure we have at least one valid training/testing cycle
    if len(data) < train_window + test_window:
        print(f"Warning: Not enough data points (need {train_window + test_window}, got {len(data)})")
        train_window = max(1, len(data) // 2)  # Ensure at least 1 point for training
        test_window = max(1, len(data) // 4)   # Ensure at least 1 point for testing
        if test_window == 0:
            test_window = 1  # Minimum test window is 1
        print(f"Adjusted to train_window={train_window}, test_window={test_window}")

    # Ensure test_window is not zero to avoid division by zero in range
    if test_window == 0:
        test_window = 1

    valid_ranges = range(0, len(data) - train_window - test_window + 1, test_window)
    print(f"Performing {len(valid_ranges)} walk-forward cycles")
    
    for start in valid_ranges:
        print(f"  Processing cycle starting at index {start}")
        train = data.iloc[start : start + train_window]
        test  = data.iloc[start + train_window : start + train_window + test_window]
        
        print(f"  Train set: {len(train)} rows, Test set: {len(test)} rows")

        # Make sure 'close' column exists in lowercase (yfinance gives uppercase 'Close')
        if 'close' not in train.columns and 'Close' in train.columns:
            train = train.copy()
            train['close'] = train['Close']
        if 'close' not in test.columns and 'Close' in test.columns:
            test = test.copy()
            test['close'] = test['Close']

        # Skip this cycle if we don't have the required data
        if 'close' not in train.columns or 'close' not in test.columns:
            print(f"  Skipping cycle - missing 'close' column in data")
            continue

        # Create target variable: 1 if next day close is higher, 0 otherwise
        y_train = (train['close'] > train['close'].shift(1)).astype(int)
        y_train = y_train.fillna(0)  # Fill NaN for first row
        
        # Remove the 'close' column from features 
        X_train = train.drop(columns=['close'])
        if 'Close' in X_train.columns:
            X_train = X_train.drop(columns=['Close'])

        # train your model
        try:
            # support both MLModel instances and raw pipelines
            if hasattr(model, "fit"):
                model.fit(X_train, y_train)
            elif hasattr(model, "model") and hasattr(model.model, "fit"):
                model.model.fit(X_train, y_train)
            else:
                raise AttributeError("Neither model.fit nor model.model.fit is available")
            
            # Remove the 'close' column from features for prediction
            X_test = test.drop(columns=['close'])
            if 'Close' in X_test.columns:
                X_test = X_test.drop(columns=['Close'])
            
            # get raw probabilities
            if hasattr(model, 'predict_proba'):
                probs = model.predict_proba(X_test)
                if isinstance(probs, np.ndarray) and probs.ndim > 1:
                    probs = probs[:,1]  # Get probability of positive class
            else:
                # Simple prediction model returns predictions directly
                probs = model.model.predict(X_test)
                
            # Generate clear directional signals (-1 for short, 0 for neutral, 1 for long)
            # Use a more cautious approach with upper and lower thresholds
            thresh_long = 0.65   # More confident to go long (65% or higher)
            thresh_short = 0.35  # More confident to go short (35% or lower)
            
            # Default signal is 0 (neutral)
            signals = np.zeros(len(probs))
            
            # Long signal (1) if probability is above thresh_long
            signals[probs > thresh_long] = 1
            
            # Short signal (-1) if probability is below thresh_short
            signals[probs < thresh_short] = -1
            
            # Only enter trades where model has high conviction (stronger signals)
            
            # smooth signals to reduce whipsaw losses
            # require 2 consecutive same signals before trading
            smoothed_signals = np.zeros_like(signals)
            for i in range(1, len(signals)):
                if signals[i] == signals[i-1] and signals[i] != 0:
                    smoothed_signals[i] = signals[i]

            # run backtest on this fold only
            fold_results = run_backtest(
                price=test['close'],
                signals=smoothed_signals,
                **bt_kwargs
            )
            
            print(f"  Cycle completed successfully, equity: {fold_results['equity'].iloc[-1]:.2f}")
            all_results.append(fold_results)
        except Exception as e:
            print(f"  Error processing cycle: {e}")
            continue

    # Warn if no results were generated
    if not all_results:
        raise ValueError("No valid backtest results were generated. Check your data.")
        
    # concatenate equity curves end‑to‑end
    return pd.concat(all_results)

# Patch MLModel class with the walk-forward backtest method
MLModel.perform_walk_forward_backtest = perform_walk_forward_backtest

# Add a method to MLModel for fitting with parameter search and optimal threshold finding
def _fit_with_search(self, X, y):
    """
    Fit the model with hyperparameter search and find optimal threshold.
    
    Args:
        X: Feature matrix
        y: Target vector
    """
    # Store feature names for later use
    self.trained_feature_names_ = list(X.columns)
    
    # Create basic pipeline with scaler and model
    self._build_pipeline()
    
    # Determine whether to use Optuna or RandomizedSearchCV
    if self.behavior_config.use_optuna and OPTUNA_AVAILABLE:
        # Set up study
        study = optuna.create_study(direction="maximize")
        
        # Define objective function for Optuna
        def objective(trial):
            # Define hyperparameters for this trial
            params = {}
            
            # Common parameters for XGBoost
            if self.model_type == "xgboost":
                params = {
                    "classifier__n_estimators": trial.suggest_int("n_estimators", 50, 100),
                    "classifier__max_depth": trial.suggest_int("max_depth", 2, 3),
                    "classifier__learning_rate": trial.suggest_float("learning_rate", 0.01, 0.1),
                    "classifier__min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
                    "classifier__subsample": trial.suggest_float("subsample", 0.5, 1.0),
                    "classifier__colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
                    "classifier__gamma": trial.suggest_float("gamma", 0, 0.5),
                    "classifier__reg_alpha": trial.suggest_float("reg_alpha", 0, 1.0),
                    "classifier__reg_lambda": trial.suggest_float("reg_lambda", 0.1, 2.0),
                }
            
            # Set parameters to pipeline
            self.model.set_params(**params)
            
            # Use TimeSeriesSplit for evaluation
            if self.behavior_config.use_time_series_cv:
                cv = TimeSeriesSplit(n_splits=self.behavior_config.time_series_n_splits)
            else:
                cv = 5  # Standard 5-fold CV
            
            # Use P&L-based scoring if specified
            if self.behavior_config.primary_metric == "sharpe_ratio":
                # Custom evaluation using backtest engine
                scores = []
                for train_idx, val_idx in cv.split(X):
                    X_train_cv, X_val_cv = X.iloc[train_idx], X.iloc[val_idx]
                    y_train_cv, y_val_cv = y.iloc[train_idx], y.iloc[val_idx]
                    
                    # Fit on training data
                    self.model.fit(X_train_cv, y_train_cv)
                    
                    # Get probabilities on validation data
                    y_pred_proba = self.model.predict_proba(X_val_cv)[:, 1]
                    
                    # Calculate P&L-based score
                    sharpe = pnl_scoring_function(y_val_cv, y_pred_proba, X_val_cv, threshold=0.5)
                    scores.append(sharpe)
                
                return np.mean(scores)
            else:
                # Standard scikit-learn scoring
                from sklearn.model_selection import cross_val_score
                scores = cross_val_score(
                    self.model, X, y, 
                    cv=cv, 
                    scoring=self.behavior_config.primary_metric
                )
                return np.mean(scores)
        
        # Run Optuna optimization
        study.optimize(
            objective, 
            n_trials=self.behavior_config.optuna_n_trials,
            show_progress_bar=True
        )
        
        # Get best parameters
        best_params = study.best_params
        best_params_with_prefix = {f"classifier__{k}": v for k, v in best_params.items()}
        
        # Set best parameters
        self.model.set_params(**best_params_with_prefix)
    
    else:
        # Use RandomizedSearchCV
        from sklearn.model_selection import RandomizedSearchCV
        
        # Define parameter grid
        param_grid = {
            "classifier__n_estimators": [50, 75, 100],
            "classifier__max_depth": [2, 3],
            "classifier__learning_rate": [0.01, 0.05, 0.1],
            "classifier__min_child_weight": [1, 5, 10],
            "classifier__subsample": [0.8, 0.9, 1.0],
            "classifier__colsample_bytree": [0.8, 0.9, 1.0],
            "classifier__gamma": [0, 0.1, 0.2],
            "classifier__reg_alpha": [0, 0.1, 0.5],
            "classifier__reg_lambda": [0.1, 1.0, 2.0],
        }
        
        # Use TimeSeriesSplit for evaluation
        if self.behavior_config.use_time_series_cv:
            cv = TimeSeriesSplit(n_splits=self.behavior_config.time_series_n_splits)
        else:
            cv = 5  # Standard 5-fold CV
        
        # Initialize search
        search = RandomizedSearchCV(
            self.model,
            param_grid,
            n_iter=self.behavior_config.n_iter,
            cv=cv,
            scoring=self.behavior_config.primary_metric,
            random_state=self.behavior_config.random_state,
            n_jobs=-1,
        )
        
        # Fit search
        search.fit(X, y)
        
        # Get best model
        self.model = search.best_estimator_
    
    # Final fit on full dataset
    self.model.fit(X, y)
    
    # Find optimal probability threshold
    self.find_optimal_threshold(X, y)
    
    # Store scaled data for SHAP analysis
    if hasattr(self.model, "named_steps") and "scaler" in self.model.named_steps:
        self.scaled_training_data_for_shap_ = self.model.named_steps["scaler"].transform(X)
    
    return self

# Patch MLModel class with the _fit_with_search method
MLModel._fit_with_search = _fit_with_search
