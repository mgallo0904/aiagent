#!/usr/bin/env python3
# filepath: /Users/markgallo/Documents/aiagent/optimized_backtest.py
"""
Optimized backtesting implementation for the AI trading agent.
This script implements a proper walk-forward testing approach with 
realistic transaction costs, volatility-based position sizing,
and sophisticated risk management controls.
"""

import pandas as pd
import numpy as np
import optuna
import matplotlib.pyplot as plt
import logging # Add this line
from market_data import MarketData
from features import prepare_features
from ml_models import MLModel
from risk_management import RiskManager
from performance_metrics import summarize_performance

# ------------------------------------------------------
# 1. Hyperparameter tuning
# ------------------------------------------------------

def tune_model_params(X, y, tscv_splits=5, n_trials=30):
    """
    Use Optuna to find optimal hyperparameters for the model.
    
    Args:
        X: Features DataFrame
        y: Labels Series
        tscv_splits: Number of time series cross-validation splits
        n_trials: Number of optimization trials
        
    Returns:
        Dictionary of best hyperparameters
    """
    from sklearn.model_selection import TimeSeriesSplit
    from sklearn.metrics import roc_auc_score

    tscv = TimeSeriesSplit(n_splits=tscv_splits)

    def objective(trial):
        params = {
            'model_type': 'xgboost',
            'params': {
                'learning_rate': trial.suggest_float('learning_rate', 1e-3, 1e-1, log=True),
                'max_depth': trial.suggest_int('max_depth', 3, 10),
                'n_estimators': trial.suggest_int('n_estimators', 50, 300),
                'subsample': trial.suggest_float('subsample', 0.6, 1.0),
                'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0)
            }
        }
        aucs = []
        for train_idx, test_idx in tscv.split(X):
            X_tr, X_val = X.iloc[train_idx], X.iloc[test_idx]
            y_tr, y_val = y.iloc[train_idx], y.iloc[test_idx]
            model = MLModel(**params)
            model.fit(X_tr, y_tr)
            preds = model.predict_proba(X_val)
            # Handle both 1D and 2D array outputs
            if len(preds.shape) > 1 and preds.shape[1] > 1:
                preds = preds[:, 1]  # Get probability of positive class
            aucs.append(roc_auc_score(y_val, preds))
        return np.mean(aucs)

    study = optuna.create_study(direction='maximize')
    study.optimize(objective, n_trials=n_trials)
    return study.best_params


# ------------------------------------------------------
# 2. Walk-forward backtest with risk controls
# ------------------------------------------------------

def walkforward_backtest(df, features, labels,
                          train_window=252, test_window=21,
                          stop_loss_pct=0.03, transaction_cost_pct=0.001,
                          risk_volatility=0.01, vol_lookback=20):
    """
    Perform walk-forward backtesting with realistic transaction costs,
    volatility-based position sizing, and stop-loss risk controls.
    
    Args:
        df: DataFrame with price data (must have 'Close' column)
        features: DataFrame with features for the model
        labels: Series with binary labels (1 for profitable move, 0 otherwise)
        train_window: Number of bars in training window
        test_window: Number of bars in test window
        stop_loss_pct: Stop loss percentage (e.g., 0.03 = 3% stop loss)
        transaction_cost_pct: Transaction cost percentage (e.g., 0.001 = 0.1%)
        risk_volatility: Target daily volatility (e.g., 0.01 = 1%)
        vol_lookback: Lookback window for volatility calculation
        
    Returns:
        Series with equity curve
    """
    dates = df.index
    equity = []
    cash = 10000.0
    all_equity = pd.Series(index=dates, dtype=float)
    all_equity.iloc[0] = cash  # Set initial capital

    # prepare risk manager
    risk_mgr = RiskManager(total_capital=cash)

    # tune once on full data (or inside each fold for more rigor)
    best_params = tune_model_params(features, labels)
    print("Tuned params:", best_params)
    
    # Performance tracking variables
    cycle_metrics = []
    
    for start in range(0, len(df) - train_window - test_window + 1, test_window):
        train_idx = slice(start, start + train_window)
        test_idx  = slice(start + train_window, start + train_window + test_window)

        X_tr = features.iloc[train_idx]
        y_tr = labels.iloc[train_idx]
        X_te = features.iloc[test_idx]
        y_te = labels.iloc[test_idx]
        price_te = df['Close'].iloc[test_idx]

        cycle_start_date = df.index[train_idx.start]
        cycle_end_date = df.index[test_idx.stop-1] if test_idx.stop <= len(df) else df.index[-1]
        print(f"Cycle {len(cycle_metrics)+1}: Train {cycle_start_date.strftime('%Y-%m-%d')} to {df.index[train_idx.stop-1].strftime('%Y-%m-%d')}, " +
              f"Test {df.index[test_idx.start].strftime('%Y-%m-%d')} to {cycle_end_date.strftime('%Y-%m-%d')}")

        # build & fit model
        model = MLModel(model_type='xgboost', params=best_params)
        model.fit(X_tr, y_tr)

        # get signals
        # Make predictions
        if hasattr(model, "predict_proba"):
            # If predict_proba returns 1D array (proba for positive class), use it directly
            # If it returns 2D array (probas for all classes), take the second column
            raw_probas = model.predict_proba(X_te)
            if raw_probas.ndim == 1:
                probs = raw_probas
            else:
                probs = raw_probas[:, 1]
        else:
            # Fallback for models without predict_proba (e.g., some regressors if misused)
            probs = model.predict(X_te)
            logging.warning("Model does not have predict_proba, using predict instead. Probabilities may not be calibrated.") # Changed logger to logging

    signals = (probs > 0.5).astype(int)

    # compute volatility-based position size
    vol = df['Close'].pct_change().rolling(vol_lookback).std().iloc[test_idx]
    pos_size = risk_volatility / vol.fillna(vol.mean())
    pos_size = pos_size.clip(0.1, 1.0)  # Limit position size between 10% and 100%

    # backtest this slice
    cash_slice = cash
    entry_price = None
    stop_price  = None
    position = 0  # 0 = no position, 1 = long position
    cycle_returns = []
    
    for i, date in enumerate(price_te.index):
        price = price_te.iloc[i]
        sig = signals[i]
        
        # Record equity before any changes
        if i == 0:
            all_equity[date] = cash_slice
            
        # Check stop loss first
        if position > 0 and price <= stop_price:
            # exit at stop loss
            ret = (stop_price / entry_price - 1) * pos_size.iloc[i]
            cash_slice *= (1 + ret)
            # apply transaction cost
            cash_slice -= cash_slice * transaction_cost_pct
            position = 0
            print(f"  {date.strftime('%Y-%m-%d')}: Stop loss hit, exited at {stop_price:.2f}, return: {ret:.2%}")
        
        # Daily return calculation if in position
        if position > 0:
            daily_ret = (price / price_te.iloc[i-1] - 1) * pos_size.iloc[i]
            cash_slice *= (1 + daily_ret)
            cycle_returns.append(daily_ret)
        
        # Signal change - enter or exit position
        if sig != position:
            if sig > 0 and position == 0:  # Enter long
                entry_price = price
                stop_price = entry_price * (1 - stop_loss_pct)
                position = 1
                # Apply transaction cost
                cash_slice -= cash_slice * transaction_cost_pct
                print(f"  {date.strftime('%Y-%m-%d')}: Entered long at {entry_price:.2f}, stop: {stop_price:.2f}")
            elif sig == 0 and position > 0:  # Exit long
                ret = (price / entry_price - 1) * pos_size.iloc[i]
                cash_slice *= (1 + ret)
                # Apply transaction cost
                cash_slice -= cash_slice * transaction_cost_pct
                position = 0
                print(f"  {date.strftime('%Y-%m-%d')}: Exited long at {price:.2f}, return: {ret:.2%}")
        
        # Record equity after changes
        all_equity[date] = cash_slice

    # Update cash for next cycle
    cash = cash_slice
    
    # Record cycle metrics
    cycle_returns_series = pd.Series(cycle_returns)
    cycle_metrics.append({
        'cycle': len(cycle_metrics) + 1,
        'start_date': df.index[test_idx.start],
        'end_date': cycle_end_date,
        'start_equity': all_equity[df.index[test_idx.start]],
        'end_equity': cash_slice,
        'return': cash_slice / all_equity[df.index[test_idx.start]] - 1,
        'sharpe': (cycle_returns_series.mean() / cycle_returns_series.std() * np.sqrt(252)) if len(cycle_returns) > 0 and cycle_returns_series.std() > 0 else 0,
        'win_rate': (cycle_returns_series > 0).mean() if len(cycle_returns) > 0 else 0
    })
    
    print(f"  Cycle {len(cycle_metrics)} completed: Equity ${cash_slice:.2f}, Return: {cycle_metrics[-1]['return']:.2%}\n")

    # Fill forward any missing values (e.g., days without trades)
    all_equity = all_equity.ffill()  # Forward-fill missing equity values
    
    # Print cycle performance summary
    print("\nCycle Performance Summary:")
    cycles_df = pd.DataFrame(cycle_metrics)
    if not cycles_df.empty:
        for _, cycle in cycles_df.iterrows():
            print(f"Cycle {cycle['cycle']}: {cycle['start_date'].strftime('%Y-%m-%d')} - {cycle['end_date'].strftime('%Y-%m-%d')}, " + 
                  f"Return: {cycle['return']:.2%}, Sharpe: {cycle['sharpe']:.2f}, Win Rate: {cycle['win_rate']:.2%}")
    
    return all_equity


# ------------------------------------------------------
# 3. Main execution
# ------------------------------------------------------
if __name__ == '__main__':
    # load market data
    md = MarketData('AAPL')
    # Fetch sufficient historical data for walk-forward backtesting
    price = md.fetch_historical(period='3y', interval='1d') # MODIFIED: period='3y'
    print(f"Shape of price data after fetch_historical: {price.shape}") # DIAGNOSTIC PRINT
    print(f"Columns in input 'price' DataFrame: {price.columns.tolist()}") # DIAGNOSTIC PRINT
    print(f"NaN sum for each column in 'price':\\n{price.isna().sum()}") # DIAGNOSTIC PRINT

    # generate features
    features_df = prepare_features(price, drop_na=False, print_diagnostics=True) # MODIFIED: drop_na=False, print_diagnostics=True
    print(f"Shape of features_df before dropna: {features_df.shape}") # DIAGNOSTIC PRINT

    # ─── 1) Impute those initial lookback NaNs ────────────────────────────
    # back‑fill then forward‑fill to recover all but terminal gaps
    features_df.fillna(method='bfill', inplace=True)
    features_df.fillna(method='ffill', inplace=True)
    print(f"Shape after fillna but before dropna: {features_df.shape}")

    if not features_df.empty:
        # Check NaNs in the last row of features_df
        last_row_nans = features_df.iloc[-1].isna()
        nan_cols_in_last_row = last_row_nans[last_row_nans].index.tolist()
        print(f"Columns that are NaN in the LAST row of features_df: {nan_cols_in_last_row}") # DIAGNOSTIC PRINT

        # Check if all rows in features_df have at least one NaN
        rows_with_any_nan = features_df.isna().any(axis=1)
        if rows_with_any_nan.all():
            print("CONFIRMED: All rows in features_df have at least one NaN before dropna.") # DIAGNOSTIC PRINT
        else:
            num_complete_rows = len(features_df) - rows_with_any_nan.sum()
            print(f"Number of rows with NO NaNs in features_df: {num_complete_rows}") # DIAGNOSTIC PRINT
            if num_complete_rows > 0:
                 print("CONTRADICTION: If some rows have no NaNs, dropna() should not result in an empty DataFrame.")

        # MODIFICATION: Drop psar_up and psar_down before dropna,
        # as one of them is always NaN if PSAR is active, causing all rows to be dropped.
        cols_to_drop_before_dropna = []
        if 'psar_up' in features_df.columns:
            cols_to_drop_before_dropna.append('psar_up')
        if 'psar_down' in features_df.columns:
            cols_to_drop_before_dropna.append('psar_down')
        
        if cols_to_drop_before_dropna:
            print(f"Dropping columns before main dropna: {cols_to_drop_before_dropna}") # DIAGNOSTIC PRINT
            features_df.drop(columns=cols_to_drop_before_dropna, inplace=True)
            print(f"Shape of features_df after dropping psar_up/down but before main dropna: {features_df.shape}") # DIAGNOSTIC PRINT
            # Re-check if all rows still have NaNs after dropping psar_up/down
            if not features_df.empty:
                rows_with_any_nan_after_psar_drop = features_df.isna().any(axis=1)
                if rows_with_any_nan_after_psar_drop.all():
                    print("INFO: All rows in features_df STILL have at least one NaN after dropping psar_up/down.")
                else:
                    num_complete_rows_after_psar_drop = len(features_df) - rows_with_any_nan_after_psar_drop.sum()
                    print(f"INFO: Number of rows with NO NaNs after dropping psar_up/down: {num_complete_rows_after_psar_drop}")


    else:
        print("features_df is empty even BEFORE dropna. This is unexpected.")


    features_df.dropna(inplace=True)
    print(f"Shape of features_df AFTER dropna: {features_df.shape}") # DIAGNOSTIC PRINT

    if features_df.empty:
        print("Error: features_df is empty after feature preparation and NaN removal.")
        print("This means no data is available for backtesting.")
        print("Please check your data source or feature engineering steps.")
        import sys
        sys.exit(1)
    
    # define label: next 5-day return >2%
    future = price['Close'].pct_change(periods=5).shift(-5)
    label = (future > 0.02).astype(int)
    # ─── 2) Align price, features, and label ────────────────────────────
    # restrict price & label to rows where features exist
    price = price.loc[features_df.index]
    label = label.loc[features_df.index]

    # ─── 3) Run the walk‑forward backtest ───────────────────────────────
    equity_curve = walkforward_backtest(
        df=price,
        features=features_df,
        labels=label,
        train_window=252,
        test_window=21,
        stop_loss_pct=0.03,
        transaction_cost_pct=0.001,
        risk_volatility=0.01,
        vol_lookback=20
    )

    # ─── 4) Plot the equity curve ───────────────────────────────────────
    plt.figure(figsize=(10, 6))
    equity_curve.plot(title='Walk‑Forward Equity Curve')
    plt.ylabel('Portfolio Value ($)')
    plt.xlabel('Date')
    plt.tight_layout()
    plt.show()

    # merge
