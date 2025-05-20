#!/usr/bin/env python3
"""
AI Options Trading Agent Launcher
This script provides a clean way to launch the application with optimized settings for Apple Silicon.
"""

import os
import sys
import argparse
from typing import Dict, Any


def setup_environment(config: Dict[str, Any]) -> None:
    """
    Set up environment variables for optimal performance on Apple Silicon

    Args:
        config: Configuration dictionary
    """
    # Set number of threads for better performance on Apple Silicon
    os.environ["OMP_NUM_THREADS"] = str(config["ML_NUM_THREADS"])
    os.environ["OPENBLAS_NUM_THREADS"] = str(config["ML_NUM_THREADS"])
    os.environ["MKL_NUM_THREADS"] = str(config["ML_NUM_THREADS"])

    # Enable Metal acceleration for TensorFlow if requested
    if config["USE_METAL_ACCELERATION"]:
        os.environ["TENSORFLOW_METAL_ACCELERATION"] = "1"


def run_walk_forward_backtest(
    symbol: str,
    period: str = "2y",
    interval: str = "1d",
    n_splits: int = 10,
    proba_threshold: float = 0.5,
) -> None:
    """
    Run a walk-forward backtest simulation.
    
    Args:
        symbol: Stock symbol to test
        period: Data period to fetch (e.g., 2y)
        interval: Data interval to use (e.g., 1d)
        n_splits: Number of splits for walk-forward testing
        proba_threshold: Probability threshold for ML model
    """
    import pandas as pd
    import matplotlib.pyplot as plt
    from matplotlib.dates import DateFormatter
    import os
    import numpy as np
    from datetime import datetime
    from sklearn.model_selection import TimeSeriesSplit
    
    from market_data import MarketData
    from ml_models import MLModel, ModelBehaviorConfig
    from features import add_ta_features
    
    print(f"Starting walk-forward backtest for {symbol}")
    print(f"Data parameters: period={period}, interval={interval}, splits={n_splits}")
    
    # Initialize market data
    market = MarketData(symbol=symbol)
    
    # Fetch historical data
    print(f"Fetching historical data for {symbol}...")
    data = market.fetch_historical(period=period, interval=interval)
    
    if data is None or data.empty:
        print(f"Error: Could not fetch historical data for {symbol}")
        return
    
    # Ensure we have sufficient data for the test
    min_required_samples = n_splits * 5  # At least 5 samples per split
    if len(data) < min_required_samples:
        print(f"Warning: Not enough data points ({len(data)}) for reliable walk-forward testing.")
        print(f"Consider using a longer period or fewer splits. Using 2 splits instead.")
        n_splits = 2
    
    print(f"Historical data fetched: {len(data)} rows")
    
    # Ensure column names are lowercase
    data.columns = [col.lower() for col in data.columns]
    
    # Add technical indicators
    data = add_ta_features(data)
    
    # Generate labels with horizon and threshold
    horizon = 3  # Look ahead 3 bars
    threshold = 0.01  # 1% threshold for meaningful price move
    
    # Calculate future price and percent change
    future_price = data["close"].shift(-horizon)
    pct_change = (future_price - data["close"]) / data["close"]
    
    # Create binary labels for significant moves
    data["label"] = np.nan
    data.loc[pct_change >= threshold, "label"] = 1  # Significant upward move
    data.loc[pct_change <= -threshold, "label"] = 0  # Significant downward move
    
    # Drop NaN values from features and labels
    data = data.dropna()
    
    print(f"Data after feature generation and filtering: {len(data)} rows")
    
    # Configure ML model with simplified hyperparameters
    behavior_config = ModelBehaviorConfig(
        proba_threshold=proba_threshold,
        early_stopping_rounds=15,
        random_state=42,
        primary_metric="f1",  # F1 score for balanced precision/recall
        n_iter=10,  # Keep hyperparameter search minimal
        use_time_series_cv=True,
        time_series_n_splits=2,  # Use small number of folds for CV
        calibrate_probabilities=True,
        calibration_method="sigmoid",
        use_optuna=False,  # Disable Optuna for simpler search
        optuna_n_trials=10,  # Minimal trials
    )
    
    # Set up simplified model parameters
    model_params = {
        "n_estimators": 50,  # Reduced from higher values
        "max_depth": 3,      # Limit tree depth
        "learning_rate": 0.1,
        "random_state": 42,
        "gamma": 0.1,
    }
    
    # Initialize ML model
    model = MLModel(model_type="xgboost", params=model_params, behavior_config=behavior_config)
    
    # Create TimeSeriesSplit for walk-forward testing
    tscv = TimeSeriesSplit(n_splits=n_splits)
    
    # Prepare X and y
    X = data.drop(columns=["label"])
    y = data["label"]
    
    # Store results from each fold
    results_data = []
    
    # For each fold
    for fold, (train_idx, test_idx) in enumerate(tscv.split(X)):
        print(f"Processing fold {fold+1}/{n_splits}")
        
        # Split data
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
        
        print(f"  Training with {len(X_train)} samples, testing on {len(X_test)} samples")
        
        # Train model on this fold's training data
        model = MLModel(model_type="xgboost", params=model_params, behavior_config=behavior_config)
        model.model.fit(X_train, y_train)
        
        # Find optimal threshold on training set
        if hasattr(model, "find_optimal_threshold"):
            optimal_threshold = model.find_optimal_threshold(X_train, y_train)
            print(f"  Optimal threshold: {optimal_threshold:.3f}")
        
        # Get predictions for test set
        if hasattr(model.model, "predict_proba"):
            y_pred_proba = model.model.predict_proba(X_test)[:, 1]
        else:
            # If model doesn't support probability predictions, use binary predictions
            y_pred = model.model.predict(X_test)
            y_pred_proba = y_pred.astype(float)
            
        y_pred = (y_pred_proba >= proba_threshold).astype(int)
        
        # Store results
        for i in range(len(X_test)):
            row = {
                'date': X_test.index[i],
                'fold': fold,
                'actual': y_test.iloc[i],
                'predicted': y_pred[i],
                'probability': y_pred_proba[i]
            }
            results_data.append(row)
    
    # Convert results to DataFrame
    results = pd.DataFrame(results_data)
    
    if len(results) == 0:
        print("Error: No valid backtest results produced")
        return
    
    # Calculate performance metrics
    accuracy = (results['actual'] == results['predicted']).mean()
    
    # Save results to CSV
    os.makedirs("data", exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_filename = f"data/walkforward_{symbol}_{timestamp}.csv"
    results.to_csv(csv_filename)
    print(f"Results saved to {csv_filename}")
    
    # Plot results
    plt.figure(figsize=(12, 8))
    
    # Plot 1: Actual vs Predicted
    plt.subplot(2, 1, 1)
    plt.plot(results['date'], results['actual'], 'b-', label='Actual')
    plt.plot(results['date'], results['predicted'], 'r--', label='Predicted')
    plt.title(f'Walk-Forward Backtest Results for {symbol}')
    plt.ylabel('Value (1=Up, 0=Down)')
    plt.legend()
    plt.grid(True)
    
    # Plot 2: Prediction Probabilities
    plt.subplot(2, 1, 2)
    plt.plot(results['date'], results['probability'], 'g-', label='Probability')
    plt.axhline(y=proba_threshold, color='r', linestyle='--', label=f'Threshold ({proba_threshold})')
    plt.xlabel('Date')
    plt.ylabel('Probability')
    plt.legend()
    plt.grid(True)
    
    # Format dates nicely
    date_format = DateFormatter('%Y-%m-%d')
    plt.gca().xaxis.set_major_formatter(date_format)
    plt.gcf().autofmt_xdate()
    
    # Save figure
    plt_filename = f"data/walkforward_{symbol}_{timestamp}.png"
    plt.savefig(plt_filename)
    print(f"Plot saved to {plt_filename}")
    
    # Calculate overall metrics
    accuracy = (results['actual'] == results['predicted']).mean()
    print(f"\nOverall accuracy: {accuracy:.4f}")
    
    # Threshold analysis
    print("\nThreshold analysis:")
    for threshold in [0.3, 0.4, 0.5, 0.6, 0.7]:
        predicted_at_threshold = (results['probability'] >= threshold).astype(int)
        acc_at_threshold = (results['actual'] == predicted_at_threshold).mean()
        print(f"  Threshold {threshold:.1f}: Accuracy = {acc_at_threshold:.4f}")


def main() -> None:
    """Run the AI Trading Agent with optimized settings"""
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="AI Options Trading Agent")
    parser.add_argument("--symbol", type=str, help="Stock symbol to trade")
    parser.add_argument("--capital", type=float, help="Initial capital amount")
    parser.add_argument("--strategy", type=str, help="Trading strategy name")
    parser.add_argument(
        "--mode",
        type=str,
        choices=["live", "backtest", "walkforward"],
        help="Operational mode (live, backtest, or walkforward)",
    )
    parser.add_argument(
        "--prob-threshold",
        type=float,
        dest="proba_threshold",
        help="Probability threshold for ML model (0.0-1.0)",
    )
    parser.add_argument(
        "--period",
        type=str,
        default="2y",
        help="Data period to use (e.g., 1mo, 3mo, 6mo, 1y, 2y, 5y, max)",
    )
    parser.add_argument(
        "--interval",
        type=str,
        default="1d",
        help="Data interval to use (e.g., 1m, 5m, 15m, 30m, 60m, 1d, 1wk)",
    )
    parser.add_argument(
        "--splits", 
        type=int, 
        default=10,
        help="Number of splits for walk-forward testing"
    )
    args = parser.parse_args()

    # Import here to avoid circular imports
    from config import get_config

    config = get_config()

    # Set environment variables
    setup_environment(config)

    # Import the main application after setting environment variables
    from main import main_gui

    # Apply command line arguments or use defaults from config
    symbol = args.symbol or config["DEFAULT_SYMBOL"]
    capital = args.capital or config["DEFAULT_CAPITAL"]
    strategy = args.strategy or config["DEFAULT_STRATEGY"]
    mode = args.mode or "live"
    proba_threshold = args.proba_threshold or config["DEFAULT_PROBA_THRESHOLD"]
    data_period = args.period
    data_interval = args.interval
    splits = args.splits

    # Set any values in environment for modules that might read them
    os.environ["AIAGENT_DEFAULT_SYMBOL"] = symbol
    os.environ["AIAGENT_DEFAULT_CAPITAL"] = str(capital)
    os.environ["AIAGENT_DEFAULT_STRATEGY"] = strategy
    os.environ["AIAGENT_MODE"] = mode
    os.environ["AIAGENT_DEFAULT_PROBA_THRESHOLD"] = str(proba_threshold)
    os.environ["AIAGENT_DATA_PERIOD"] = data_period
    os.environ["AIAGENT_DATA_INTERVAL"] = data_interval

    # Launch appropriate mode
    if mode == "walkforward":
        # Run walk-forward backtest
        run_walk_forward_backtest(
            symbol=symbol,
            period=data_period,
            interval=data_interval,
            n_splits=splits,
            proba_threshold=proba_threshold
        )
    else:
        # Launch the regular application
        main_gui(cli_proba_threshold=proba_threshold, cli_mode=mode)


if __name__ == "__main__":
    main()
