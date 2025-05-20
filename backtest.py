"""
Backtesting utilities for evaluating trading strategies with P&L metrics.
This module contains functionality for realistic broker-style backtesting.
"""

import numpy as np
import pandas as pd

def run_backtest(
    price: pd.Series,
    signals: np.ndarray,
    initial_capital: float = 10_000,
    transaction_cost_pct: float = 0.001,
    target_volatility: float = 0.01,
    vol_lookback: int = 20
):
    """
    Simulate trading:
     - 'signals' is an array of -1/0/+1 after smoothing
       where -1 = short, 0 = no position, 1 = long
     - target_volatility = 1% daily
     - transaction_cost_pct = 0.1% per trade
    """
    # Input validation
    if not isinstance(price, pd.Series):
        raise TypeError("price must be a pandas Series")
    if not isinstance(signals, np.ndarray):
        raise TypeError("signals must be a numpy array")
    if len(price) != len(signals):
        raise ValueError("price and signals must have the same length")
    if initial_capital <= 0:
        raise ValueError("initial_capital must be positive")
    if transaction_cost_pct < 0 or transaction_cost_pct >= 1:
        raise ValueError("transaction_cost_pct must be between 0 and 1")
    if target_volatility <= 0:
        raise ValueError("target_volatility must be positive")
    if vol_lookback <= 0:
        raise ValueError("vol_lookback must be positive")
    if not np.all(np.isin(signals, [-1, 0, 1])):
        raise ValueError("signals must only contain values -1, 0, or 1")
    if vol_lookback <= 0:
        raise ValueError("vol_lookback must be positive")

    dates = price.index
    results = pd.DataFrame(index=dates)
    
    # precompute daily vol estimates
    daily_ret = price.pct_change().fillna(0)
    vol_est = daily_ret.rolling(vol_lookback).std().fillna(daily_ret.std())
    
    # Initialize DataFrame with proper float dtypes
    results['price']    = price
    results['signal']   = signals
    results['position'] = 0.0
    results['cash']     = float(initial_capital)
    results['holdings'] = 0.0
    results['equity']   = float(initial_capital)
    results['returns']  = 0.0

    for i in range(1, len(dates)):
        price_i = price.iloc[i]
        signal_i = signals[i]
        
        # position sizing based on volatility target
        # fraction = target_vol / est_vol, capped at 100%
        frac = target_volatility / vol_est.iloc[i]
        frac = np.clip(frac, 0.0, 1.0)

        # current position and equity
        equity_prev = results['equity'].iloc[i-1]
        current_shares = results['position'].iloc[i-1]
        
        # calculate desired position based on signal
        if signal_i == 0:  # No position
            desired_shares = 0
        else:
            # Scale position by volatility targeting and signal direction
            desired_shares = np.floor((equity_prev * frac) / price_i) * signal_i
        
        # Calculate trades and costs
        trades = desired_shares - current_shares
        cost = abs(trades) * price_i * transaction_cost_pct
        
        # Update cash based on trades and costs
        cash_prev = results['cash'].iloc[i-1]
        cash = cash_prev - (trades * price_i) - cost
        
        # Calculate holdings value (can be negative for short positions)
        holdings = desired_shares * price_i
        
        # Calculate equity (cash + holdings)
        equity = cash + holdings
        
        # Calculate returns
        returns = equity / equity_prev - 1
        
        # Update results
        results.loc[dates[i], 'position'] = desired_shares
        results.loc[dates[i], 'cash'] = cash
        results.loc[dates[i], 'holdings'] = holdings
        results.loc[dates[i], 'equity'] = equity
        results.loc[dates[i], 'returns'] = returns

    return results
