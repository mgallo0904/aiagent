"""
Backtesting utilities for evaluating trading strategies with P&L metrics.
This module contains functionality for realistic broker-style backtesting.
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from strategies import OptionPricer # Assuming OptionPricer is in strategies.py

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


def run_option_strategy_backtest(
    price_data: pd.DataFrame, # Must contain 'close' and volatility_column
    signals: pd.Series,       # Signals (1, -1, 0) indexed by date
    initial_capital: float = 100000.0,
    otm_percentage: float = 0.02,
    dte_days: int = 30,
    option_hold_days: int = 10,
    commission_per_contract: float = 0.65,
    num_contracts_per_trade: int = 1,
    option_multiplier: int = 100,
    volatility_column: str = 'annualized_volatility_20', # Name of volatility col in price_data
    risk_free_rate: float = 0.01,
    dividend_yield: float = 0.00
) -> pd.Series:
    """
    Backtests a simple option buying strategy based on ML signals.
    """
    if not isinstance(price_data, pd.DataFrame) or 'close' not in price_data.columns or volatility_column not in price_data.columns:
        raise ValueError(f"price_data must be a DataFrame with 'close' and '{volatility_column}' columns.")
    if not isinstance(signals, pd.Series):
        raise ValueError("signals must be a Pandas Series.")
    if not price_data.index.equals(signals.index):
        # Attempt to align if possible, or raise error if alignment fails significantly
        try:
            common_index = price_data.index.intersection(signals.index)
            if len(common_index) < len(signals) * 0.9: # If less than 90% overlap
                 raise ValueError("Significant mismatch between price_data and signals indices.")
            price_data = price_data.loc[common_index]
            signals = signals.loc[common_index]
            if price_data.empty:
                raise ValueError("No common data between price_data and signals after index alignment.")
        except Exception as e:
            raise ValueError(f"Index mismatch and alignment failed: price_data length {len(price_data)}, signals length {len(signals)}. Error: {e}")


    equity = pd.Series(index=price_data.index, dtype=float)
    cash = initial_capital
    active_option = None  # Stores dict: {type, strike, expiry_date, entry_date, entry_option_price, num_contracts}
    
    option_pricer = OptionPricer(risk_free_rate=risk_free_rate, dividend_yield=dividend_yield)

    for date, row in price_data.iterrows():
        current_underlying_price = row['close']
        current_volatility = row[volatility_column]
        signal = signals.loc[date]

        # 1. Check and handle existing position
        if active_option:
            close_position = False
            exit_reason = ""

            if date >= active_option['expiry_date']:
                close_position = True
                exit_reason = "Expiry reached"
            elif (date - active_option['entry_date']).days >= option_hold_days:
                close_position = True
                exit_reason = f"Max hold period ({option_hold_days} days) reached"
            elif (signal == 1 and active_option['type'] == 'put') or \
                 (signal == -1 and active_option['type'] == 'call'):
                close_position = True
                exit_reason = "Opposite signal received"

            if close_position:
                # Calculate exit option price
                remaining_dte_days = (active_option['expiry_date'] - date).days
                exit_option_price = 0.0
                if remaining_dte_days > 0:
                    try:
                        exit_option_price, _, _ = option_pricer.price_and_greeks(
                            eval_date_py=date,
                            spot=current_underlying_price,
                            strike=active_option['strike'],
                            vol=current_volatility,
                            expiry_date_py=active_option['expiry_date'],
                            option_type=active_option['type']
                        )
                    except Exception as e:
                        print(f"{date}: Error pricing option for exit ({active_option['type']} K={active_option['strike']} Exp={active_option['expiry_date']}): {e}")
                elif active_option['type'] == 'call': # Intrinsic value at expiry
                    exit_option_price = max(0, current_underlying_price - active_option['strike'])
                elif active_option['type'] == 'put': # Intrinsic value at expiry
                    exit_option_price = max(0, active_option['strike'] - current_underlying_price)
                
                pnl = (exit_option_price - active_option['entry_option_price']) * active_option['num_contracts'] * option_multiplier
                cash += (exit_option_price * active_option['num_contracts'] * option_multiplier) # Add back sale proceeds
                cash -= (commission_per_contract * active_option['num_contracts']) # Commission on exit
                # print(f"{date}: Closed {active_option['type']} @{exit_option_price:.2f} (Entry: {active_option['entry_option_price']:.2f}). P&L: {pnl:.2f}. Reason: {exit_reason}. Cash: {cash:.2f}")
                active_option = None

        # 2. Evaluate new signal if no active position or if position was just closed
        if signal != 0 and active_option is None:
            option_type_to_buy = 'call' if signal == 1 else 'put'
            strike_price_otm_factor = 1 + otm_percentage if signal == 1 else 1 - otm_percentage
            strike_price = round(current_underlying_price * strike_price_otm_factor, 2)
            expiry_date = date + timedelta(days=dte_days)
            
            try:
                entry_option_price, _, _ = option_pricer.price_and_greeks(
                    eval_date_py=date,
                    spot=current_underlying_price,
                    strike=strike_price,
                    vol=current_volatility,
                    expiry_date_py=expiry_date,
                    option_type=option_type_to_buy
                )
            except Exception as e:
                # print(f"{date}: Error pricing option for entry ({option_type_to_buy} K={strike_price} Exp={expiry_date}): {e}")
                entry_option_price = -1 # Indicate failure

            if entry_option_price > 0:
                cost_of_trade = (entry_option_price * num_contracts_per_trade * option_multiplier) + \
                                (commission_per_contract * num_contracts_per_trade)
                
                if cash >= cost_of_trade:
                    cash -= cost_of_trade
                    active_option = {
                        'type': option_type_to_buy,
                        'strike': strike_price,
                        'expiry_date': expiry_date,
                        'entry_date': date,
                        'entry_option_price': entry_option_price,
                        'num_contracts': num_contracts_per_trade
                    }
                    # print(f"{date}: Opened {option_type_to_buy} @{entry_option_price:.2f} K={strike_price} Exp={expiry_date}. Cash: {cash:.2f}")
                # else:
                    # print(f"{date}: Signal to buy {option_type_to_buy}, but insufficient cash. Required: {cost_of_trade:.2f}, Available: {cash:.2f}")
        
        # 3. Update daily equity
        current_option_value_mv = 0.0
        if active_option:
            remaining_dte_days_mv = (active_option['expiry_date'] - date).days
            if remaining_dte_days_mv > 0:
                try:
                    current_option_price_mv, _, _ = option_pricer.price_and_greeks(
                        eval_date_py=date,
                        spot=current_underlying_price,
                        strike=active_option['strike'],
                        vol=current_volatility,
                        expiry_date_py=active_option['expiry_date'],
                        option_type=active_option['type']
                    )
                    current_option_value_mv = current_option_price_mv * active_option['num_contracts'] * option_multiplier
                except Exception: # If pricing fails, assume value is what was paid or simplify
                    current_option_value_mv = active_option['entry_option_price'] * active_option['num_contracts'] * option_multiplier 
            elif active_option['type'] == 'call':
                 current_option_value_mv = max(0, current_underlying_price - active_option['strike']) * active_option['num_contracts'] * option_multiplier
            elif active_option['type'] == 'put':
                 current_option_value_mv = max(0, active_option['strike'] - current_underlying_price) * active_option['num_contracts'] * option_multiplier


        equity.loc[date] = cash + current_option_value_mv
        
    return equity

[end of backtest.py]
