#!/usr/bin/env python3
# filepath: /Users/markgallo/Documents/aiagent/performance_metrics.py
"""
Performance metrics calculation for trading strategy backtests.
This module calculates comprehensive performance metrics from backtest results.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
import os
from typing import Dict, List, Optional, Tuple, Union, Any


def calculate_metrics(returns: pd.Series) -> Dict[str, float]:
    """
    Calculate key performance metrics from a series of returns.
    
    Args:
        returns: Series of returns (typically daily)
        
    Returns:
        Dictionary of performance metrics
    """
    # Filter out any NaN values and cap extreme values to prevent numerical issues
    returns = returns.dropna()
    
    # Cap extreme return values to prevent numerical instability
    max_return_cap = 0.5  # Cap daily returns at 50%
    returns = returns.clip(lower=-max_return_cap, upper=max_return_cap)
    
    # Skip empty series
    if len(returns) == 0:
        return {
            "total_return": 0.0,
            "annualized_return": 0.0,
            "annualized_volatility": 0.0,
            "sharpe_ratio": 0.0,
            "sortino_ratio": 0.0,
            "max_drawdown": 0.0,
            "calmar_ratio": 0.0,
            "win_rate": 0.0,
            "avg_win": 0.0,
            "avg_loss": 0.0,
            "profit_factor": 0.0,
            "recovery_factor": 0.0
        }
    
    # Trading days in a year (approximate)
    trading_days = 252
    
    # Calculate basic metrics
    total_return = (1 + returns).prod() - 1
    
    # Cap total return to prevent extreme values
    total_return = min(total_return, 10.0)  # Cap at 1000%
    
    daily_mean = returns.mean()
    daily_std = returns.std()
    
    # Annualized metrics
    period_years = len(returns) / trading_days
    annualized_return = (1 + total_return) ** (1 / period_years) - 1 if period_years > 0 else 0
    
    # Cap annualized return to prevent extreme values
    annualized_return = min(annualized_return, 5.0)  # Cap at 500%
    
    annualized_volatility = daily_std * np.sqrt(trading_days)
    
    # Risk metrics - cap to reasonable values
    sharpe_ratio = (annualized_return) / annualized_volatility if annualized_volatility != 0 else 0
    sharpe_ratio = min(max(sharpe_ratio, -20), 20)  # Cap between -20 and 20
    
    # Downside volatility (only negative returns)
    downside_returns = returns[returns < 0]
    downside_deviation = downside_returns.std() * np.sqrt(trading_days) if len(downside_returns) > 0 else 0
    sortino_ratio = (annualized_return) / downside_deviation if downside_deviation != 0 else 0
    sortino_ratio = min(max(sortino_ratio, -30), 30)  # Cap between -30 and 30
    
    # Calculate cumulative returns for drawdown analysis
    cumulative_returns = (1 + returns).cumprod()
    running_max = cumulative_returns.cummax()
    drawdowns = (cumulative_returns - running_max) / running_max
    max_drawdown = drawdowns.min()
    
    # Calculate win rate
    win_rate = (returns > 0).mean()
    
    # Calculate average win and loss
    avg_win = returns[returns > 0].mean() if len(returns[returns > 0]) > 0 else 0
    avg_loss = returns[returns < 0].mean() if len(returns[returns < 0]) > 0 else 0
    
    # Calculate profit factor
    total_wins = returns[returns > 0].sum() if len(returns[returns > 0]) > 0 else 0
    total_losses = abs(returns[returns < 0].sum()) if len(returns[returns < 0]) > 0 else 0
    profit_factor = total_wins / total_losses if total_losses != 0 else 0
    profit_factor = min(profit_factor, 100)  # Cap at 100
    
    # Calmar ratio (annualized return / max drawdown)
    calmar_ratio = abs(annualized_return / max_drawdown) if max_drawdown != 0 else 0
    calmar_ratio = min(calmar_ratio, 50)  # Cap at 50
    
    # Recovery factor (total return / max drawdown)
    recovery_factor = abs(total_return / max_drawdown) if max_drawdown != 0 else 0
    recovery_factor = min(recovery_factor, 100)  # Cap at 100
    
    return {
        "total_return": total_return,
        "annualized_return": annualized_return,
        "annualized_volatility": annualized_volatility,
        "sharpe_ratio": sharpe_ratio,
        "sortino_ratio": sortino_ratio,
        "max_drawdown": max_drawdown,
        "calmar_ratio": calmar_ratio,
        "win_rate": win_rate,
        "avg_win": avg_win,
        "avg_loss": avg_loss,
        "profit_factor": profit_factor,
        "recovery_factor": recovery_factor
    }


def calculate_drawdowns(equity_curve: pd.Series) -> pd.DataFrame:
    """
    Calculate drawdown information from an equity curve.
    
    Args:
        equity_curve: Series of equity values
        
    Returns:
        DataFrame of drawdown information
    """
    # Calculate returns
    returns = equity_curve.pct_change().fillna(0)
    
    # Calculate cumulative returns
    cumulative_returns = (1 + returns).cumprod()
    
    # Calculate drawdowns
    running_max = cumulative_returns.cummax()
    drawdowns = (cumulative_returns - running_max) / running_max
    
    # Find drawdown periods
    is_drawdown = drawdowns < 0
    
    # If no drawdowns, return empty dataframe
    if not is_drawdown.any():
        return pd.DataFrame(columns=['start_date', 'end_date', 'recovery_date', 
                                     'drawdown', 'duration_days', 'recovery_days'])
    
    # Find starts of drawdowns
    start_indices = []
    for i in range(1, len(is_drawdown)):
        if is_drawdown.iloc[i] and not is_drawdown.iloc[i-1]:
            start_indices.append(i)
    
    # If no complete drawdown periods, return empty dataframe
    if not start_indices:
        return pd.DataFrame(columns=['start_date', 'end_date', 'recovery_date', 
                                     'drawdown', 'duration_days', 'recovery_days'])
    
    # Calculate drawdown periods
    drawdown_info = []
    
    for start_idx in start_indices:
        # Find the lowest point of this drawdown
        end_idx = start_idx
        lowest_val = drawdowns.iloc[start_idx]
        
        i = start_idx
        while i < len(drawdowns) and drawdowns.iloc[i] < 0:
            if drawdowns.iloc[i] < lowest_val:
                lowest_val = drawdowns.iloc[i]
                end_idx = i
            i += 1
                
        # Find recovery date (if any)
        recovery_idx = end_idx
        for j in range(end_idx, len(drawdowns)):
            if drawdowns.iloc[j] >= 0:
                recovery_idx = j
                break
                
        # Only add complete drawdowns (those that have recovered)
        if recovery_idx > end_idx and recovery_idx < len(drawdowns):
            start_date = drawdowns.index[start_idx]
            end_date = drawdowns.index[end_idx]
            recovery_date = drawdowns.index[recovery_idx]
            
            duration_days = (end_date - start_date).days
            recovery_days = (recovery_date - end_date).days
            
            drawdown_info.append({
                'start_date': start_date,
                'end_date': end_date,
                'recovery_date': recovery_date,
                'drawdown': lowest_val,
                'duration_days': duration_days,
                'recovery_days': recovery_days
            })
    
    # Convert to DataFrame and sort by drawdown amount
    drawdown_df = pd.DataFrame(drawdown_info)
    if not drawdown_df.empty:
        drawdown_df = drawdown_df.sort_values('drawdown', ascending=True)
    
    return drawdown_df


def calculate_cycle_metrics(results: pd.DataFrame, precomputed_metrics: Optional[pd.DataFrame] = None) -> pd.DataFrame:
    """
    Calculate performance metrics for each training/testing cycle.
    If precomputed_metrics are provided, use them directly.
    
    Args:
        results: DataFrame containing backtest results from walk-forward test (equity curve or returns).
        precomputed_metrics: Optional DataFrame of precomputed cycle metrics.
        
    Returns:
        DataFrame of per-cycle metrics
    """
    if precomputed_metrics is not None and not precomputed_metrics.empty:
        # Ensure standard column names if using precomputed, though they should match already
        # For example, rename 'start' to 'start_date' and 'end' to 'end_date' if necessary
        # For now, assume they are compatible or that the precomputed_metrics DataFrame is already correctly formatted.
        # Example: required_cols = ['cycle', 'start_date', 'end_date', 'return', 'sharpe_ratio', 'win_rate']
        # if not all(col in precomputed_metrics.columns for col in required_cols):
        #     print("Warning: Precomputed cycle metrics might be missing required columns or have different names.")
        return precomputed_metrics

    # If precomputed_metrics is not provided or empty, calculate them from results (equity curve/returns)
    # Ensure results is a DataFrame with a DatetimeIndex
    if not isinstance(results.index, pd.DatetimeIndex):
        print("Warning: calculate_cycle_metrics expects a DataFrame with a DatetimeIndex when precomputed_metrics are not provided.")
        return pd.DataFrame()
        
    # Identify cycles by looking for discontinuities in the index
    index_diff = results.index.to_series().diff()
    # Consider a cycle break if more than 1 day gap (adjust if interval is not daily)
    # Assuming daily data for now where a gap > 1 day means a new cycle (e.g. weekend or end of test period)
    cycle_breaks = index_diff[index_diff > pd.Timedelta(days=1.5)].index # Adjusted threshold
    
    cycle_ranges = []
    if len(results) == 0:
        return pd.DataFrame()

    if len(cycle_breaks) == 0:
        # No breaks found, treat as a single cycle
        if len(results) > 0:
            cycle_ranges = [(results.index[0], results.index[-1])]
    else:
        # First cycle
        if len(results) > 0:
            cycle_ranges.append((results.index[0], cycle_breaks[0] - pd.Timedelta(days=1))) # End before the break
            
            # Middle cycles
            for i in range(len(cycle_breaks) - 1):
                cycle_ranges.append((cycle_breaks[i], cycle_breaks[i+1] - pd.Timedelta(days=1)))
                
            # Last cycle
            if cycle_breaks[-1] < results.index[-1]:
                cycle_ranges.append((cycle_breaks[-1], results.index[-1]))
    
    # Calculate metrics for each cycle
    calculated_cycle_metrics = [] # Renamed to avoid conflict
    
    for i, (start_date, end_date) in enumerate(cycle_ranges):
        # Ensure start_date is not after end_date
        if start_date > end_date:
            continue
            
        cycle_data = results.loc[start_date:end_date]
        
        # Skip cycles with no data or too little data for metrics
        if len(cycle_data) < 2:
            continue
            
        # Get returns for this cycle
        # If 'results' is an equity curve, calculate returns for the cycle
        if 'equity' in cycle_data.columns or cycle_data.name == 'equity' or isinstance(cycle_data, pd.Series):
            current_cycle_equity = cycle_data.squeeze() # Ensure it's a Series
            cycle_returns = current_cycle_equity.pct_change().dropna()
        elif 'return' in cycle_data.columns: # If 'results' already contains returns
            cycle_returns = cycle_data['return'].dropna()
        else:
            print(f"Warning: Cycle data for {start_date}-{end_date} does not seem to be equity or returns.")
            continue

        if len(cycle_returns) < 2:
            metrics = {
                'total_return': 0.0, 'sharpe_ratio': 0.0, 'win_rate': 0.0
            }
        else:
            metrics = calculate_metrics(cycle_returns) # Use the main metrics calculator
        
        calculated_cycle_metrics.append({
            'cycle': i + 1,
            'start_date': start_date.date(),
            'end_date': end_date.date(),
            'return': metrics.get('total_return', 0.0),
            'sharpe_ratio': metrics.get('sharpe_ratio', 0.0),
            'win_rate': metrics.get('win_rate', 0.0)
        })
    
    return pd.DataFrame(calculated_cycle_metrics)


def calculate_trade_statistics(
    returns: Optional[pd.Series] = None, 
    trades_list: Optional[List[Dict[str, Any]]] = None
) -> Dict[str, Any]:
    """
    Calculate trade statistics either from a list of trades or inferred from returns.
    Using trades_list is preferred for accuracy.
    
    Args:
        returns: Series of returns (used if trades_list is not provided)
        trades_list: List of dictionaries, where each dictionary represents a trade.
                     Expected keys: 'return_pct', 'entry_date', 'exit_date', 'position_size' (optional)
        
    Returns:
        Dictionary of trade statistics
    """
    if trades_list is not None and len(trades_list) > 0:
        # Calculate from trades_list
        df_trades = pd.DataFrame(trades_list)
        if 'return_pct' not in df_trades.columns:
            print("Warning: 'return_pct' not found in trades_list. Cannot calculate trade statistics.")
            return {
                'number_of_trades': 0,
                'avg_trade_return': 0,
                'win_rate': 0,
                'loss_rate': 0,
                'avg_win_return': 0,
                'avg_loss_return': 0,
                'profit_factor': 0,
                'avg_trade_duration_days': 0,
                'exposure_pct': 0 # Placeholder, actual exposure calculation needs total backtest duration
            }

        # Ensure 'entry_date' and 'exit_date' are datetime objects for duration calculation
        if 'entry_date' in df_trades.columns and 'exit_date' in df_trades.columns:
            df_trades['entry_date'] = pd.to_datetime(df_trades['entry_date'])
            df_trades['exit_date'] = pd.to_datetime(df_trades['exit_date'])
            df_trades['duration'] = (df_trades['exit_date'] - df_trades['entry_date']).dt.days
            avg_duration = df_trades['duration'].mean()
        else:
            avg_duration = 0
            
        number_of_trades = len(df_trades)
        avg_trade_return = df_trades['return_pct'].mean()
        wins = df_trades[df_trades['return_pct'] > 0]
        losses = df_trades[df_trades['return_pct'] < 0]
        
        win_rate = len(wins) / number_of_trades if number_of_trades > 0 else 0
        loss_rate = len(losses) / number_of_trades if number_of_trades > 0 else 0
        
        avg_win_return = wins['return_pct'].mean() if len(wins) > 0 else 0
        avg_loss_return = losses['return_pct'].mean() if len(losses) > 0 else 0 # Will be negative
        
        total_profit_from_wins = wins['return_pct'].sum()
        total_loss_from_losses = abs(losses['return_pct'].sum())
        
        profit_factor = total_profit_from_wins / total_loss_from_losses if total_loss_from_losses > 0 else np.inf
        if profit_factor == np.inf and total_profit_from_wins == 0: # No wins, no losses, avoid inf if possible
             profit_factor = 0
        elif profit_factor == np.inf and total_profit_from_wins > 0: # All wins, no losses
            pass # keep as inf or a large number


        return {
            'number_of_trades': number_of_trades,
            'avg_trade_return': avg_trade_return,
            'win_rate': win_rate,
            'loss_rate': loss_rate,
            'avg_win_return': avg_win_return,
            'avg_loss_return': avg_loss_return,
            'profit_factor': profit_factor,
            'avg_trade_duration_days': avg_duration,
            # 'exposure_pct': exposure_pct # Requires total backtest duration
        }

    # Fallback: Infer from returns series (less accurate)
    if returns is None or returns.empty:
        return {
            'number_of_trades': 0, 'avg_trade_return': 0, 'win_rate': 0, 'loss_rate': 0,
            'avg_win_return': 0, 'avg_loss_return': 0, 'profit_factor': 0,
            'avg_trade_duration_days': 0, 'exposure_pct': 0
        }

    trades = returns[returns != 0]  # Consider non-zero returns as trades
    number_of_trades = len(trades)
    
    if number_of_trades == 0:
        return {
            'number_of_trades': 0, 'avg_trade_return': 0, 'win_rate': 0, 'loss_rate': 0,
            'avg_win_return': 0, 'avg_loss_return': 0, 'profit_factor': 0,
            'avg_trade_duration_days': 0, 'exposure_pct': 0
        }
        
    avg_trade_return = trades.mean()
    win_rate = (trades > 0).sum() / number_of_trades
    loss_rate = (trades < 0).sum() / number_of_trades
    avg_win_return = trades[trades > 0].mean() if (trades > 0).any() else 0
    avg_loss_return = trades[trades < 0].mean() if (trades < 0).any() else 0
    
    total_profit_from_wins = trades[trades > 0].sum()
    total_loss_from_losses = abs(trades[trades < 0].sum())
    profit_factor = total_profit_from_wins / total_loss_from_losses if total_loss_from_losses > 0 else np.inf
    if profit_factor == np.inf and total_profit_from_wins == 0:
        profit_factor = 0
        
    return {
        'number_of_trades': number_of_trades,
        'avg_trade_return': avg_trade_return,
        'win_rate': win_rate,
        'loss_rate': loss_rate,
        'avg_win_return': avg_win_return,
        'avg_loss_return': avg_loss_return,
        'profit_factor': profit_factor,
        'avg_trade_duration_days': np.nan,  # Cannot determine duration from returns only
        'exposure_pct': np.nan # Cannot determine from returns only
    }

def summarize_performance(
    returns_or_equity: pd.Series,
    initial_capital: float = 10_000,
    trades_list: Optional[List[Dict[str, Any]]] = None,  # Added
    precomputed_cycle_metrics: Optional[pd.DataFrame] = None  # Added
) -> Dict[str, Any]:
    """
    Summarize backtest performance metrics comprehensively.
    
    Args:
        returns_or_equity: Series of returns or equity values
        initial_capital: Initial capital amount
        trades_list: List of dictionaries, where each dictionary represents a trade.
        precomputed_cycle_metrics: Optional DataFrame of precomputed cycle metrics.
        
    Returns:
        Dictionary containing all performance metrics
    """
    # Equity curve and returns calculation
    if isinstance(returns_or_equity.index, pd.MultiIndex):
        print("Warning: MultiIndex found in returns_or_equity. Performance metrics might be incorrect.")
        # Attempt to use the first level of the index if it's a DatetimeIndex
        if isinstance(returns_or_equity.index.levels[0], pd.DatetimeIndex):
            returns_or_equity.index = returns_or_equity.index.levels[0]
        else:
            # Fallback or raise error if unable to resolve index
            pass # Or handle error appropriately

    series_name = returns_or_equity.name if returns_or_equity.name is not None else ""
    if series_name == 'equity' or 'equity' in series_name.lower() or (returns_or_equity > initial_capital * 0.1).all(): # Heuristic for equity curve
        equity_curve = returns_or_equity.copy()
        equity_curve.iloc[0] = initial_capital # Ensure initial capital is set correctly
        returns = equity_curve.pct_change().fillna(0)
        # Replace inf/-inf with 0, can happen if equity goes to 0 or from 0
        returns.replace([np.inf, -np.inf], 0, inplace=True)
    else:
        returns = returns_or_equity.copy()
        # Replace inf/-inf with 0 from returns series directly
        returns.replace([np.inf, -np.inf], 0, inplace=True)
        equity_curve = (1 + returns).cumprod() * initial_capital
        equity_curve.iloc[0] = initial_capital # Ensure initial capital is set for derived equity curve

    # Basic portfolio metrics
    portfolio_summary = calculate_metrics(returns)
    portfolio_summary['initial_capital'] = initial_capital
    portfolio_summary['final_capital'] = equity_curve.iloc[-1]

    # Drawdown analysis
    drawdowns_df = calculate_drawdowns(equity_curve)

    # Trade statistics
    trade_stats = calculate_trade_statistics(returns, trades_list=trades_list) # Pass trades_list

    # Cycle metrics
    # Use precomputed if available, otherwise calculate from equity_curve
    final_cycle_metrics = calculate_cycle_metrics(equity_curve, precomputed_metrics=precomputed_cycle_metrics)

    return {
        "summary": portfolio_summary,
        "drawdowns": drawdowns_df,
        "trade_statistics": trade_stats,
        "cycle_metrics": final_cycle_metrics,
        "equity_curve": equity_curve,
        "returns": returns
    }


def print_performance_report(performance_summary: Dict[str, Any]) -> None:
    """
    Print a formatted performance report from a performance summary.
    
    Args:
        performance_summary: Output from summarize_performance()
    """
    summary = performance_summary['summary']
    metrics = summary  # Detailed metrics are stored in summary
    drawdowns = performance_summary['drawdowns']
    cycle_metrics = performance_summary['cycle_metrics']
    trade_stats = performance_summary['trade_statistics']
    
    print("\n" + "="*50)
    print("PERFORMANCE REPORT")
    print("="*50)
    
    print("\nSUMMARY:")
    print(f"Initial Capital: ${summary['initial_capital']:,.2f}")
    print(f"Final Capital: ${summary['final_capital']:,.2f}")
    print(f"Total Return: {summary['total_return']:.2%}")
    print(f"Annualized Return: {summary['annualized_return']:.2%}")
    print(f"Sharpe Ratio: {summary['sharpe_ratio']:.2f}")
    print(f"Maximum Drawdown: {summary['max_drawdown']:.2%}")
    print(f"Win Rate: {summary['win_rate']:.2%}")
    print(f"Profit Factor: {summary['profit_factor']:.2f}")
    
    print("\nDETAILED METRICS:")
    print(f"Annualized Volatility: {metrics['annualized_volatility']:.2%}")
    print(f"Sortino Ratio: {metrics['sortino_ratio']:.2f}")
    print(f"Calmar Ratio: {metrics['calmar_ratio']:.2f}")
    print(f"Recovery Factor: {metrics['recovery_factor']:.2f}")
    print(f"Average Win: {metrics['avg_win']:.2%}")
    print(f"Average Loss: {metrics['avg_loss']:.2%}")
    
    print("\nTRADE STATISTICS:")
    print(f"Number of Trades: {trade_stats['number_of_trades']}")
    print(f"Average Trade Return: {trade_stats['avg_trade_return']:.2%}")
    print(f"Average Trade Duration: {trade_stats['avg_trade_duration_days']:.1f} days")
    print(f"Market Exposure: {trade_stats['exposure_pct']:.2%}")
    
    print("\nCYCLE PERFORMANCE:")
    if not cycle_metrics.empty:
        for _, cycle in cycle_metrics.iterrows():
            start_date = cycle['start_date'].strftime('%Y-%m-%d')
            end_date = cycle['end_date'].strftime('%Y-%m-%d')
            print(f"Cycle {cycle['cycle']} ({start_date} to {end_date}):")
            print(f"  Return: {cycle['return']:.2%}, Sharpe: {cycle['sharpe_ratio']:.2f}, " + 
                  f"Max DD: {cycle['max_drawdown']:.2%}, Win Rate: {cycle['win_rate']:.2%}")
    else:
        print("No cycle metrics available")
    
    print("\nLARGEST DRAWDOWNS:")
    if not drawdowns.empty and len(drawdowns) > 0:
        top_n = min(5, len(drawdowns))
        for i, dd in drawdowns.head(top_n).iterrows():
            start = dd['start_date'].strftime('%Y-%m-%d')
            end = dd['end_date'].strftime('%Y-%m-%d')
            recovery = dd['recovery_date'].strftime('%Y-%m-%d')
            print(f"{i+1}. {dd['drawdown']:.2%} from {start} to {end} (recovered: {recovery})")
    else:
        print("No significant drawdowns")
    
    print("\n" + "="*50)


def visualize_performance(results: pd.DataFrame, 
                         performance_summary: Dict[str, Any],
                         filename: Optional[str] = None) -> None:
    """
    Create detailed performance visualization.
    
    Args:
        results: DataFrame of backtest results
        performance_summary: Output from summarize_performance()
        filename: Optional filename to save the plot
    """
    cycle_metrics = performance_summary['cycle_metrics']
    
    # Create figure with subplots
    fig = plt.figure(figsize=(15, 12))
    
    # Define grid layout
    gs = plt.GridSpec(4, 2, figure=fig, height_ratios=[3, 1, 1.5, 1.5])
    
    # Equity curve
    ax1 = fig.add_subplot(gs[0, :])
    results['equity'].plot(ax=ax1, linewidth=2, color='blue')
    ax1.set_title('Equity Curve')
    ax1.set_ylabel('Account Value ($)')
    ax1.grid(True)
    
    # Drawdown
    ax2 = fig.add_subplot(gs[1, :], sharex=ax1)
    equity_curve = results['equity']
    returns = equity_curve.pct_change().fillna(0)
    cumulative_returns = (1 + returns).cumprod()
    running_max = cumulative_returns.cummax()
    drawdowns = (cumulative_returns - running_max) / running_max * 100  # Convert to percentage
    drawdowns.plot(ax=ax2, color='red', linewidth=1.5)
    ax2.set_title('Drawdowns')
    ax2.set_ylabel('Drawdown %')
    ax2.grid(True)
    ax2.fill_between(drawdowns.index, drawdowns, 0, color='red', alpha=0.3)
    
    # Cycle returns comparison (bar chart)
    ax3 = fig.add_subplot(gs[2, 0])
    if not cycle_metrics.empty:
        cycle_metrics['return'].plot(kind='bar', ax=ax3, color='green')
        ax3.set_title('Returns by Cycle')
        ax3.set_ylabel('Return %')
        ax3.set_xlabel('Cycle')
        ax3.grid(True, axis='y')
        # Format y-axis as percentage
        ax3.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'{x:.1%}'))
        # Add cycle numbers
        ax3.set_xticklabels([f'Cycle {i}' for i in cycle_metrics['cycle']], rotation=45)
    else:
        ax3.text(0.5, 0.5, 'No cycle metrics available', ha='center', va='center')
    
    # Cycle Sharpe ratios
    ax4 = fig.add_subplot(gs[2, 1])
    if not cycle_metrics.empty:
        cycle_metrics['sharpe_ratio'].plot(kind='bar', ax=ax4, color='purple')
        ax4.set_title('Sharpe Ratio by Cycle')
        ax4.set_ylabel('Sharpe Ratio')
        ax4.set_xlabel('Cycle')
        ax4.grid(True, axis='y')
        # Add cycle numbers
        ax4.set_xticklabels([f'Cycle {i}' for i in cycle_metrics['cycle']], rotation=45)
    else:
        ax4.text(0.5, 0.5, 'No cycle metrics available', ha='center', va='center')
    
    # Monthly returns heatmap
    ax5 = fig.add_subplot(gs[3, 0])
    # Convert daily returns to monthly
    monthly_returns = results['returns'].resample('M').apply(lambda x: (1 + x).prod() - 1)
    # Create a pivot table with years as rows and months as columns
    monthly_returns.index = monthly_returns.index.to_period('M')
    monthly_pivot = pd.pivot_table(
        monthly_returns.reset_index(),
        values='returns',
        index=monthly_returns.index.year,
        columns=monthly_returns.index.month,
        aggfunc='first'
    )
    
    if not monthly_pivot.empty:
        # Create heatmap
        im = ax5.imshow(monthly_pivot.values, cmap='RdYlGn')
        ax5.set_title('Monthly Returns Heatmap')
        ax5.set_ylabel('Year')
        ax5.set_xlabel('Month')
        
        # Add text annotations
        for i in range(monthly_pivot.shape[0]):
            for j in range(monthly_pivot.shape[1]):
                value = monthly_pivot.values[i, j]
                if not np.isnan(value):
                    text = ax5.text(j, i, f"{value:.1%}",
                                   ha="center", va="center", 
                                   color="black" if abs(value) < 0.1 else "white")
        
        # Set tick labels
        month_labels = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 
                         'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
        ax5.set_xticks(np.arange(12))
        ax5.set_xticklabels(month_labels)
        ax5.set_yticks(np.arange(len(monthly_pivot.index)))
        ax5.set_yticklabels(monthly_pivot.index)
        
        # Add colorbar
        cbar = plt.colorbar(im, ax=ax5)
        cbar.set_label('Returns')
    else:
        ax5.text(0.5, 0.5, 'Insufficient data for monthly heatmap', ha='center', va='center')
    
    # Distribution of daily returns
    ax6 = fig.add_subplot(gs[3, 1])
    daily_returns = results['returns'].iloc[1:]  # Skip first row (typically zero)
    daily_returns.hist(bins=50, ax=ax6, alpha=0.6, color='blue')
    
    # Add normal distribution curve
    if len(daily_returns) > 1:
        mean = daily_returns.mean()
        std = daily_returns.std()
        x = np.linspace(mean - 3*std, mean + 3*std, 100)
        y = len(daily_returns) * (1/(std * np.sqrt(2*np.pi))) * np.exp(-(x-mean)**2 / (2*std**2)) * (daily_returns.max() - daily_returns.min())/50
        ax6.plot(x, y, 'r-', linewidth=2)
    
    ax6.set_title('Distribution of Daily Returns')
    ax6.set_xlabel('Daily Return')
    ax6.set_ylabel('Frequency')
    ax6.grid(True)
    
    # Add summary text to the plot
    summary = performance_summary['summary']
    summary_text = (
        f"Total Return: {summary['total_return']:.2%}\n"
        f"Annualized Return: {summary['annualized_return']:.2%}\n"
        f"Sharpe Ratio: {summary['sharpe_ratio']:.2f}\n"
        f"Max Drawdown: {summary['max_drawdown']:.2%}\n"
        f"Win Rate: {summary['win_rate']:.2%}\n"
        f"Profit Factor: {summary['profit_factor']:.2f}"
    )
    
    fig.text(0.02, 0.02, summary_text, fontsize=10,
             bbox=dict(facecolor='white', edgecolor='gray', alpha=0.8))
    
    plt.tight_layout()
    
    if filename:
        plt.savefig(filename, dpi=150, bbox_inches='tight')
        print(f"Performance visualization saved to {filename}")
    
    plt.show()


def generate_performance_report(results: pd.DataFrame, 
                               initial_capital: float = 10_000, 
                               output_dir: str = 'data',
                               symbol: str = 'BACKTEST') -> Dict[str, Any]:
    """
    Generate and save complete performance report.
    
    Args:
        results: DataFrame containing backtest results
        initial_capital: Initial capital amount
        output_dir: Directory to save report files
        symbol: Symbol or identifier for the backtest
        
    Returns:
        Performance summary dictionary
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Generate performance summary
    performance_summary = summarize_performance(results, initial_capital)
    
    # Generate timestamp for filenames
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Save performance report to CSV files
    cycle_metrics = performance_summary['cycle_metrics']
    drawdowns = performance_summary['drawdowns']
    
    # Save cycle metrics
    if not cycle_metrics.empty:
        cycle_metrics.to_csv(f"{output_dir}/cycles_{symbol}_{timestamp}.csv")
    
    # Save drawdowns
    if not drawdowns.empty:
        drawdowns.to_csv(f"{output_dir}/drawdowns_{symbol}_{timestamp}.csv")
    
    # Create visualization and save
    visualize_performance(
        results, 
        performance_summary,
        filename=f"{output_dir}/performance_{symbol}_{timestamp}.png"
    )
    
    # Print report to console
    print_performance_report(performance_summary)
    
    return performance_summary


if __name__ == "__main__":
    # Example usage - this will run if the script is executed directly
    print("Performance metrics module loaded.")
    print("This module provides functions for calculating trading strategy performance metrics.")
    print("To use it, import the functions into your backtest script.")
