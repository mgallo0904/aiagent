import pandas as pd
import numpy as np
# import optuna # Optuna is now handled within MLModel
import matplotlib.pyplot as plt
import os

from market_data import MarketData
from features import prepare_features_for_model # Use the new comprehensive feature prep
from ml_models import MLModel, ModelBehaviorConfig # Use the enhanced MLModel
from risk_management import RiskManager # RiskManager will be used for position sizing
from performance_metrics import summarize_performance, print_performance_report, visualize_performance # Keep these
from strategies import TradingStrategy # To select options
from config.loader import get_config # Load global config

CONFIG = get_config()


# The tune_model_params function is now largely handled by MLModel's fit method with HPO.
# It can be removed or adapted if a pre-HPO step is ever needed.
# For now, we assume MLModel handles its tuning.

def simulate_option_pnl(option_details: pd.Series, underlying_price_series: pd.Series, initial_premium_paid: float, quantity: int, transaction_cost_per_contract: float = 0.65):
    """
    Simulates P&L for a single option contract over its holding period.
    This is a simplified P&L simulation, primarily based on intrinsic value.
    A full model would require IV changes, time decay (theta), etc.

    Args:
        option_details (pd.Series): Details of the chosen option (strike, optionType, ExpiryDate etc.).
        underlying_price_series (pd.Series): Series of underlying prices during the holding period.
        initial_premium_paid (float): Premium paid per share for the option.
        quantity (int): Number of contracts.
        transaction_cost_per_contract (float): Cost per contract for entry/exit.

    Returns:
        pd.Series: P&L series for the option.
    """
    strike = option_details['strike']
    option_type = option_details['optionType'] # 'call' or 'put'
    
    pnl = pd.Series(index=underlying_price_series.index, dtype=float)
    initial_total_cost = (initial_premium_paid * 100 * quantity) + (transaction_cost_per_contract * quantity) # Entry cost

    for date, underlying_price in underlying_price_series.items():
        intrinsic_value = 0
        if option_type == 'call':
            intrinsic_value = max(0, underlying_price - strike)
        elif option_type == 'put':
            intrinsic_value = max(0, strike - underlying_price)
        
        # Simplified current value: intrinsic value.
        # A better model would add estimated time value (e.g. from a pricing model or market data if available).
        # For this simulation, let's assume time value decays linearly to zero by a hypothetical expiry
        # or we just track intrinsic value for simplicity of illustration here.
        # For now, P&L based on change from initial cost vs current intrinsic value.
        current_option_value_per_share = intrinsic_value # Simplified: value is intrinsic
        
        # Gross P&L based on current option value vs premium paid
        gross_pnl_for_day = (current_option_value_per_share - initial_premium_paid) * 100 * quantity
        pnl[date] = gross_pnl_for_day # Net P&L considering initial cost is handled by equity curve logic

    # The P&L series returned here represents the change in option value from the entry premium.
    # The initial cost is accounted for when adjusting cash in the equity curve.
    return pnl


def walkforward_options_backtest(
    symbol: str,
    train_window: int = 252, # ~1 year
    test_window: int = 21,   # ~1 month (trades held for this duration max)
    initial_capital: float = 100000,
    # Option selection parameters (could come from CONFIG)
    option_strategy_name: str = CONFIG.get("DEFAULT_STRATEGY", "Intelligent Call Buyer"), # e.g., "Intelligent Call Buyer"
    # ML model config (can be fine-tuned)
    ml_model_proba_threshold_ui: float = CONFIG.get("DEFAULT_PROBA_THRESHOLD", 0.6),
    transaction_cost_options: float = 0.65, # Per contract
    max_holding_period_option: int = 21 # Max days to hold an option trade
):
    """
    Performs walk-forward backtesting simulating options trading.
    """
    logger = plt.set_loglevel('WARNING') # Suppress matplotlib debug prints if any

    md = MarketData(symbol=symbol)
    # Fetch enough data for multiple train/test cycles + feature lookbacks
    # Total period should accommodate num_cycles * test_window + train_window + feature_warmup
    # Example: 5 cycles of 1 month test, 1 year train = 5*21 + 252 + ~50_warmup = ~420 trading days ~ 2 years.
    # Let's use a fixed long period and let the cycles determine usage.
    data_fetch_period = "5y" # Ample data for several folds
    underlying_hist_data_full = md.fetch_historical(period=data_fetch_period, interval="1d")
    
    md.fetch_options_chain(min_dte=CONFIG.get("DEFAULT_OPTION_DTE_MIN", 15), 
                           max_dte=CONFIG.get("DEFAULT_OPTION_DTE_MAX", 60))
    options_snapshot_df_full = md.get_all_options_df()
    md.calculate_iv_rank_percentile() # Calculate once based on full history fetched


    if underlying_hist_data_full is None or underlying_hist_data_full.empty:
        print(f"Failed to fetch underlying data for {symbol}. Aborting backtest.")
        return pd.Series([initial_capital], index=[pd.Timestamp.now().normalize()])

    # Prepare features for the entire dataset once
    # CONFIG should be passed to prepare_features_for_model for its internal settings.
    X_all, y_all = prepare_features_for_model(
        underlying_data=underlying_hist_data_full,
        options_data_all_expirations=options_snapshot_df_full, # Pass snapshot
        config=CONFIG,
        drop_na=False, # Handle NaNs carefully within walk-forward
        print_diagnostics=True # Enable for debugging
    )
    
    # Align X_all, y_all, and underlying prices
    price_data_full = underlying_hist_data_full['Close']
    common_idx = X_all.index.intersection(y_all.dropna().index).intersection(price_data_full.index)
    X_all = X_all.loc[common_idx].fillna(0) # Fill any remaining NaNs in features with 0
    y_all = y_all.loc[common_idx]
    price_data_full = price_data_full.loc[common_idx]

    if X_all.empty or y_all.empty or len(X_all) < (train_window + test_window):
        print(f"Not enough data after feature prep for walk-forward. X_all: {len(X_all)}. Required: {train_window + test_window}")
        return pd.Series([initial_capital], index=[price_data_full.index[0] if not price_data_full.empty else pd.Timestamp.now().normalize()])


    # Initialize equity curve, cash, holdings, etc.
    dates_all = price_data_full.index
    equity_curve = pd.Series(index=dates_all, dtype=float)
    equity_curve.iloc[0] = initial_capital
    cash = initial_capital
    current_holdings_value = 0.0
    
    # Active trade details
    active_option_trade = None # Stores details of the currently held option
    option_entry_date = None
    option_purchase_price_per_share = 0.0 # Premium paid per share for the option
    option_quantity = 0

    # Risk Manager (re-initialize per fold or use one that updates capital?) For now, one instance.
    # risk_mgr = RiskManager(total_capital=initial_capital, ...) # Loaded from CONFIG

    # ML Model configuration for MLTradingStrategy
    # No separate TradingStrategy instance like `option_strategy` needed if MLTradingStrategy is primary.
    # MLTradingStrategy will be instantiated per fold with the trained model.
    
    model_behavior_cfg = ModelBehaviorConfig(
        proba_threshold=ml_model_proba_threshold_ui, # From UI/config
        # Use other settings from CONFIG for MLModel
        use_optuna=CONFIG.get("USE_OPTUNA", True),
        optuna_n_trials=CONFIG.get("OPTUNA_N_TRIALS", 50), # Smaller for backtest speed
        n_iter_random_search=CONFIG.get("ML_N_ITER_RANDOM_SEARCH", 20),
        primary_metric_tuning=CONFIG.get("ML_PRIMARY_METRIC_TUNING", "roc_auc"),
        # ... other behavior config fields
    )
    base_model_params = CONFIG.get("DEFAULT_MODEL_PARAMS", {})

    # Storing cycle metrics for later analysis
    cycle_metrics_list = []
    all_trades_list = [] # To store details of each trade for metrics calculation

    # Create a MarketData instance that MLTradingStrategy can use internally for fetching.
    # This instance is symbol-specific and will be passed to MLTradingStrategy.
    # Its state will be managed by MLTradingStrategy's _get_latest_features method.
    market_data_for_strategy = MarketData(symbol=symbol)
    # We need to ensure that this market_data_for_strategy.data is populated with enough history
    # for feature calculation when _get_latest_features is called.
    # MLTradingStrategy._get_latest_features() calls fetch_historical("1y"), which should be sufficient.

    num_cycles = (len(X_all) - train_window) // test_window
    print(f"Starting Optimized Walk-Forward Options Backtest for {symbol} with {num_cycles} cycles using MLTradingStrategy.")

    for cycle_num in range(num_cycles):
        train_start_idx = cycle_num * test_window
        train_end_idx = train_start_idx + train_window
        
        test_start_idx = train_end_idx
        test_end_idx = test_start_idx + test_window

        if test_end_idx > len(X_all): break # Ensure we don't go out of bounds

        X_train = X_all.iloc[train_start_idx:train_end_idx]
        y_train = y_all.iloc[train_start_idx:train_end_idx]
        X_test_cycle = X_all.iloc[test_start_idx:test_end_idx]
        prices_test_cycle = price_data_full.iloc[test_start_idx:test_end_idx]
        
        cycle_start_date = prices_test_cycle.index[0]
        cycle_end_date = prices_test_cycle.index[-1]
        print(f"\nCycle {cycle_num+1}/{num_cycles}: Train {X_train.index[0].date()} to {X_train.index[-1].date()}, Test {cycle_start_date.date()} to {cycle_end_date.date()}")

        # Initialize and train ML Model for this fold
        ml_model_fold = MLModel(model_type=CONFIG.get("DEFAULT_MODEL_TYPE", "xgboost"),
                                params=base_model_params,
                                behavior_config=model_behavior_cfg)
        # Provide a portion of X_train for early stopping if XGBoost and if desired
        # For simplicity, HPO inside fit will use its own CV splits.
        ml_model_fold.train(X_train, y_train) # Changed from fit to train, if MLModel was updated to use 'train'
        print(f"  ML Model trained for cycle {cycle_num+1}.")

        # Instantiate MLTradingStrategy for this fold
        ml_strategy_fold = MLTradingStrategy(
            ml_model=ml_model_fold,
            market_data_provider=market_data_for_strategy, # Pass the dedicated MD instance
            strategy_params={
                "option_dte": CONFIG.get("DEFAULT_OPTION_DTE_MAX", 45),
                "prediction_horizon": CONFIG.get("ML_PREDICTION_HORIZON", 1), # Ensure these match model
                "dynamic_threshold_factor": CONFIG.get("ML_THRESHOLD_FACTOR", 0.5),
                "commission_per_contract": transaction_cost_options, # Use backtest param
                # action_mapping can be customized here if needed
            }
        )
        print(f"  MLTradingStrategy instantiated for cycle {cycle_num+1}.")

        # --- Simulate trading within the test window (day by day) ---
        for i in range(len(prices_test_cycle)):
            current_date = prices_test_cycle.index[i]
            current_underlying_price = prices_test_cycle.iloc[i]
            
            # Update equity with current cash and holdings value
            # If holding an option, its value changes. For simplicity, let's update holding value at exit.
            # Or, estimate daily option value change (more complex).
            # For now, holdings_value is option's current market value (if we could get it) or estimated.
            # Let's simplify: equity = cash + (current option value if held, else 0)
            # This means we need to mark-to-market the option daily or P&L is only realized at close.
            # For this simplified version, let's realize P&L at exit or end of test window for the option.
            # So, current_holdings_value represents the value of the *active* option if we were to sell it now.
            
            # If an option is held, check exit conditions
            if active_option_trade:
                # Check max holding period or other exit rules (e.g. target profit, stop loss on option premium)
                days_held = (current_date - option_entry_date).days
                
                # Simplified exit: if max holding period reached OR end of test window for this cycle
                # More advanced: ATR stop on underlying, % profit/loss on option premium
                force_exit = (days_held >= max_holding_period_option) or (i == len(prices_test_cycle) - 1)
                
                # Simulate option's current value (highly simplified - only intrinsic)
                current_option_intrinsic_value_per_share = 0
                if active_option_trade['optionType'] == 'call':
                    current_option_intrinsic_value_per_share = max(0, current_underlying_price - active_option_trade['strike'])
                elif active_option_trade['optionType'] == 'put':
                    current_option_intrinsic_value_per_share = max(0, active_option_trade['strike'] - current_underlying_price)
                
                current_option_total_value = current_option_intrinsic_value_per_share * 100 * option_quantity
                current_holdings_value = current_option_total_value # Update holding value

                if force_exit:
                    print(f"  {current_date.date()}: Exiting option {active_option_trade['contractSymbol']} due to {'max hold period' if days_held >= max_holding_period_option else 'end of cycle test window'}.")
                    pnl_on_trade = (current_option_intrinsic_value_per_share - option_purchase_price_per_share) * 100 * option_quantity
                    pnl_on_trade -= (transaction_cost_options * option_quantity) # Exit cost
                    
                    cash += (option_purchase_price_per_share * 100 * option_quantity) # Add back initial cost basis from cash
                    cash += pnl_on_trade # Add P&L

                    all_trades_list.append({
                        'entry_date': option_entry_date, 'exit_date': current_date,
                        'symbol': active_option_trade['contractSymbol'], 'type': active_option_trade['optionType'],
                        'strike': active_option_trade['strike'], 'entry_premium': option_purchase_price_per_share,
                        'exit_price_option_est': current_option_intrinsic_value_per_share, # This is simplified
                        'quantity': option_quantity, 'pnl': pnl_on_trade,
                        'underlying_at_entry': prices_test_cycle.loc[option_entry_date] if option_entry_date in prices_test_cycle.index else None,
                        'underlying_at_exit': current_underlying_price
                    })
                    
                    active_option_trade = None
                    current_holdings_value = 0
                    option_quantity = 0


            # If no active trade, check for entry signals for the current day
            if not active_option_trade:
                # Use MLTradingStrategy to generate signal
                # The market_data_for_strategy's internal state will be updated by generate_signal -> _get_latest_features
                # to reflect data up to current_date (or rather, previous day for feature generation)
                current_trade_action = ml_strategy_fold.generate_signal()
                action_type = current_trade_action.get("action")
                
                if action_type and action_type not in [ACTION_HOLD, "hold", "HOLD", None]: # Check if action is not HOLD
                    option_details_from_signal = current_trade_action.get("details")
                    if option_details_from_signal:
                        premium_per_share = option_details_from_signal.get("estimated_premium", 0)
                        if premium_per_share <= 0: # Cannot trade if premium is not positive
                            print(f"  {current_date.date()}: Estimated premium is {premium_per_share:.2f} for {option_details_from_signal.get('symbol')}, cannot enter trade. Signal: {action_type}")
                        else:
                            # Position Sizing
                            # For MLTradingStrategy, confidence could be derived from model's raw output if it were probabilities
                            # For now, using a fixed confidence for buy/sell signals from ternary model
                            model_confidence_for_sizing = 0.75 # Default confidence for BUY_CALL/BUY_PUT
                            if option_details_from_signal.get("ml_prediction_raw") == 0: # Should not happen if action_type is not HOLD
                                model_confidence_for_sizing = 0.5 
                            
                            temp_risk_manager = RiskManager(total_capital=cash, config=CONFIG) # Pass CONFIG to RiskManager
                            
                            size_params = temp_risk_manager.PositionSizeParams(
                                trade_type="option",
                                option_premium=premium_per_share,
                                model_confidence_scalar=model_confidence_for_sizing,
                                current_price=current_underlying_price, # Pass current underlying price
                                # Add other params as required by your RiskManager's calculate_position_size
                            )
                            num_contracts_to_trade = temp_risk_manager.calculate_position_size(size_params)
                            
                            if num_contracts_to_trade > 0:
                                total_trade_cost = premium_per_share * 100 * num_contracts_to_trade + (transaction_cost_options * num_contracts_to_trade)
                                if cash >= total_trade_cost:
                                    # Construct active_option_trade from signal details
                                    active_option_trade = {
                                        'contractSymbol': f"{symbol}{current_date.strftime('%y%m%d')}{option_details_from_signal['optionType'][0].upper()}{int(option_details_from_signal['strike_price']*1000):08d}", # Example symbol
                                        'strike': option_details_from_signal['strike_price'],
                                        'optionType': option_details_from_signal['optionType'], # 'call' or 'put'
                                        'estimated_premium_on_entry': premium_per_share # Store the estimated premium
                                        # Add other details like DTE if needed for P&L or exit logic later
                                    }
                                    option_entry_date = current_date
                                    option_purchase_price_per_share = premium_per_share # This is the actual entry premium (estimated)
                                    option_quantity = num_contracts_to_trade
                                    
                                    cash -= total_trade_cost
                                    current_holdings_value = premium_per_share * 100 * option_quantity
                                    
                                    print(f"  {current_date.date()}: Entered {option_quantity} contracts of {active_option_trade['contractSymbol']} "
                                          f"({active_option_trade['optionType']}) at {premium_per_share:.2f}. Cost: {total_trade_cost:.2f}. Cash left: {cash:.2f}. Reason: {option_details_from_signal.get('reason')}")
                                else:
                                    print(f"  {current_date.date()}: Insufficient cash for option trade. Need {total_trade_cost:.2f}, have {cash:.2f}. Signal: {action_type}")
                            else:
                                print(f"  {current_date.date()}: Position size is 0 for potential option trade. Signal: {action_type}")
                    else:
                        print(f"  {current_date.date()}: Action {action_type} but no details from MLTradingStrategy. Holding.")
                # else:
                    # This case means MLTradingStrategy returned ACTION_HOLD or similar
                    # print(f"  {current_date.date()}: MLTradingStrategy signals HOLD.")
            
            # Update equity curve for the day
            equity_curve[current_date] = cash + current_holdings_value # current_holdings_value is mark-to-market of option

        # End of cycle: record metrics
        cycle_end_equity = equity_curve[prices_test_cycle.index[-1]]
        cycle_start_equity_val = equity_curve[prices_test_cycle.index[0]] if prices_test_cycle.index[0] in equity_curve.index and not pd.isna(equity_curve[prices_test_cycle.index[0]]) else (equity_curve.ffill().loc[prices_test_cycle.index[0]] if prices_test_cycle.index[0] in equity_curve.ffill().index else initial_capital) # Handle potential NaNs
        
        cycle_return = (cycle_end_equity / cycle_start_equity_val) - 1 if cycle_start_equity_val != 0 else 0
        
        # For Sharpe and Win Rate, we need daily returns of this cycle's option trades (if any)
        # This part is complex as it depends on how P&L is being tracked for options daily.
        # For now, let's use the overall equity curve's returns for this cycle portion.
        cycle_equity_series = equity_curve.loc[prices_test_cycle.index].dropna()
        cycle_daily_returns = cycle_equity_series.pct_change().fillna(0)
        
        sharpe_ratio_cycle = (cycle_daily_returns.mean() / cycle_daily_returns.std() * np.sqrt(252)) if cycle_daily_returns.std() != 0 and len(cycle_daily_returns) >1 else 0
        win_rate_cycle = (cycle_daily_returns > 0).mean() if len(cycle_daily_returns) > 0 else 0

        cycle_metrics_list.append({
            'cycle': cycle_num + 1, 'start_date': cycle_start_date, 'end_date': cycle_end_date,
            'start_equity': cycle_start_equity_val, 'end_equity': cycle_end_equity,
            'return': cycle_return, 'sharpe_ratio': sharpe_ratio_cycle, 'win_rate': win_rate_cycle
        })
        print(f"  Cycle {cycle_num+1} End. Equity: {cycle_end_equity:.2f}. Return: {cycle_return:.2%}")

    # Fill forward any NaNs in equity curve that might have occurred if no trading days matched
    equity_curve = equity_curve.ffill().fillna(initial_capital) # Fill remaining NaNs with initial capital or last known good value
    
    # --- Performance Summary ---
    print("\n--- Overall Performance (Options Trading Simulation) ---")
    # The summarize_performance function expects a returns series or equity curve.
    # We pass the equity_curve Series.
    # It also takes trades_list for more detailed trade statistics.
    performance_summary = summarize_performance(equity_curve, initial_capital=initial_capital, trades_list=all_trades_list, precomputed_cycle_metrics=pd.DataFrame(cycle_metrics_list))
    
    # Print the report using the dedicated function
    # The print_performance_report in performance_metrics.py might need slight adjustment
    # if the structure of performance_summary or its sub-dictionaries changed.
    # For now, assuming it can parse the output of summarize_performance.
    
    # Manually print key summary stats for now
    final_summary_metrics = performance_summary.get('summary', {})
    print("\nKey Performance Metrics:")
    for k, v in final_summary_metrics.items():
        if isinstance(v, (float, np.float_)):
            print(f"  {k.replace('_', ' ').title()}: {v:.2f}" if 'ratio' in k or 'factor' in k else f"  {k.replace('_', ' ').title()}: {v:,.2f}")
            if '%' in k or 'return' in k or 'drawdown' in k : print(f"  {k.replace('_', ' ').title()}: {v:.2%}")
        else:
            print(f"  {k.replace('_', ' ').title()}: {v}")
            
    trade_stats = performance_summary.get('trade_statistics', {})
    print("\nTrade Statistics:")
    for k,v in trade_stats.items():
        print(f"  {k.replace('_', ' ').title()}: {v:.2f}" if isinstance(v, (float,np.float_)) else f"  {k.replace('_', ' ').title()}: {v}")


    # Save equity curve and performance plot
    output_dir = 'data'
    os.makedirs(output_dir, exist_ok=True)
    equity_curve.to_csv(f"{output_dir}/equity_curve_options_optimized_{symbol}.csv")
    
    # Visualize performance
    # The visualize_performance function needs a DataFrame with 'equity' and 'returns' columns.
    # Our equity_curve is a Series. Need to package it correctly.
    results_df_for_viz = pd.DataFrame({
        'equity': equity_curve,
        'returns': equity_curve.pct_change().fillna(0)
    })
    
    # Ensure the 'cycle_metrics' DataFrame is correctly formatted if passed to visualize_performance
    # The summarize_performance should return it in the right format.
    
    visualize_performance(
        results_df_for_viz,
        performance_summary, # Pass the full summary dict
        filename=f"{output_dir}/performance_options_optimized_{symbol}.png"
    )
    plt.show()

    return equity_curve


if __name__ == '__main__':
    target_symbol = "AAPL" # Example symbol
    print(f"Starting optimized backtest for {target_symbol}...")
    
    # Run the options-focused walk-forward backtest
    final_equity = walkforward_options_backtest(
        symbol=target_symbol,
        train_window=CONFIG.get("TRAIN_WINDOW_OPTIMIZED", 252 * 2), # 2 years train
        test_window=CONFIG.get("TEST_WINDOW_OPTIMIZED", 21 * 3),   # 3 months test/hold
        initial_capital=CONFIG.get("DEFAULT_CAPITAL", 100000),
        option_strategy_name=CONFIG.get("DEFAULT_STRATEGY", "Intelligent Call Buyer"),
        ml_model_proba_threshold_ui=CONFIG.get("DEFAULT_PROBA_THRESHOLD", 0.6),
        transaction_cost_options = CONFIG.get("TRANSACTION_COST_OPTIONS", 0.65), # Per contract
        max_holding_period_option = CONFIG.get("MAX_HOLDING_PERIOD_OPTION", 45) # Max days to hold option
    )

    if not final_equity.empty:
        print("\nBacktest finished.")
        print(f"Final equity for {target_symbol}: ${final_equity.iloc[-1]:,.2f}")
    else:
        print("Backtest did not produce results.")
