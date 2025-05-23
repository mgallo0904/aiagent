import argparse
import sys
import logging # Added for logging configuration

from PyQt6.QtWidgets import QApplication

# Local application imports
from security import SecurityManager
from ui import TradingUI
from trading_engine import TradingEngine, configure_logging as configure_engine_logging # Import TradingEngine and its logging config

# Logger for main.py
logger = logging.getLogger(__name__)


class Application:
    def __init__(self, operational_mode="live"):
        self.ui = TradingUI(main_app_logic=self)
        self.security_manager = SecurityManager()
        self.operational_mode = operational_mode
        
        # TradingEngine instance will be created when run_trading_logic_from_ui is called
        self.trading_engine: TradingEngine = None

        # Log application start via UI if available, otherwise console (though UI not shown yet)
        self.ui.log_message(
            f"Application initialized in {self.operational_mode.upper()} mode."
        )
        logger.info(f"Application initialized in {self.operational_mode.upper()} mode.")


    def authenticate_and_show_ui(self):
        self.ui.show()
        self.ui.set_operational_mode_display(self.operational_mode)
        return True

    def run_trading_logic_from_ui(self):
        self.ui.clear_all_outputs()
        self.ui.log_message(
            f"Starting new trading analysis from UI (Mode: {self.operational_mode.upper()})..."
        )
        logger.info(f"Starting new trading analysis from UI (Mode: {self.operational_mode.upper()})...")

        inputs = self.ui.get_inputs()
        user_symbol = inputs["symbol"]
        initial_capital = inputs["capital"]
        strategy_name = inputs["strategy_name"]
        ml_proba_threshold = inputs["proba_threshold"]
        data_period = inputs.get("data_period", "2y")
        data_interval = inputs.get("data_interval", "1d")

        if not user_symbol:
            self.ui.update_status("Error: Stock symbol cannot be empty.")
            self.ui.log_message("Error: Stock symbol was empty.")
            logger.error("Error: Stock symbol was empty.")
            return

        self.ui.update_status(
            f"Processing for {user_symbol} with {initial_capital} capital, using {strategy_name} strategy, ML threshold {ml_proba_threshold}."
        )
        self.ui.log_message(
            f"User inputs: Symbol={user_symbol}, Capital={initial_capital}, Strategy={strategy_name}, ML Threshold={ml_proba_threshold}, Period={data_period}, Interval={data_interval}"
        )
        logger.info(
            f"User inputs: Symbol={user_symbol}, Capital={initial_capital}, Strategy={strategy_name}, ML Threshold={ml_proba_threshold}, Period={data_period}, Interval={data_interval}"
        )
        
        try:
            # Instantiate and run the TradingEngine
            self.trading_engine = TradingEngine(
                symbol=user_symbol,
                initial_capital=initial_capital,
                strategy_name=strategy_name,
                ml_model_proba_threshold=ml_proba_threshold,
                operational_mode=self.operational_mode,
            )
            
            # The TradingEngine's methods will use its own logger.
            # For the UI, we need to bridge the information.
            # This version assumes TradingEngine logs sufficiently.
            # For more detailed UI updates, TradingEngine methods would need to return specific data.

            self.trading_engine._initialize_modules() # Call initialize separately
            
            # --- Data Fetching ---
            self.ui.log_message(f"Fetching market data for {user_symbol}...")
            # Use engine's market object after initialization
            historical_data = self.trading_engine._fetch_market_data(period=data_period, interval=data_interval)
            if historical_data is None or historical_data.empty:
                msg = f"Could not fetch historical data for {user_symbol} via engine."
                self.ui.update_status(f"Error: {msg}")
                self.ui.log_message(msg)
                self.ui.display_market_data({}) # Clear or show empty data
                return
            
            self.ui.log_message(f"Historical data fetched. Rows: {len(historical_data)}")
            # Display data using UI methods (requires market object from engine)
            market_display_data = {"historical_data": historical_data}
            market_display_data["ma_data"] = self.trading_engine.market.calculate_moving_average()
            market_display_data["rsi_data"] = self.trading_engine.market.calculate_rsi()
            upper, lower, mavg = self.trading_engine.market.calculate_bollinger_bands()
            market_display_data["bollinger_upper"] = upper
            market_display_data["bollinger_lower"] = lower
            market_display_data["bollinger_middle"] = mavg
            market_display_data["volatility_data"] = self.trading_engine.market.calculate_volatility()
            self.ui.display_market_data(market_display_data)
            self.ui.log_message("Market data and indicators displayed.")

            # --- ML Model Training ---
            self.ui.log_message(f"Training ML model for {user_symbol} via engine...")
            # _train_ml_model now returns a dictionary of metrics
            returned_metrics = self.trading_engine._train_ml_model(historical_data)
            
            accuracy_from_metrics = 0.0 # Default
            status_msg = "Model training completed or skipped."

            if isinstance(returned_metrics, dict):
                accuracy_from_metrics = returned_metrics.get('accuracy', 0.0)
                # Use the full metrics dictionary for UI display
                # Status message can be more generic or also derived from metrics
                if "error" in returned_metrics:
                    status_msg = f"ML training/evaluation error: {returned_metrics['error']}"
                elif accuracy_from_metrics > 0:
                    status_msg = f"Model trained. Test Accuracy: {accuracy_from_metrics:.2f}."
                else:
                    status_msg = "Model trained, but test accuracy is 0 or not available."
                
                # Log the detailed metrics received from the engine
                self.ui.log_message(f"Engine ML metrics: {returned_metrics}")
            else: # Fallback if returned_metrics is not a dict (e.g. old float accuracy)
                accuracy_from_metrics = float(returned_metrics) if isinstance(returned_metrics, (int, float)) else 0.0
                status_msg = f"Model trained. Accuracy (legacy): {accuracy_from_metrics:.2f}."
                returned_metrics = {"accuracy": accuracy_from_metrics, "status": "Legacy metrics format"}


            if self.trading_engine.ml_model and len(historical_data) < 20: # Check engine's state if needed
                status_msg = f"Not enough data for {user_symbol} to train ML model (need 20+). Skipping ML."
                # Ensure returned_metrics reflects this if it's not already an error from engine
                if not (isinstance(returned_metrics, dict) and "error" in returned_metrics):
                     returned_metrics = {"accuracy": 0.0, "status": status_msg}

            self.ui.display_ml_output(accuracy_from_metrics, returned_metrics) # Pass full metrics dict
            self.ui.log_message(f"ML Model training attempt finished. UI updated. {status_msg}")
            
            # --- Strategy Execution ---
            self.ui.log_message(f"Executing strategy {strategy_name} via engine...")
            # _execute_strategy_and_trades in engine now handles logging of its actions.
            # It might need to return action and trade params for UI display.
            # For now, we assume UI logs from Application are sufficient guide, and engine logs details.
            
            # The TradingEngine's _execute_strategy_and_trades method needs to be called.
            # It internally calls _process_trade_action, which logs.
            # To display results in UI, we might need _execute_strategy_and_trades to return info.
            # Let's assume for now that engine logging is primary for headless, and UI logs guide user.
            # This is a simplification point.
            
            # We need to get the action from the strategy via the engine for UI display
            action = self.trading_engine.strategy.execute(self.trading_engine.market, self.trading_engine.ml_model)
            self.ui.display_strategy_action(action)
            self.ui.log_message(f"Strategy action from engine: {action}")

            is_valid_action = False
            if action and isinstance(action, dict):
                action_type = action.get("action", "").lower()
                if action_type and action_type != "hold":
                    is_valid_action = True
            
            if is_valid_action:
                # Process trade action using engine's method. It logs internally.
                # To update UI with execution details, _process_trade_action would need to return status.
                # For now, Application logs the intent, engine logs the execution details.
                self.ui.log_message(f"Processing trade action via engine: {action}")
                self.trading_engine._process_trade_action(action) 
                # How to get trade execution status back to UI?
                # This is a gap if TradingEngine._process_trade_action does not return status for UI.
                # Assuming for now that engine logs are enough and UI gets a generic message.
                # A better way: engine returns dict, UI displays it.
                # For now, let's add a generic message.
                self.ui.display_trade_execution({
                    "status": "Processed by Engine",
                    "message": f"Trade action {action.get('action')} for {action.get('details', {}).get('symbol')} processed by engine. Check engine logs for details."
                })

            else:
                msg = f"No trade action (or 'hold') to execute for {self.trading_engine.market.symbol}."
                self.ui.log_message(msg)
                self.ui.display_trade_execution({"status": "Hold or No Action", "message": msg})

            # --- Final Checks (Risk and Compliance) ---
            self.ui.log_message("Performing final checks (Risk via engine, Compliance via Application)...")
            # Risk assessment is done by the engine
            risk_level = self.trading_engine.risk_manager.assess_risk(
                market_data=self.trading_engine.market, position_details=None # Engine might manage positions internally
            )
            self.ui.display_risk_assessment(risk_level) # Update UI with risk level
            self.ui.log_message(f"Engine risk assessment: {risk_level}")

            # Compliance check remains with Application (can be debated)
            if not self.security_manager.ensure_compliance():
                self.ui.log_message("Compliance check failed.")
                self.ui.update_status("Compliance Check FAILED.")
            else:
                self.ui.log_message("System is compliant.")
                self.ui.update_status(
                    f"Processing for {user_symbol} complete. System compliant."
                )
            
            self.ui.log_message("Trading analysis from UI finished.")
            logger.info("Trading analysis from UI finished.")

        except Exception as e:
            self.ui.log_message(f"Error during trading logic execution: {e}")
            self.ui.update_status(f"Error: {e}")
            logger.error(f"Error during trading logic execution: {e}", exc_info=True)


def main_gui(cli_args):
    app = QApplication.instance()
    if not app:
        app = QApplication(sys.argv)

    main_app = Application(operational_mode=cli_args.mode)

    if cli_args.proba_threshold is not None:
        main_app.ui.set_default_proba_threshold(cli_args.proba_threshold)
    
    # Configure engine logging to console by default for GUI mode, or as per args if specified
    # This is tricky because UI also has its own logging.
    # Let's make engine log to console if GUI is running.
    log_file = cli_args.log_file if hasattr(cli_args, 'log_file') and cli_args.log_file else None
    configure_engine_logging(log_to_file=bool(log_file), log_file_path=log_file)
    logger.info("Standard logging configured for GUI mode (engine logs to console/file).")


    if main_app.authenticate_and_show_ui():
        sys.exit(app.exec())
    else:
        logger.error("Failed to initialize or authenticate GUI application.")
        sys.exit(1)

def main_headless(cli_args):
    # Configure logging for headless mode (file or console)
    configure_engine_logging(log_to_file=bool(cli_args.log_file), log_file_path=cli_args.log_file)
    logger.info(f"Headless mode started. Engine logging to: {'File ({cli_args.log_file})' if cli_args.log_file else 'Console'}")
    
    if not cli_args.symbol:
        logger.error("Symbol must be provided for headless mode. Use --symbol <SYMBOL>")
        print("Error: --symbol is required for headless mode.", file=sys.stderr)
        sys.exit(1)

    try:
        engine = TradingEngine(
            symbol=cli_args.symbol,
            initial_capital=cli_args.capital,
            strategy_name=cli_args.strategy,
            ml_model_proba_threshold=cli_args.proba_threshold if cli_args.proba_threshold is not None else 0.5, # Default if not set
            operational_mode=cli_args.mode,
        )
        # Pass data period and interval from args
        engine.run_trading_cycle(data_period=cli_args.data_period, data_interval=cli_args.data_interval)
        logger.info(f"Headless trading cycle for {cli_args.symbol} completed.")

    except Exception as e:
        logger.error(f"Error in headless mode execution: {e}", exc_info=True)
        print(f"Error during headless execution: {e}", file=sys.stderr)
        sys.exit(1)
    
    sys.exit(0)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="AI Trading Agent")
    
    # Common arguments for both modes
    parser.add_argument(
        "--proba_threshold",
        type=float,
        help="Probability threshold for ML model (0.0 to 1.0). Overrides UI/config default.",
    )
    parser.add_argument(
        "--mode",
        type=str,
        choices=["live", "backtest", "paper"], # Added paper
        default="live",
        help="Operational mode for TradingEngine.",
    )
    parser.add_argument(
        "--log_file",
        type=str,
        default=None, # None means console logging for the engine
        help="Path to a file for engine logging. If not set, logs to console."
    )

    # Subparsers for different run modes (gui vs headless)
    subparsers = parser.add_subparsers(title="run_mode", dest="run_mode", required=True,
                                       help="Specify 'gui' to run with UI, or 'headless' for command-line execution.")

    # GUI mode parser (can have gui-specific args if needed later)
    parser_gui = subparsers.add_parser("gui", help="Run the application with the PyQt6 GUI.")
    # GUI mode doesn't need symbol, capital etc. here as they are input via UI.
    
    # Headless mode parser
    parser_headless = subparsers.add_parser("headless", help="Run the trading engine in headless mode.")
    parser_headless.add_argument(
        "--symbol",
        type=str,
        required=True, # Symbol is required for headless
        help="Stock symbol (e.g., AAPL) for headless trading."
    )
    parser_headless.add_argument(
        "--capital",
        type=float,
        default=10000.0,
        help="Initial capital for headless trading."
    )
    parser_headless.add_argument(
        "--strategy",
        type=str,
        default="DefaultStrategy", # Make sure this strategy exists
        help="Trading strategy name for headless mode."
    )
    parser_headless.add_argument(
        "--data_period",
        type=str,
        default="2y",
        help="Historical data period (e.g., '1y', '6mo') for headless mode."
    )
    parser_headless.add_argument(
        "--data_interval",
        type=str,
        default="1d",
        help="Historical data interval (e.g., '1d', '1h') for headless mode."
    )

    args = parser.parse_args()

    # Basic logging setup for the main.py itself (distinct from engine's logger)
    # This will also be affected by configure_engine_logging if it resets all handlers.
    # It's better if configure_engine_logging in trading_engine.py takes a logger name.
    # For now, let's assume configure_engine_logging sets up for the 'trading_engine' logger.
    # And here we set up a basic one for 'main' or root logger.
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s', stream=sys.stdout)

    if args.run_mode == "gui":
        main_gui(args)
    elif args.run_mode == "headless":
        main_headless(args)
    else:
        # Should not happen due to 'required=True' in subparsers
        logger.error(f"Invalid run_mode: {args.run_mode}")
        parser.print_help()
        sys.exit(1)
