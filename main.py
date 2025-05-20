import argparse  # Added argparse
import sys

import numpy as np
from PyQt6.QtWidgets import QApplication

from market_data import MarketData
from ml_models import MLModel, ModelBehaviorConfig  # Added ModelBehaviorConfig
from risk_management import RiskManager
from security import SecurityManager
from strategies import TradingStrategy
from trade_executor import TradeExecutor
from ui import TradingUI
from ml_extensions import prepare_features


class Application:
    def __init__(self, operational_mode="live"):  # Added operational_mode
        self.ui = TradingUI(main_app_logic=self)
        self.security_manager = SecurityManager()
        self.market = None
        self.strategy = None
        self.ml_model = None
        self.risk_manager = None
        self.executor = None
        self.operational_mode = operational_mode  # Store operational mode
        self.ui.log_message(
            f"Application started in {self.operational_mode.upper()} mode."
        )

    def authenticate_and_show_ui(self):
        self.ui.show()
        self.ui.set_operational_mode_display(
            self.operational_mode
        )  # Update UI with mode
        return True

    def _initialize_modules(
        self, user_symbol, initial_capital, strategy_name, ml_model_proba_threshold
    ):
        self.ui.log_message("Initializing core modules...")
        self.market = MarketData(symbol=user_symbol)
        self.strategy = TradingStrategy(name=strategy_name)

        # Create ModelBehaviorConfig instance
        behavior_config = ModelBehaviorConfig(
            proba_threshold=ml_model_proba_threshold,
            early_stopping_rounds=15,  # Increased from default 10
            random_state=42,
            primary_metric="accuracy",
            n_iter=50,  # Number of hyperparameter combinations to try (set to 50 for diagnostics)
            # Time-series aware CV
            use_time_series_cv=True,
            time_series_n_splits=5,
            # Probability calibration
            calibrate_probabilities=True,
            calibration_method="isotonic",
        )
        # Instantiate MLModel with the behavior_config
        # Default to XGBoost if not specified, or allow selection via strategy/config later
        model_type_to_use = "xgboost"  # PENDING TASK 4: Prefer XGBoost
        model_params_to_use = {
            "n_estimators": 150,
            "max_depth": 4,
            "learning_rate": 0.05,
            "random_state": 42,  # Ensure reproducibility
            "reg_alpha": 0.1,     # L1 regularization for sparsity
            "reg_lambda": 1.0,    # L2 regularization
            "min_child_weight": 5, # Prevents tiny leaves
            "gamma": 0.1,          # Min loss reduction for splits
        }
        self.ml_model = MLModel(
            model_type=model_type_to_use,
            params=model_params_to_use,
            behavior_config=behavior_config,
        )
        self.ui.log_message(
            f"ML Model ({model_type_to_use}) initialized with custom params and threshold: {ml_model_proba_threshold}"
        )

        self.risk_manager = RiskManager(total_capital=initial_capital)
        self.executor = TradeExecutor()
        self.ui.log_message("Core modules initialized.")

    def _fetch_and_display_market_data(self):
        self.ui.log_message(
            f"Fetching historical data for {self.market.symbol}...")
            
        # Get data settings from UI if available, otherwise use defaults
        inputs = self.ui.get_inputs()
        data_period = inputs.get("data_period", "2y")  # Default 2 years instead of 6mo
        data_interval = inputs.get("data_interval", "1d")  # Default daily
        
        self.ui.log_message(f"Using data parameters: period={data_period}, interval={data_interval}")
            
        data = self.market.fetch_historical(period=data_period, interval=data_interval)
        if data is None or data.empty:
            msg = f"Could not fetch historical data for {self.market.symbol}."
            self.ui.update_status(f"Error: {msg}")
            self.ui.log_message(msg)
            self.ui.display_market_data({})
            return None
        self.ui.log_message(f"Historical data fetched. Rows: {len(data)}")

        market_display_data = {"historical_data": data}
        self.ui.log_message("Calculating technical indicators...")
        market_display_data["ma_data"] = self.market.calculate_moving_average()
        market_display_data["rsi_data"] = self.market.calculate_rsi()
        upper, lower, mavg = self.market.calculate_bollinger_bands()
        market_display_data["bollinger_upper"] = upper
        market_display_data["bollinger_lower"] = lower
        market_display_data["bollinger_middle"] = mavg
        market_display_data["volatility_data"] = self.market.calculate_volatility(
        )
        self.ui.display_market_data(market_display_data)
        self.ui.log_message("Technical indicators calculated and displayed.")
        return data

    def _train_ml_model(self, data):
        if len(data) < 20:
            msg = f"Not enough data for {self.market.symbol} to train ML model (need 20+). Skipping ML."
            self.ui.log_message(msg)
            self.ui.display_ml_output(0.0, {"status": msg})
            return 0.0

        self.ui.log_message(f"Training ML model for {self.market.symbol}...")
        # Use prepare_features from ml_extensions to get X, y
        X, y = prepare_features(data)
        result = self.ml_model.fit(X, y)
        # Optionally, you can add evaluation here if needed
        accuracy = None
        if hasattr(self.ml_model.model, 'score'):
            accuracy = self.ml_model.model.score(X, y)
            self.ui.display_ml_output(accuracy, {"status": "Model trained."})
            self.ui.log_message(f"ML Model trained. Accuracy: {accuracy:.2f}")
            return accuracy
        self.ui.display_ml_output(0.0, {"status": "Training complete, but no accuracy available."})
        return 0.0

    def _extract_trade_details(self, action: dict) -> tuple:
        """Helper to extract trade details from an action dictionary."""
        order_type = action.get("action", "").upper()
        action_details = action.get("details", {})
        trade_symbol = action_details.get("symbol", self.market.symbol)
        quantity = action_details.get("quantity", 0)
        option_type = action_details.get("option_type")
        strike_price = action_details.get("strike")
        expiry_date = action_details.get("expiry")

        self.ui.log_message(
            f"Preparing trade: {order_type} {quantity} of {trade_symbol} (Mode: {self.operational_mode.upper()})"
        )
        if option_type:
            self.ui.log_message(
                f"Option details: Type={option_type}, Strike={strike_price}, Expiry={expiry_date}"
            )

        return (
            order_type,
            trade_symbol,
            quantity,
            option_type,
            strike_price,
            expiry_date,
        )

    def _calculate_trade_parameters(
        self, order_type: str, trade_symbol: str
    ) -> tuple | None:
        """Helper to fetch price and calculate stop-loss/take-profit."""
        current_price = self.market.get_current_price(
            symbol_override=trade_symbol
        )  # Allow override for options
        if current_price is None:
            msg = f"Could not retrieve current price for {trade_symbol} (or its underlying). Skipping trade."
            self.ui.log_message(msg)
            self.ui.display_trade_execution(
                {"status": "Error", "message": msg})
            return None

        self.ui.log_message(
            f"Reference price for {trade_symbol} (or underlying): {current_price}"
        )
        stop_loss_price = self.risk_manager.calculate_stop_loss(
            current_price, order_type
        )
        take_profit_price = self.risk_manager.calculate_take_profit(
            current_price, order_type
        )
        self.ui.log_message(
            f"Calculated SL: {stop_loss_price}, TP: {take_profit_price} (based on reference price)"
        )
        return current_price, stop_loss_price, take_profit_price

    def _execute_or_simulate_trade(self, final_trade_params: dict):
        """Helper to execute or simulate the trade based on operational mode."""
        if self.operational_mode == "backtest":
            self.ui.log_message(
                f"BACKTEST MODE: Simulated trade execution. Parameters: {final_trade_params}"
            )
            self.ui.display_trade_execution(
                {
                    "status": "Simulated (Backtest)",
                    "message": "Trade simulated in backtest mode.",
                    "details": final_trade_params,
                }
            )
        else:  # Live mode
            # In a real scenario, this would call self.executor.execute_trade(final_trade_params)
            self.ui.log_message(
                f"LIVE MODE: Trade execution SKIPPED (MANUAL OVERRIDE). Parameters: {final_trade_params}"
            )
            self.ui.display_trade_execution(
                {
                    "status": "Skipped (Manual - Live Mode)",
                    "message": "Trade execution is handled manually.",
                    "details": final_trade_params,
                }
            )

    def _get_model_confidence_scalar(self, order_type: str) -> float | None:
        """Determines a confidence scalar based on ML model output for the given order type."""
        model_confidence_scalar = None
        if self.ml_model and self.ml_model.last_prediction_proba is not None:
            proba_positive_class = self.ml_model.last_prediction_proba

            if (
                order_type == "BUY_CALL"
            ):  # Assumes positive class (e.g., price up) is favorable
                model_confidence_scalar = proba_positive_class
            elif (
                order_type
                == "BUY_PUT"
                # Assumes lower proba of positive class (price up) is favorable for puts
            ):
                model_confidence_scalar = 1.0 - proba_positive_class
            # Add other order types if ML model provides relevant probabilities for them

            if model_confidence_scalar is not None:
                model_confidence_scalar = np.clip(
                    model_confidence_scalar, 0.0, 1.0
                )  # Ensure 0-1 range
                self.ui.log_message(
                    f"Derived ML model confidence scalar: {model_confidence_scalar:.4f} for order type {order_type}."
                )
            else:
                self.ui.log_message(
                    f"No specific ML confidence logic for order type {order_type} or probability not applicable. Using default risk."
                )
        else:
            self.ui.log_message(
                "ML model or last_prediction_proba not available. Using default risk for position sizing."
            )
        return model_confidence_scalar

    def _calculate_risk_adjusted_quantity(
        self,
        current_price: float,
        order_type: str,
        stop_loss_price: float | None,
        model_confidence_scalar: float | None,
    ) -> int:
        """Calculates position size using RiskManager, incorporating model confidence."""
        position_size_params = self.risk_manager.PositionSizeParams(
            current_price=current_price,
            order_type=order_type,
            stop_loss_price=stop_loss_price,
            atr_value=None,  # Placeholder for ATR if it becomes available/necessary here
            model_confidence_scalar=model_confidence_scalar,
        )
        calculated_quantity = self.risk_manager.calculate_position_size(
            params=position_size_params
        )
        self.ui.log_message(
            f"RiskManager calculated position size: {calculated_quantity} (incorporating confidence: {model_confidence_scalar})"
        )
        return calculated_quantity

    def _determine_final_trade_quantity(
        self, strategy_quantity: int, risk_adjusted_quantity: int, is_option_trade: bool
    ) -> int:
        """Determines the final quantity for a trade based on strategy and risk assessment."""
        final_quantity: int

        if risk_adjusted_quantity <= 0:  # Corrected from `== 0` to `<=0` for safety
            self.ui.log_message(
                f"Risk-adjusted position size is {risk_adjusted_quantity}. No trade will be placed, overriding strategy quantity {strategy_quantity}."
            )
            return 0

        if is_option_trade:
            final_quantity = min(strategy_quantity, risk_adjusted_quantity)
            if final_quantity != strategy_quantity:
                self.ui.log_message(
                    f"Strategy option quantity {strategy_quantity} adjusted to {final_quantity} "
                    f"based on risk/confidence assessment (max allowed: {risk_adjusted_quantity})."
                )
        else:  # For stock trades
            final_quantity = risk_adjusted_quantity
            if final_quantity != strategy_quantity and strategy_quantity > 0:
                self.ui.log_message(
                    f"Strategy stock quantity {strategy_quantity} overridden by risk-adjusted quantity {final_quantity}."
                )

        if final_quantity < 0:
            self.ui.log_message(
                f"Warning: Final quantity calculated as {final_quantity}. Setting to 0."
            )
            return 0

        return final_quantity

    def _process_trade_action(self, action):
        (
            order_type,
            trade_symbol,
            strategy_quantity,
            option_type,
            strike_price,
            expiry_date,
        ) = self._extract_trade_details(action)

        if strategy_quantity <= 0:
            msg = f"Strategy provided non-positive quantity ({strategy_quantity}) for trade action: {action}. Skipping."
            self.ui.log_message(msg)
            self.ui.display_trade_execution(
                {"status": "Skipped", "message": msg})
            return

        price_params = self._calculate_trade_parameters(
            order_type, trade_symbol)
        if price_params is None:
            return

        current_price, stop_loss_price, take_profit_price = price_params

        model_confidence_scalar = self._get_model_confidence_scalar(order_type)

        risk_adj_order_type = "BUY"
        if "SELL" in order_type or "PUT" in order_type:
            risk_adj_order_type = "SELL"

        calculated_risk_adj_quantity = self._calculate_risk_adjusted_quantity(
            current_price, risk_adj_order_type, stop_loss_price, model_confidence_scalar
        )

        is_option = bool(option_type)
        final_quantity = self._determine_final_trade_quantity(
            strategy_quantity, calculated_risk_adj_quantity, is_option
        )

        if final_quantity <= 0:
            msg = f"Final calculated quantity is {final_quantity} (strategy: {strategy_quantity}, risk-adj: {calculated_risk_adj_quantity}). No trade will be placed."
            self.ui.log_message(msg)
            self.ui.display_trade_execution(
                {"status": "Skipped", "message": msg})
            return

        trade_parameters = {
            "symbol": trade_symbol,
            "quantity": final_quantity,
            "order_type": order_type,
            "price": None,  # Market order assumed, or use current_price for limit if supported
            "stop_loss": stop_loss_price,
            "take_profit": take_profit_price,
            "option_type": option_type,
            "strike": strike_price,
            "expiry": expiry_date,
        }
        final_trade_params = {
            k: v for k, v in trade_parameters.items() if v is not None
        }

        self._execute_or_simulate_trade(final_trade_params)

    def _execute_strategy_and_trades(self):
        self.ui.log_message(
            f"Executing trading strategy: {self.strategy.name}...")
        action = self.strategy.execute(self.market)
        self.ui.display_strategy_action(action)
        self.ui.log_message(f"Strategy action: {action}")

        is_valid_action = False
        if action and isinstance(action, dict):
            action_type = action.get("action", "").lower()
            if action_type and action_type != "hold":
                is_valid_action = True

        if is_valid_action:
            self._process_trade_action(action)
        else:
            msg = f"No trade action (or 'hold') to execute for {self.market.symbol}."
            self.ui.log_message(msg)
            self.ui.display_trade_execution(
                {"status": "Hold or No Action", "message": msg}
            )

    def _perform_final_checks(self):
        self.ui.log_message("Assessing risk...")
        risk_level = self.risk_manager.assess_risk(
            market_data=self.market, position_details=None
        )
        self.ui.display_risk_assessment(risk_level)
        self.ui.log_message(f"Risk assessment: {risk_level}")

        self.ui.log_message("Performing compliance check...")
        if not self.security_manager.ensure_compliance():
            self.ui.log_message("Compliance check failed.")
            self.ui.update_status("Compliance Check FAILED.")
        else:
            self.ui.log_message("System is compliant.")
            self.ui.update_status(
                f"Processing for {self.market.symbol} complete. System compliant."
            )

    def run_trading_logic_from_ui(self):
        self.ui.clear_all_outputs()  # Ensure logs are cleared first
        self.ui.log_message(
            f"Starting new trading analysis from UI (Mode: {self.operational_mode.upper()})..."
        )

        inputs = self.ui.get_inputs()
        user_symbol = inputs["symbol"]
        initial_capital = inputs["capital"]
        strategy_name = inputs["strategy_name"]
        # Changed key to match ui.py
        ml_proba_threshold = inputs["proba_threshold"]

        if not user_symbol:
            self.ui.update_status("Error: Stock symbol cannot be empty.")
            self.ui.log_message("Error: Stock symbol was empty.")
            return

        self.ui.update_status(
            f"Processing for {user_symbol} with {initial_capital} capital, using {strategy_name} strategy, ML threshold {ml_proba_threshold}."
        )
        self.ui.log_message(
            f"User inputs: Symbol={user_symbol}, Capital={initial_capital}, Strategy={strategy_name}, ML Threshold={ml_proba_threshold}"
        )

        self._initialize_modules(
            user_symbol, initial_capital, strategy_name, ml_proba_threshold
        )

        historical_data = self._fetch_and_display_market_data()
        if historical_data is None:
            return

        self._train_ml_model(historical_data)
        self._execute_strategy_and_trades()
        self._perform_final_checks()

        self.ui.log_message("Trading analysis from UI finished.")


def main_gui(
    cli_proba_threshold=None, cli_mode="live"
):  # Added cli_mode, default to 'live'
    app = QApplication.instance()
    if not app:
        app = QApplication(sys.argv)

    # Pass mode to Application
    main_app = Application(operational_mode=cli_mode)

    if cli_proba_threshold is not None:
        main_app.ui.set_default_proba_threshold(cli_proba_threshold)

    if main_app.authenticate_and_show_ui():
        sys.exit(app.exec())
    else:
        print("Failed to initialize or authenticate application.")
        sys.exit(1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="AI Trading Agent")
    parser.add_argument(
        "--proba_threshold",
        type=float,
        help="Probability threshold for ML model classification (0.0 to 1.0). Overrides UI default if set.",
    )
    parser.add_argument(
        "--mode",
        type=str,
        choices=["live", "backtest"],
        default="live",
        help="Operational mode: 'live' for real trading (manual override currently) or 'backtest' for simulated trading.",
    )
    args = parser.parse_args()

    main_gui(cli_proba_threshold=args.proba_threshold, cli_mode=args.mode)
