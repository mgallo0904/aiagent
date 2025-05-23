import logging
import numpy as np

from market_data import MarketData
from ml_models import MLModel, ModelBehaviorConfig
from risk_management import RiskManager
from strategies import TradingStrategy
from trade_executor import TradeExecutor
from ml_extensions import prepare_features

# Configure basic logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class TradingEngine:
    def __init__(self, symbol: str, initial_capital: float, strategy_name: str, ml_model_proba_threshold: float, operational_mode: str = "live"):
        self.symbol = symbol
        self.initial_capital = initial_capital
        self.strategy_name = strategy_name
        self.ml_model_proba_threshold = ml_model_proba_threshold
        self.operational_mode = operational_mode

        self.market: MarketData = None
        self.strategy: TradingStrategy = None
        self.ml_model: MLModel = None
        self.risk_manager: RiskManager = None
        self.executor: TradeExecutor = None

        logger.info(f"TradingEngine initialized for {self.symbol} in {self.operational_mode.upper()} mode.")

    def _initialize_modules(self):
        logger.info("Initializing core modules...")
        self.market = MarketData(symbol=self.symbol)
        
        # Initialize OptionPricer first, as it might be needed by the strategy
        # For now, OptionPricer uses default rf_rate and div_yield.
        # These could be made configurable in TradingEngine or passed via params if needed.
        option_pricer = strategies.OptionPricer() # Import from strategies was implicit, make explicit if needed

        # Initialize ML Model (already done before strategy)
        # This is a bit of a chicken-and-egg if ML model itself was a strategy choice,
        # but here ML model is a component used BY a strategy.
        behavior_config = ModelBehaviorConfig(
            proba_threshold=self.ml_model_proba_threshold,
            early_stopping_rounds=15,
            random_state=42,
            primary_metric="accuracy",
            n_iter=50,
            use_time_series_cv=True,
            time_series_n_splits=5,
            calibrate_probabilities=True,
            calibration_method="isotonic",
        )
        
        model_type_to_use = "xgboost" # This could be made configurable
        model_params_to_use = { # These could also be part of engine config
            "n_estimators": 150, "max_depth": 4, "learning_rate": 0.05,
            "random_state": 42, "reg_alpha": 0.1, "reg_lambda": 1.0,
            "min_child_weight": 5, "gamma": 0.1,
        }
        # Ensure self.ml_model is initialized before being passed to a strategy
        if self.ml_model is None: # Should always be None at this stage based on constructor
            self.ml_model = MLModel(
                model_type=model_type_to_use,
                # config= appropriate ModelConfig instance, # TODO: MLModel expects ModelConfig, not just params
                behavior_config=behavior_config,
                params=model_params_to_use 
            )
            logger.info(
                f"ML Model ({model_type_to_use}) initialized with custom params and threshold: {self.ml_model_proba_threshold}"
            )
        else:
            logger.info("ML Model already initialized.")


        # Initialize Strategy, potentially passing the ml_model and option_pricer
        # The strategy constructor in strategies.py was updated to accept these.
        # Specific params for the strategy (like DTE, OTM%, thresholds) can be passed via a 'params' dict
        # For now, they will use defaults set in TradingStrategy.__init__ or MLSignalStrategy specific defaults.
        strategy_params = {
            "proba_threshold_long": self.ml_model_proba_threshold, # Example: reuse engine's proba threshold
            "proba_threshold_short": self.ml_model_proba_threshold, # Or have separate config for short
            # Add other strategy-specific params here if they need to be configured from engine level
        }

        self.strategy = TradingStrategy(
            name=self.strategy_name,
            ml_model_instance=self.ml_model if self.strategy_name == strategies.STRATEGY_ML_SIGNAL else None,
            option_pricer_instance=option_pricer,
            params=strategy_params 
        )
        logger.info(f"Strategy '{self.strategy_name}' initialized. ML model linked: {self.strategy.ml_model is not None}.")

        self.risk_manager = RiskManager(total_capital=self.initial_capital)
        self.executor = TradeExecutor()
        logger.info("Core modules initialized.")

    def run_trading_cycle(self, data_period="2y", data_interval="1d"):
        logger.info(f"Starting trading cycle for {self.symbol}...")
        self._initialize_modules()
        
        historical_data = self._fetch_market_data(period=data_period, interval=data_interval)
        if historical_data is None or historical_data.empty:
            logger.error("Halting trading cycle due to failure in fetching market data.")
            return

        self._train_ml_model(historical_data)
        self._execute_strategy_and_trades()
        self._perform_final_checks()
        logger.info(f"Trading cycle for {self.symbol} finished.")

    def _fetch_market_data(self, period="2y", interval="1d"):
        logger.info(f"Fetching historical data for {self.market.symbol} with period={period}, interval={interval}...")
        data = self.market.fetch_historical(period=period, interval=interval)
        if data is None or data.empty:
            msg = f"Could not fetch historical data for {self.market.symbol}."
            logger.error(msg)
            return None
        
        logger.info(f"Historical data fetched. Rows: {len(data)}")
        # Technical indicators are calculated within MarketData as needed by strategies or analysis
        # For now, we just return the raw data. If specific indicators are needed by other
        # engine components directly, this method can be expanded.
        return data

    def _train_ml_model(self, data):
        if len(data) < 20:
            msg = f"Not enough data for {self.market.symbol} to train ML model (need 20+). Skipping ML."
            logger.warning(msg)
            return 0.0

        logger.info(f"Training ML model for {self.market.symbol}...")
        X, y = prepare_features(data) # prepare_features uses default n_forward=3, threshold=0.01
        
        # MLModel.fit now performs an 80/20 split, trains on 80%, evaluates on 20%, and returns test metrics
        test_metrics = self.ml_model.fit(X, y) 
        
        if test_metrics and not isinstance(test_metrics, dict):
            # This case might occur if fit returned something unexpected (e.g. old accuracy float)
            logger.warning(f"MLModel.fit returned an unexpected type: {type(test_metrics)}. Expected dict of metrics.")
            # Fallback for backward compatibility or error handling
            return {"accuracy": float(test_metrics) if isinstance(test_metrics, (int,float)) else 0.0, "status": "Metrics format error"}

        if "error" in test_metrics:
            logger.error(f"ML model training/evaluation failed: {test_metrics['error']}")
            return {"accuracy": 0.0, "status": f"Error: {test_metrics['error']}"}

        logger.info(f"ML Model training and out-of-sample evaluation complete. Test Metrics: {test_metrics}")
        
        # Return the dictionary of test metrics.
        # The UI will need to be updated to display these richer metrics.
        # For now, ensure a consistent return type (dict).
        # If 'accuracy' is not in test_metrics, add a default or handle it in the caller.
        if 'accuracy' not in test_metrics:
            # This shouldn't happen if evaluate() always includes it for classification
            # or a primary metric for regression.
            logger.warning("Accuracy not found in returned test metrics dict. Setting to 0.0 for compatibility.")
            test_metrics['accuracy'] = 0.0 
            
        return test_metrics


    def _extract_trade_details(self, action: dict) -> tuple:
        order_type = action.get("action", "").upper()
        action_details = action.get("details", {})
        trade_symbol = action_details.get("symbol", self.market.symbol)
        quantity = action_details.get("quantity", 0)
        option_type = action_details.get("option_type")
        strike_price = action_details.get("strike")
        expiry_date = action_details.get("expiry")

        logger.info(
            f"Preparing trade: {order_type} {quantity} of {trade_symbol} (Mode: {self.operational_mode.upper()})"
        )
        if option_type:
            logger.info(
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
        current_price = self.market.get_current_price(
            symbol_override=trade_symbol
        )
        if current_price is None:
            msg = f"Could not retrieve current price for {trade_symbol} (or its underlying). Skipping trade."
            logger.error(msg)
            return None

        logger.info(
            f"Reference price for {trade_symbol} (or underlying): {current_price}"
        )
        stop_loss_price = self.risk_manager.calculate_stop_loss(
            current_price, order_type
        )
        take_profit_price = self.risk_manager.calculate_take_profit(
            current_price, order_type
        )
        logger.info(
            f"Calculated SL: {stop_loss_price}, TP: {take_profit_price} (based on reference price)"
        )
        return current_price, stop_loss_price, take_profit_price

    def _execute_or_simulate_trade(self, final_trade_params: dict):
        if self.operational_mode == "backtest":
            logger.info(
                f"BACKTEST MODE: Simulated trade execution. Parameters: {final_trade_params}"
            )
            # In a real headless backtest, you might record this to a results store
        elif self.operational_mode == "live":
            logger.info(
                f"LIVE MODE: Trade execution SKIPPED (MANUAL OVERRIDE). Parameters: {final_trade_params}"
            )
            # In a real live headless engine, this would call:
            # self.executor.execute_trade(final_trade_params)
        else: # Paper trading or other modes
            logger.info(
                f"{self.operational_mode.upper()} MODE: Simulated trade. Parameters: {final_trade_params}"
            )


    def _get_model_confidence_scalar(self, order_type: str) -> float | None:
        model_confidence_scalar = None
        if self.ml_model and self.ml_model.last_prediction_proba is not None:
            proba_positive_class = self.ml_model.last_prediction_proba

            if order_type == "BUY_CALL":
                model_confidence_scalar = proba_positive_class
            elif order_type == "BUY_PUT":
                model_confidence_scalar = 1.0 - proba_positive_class
            
            if model_confidence_scalar is not None:
                model_confidence_scalar = np.clip(model_confidence_scalar, 0.0, 1.0)
                logger.info(
                    f"Derived ML model confidence scalar: {model_confidence_scalar:.4f} for order type {order_type}."
                )
            else:
                logger.info(
                    f"No specific ML confidence logic for order type {order_type} or probability not applicable. Using default risk."
                )
        else:
            logger.info(
                "ML model or last_prediction_proba not available. Using default risk for position sizing."
            )
        return model_confidence_scalar

    def _calculate_risk_adjusted_quantity(
        self,
        current_price: float,
        order_type: str, # Should be 'BUY' or 'SELL' for risk calc
        stop_loss_price: float | None,
        model_confidence_scalar: float | None,
    ) -> int:
        position_size_params = self.risk_manager.PositionSizeParams(
            current_price=current_price,
            order_type=order_type, # This should be 'BUY' or 'SELL'
            stop_loss_price=stop_loss_price,
            atr_value=None, 
            model_confidence_scalar=model_confidence_scalar,
        )
        calculated_quantity = self.risk_manager.calculate_position_size(
            params=position_size_params
        )
        logger.info(
            f"RiskManager calculated position size: {calculated_quantity} (incorporating confidence: {model_confidence_scalar})"
        )
        return calculated_quantity

    def _determine_final_trade_quantity(
        self, strategy_quantity: int, risk_adjusted_quantity: int, is_option_trade: bool
    ) -> int:
        final_quantity: int

        if risk_adjusted_quantity <= 0:
            logger.warning(
                f"Risk-adjusted position size is {risk_adjusted_quantity}. No trade will be placed, overriding strategy quantity {strategy_quantity}."
            )
            return 0

        if is_option_trade:
            final_quantity = min(strategy_quantity, risk_adjusted_quantity)
            if final_quantity != strategy_quantity:
                logger.info(
                    f"Strategy option quantity {strategy_quantity} adjusted to {final_quantity} "
                    f"based on risk/confidence assessment (max allowed: {risk_adjusted_quantity})."
                )
        else:  # For stock trades
            final_quantity = risk_adjusted_quantity
            if final_quantity != strategy_quantity and strategy_quantity > 0 : # only log if strategy proposed a trade
                logger.info(
                    f"Strategy stock quantity {strategy_quantity} overridden by risk-adjusted quantity {final_quantity}."
                )
        
        if final_quantity < 0: # Should not happen if risk_adjusted_quantity is positive and strategy_quantity is positive
            logger.warning(f"Warning: Final quantity calculated as {final_quantity}. Setting to 0.")
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
            logger.info(msg)
            return

        price_params = self._calculate_trade_parameters(
            order_type, trade_symbol)
        if price_params is None:
            return # Error already logged

        current_price, stop_loss_price, take_profit_price = price_params

        model_confidence_scalar = self._get_model_confidence_scalar(order_type)
        
        # Determine risk adjustment type ('BUY' or 'SELL')
        # This is simplified; complex strategies might need more nuance
        risk_adj_order_type = "BUY" 
        if "SELL" in order_type or "PUT" in order_type: # Covers SELL_SHORT, BUY_PUT (as it's bearish)
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
            logger.info(msg)
            return

        trade_parameters = {
            "symbol": trade_symbol,
            "quantity": final_quantity,
            "order_type": order_type,
            "price": None, 
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
        logger.info(
            f"Executing trading strategy: {self.strategy.name}..."
        )
        # Pass the ml_model instance to the strategy's execute method
        action = self.strategy.execute(market_data=self.market, ml_model=self.ml_model)
        logger.info(f"Strategy action: {action}")

        is_valid_action = False
        if action and isinstance(action, dict):
            action_type = action.get("action", "").lower()
            if action_type and action_type != "hold":
                is_valid_action = True
        
        if is_valid_action:
            self._process_trade_action(action)
        else:
            msg = f"No trade action (or 'hold') to execute for {self.market.symbol}."
            logger.info(msg)

    def _perform_final_checks(self):
        logger.info("Assessing risk...")
        # TODO: position_details might need to be fetched or tracked by the engine
        risk_level = self.risk_manager.assess_risk(
            market_data=self.market, position_details=None 
        )
        logger.info(f"Risk assessment: {risk_level}")

        # SecurityManager is not part of TradingEngine in this design,
        # as it was primarily UI/Application level in the original code.
        # If compliance checks are needed for headless, SecurityManager would need refactoring.
        logger.info("TradingEngine final checks complete. Compliance checks are handled by the Application layer if UI is present.")

# Example of how to configure logging to a file (can be called from main.py)
def configure_logging(log_to_file=False, log_file_path="trading_engine.log"):
    global logger # Use the logger defined at the module level
    for handler in logger.handlers[:]: # Remove existing handlers
        logger.removeHandler(handler)
    
    if log_to_file:
        file_handler = logging.FileHandler(log_file_path)
        file_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
        logger.addHandler(file_handler)
    else:
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
        logger.addHandler(console_handler)
    
    logger.info(f"Logging configured. Output to: {'File ({log_file_path})' if log_to_file else 'Console'}")
