import operator
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Optional # Added Optional
from datetime import datetime, timedelta 
import QuantLib as ql 
import numpy as np # Added for MLSignalStrategy
import pandas as pd # Added for MLSignalStrategy type hints

# Define constants for repeated string literals
REASON_INSUFFICIENT_DATA = "Insufficient data for underlying stock"
ACTION_HOLD = "hold"
ACTION_BUY_CALL = "BUY_CALL"
ACTION_SELL_CALL = "SELL_CALL"
ACTION_BUY_PUT = "BUY_PUT"
ACTION_SELL_PUT = "SELL_PUT"
ACTION_BUY_STRADDLE = (
    "BUY_STRADDLE"
)
ACTION_BUY_STRANGLE = "BUY_STRANGLE"
ACTION_CLOSE_OPTION_POSITION = (
    "CLOSE_OPTION_POSITION"
)

# Strategy Names
STRATEGY_COVERED_CALL = "Covered Call"
STRATEGY_STRADDLE = "Straddle"
STRATEGY_STRANGLE = "Strangle"
STRATEGY_ATR_TRAILING_STOP = "ATR Trailing Stop"
STRATEGY_SIMPLE_CALL_BUYER = "Simple Call Buyer"
STRATEGY_SIMPLE_PUT_BUYER = "Simple Put Buyer"
STRATEGY_CALL_ALIAS = "Call"
STRATEGY_ML_SIGNAL = "ML Signal Strategy" # New Strategy Name


class OptionPricer:
    def __init__(self, risk_free_rate=0.02, dividend_yield=0.0):
        self.rf_rate = risk_free_rate
        self.div_yield = dividend_yield
        self.day_count = ql.Actual365Fixed()
        self.calendar = ql.TARGET()

    def _to_ql_date(self, date_py: datetime) -> ql.Date:
        return ql.Date(date_py.day, date_py.month, date_py.year)

    def price_and_greeks(self, eval_date_py: datetime, spot: float, strike: float, vol: float, expiry_date_py: datetime, option_type: str = 'call'):
        eval_date_ql = self._to_ql_date(eval_date_py)
        ql.Settings.instance().evaluationDate = eval_date_ql
        maturity_ql = self._to_ql_date(expiry_date_py)
        spot_handle = ql.QuoteHandle(ql.SimpleQuote(spot))
        vol_handle = ql.BlackVolTermStructureHandle(
            ql.BlackConstantVol(eval_date_ql, self.calendar, vol, self.day_count)
        )
        rf_handle = ql.YieldTermStructureHandle(
            ql.FlatForward(eval_date_ql, self.calendar, self.rf_rate, self.day_count)
        )
        dv_handle = ql.YieldTermStructureHandle(
            ql.FlatForward(eval_date_ql, self.calendar, self.div_yield, self.day_count)
        )
        payoff = ql.PlainVanillaPayoff(
            ql.Option.Call if option_type.lower() == 'call' else ql.Option.Put,
            strike
        )
        exercise = ql.EuropeanExercise(maturity_ql)
        bs_process = ql.BlackScholesMertonProcess(
            spot_handle, dv_handle, rf_handle, vol_handle
        )
        option = ql.VanillaOption(payoff, exercise)
        option.setPricingEngine(ql.AnalyticEuropeanEngine(bs_process))
        option.recalculate()
        price = option.NPV()
        delta = option.delta()
        vega = option.vega() / 100 
        if price is None or price < 0: price = 0.0
        if delta is None: delta = 0.0 if option_type.lower() == 'call' and spot < strike else (-1.0 if option_type.lower() == 'put' and spot > strike else 0.5)
        if vega is None: vega = 0.0
        return price, delta, vega


@dataclass
class SimpleOptionStrategyConfig:
    strategy_name: str
    action_type: str
    otm_percentage_direction: int
    rsi_threshold_key: str
    rsi_operator: Callable[[Any, Any], bool]
    ma_operator: Callable[[Any, Any], bool]


@dataclass
class ATRStopParams:
    option_type: str
    entry_price: float
    current_price: float
    atr: float
    atr_multiplier: float


@dataclass
class ATRPreCheckResult:
    should_proceed: bool
    action_details: Dict[str, Any] = field(default_factory=dict)
    position_open: bool = False
    underlying_entry_price: float | None = None
    option_type_held: str | None = None


def _is_data_insufficient(market_data_sources):
    for data_source in market_data_sources:
        is_none = data_source is None
        is_empty_attr = False
        if not is_none and hasattr(data_source, "empty"):
            is_empty_attr = data_source.empty
        if is_none or is_empty_attr:
            return True
    return False


class TradingStrategy:
    def __init__(self, name: str, ml_model_instance = None, option_pricer_instance = None, params: Optional[Dict[str, Any]] = None): # Added ml_model, option_pricer, params
        self.name = name
        self.params = {} 
        self.ml_model = ml_model_instance # Store ML model if provided
        self.option_pricer = option_pricer_instance if option_pricer_instance else OptionPricer() # Use provided or default pricer

        # Default parameters
        self.params.setdefault("rsi_oversold_threshold", 30)
        self.params.setdefault("rsi_overbought_threshold", 70)
        self.params.setdefault("volatility_threshold", 0.03) # For straddle
        self.params.setdefault("strangle_vol_threshold", 0.05) # For strangle
        self.params.setdefault("ma_trend_window", 5)
        self.params.setdefault("option_dte", 30)
        self.params.setdefault("covered_call_otm_pct", 0.05)
        self.params.setdefault("straddle_atm_offset", 0)
        self.params.setdefault("strangle_otm_pct", 0.05)
        self.params.setdefault("simple_option_otm_pct", 0.02)
        self.params.setdefault("option_quantity", 1)
        self.params.setdefault("risk_free_rate", 0.02) # For OptionPricer if not passed directly
        self.params.setdefault("dividend_yield", 0.00) # For OptionPricer
        self.params.setdefault("target_option_vega_exposure", None)
        self.params.setdefault("commission_per_contract", 0.65)
        self.params.setdefault("slippage_per_contract", 0.01)
        self.params.setdefault("atr_multiplier", 2.0)
        self.params.setdefault("position_open", False)
        self.params.setdefault("entry_price", None)
        self.params.setdefault("option_type_held", None)
        self.params.setdefault("option_strike_held", None)
        self.params.setdefault("option_expiration_held", None)
        
        # ML Signal Strategy specific defaults
        self.params.setdefault("proba_threshold_long", 0.6)
        self.params.setdefault("proba_threshold_short", 0.6)
        self.params.setdefault("ml_otm_pct", 0.02) # OTM percentage for ML signals
        self.params.setdefault("ml_option_dte", 30) # DTE for ML signals

        if params: # Update with any externally provided params
            self.params.update(params)

        if not self.option_pricer and self.name != STRATEGY_ML_SIGNAL : # MLSignalStrategy might not always use it if not pricing options
             self.option_pricer = OptionPricer(
                risk_free_rate=self.params["risk_free_rate"],
                dividend_yield=self.params["dividend_yield"]
            )
        
        # Ensure MLSignalStrategy gets its specific pricer if one was passed to constructor
        if self.name == STRATEGY_ML_SIGNAL and option_pricer_instance:
            self.option_pricer = option_pricer_instance


        self._strategy_executors = {
            STRATEGY_COVERED_CALL: self._execute_covered_call,
            STRATEGY_STRADDLE: self._execute_straddle,
            STRATEGY_STRANGLE: self._execute_strangle,
            STRATEGY_ATR_TRAILING_STOP: self._execute_atr_trailing_stop,
            STRATEGY_SIMPLE_CALL_BUYER: self._execute_simple_call_buyer,
            STRATEGY_CALL_ALIAS: self._execute_simple_call_buyer,
            STRATEGY_SIMPLE_PUT_BUYER: self._execute_simple_put_buyer,
            STRATEGY_ML_SIGNAL: self._execute_ml_signal, # Added ML Signal executor
        }

    def _get_common_indicator_data(self, market_data):
        rsi = market_data.calculate_rsi()
        ma = market_data.calculate_moving_average()
        current_price = market_data.get_current_price()
        return rsi, ma, current_price

    def _get_volatility_and_price_data(self, market_data):
        volatility = market_data.calculate_volatility()
        current_price = market_data.get_current_price()
        return volatility, current_price

    def _check_rsi_ma_trade_conditions(
        self, rsi_val: float, ma_val: float, ma_series, config: SimpleOptionStrategyConfig
    ) -> bool:
        rsi_condition_met = config.rsi_operator(
            rsi_val, self.params[config.rsi_threshold_key]
        )
        ma_trend_window = self.params["ma_trend_window"]
        ma_condition_met = False
        if len(ma_series) > ma_trend_window:
            previous_ma_val = (
                ma_series.iloc[-(ma_trend_window + 1)]
                if len(ma_series) > ma_trend_window
                else ma_val
            )
            ma_condition_met = config.ma_operator(ma_val, previous_ma_val)
        return rsi_condition_met and ma_condition_met

    def _create_option_action_details_dict(
        self, base_details: Dict[str, Any], action_specific_details: Dict[str, Any]
    ) -> Dict[str, Any]:
        details = base_details.copy()
        details.update(action_specific_details)
        action = details.pop("action_type")
        for key in ["strike_price", "call_strike_price", "put_strike_price"]:
            if key in details and isinstance(details[key], (float, int)):
                details[key] = round(details[key], 2)
        return {"action": action, "details": details}

    def _evaluate_simple_option_strategy(
        self, market_data, config: SimpleOptionStrategyConfig, ml_model = None # ml_model not used by this one
    ) -> Dict[str, Any]:
        # (Implementation as before, slightly adapted for current_date if needed)
        rsi, ma, current_price = self._get_common_indicator_data(market_data)
        try: # market_data.data is a DataFrame, index is DatetimeIndex
            current_date = market_data.data.index[-1].to_pydatetime()
            if not isinstance(current_date, datetime): current_date = current_date.to_pydatetime()
        except Exception: # Fallback if data or index is not as expected
            current_date = datetime.now()

        if _is_data_insufficient([rsi, ma, current_price]):
            return {"action": ACTION_HOLD, "details": {"reason": REASON_INSUFFICIENT_DATA, "strategy": config.strategy_name}}
        if current_price is None or not isinstance(current_price, (int, float)):
            return {"action": ACTION_HOLD, "details": {"reason": "Invalid current price for strategy evaluation.", "strategy": config.strategy_name}}

        rsi_val = rsi.iloc[-1]
        ma_val = ma.iloc[-1]

        if self._check_rsi_ma_trade_conditions(rsi_val, ma_val, ma, config):
            strike_otm_pct = self.params["simple_option_otm_pct"]
            strike_price = current_price * (1 + config.otm_percentage_direction * strike_otm_pct)
            estimated_option_price, estimated_delta, estimated_vega = None, None, None
            quantity = self.params["option_quantity"]
            try:
                current_volatility_series = market_data.calculate_volatility()
                if _is_data_insufficient([current_volatility_series]) or current_volatility_series.empty: raise ValueError("Insufficient volatility data.")
                current_vol = current_volatility_series.iloc[-1]
                if current_vol <= 0: raise ValueError(f"Invalid volatility: {current_vol}")
                expiry_date = current_date + timedelta(days=self.params["option_dte"])
                opt_price, delta, vega = self.option_pricer.price_and_greeks(
                    eval_date_py=current_date, spot=current_price, strike=strike_price, vol=current_vol,
                    expiry_date_py=expiry_date, option_type=config.action_type.split('_')[-1].lower()
                )
                estimated_option_price = round(opt_price, 2)
                estimated_delta = round(delta, 4)
                estimated_vega = round(vega, 4)
                target_vega_exposure = self.params.get("target_option_vega_exposure")
                if target_vega_exposure is not None and estimated_vega != 0:
                    calculated_qty = target_vega_exposure / abs(estimated_vega)
                    quantity = max(1, round(calculated_qty))
            except Exception as e:
                print(f"Strategy {config.strategy_name} for {market_data.symbol}: Error during option pricing/sizing: {e}")
            reason_suffix = ""
            if estimated_option_price is not None:
                reason_suffix = f" Est. Option Price: ${estimated_option_price:.2f}, Delta: {estimated_delta:.4f}, Vega: {estimated_vega:.4f}. Quantity: {quantity}."
            reason = f"Conditions met for {config.strategy_name}: RSI {rsi_val:.2f} ({config.rsi_operator.__name__} {self.params[config.rsi_threshold_key]}), MA trend favorable (MA {ma_val:.2f} {config.ma_operator.__name__} previous MA), Current Price {current_price:.2f}.{reason_suffix}"
            base_details = {"action_type": config.action_type, "symbol": market_data.symbol, "dte": self.params["option_dte"], "reason": reason}
            action_specific_details = {
                "strike_price": strike_price, "quantity": quantity, "estimated_option_price": estimated_option_price,
                "estimated_delta": estimated_delta, "estimated_vega": estimated_vega,
                "commission_per_contract": self.params["commission_per_contract"], "slippage_per_contract": self.params["slippage_per_contract"]
            }
            return self._create_option_action_details_dict(base_details, action_specific_details)
        return {"action": ACTION_HOLD, "details": {"reason": f"Conditions not met for {config.strategy_name}", "strategy": config.strategy_name}}

    # Modified execute to accept ml_model
    def execute(self, market_data, ml_model = None) -> Dict[str, Any]:
        executor = self._strategy_executors.get(self.name)
        if executor:
            # Pass ml_model to executor if it's the ML strategy
            if self.name == STRATEGY_ML_SIGNAL:
                return executor(market_data, ml_model)
            return executor(market_data) # Other strategies don't need it for now
        else:
            print(f"Unknown strategy: {self.name}")
            return {"action": ACTION_HOLD, "details": {"reason": f"Unknown strategy: {self.name}"}}

    def _execute_ml_signal(self, market_data, ml_model) -> Dict[str, Any]:
        strategy_name = STRATEGY_ML_SIGNAL
        if ml_model is None:
            return {"action": ACTION_HOLD, "details": {"reason": "ML model not provided for MLSignalStrategy.", "strategy": strategy_name}}
        if not hasattr(ml_model, 'label_encoder') or ml_model.label_encoder is None or not hasattr(ml_model.label_encoder, 'classes_'):
             return {"action": ACTION_HOLD, "details": {"reason": "ML model label encoder not available or not fitted.", "strategy": strategy_name}}


        latest_features = market_data.get_latest_features()
        current_price = market_data.get_current_price() # Get current price for strike calculation

        if latest_features is None or latest_features.empty:
            return {"action": ACTION_HOLD, "details": {"reason": REASON_INSUFFICIENT_DATA + " (latest features)", "strategy": strategy_name}}
        if current_price is None:
            return {"action": ACTION_HOLD, "details": {"reason": REASON_INSUFFICIENT_DATA + " (current price)", "strategy": strategy_name}}

        # Ensure features are in the order expected by the model
        if hasattr(ml_model, 'trained_feature_names_') and ml_model.trained_feature_names_:
            try:
                latest_features = latest_features[ml_model.trained_feature_names_]
            except KeyError as e:
                missing_cols = set(ml_model.trained_feature_names_) - set(latest_features.columns)
                return {"action": ACTION_HOLD, "details": {"reason": f"Missing features for ML prediction: {missing_cols}. Error: {e}", "strategy": strategy_name}}
        
        probas = ml_model.predict_proba(latest_features)
        if probas is None or len(probas) == 0:
            return {"action": ACTION_HOLD, "details": {"reason": "Failed to get probability predictions from ML model.", "strategy": strategy_name}}
        
        probas_for_latest = probas[0] # predict_proba returns array of arrays, get first for single row input
        
        le_classes = ml_model.label_encoder.classes_
        prob_dict = dict(zip(le_classes, probas_for_latest))

        # Ternary labels are assumed to be -1 (Down), 0 (Hold), 1 (Up)
        prob_up = prob_dict.get(1, 0.0)    # Probability of Up (label 1)
        prob_down = prob_dict.get(-1, 0.0) # Probability of Down (label -1)

        proba_threshold_long = self.params.get('proba_threshold_long', 0.6)
        proba_threshold_short = self.params.get('proba_threshold_short', 0.6)
        
        action_type = ACTION_HOLD
        reason = f"ML Signal: P(Up)={prob_up:.2f}, P(Down)={prob_down:.2f}. Thresholds L:{proba_threshold_long}, S:{proba_threshold_short}."
        strike_price = None
        otm_pct = self.params.get('ml_otm_pct', 0.02)
        option_action_type = None

        if prob_up > proba_threshold_long:
            option_action_type = ACTION_BUY_CALL
            strike_price = current_price * (1 + otm_pct)
            reason += " BUY_CALL triggered."
        elif prob_down > proba_threshold_short:
            option_action_type = ACTION_BUY_PUT
            strike_price = current_price * (1 - otm_pct)
            reason += " BUY_PUT triggered."
        else:
            reason += " HOLD triggered."
            return {"action": ACTION_HOLD, "details": {"reason": reason, "strategy": strategy_name}}

        base_details = {
            "action_type": option_action_type,
            "symbol": market_data.symbol,
            "dte": self.params.get('ml_option_dte', 30),
            "reason": reason,
            "strategy": strategy_name, # Ensure strategy name is in details
        }
        action_specific_details = {
            "strike_price": strike_price,
            "quantity": self.params.get('option_quantity', 1), # Use general option_quantity or make specific one
            "estimated_option_price": None, # Can be populated if pricer is used
            "estimated_delta": None,
            "estimated_vega": None,
            "commission_per_contract": self.params["commission_per_contract"],
            "slippage_per_contract": self.params["slippage_per_contract"]
        }
        return self._create_option_action_details_dict(base_details, action_specific_details)

    def _execute_covered_call(self, market_data, ml_model = None) -> Dict[str, Any]:
        # (Implementation as before)
        rsi, ma, current_price = self._get_common_indicator_data(market_data)
        strategy_name = STRATEGY_COVERED_CALL
        if _is_data_insufficient([rsi, ma, current_price]):
            return {"action": ACTION_HOLD, "details": {"reason": REASON_INSUFFICIENT_DATA, "strategy": strategy_name}}
        if current_price is None or not isinstance(current_price, (int, float)):
            return {"action": ACTION_HOLD, "details": {"reason": f"Invalid current price for {strategy_name}.", "strategy": strategy_name}}
        is_not_overbought = rsi.iloc[-1] < self.params["rsi_overbought_threshold"]
        ma_trend_window = self.params["ma_trend_window"]
        is_ma_trending_up = (ma.iloc[-1] > ma.iloc[-(ma_trend_window + 1)] if len(ma) > ma_trend_window else False)
        if is_not_overbought and is_ma_trending_up:
            strike_price = current_price * (1 + self.params["covered_call_otm_pct"])
            reason = f"Stock stable/bullish (RSI {rsi.iloc[-1]:.2f} not overbought, MA up: {is_ma_trending_up}), selling OTM call."
            base_details = {"action_type": ACTION_SELL_CALL, "symbol": market_data.symbol, "dte": self.params["option_dte"], "reason": reason}
            action_specific_details = {"strike_price": strike_price, "quantity": self.params["option_quantity"]}
            return self._create_option_action_details_dict(base_details, action_specific_details)
        return {"action": ACTION_HOLD, "details": {"reason": f"Conditions not met for {strategy_name}", "strategy": strategy_name}}

    def _execute_volatility_strategy(
        self, market_data, strategy_name: str, volatility_param_key: str, action_type: str, ml_model = None
    ) -> Dict[str, Any]:
        # (Implementation as before)
        volatility, current_price = self._get_volatility_and_price_data(market_data)
        if _is_data_insufficient([volatility, current_price]):
            return {"action": ACTION_HOLD, "details": {"reason": REASON_INSUFFICIENT_DATA, "strategy": strategy_name}}
        if current_price is None or not isinstance(current_price, (int, float)):
            return {"action": ACTION_HOLD, "details": {"reason": f"Invalid current price for {strategy_name}.", "strategy": strategy_name}}
        is_condition_met = volatility.iloc[-1] > self.params[volatility_param_key]
        if is_condition_met:
            base_details = {"action_type": action_type, "symbol": market_data.symbol, "dte": self.params["option_dte"], "quantity_per_leg": self.params["option_quantity"]}
            action_specific_details = {}
            if strategy_name == STRATEGY_STRADDLE:
                strike_price = round(current_price + self.params["straddle_atm_offset"], 0)
                reason = f"High expected volatility (Vol: {volatility.iloc[-1]:.2f}), buying ATM straddle."
                action_specific_details["strike_price"] = strike_price
            elif strategy_name == STRATEGY_STRANGLE:
                call_strike = current_price * (1 + self.params["strangle_otm_pct"])
                put_strike = current_price * (1 - self.params["strangle_otm_pct"])
                reason = f"Very high expected volatility (Vol: {volatility.iloc[-1]:.2f}), buying OTM strangle."
                action_specific_details["call_strike_price"] = call_strike
                action_specific_details["put_strike_price"] = put_strike
            else: return {"action": ACTION_HOLD, "details": {"reason": f"Unknown volatility strategy configuration for {strategy_name}"}}
            base_details["reason"] = reason
            return self._create_option_action_details_dict(base_details, action_specific_details)
        vol_level_reason = ("low volatility" if strategy_name == STRATEGY_STRADDLE else "volatility not high enough")
        return {"action": ACTION_HOLD, "details": {"reason": f"Conditions not met for {strategy_name} ({vol_level_reason})", "strategy": strategy_name}}

    def _execute_straddle(self, market_data, ml_model = None) -> Dict[str, Any]:
        return self._execute_volatility_strategy(market_data, STRATEGY_STRADDLE, "volatility_threshold", ACTION_BUY_STRADDLE)

    def _execute_strangle(self, market_data, ml_model = None) -> Dict[str, Any]:
        return self._execute_volatility_strategy(market_data, STRATEGY_STRANGLE, "strangle_vol_threshold", ACTION_BUY_STRANGLE)

    def _calculate_trailing_stop_and_reason(self, atr_params: ATRStopParams):
        # (Implementation as before)
        option_type, entry_price, current_price, atr_val, atr_multiplier = atr_params.option_type, atr_params.entry_price, atr_params.current_price, atr_params.atr, atr_params.atr_multiplier
        stop_price, triggered, reason_segment = None, False, ""
        if option_type == "call":
            stop_price = entry_price - (atr_val * atr_multiplier)
            triggered = current_price < stop_price
            reason_segment = f"fell below ATR stop level {stop_price:.2f}"
        elif option_type == "put":
            stop_price = entry_price + (atr_val * atr_multiplier)
            triggered = current_price > stop_price
            reason_segment = f"rose above ATR stop level {stop_price:.2f}"
        full_reason = f"Underlying price {current_price:.2f} {reason_segment} (based on entry underlying price {entry_price:.2f} and ATR {atr_val:.2f}) for {option_type.upper()} option." if triggered else ""
        return triggered, full_reason, stop_price

    def _get_atr_stop_details(self, market_data, current_underlying_price, current_atr):
        # (Implementation as before)
        stop_loss_multiplier, underlying_entry_price_at_option_purchase, option_type_held, strategy_name = self.params.get("atr_multiplier", 2.0), self.params.get("entry_price"), self.params.get("option_type_held"), STRATEGY_ATR_TRAILING_STOP
        default_reason = f"Holding {option_type_held} option. Underlying price {current_underlying_price:.2f}. ATR stop conditions not met."
        action_details_result = {"action": ACTION_HOLD, "details": {"reason": default_reason, "strategy": strategy_name}}
        if not option_type_held or underlying_entry_price_at_option_purchase is None:
            action_details_result["details"]["reason"] = "ATR stop check skipped: Missing option type or entry price."
            return action_details_result
        atr_calc_params = ATRStopParams(option_type_held, underlying_entry_price_at_option_purchase, current_underlying_price, current_atr, stop_loss_multiplier)
        triggered, reason_for_close, stop_price = self._calculate_trailing_stop_and_reason(atr_calc_params)
        if triggered:
            self.params["position_open"] = False
            base_details = {"action_type": ACTION_CLOSE_OPTION_POSITION, "symbol": market_data.symbol, "dte": self.params.get("option_dte", 0), "reason": reason_for_close}
            action_specific_details = {"quantity": self.params["option_quantity"], "option_type_to_close": option_type_held.upper(), "strike_price_held": self.params.get("option_strike_held", "N/A"), "expiration_held": self.params.get("option_expiration_held", "N/A"), "extra_details": {"stop_price_triggered_at": stop_price, "underlying_price_at_stop": current_underlying_price}}
            return self._create_option_action_details_dict(base_details, action_specific_details)
        if self.params.get("position_open"): action_details_result["details"]["reason"] = f"Holding {option_type_held} option. Underlying price {current_underlying_price:.2f}. ATR stop ({stop_price:.2f} for {option_type_held}) not hit."
        return action_details_result

    def _validate_atr_inputs(self, current_atr: float | None, strategy_name: str) -> tuple[bool, Dict[str, Any] | None]:
        # (Implementation as before)
        if current_atr is None or not isinstance(current_atr, (int, float)): return False, {"action": ACTION_HOLD, "details": {"reason": f"Invalid ATR value (None or not a number) for {strategy_name}.", "strategy": strategy_name}}
        if current_atr <= 0: return False, {"action": ACTION_HOLD, "details": {"reason": f"Invalid ATR value ({current_atr} <= 0) for {strategy_name}.", "strategy": strategy_name}}
        return True, None

    def _perform_atr_pre_checks(self, strategy_name: str) -> ATRPreCheckResult:
        # (Implementation as before)
        position_open, underlying_entry_price, option_type_held = self.params.get("position_open", False), self.params.get("entry_price"), self.params.get("option_type_held")
        if not position_open: return ATRPreCheckResult(False, {"action": ACTION_HOLD, "details": {"reason": "No open option position to manage.", "strategy": strategy_name}})
        if underlying_entry_price is None: return ATRPreCheckResult(False, {"action": ACTION_HOLD, "details": {"reason": "Missing underlying entry price for ATR stop.", "strategy": strategy_name}})
        if option_type_held is None: return ATRPreCheckResult(False, {"action": ACTION_HOLD, "details": {"reason": "Missing option type for ATR stop.", "strategy": strategy_name}})
        return ATRPreCheckResult(True, position_open=position_open, underlying_entry_price=underlying_entry_price, option_type_held=option_type_held)

    def _execute_atr_trailing_stop(self, market_data, ml_model = None) -> Dict[str, Any]:
        # (Implementation as before)
        strategy_name = STRATEGY_ATR_TRAILING_STOP
        pre_check_result = self._perform_atr_pre_checks(strategy_name)
        if not pre_check_result.should_proceed: return pre_check_result.action_details
        atr_series, current_underlying_price = market_data.calculate_atr(), market_data.get_current_price()
        if _is_data_insufficient([atr_series, current_underlying_price]): return {"action": ACTION_HOLD, "details": {"reason": REASON_INSUFFICIENT_DATA, "strategy": strategy_name}}
        if current_underlying_price is None or not isinstance(current_underlying_price, (int, float)): return {"action": ACTION_HOLD, "details": {"reason": f"Invalid current underlying price for {strategy_name}.", "strategy": strategy_name}}
        current_atr = atr_series.iloc[-1]
        is_valid_atr, atr_error_details = self._validate_atr_inputs(current_atr, strategy_name)
        if not is_valid_atr: return atr_error_details
        return self._get_atr_stop_details(market_data, current_underlying_price, current_atr)

    def _execute_simple_call_buyer(self, market_data, ml_model = None) -> Dict[str, Any]:
        config = SimpleOptionStrategyConfig(STRATEGY_SIMPLE_CALL_BUYER, ACTION_BUY_CALL, 1, "rsi_oversold_threshold", operator.lt, operator.gt)
        return self._evaluate_simple_option_strategy(market_data, config)

    def _execute_simple_put_buyer(self, market_data, ml_model = None) -> Dict[str, Any]:
        config = SimpleOptionStrategyConfig(STRATEGY_SIMPLE_PUT_BUYER, ACTION_BUY_PUT, -1, "rsi_overbought_threshold", operator.gt, operator.lt)
        return self._evaluate_simple_option_strategy(market_data, config)

    def customize(self, params: Dict[str, Any]):
        self.params.update(params)
        print(f"Strategy {self.name} parameters updated: {self.params}")

# It's cleaner to define MLSignalStrategy as its own class if it has a distinct constructor signature
# However, to fit the current TradingStrategy.__init__ that now accepts ml_model and option_pricer,
# we can make _execute_ml_signal a method of TradingStrategy and select it via self.name.
# The TradingStrategy constructor was updated to accept ml_model and option_pricer instances.
# And specific params for MLSignalStrategy were added to its defaults.
# The _strategy_executors map now includes STRATEGY_ML_SIGNAL pointing to _execute_ml_signal.
# This avoids needing a separate class while achieving functional separation.
# If MLSignalStrategy needed a very different __init__ (e.g. mandatory ml_model), a new class would be better.
# For now, this integration is less disruptive.

[end of strategies.py]
