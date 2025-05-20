import operator
from dataclasses import dataclass, field
from typing import Any, Callable, Dict

# Define constants for repeated string literals
REASON_INSUFFICIENT_DATA = "Insufficient data for underlying stock"
ACTION_HOLD = "hold"
ACTION_BUY_CALL = "BUY_CALL"
ACTION_SELL_CALL = "SELL_CALL"
ACTION_BUY_PUT = "BUY_PUT"
ACTION_SELL_PUT = "SELL_PUT"
ACTION_BUY_STRADDLE = (
    "BUY_STRADDLE"  # Represents buying a call and a put at the same strike
)
# Represents buying an OTM call and an OTM put
ACTION_BUY_STRANGLE = "BUY_STRANGLE"
ACTION_CLOSE_OPTION_POSITION = (
    "CLOSE_OPTION_POSITION"  # Generic action to close an option
)

# Strategy Names
STRATEGY_COVERED_CALL = "Covered Call"
STRATEGY_STRADDLE = "Straddle"
STRATEGY_STRANGLE = "Strangle"
STRATEGY_ATR_TRAILING_STOP = "ATR Trailing Stop"
STRATEGY_SIMPLE_CALL_BUYER = "Simple Call Buyer"
STRATEGY_SIMPLE_PUT_BUYER = "Simple Put Buyer"
STRATEGY_CALL_ALIAS = "Call"  # Alias for Simple Call Buyer


@dataclass
class SimpleOptionStrategyConfig:
    strategy_name: str
    action_type: str
    otm_percentage_direction: int
    rsi_threshold_key: str  # e.g., 'rsi_oversold_threshold'
    rsi_operator: Callable[[Any, Any], bool]  # e.g., operator.lt
    ma_operator: Callable[[Any, Any], bool]  # e.g., operator.gt


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
    """Helper to check if any market data source is empty or None."""
    for data_source in market_data_sources:
        is_none = data_source is None
        # Check for 'empty' attribute only if data_source is not None and has the attribute
        is_empty_attr = False
        if not is_none and hasattr(data_source, "empty"):
            is_empty_attr = data_source.empty

        if is_none or is_empty_attr:
            return True
    return False


class TradingStrategy:
    def __init__(self, name: str):  # Removed ui parameter
        self.name = name
        self.params = {}

        # Default parameters for stock-based decisions (can also inform option decisions)
        self.params.setdefault("rsi_oversold_threshold", 30)
        self.params.setdefault("rsi_overbought_threshold", 70)
        self.params.setdefault("volatility_threshold", 0.03)
        self.params.setdefault("strangle_vol_threshold", 0.05)
        self.params.setdefault("ma_trend_window", 5)

        # Default parameters for options
        self.params.setdefault("option_dte", 30)  # Days to expiration
        self.params.setdefault(
            "covered_call_otm_pct", 0.05
        )  # 5% OTM for covered call strike
        # 0 for ATM, or small $ amount
        self.params.setdefault("straddle_atm_offset", 0)
        # 5% OTM for strangle strikes
        self.params.setdefault("strangle_otm_pct", 0.05)
        self.params.setdefault(
            "simple_option_otm_pct", 0.02
        )  # 2% OTM for simple call/put buys
        self.params.setdefault("option_quantity", 1)  # Number of contracts

        # Parameters for ATR Trailing Stop (can apply to underlying or option price)
        self.params.setdefault("atr_multiplier", 2.0)
        self.params.setdefault("position_open", False)
        self.params.setdefault(
            "entry_price", None
        )  # Underlying's price when option was bought
        self.params.setdefault("option_type_held", None)  # 'call' or 'put'
        self.params.setdefault("option_strike_held", None)
        self.params.setdefault("option_expiration_held", None)

        self._strategy_executors = {
            STRATEGY_COVERED_CALL: self._execute_covered_call,
            STRATEGY_STRADDLE: self._execute_straddle,
            STRATEGY_STRANGLE: self._execute_strangle,
            STRATEGY_ATR_TRAILING_STOP: self._execute_atr_trailing_stop,
            STRATEGY_SIMPLE_CALL_BUYER: self._execute_simple_call_buyer,
            STRATEGY_CALL_ALIAS: self._execute_simple_call_buyer,  # Alias
            STRATEGY_SIMPLE_PUT_BUYER: self._execute_simple_put_buyer,
        }

    def _get_common_indicator_data(self, market_data):
        """Fetches RSI, MA, and current price."""
        rsi = market_data.calculate_rsi()
        ma = market_data.calculate_moving_average()
        current_price = market_data.get_current_price()
        return rsi, ma, current_price

    def _get_volatility_and_price_data(self, market_data):
        """Fetches volatility and current price."""
        volatility = market_data.calculate_volatility()
        current_price = market_data.get_current_price()
        return volatility, current_price

    def _check_rsi_ma_trade_conditions(
        self,
        rsi_val: float,
        ma_val: float,
        ma_series,
        config: SimpleOptionStrategyConfig,
    ) -> bool:
        """Checks RSI and MA conditions based on the provided configuration."""
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
        """Creates a standardized dictionary for option trade actions."""
        details = (
            base_details.copy()
        )  # Start with base details like symbol, dte, reason
        details.update(
            action_specific_details
        )  # Add action-specifics like strike, quantity

        action = details.pop(
            "action_type"
        )  # Extract action from details to form the final structure

        # Round strike prices if they exist
        for key in ["strike_price", "call_strike_price", "put_strike_price"]:
            if key in details and isinstance(details[key], (float, int)):
                details[key] = round(details[key], 2)

        return {"action": action, "details": details}

    def _evaluate_simple_option_strategy(
        self, market_data, config: SimpleOptionStrategyConfig
    ) -> Dict[str, Any]:
        rsi, ma, current_price = self._get_common_indicator_data(market_data)

        if _is_data_insufficient([rsi, ma, current_price]):
            return {
                "action": ACTION_HOLD,
                "details": {
                    "reason": REASON_INSUFFICIENT_DATA,
                    "strategy": config.strategy_name,
                },
            }

        if current_price is None or not isinstance(current_price, (int, float)):
            return {
                "action": ACTION_HOLD,
                "details": {
                    "reason": "Invalid current price for strategy evaluation.",
                    "strategy": config.strategy_name,
                },
            }

        rsi_val = rsi.iloc[-1]
        ma_val = ma.iloc[-1]

        if self._check_rsi_ma_trade_conditions(rsi_val, ma_val, ma, config):
            strike_otm_pct = self.params["simple_option_otm_pct"]
            strike_price = current_price * (
                1 + config.otm_percentage_direction * strike_otm_pct
            )
            reason = f"Conditions met for {config.strategy_name}: RSI {rsi_val:.2f} ({config.rsi_operator.__name__} {self.params[config.rsi_threshold_key]}), MA trend favorable (MA {ma_val:.2f} {config.ma_operator.__name__} previous MA), Current Price {current_price:.2f}."

            base_details = {
                "action_type": config.action_type,
                "symbol": market_data.symbol,
                "dte": self.params["option_dte"],
                "reason": reason,
            }
            action_specific_details = {
                "strike_price": strike_price,
                "quantity": self.params["option_quantity"],
            }
            return self._create_option_action_details_dict(
                base_details, action_specific_details
            )
        return {
            "action": ACTION_HOLD,
            "details": {
                "reason": f"Conditions not met for {config.strategy_name}",
                "strategy": config.strategy_name,
            },
        }

    def execute(self, market_data) -> Dict[str, Any]:
        executor = self._strategy_executors.get(self.name)
        if executor:
            return executor(market_data)
        else:
            print(f"Unknown strategy: {self.name}")
            return {
                "action": ACTION_HOLD,
                "details": {"reason": f"Unknown strategy: {self.name}"},
            }

    def _execute_covered_call(
        self, market_data
    ) -> Dict[str, Any]:  # Renamed from covered_call
        rsi, ma, current_price = self._get_common_indicator_data(market_data)

        strategy_name = STRATEGY_COVERED_CALL
        if _is_data_insufficient([rsi, ma, current_price]):
            print(
                f"Strategy {strategy_name} for {market_data.symbol}: Not enough data."
            )
            return {
                "action": ACTION_HOLD,
                "details": {
                    "reason": REASON_INSUFFICIENT_DATA,
                    "strategy": strategy_name,
                },
            }

        if current_price is None or not isinstance(current_price, (int, float)):
            return {
                "action": ACTION_HOLD,
                "details": {
                    "reason": f"Invalid current price for {strategy_name}.",
                    "strategy": strategy_name,
                },
            }

        is_not_overbought = rsi.iloc[-1] < self.params["rsi_overbought_threshold"]
        ma_trend_window = self.params["ma_trend_window"]
        is_ma_trending_up = (
            ma.iloc[-1] > ma.iloc[-(ma_trend_window + 1)]
            if len(ma) > ma_trend_window
            else False
        )

        if is_not_overbought and is_ma_trending_up:
            strike_price = current_price * \
                (1 + self.params["covered_call_otm_pct"])
            reason = f"Stock stable/bullish (RSI {rsi.iloc[-1]:.2f} not overbought, MA up: {is_ma_trending_up}), selling OTM call."
            print(
                f"Strategy {strategy_name} for {market_data.symbol}: Favorable conditions. {reason}"
            )

            base_details = {
                "action_type": ACTION_SELL_CALL,
                "symbol": market_data.symbol,
                "dte": self.params["option_dte"],
                "reason": reason,
            }
            action_specific_details = {
                "strike_price": strike_price,
                "quantity": self.params["option_quantity"],
            }
            return self._create_option_action_details_dict(
                base_details, action_specific_details
            )
        return {
            "action": ACTION_HOLD,
            "details": {
                "reason": f"Conditions not met for {strategy_name}",
                "strategy": strategy_name,
            },
        }

    def _execute_volatility_strategy(
        self,
        market_data,
        strategy_name: str,
        volatility_param_key: str,
        action_type: str,
    ) -> Dict[str, Any]:
        volatility, current_price = self._get_volatility_and_price_data(
            market_data)

        if _is_data_insufficient([volatility, current_price]):
            print(
                f"Strategy {strategy_name} for {market_data.symbol}: Not enough data."
            )
            return {
                "action": ACTION_HOLD,
                "details": {
                    "reason": REASON_INSUFFICIENT_DATA,
                    "strategy": strategy_name,
                },
            }

        if current_price is None or not isinstance(current_price, (int, float)):
            return {
                "action": ACTION_HOLD,
                "details": {
                    "reason": f"Invalid current price for {strategy_name}.",
                    "strategy": strategy_name,
                },
            }

        is_condition_met = volatility.iloc[-1] > self.params[volatility_param_key]

        if is_condition_met:
            base_details = {
                "action_type": action_type,
                "symbol": market_data.symbol,
                "dte": self.params["option_dte"],
                "quantity_per_leg": self.params["option_quantity"],
            }
            action_specific_details = {}

            if strategy_name == STRATEGY_STRADDLE:
                strike_price = round(
                    current_price + self.params["straddle_atm_offset"], 0
                )
                reason = f"High expected volatility (Vol: {volatility.iloc[-1]:.2f}), buying ATM straddle."
                action_specific_details["strike_price"] = strike_price
            elif strategy_name == STRATEGY_STRANGLE:
                call_strike = current_price * \
                    (1 + self.params["strangle_otm_pct"])
                put_strike = current_price * \
                    (1 - self.params["strangle_otm_pct"])
                reason = f"Very high expected volatility (Vol: {volatility.iloc[-1]:.2f}), buying OTM strangle."
                action_specific_details["call_strike_price"] = call_strike
                action_specific_details["put_strike_price"] = put_strike
            else:  # Should not happen with current setup
                return {
                    "action": ACTION_HOLD,
                    "details": {
                        "reason": f"Unknown volatility strategy configuration for {strategy_name}"
                    },
                }

            base_details["reason"] = reason
            print(
                f"Strategy {strategy_name} for {market_data.symbol}: {reason}")
            return self._create_option_action_details_dict(
                base_details, action_specific_details
            )

        vol_level_reason = (
            "low volatility"
            if strategy_name == STRATEGY_STRADDLE
            else "volatility not high enough"
        )
        return {
            "action": ACTION_HOLD,
            "details": {
                "reason": f"Conditions not met for {strategy_name} ({vol_level_reason})",
                "strategy": strategy_name,
            },
        }

    # Renamed from straddle
    def _execute_straddle(self, market_data) -> Dict[str, Any]:
        return self._execute_volatility_strategy(
            market_data, STRATEGY_STRADDLE, "volatility_threshold", ACTION_BUY_STRADDLE
        )

    # Renamed from strangle
    def _execute_strangle(self, market_data) -> Dict[str, Any]:
        return self._execute_volatility_strategy(
            market_data,
            STRATEGY_STRANGLE,
            "strangle_vol_threshold",
            ACTION_BUY_STRANGLE,
        )

    def _calculate_trailing_stop_and_reason(self, atr_params: ATRStopParams):
        """Calculates if ATR trailing stop is triggered and provides a reason."""
        option_type = atr_params.option_type
        entry_price = atr_params.entry_price
        current_price = atr_params.current_price
        atr_val = atr_params.atr
        atr_multiplier = atr_params.atr_multiplier

        stop_price = None
        triggered = False
        reason_segment = ""

        if option_type == "call":
            stop_price = entry_price - (atr_val * atr_multiplier)
            triggered = current_price < stop_price
            reason_segment = f"fell below ATR stop level {stop_price:.2f}"
        elif option_type == "put":
            stop_price = entry_price + (atr_val * atr_multiplier)
            triggered = current_price > stop_price
            reason_segment = f"rose above ATR stop level {stop_price:.2f}"

        full_reason = ""
        if triggered:
            full_reason = (
                f"Underlying price {current_price:.2f} {reason_segment} "
                f"(based on entry underlying price {entry_price:.2f} and ATR {atr_val:.2f}) for {option_type.upper()} option."
            )
        return triggered, full_reason, stop_price

    def _get_atr_stop_details(self, market_data, current_underlying_price, current_atr):
        stop_loss_multiplier = self.params.get("atr_multiplier", 2.0)
        underlying_entry_price_at_option_purchase = self.params.get(
            "entry_price")
        option_type_held = self.params.get(
            "option_type_held")  # 'call' or 'put'
        strategy_name = STRATEGY_ATR_TRAILING_STOP

        default_reason = f"Holding {option_type_held} option. Underlying price {current_underlying_price:.2f}. ATR stop conditions not met."
        action_details_result = {
            "action": ACTION_HOLD,
            "details": {"reason": default_reason, "strategy": strategy_name},
        }

        if not option_type_held or underlying_entry_price_at_option_purchase is None:
            action_details_result["details"][
                "reason"
            ] = "ATR stop check skipped: Missing option type or entry price."
            return action_details_result

        atr_calc_params = ATRStopParams(
            option_type=option_type_held,
            entry_price=underlying_entry_price_at_option_purchase,
            current_price=current_underlying_price,
            atr=current_atr,
            atr_multiplier=stop_loss_multiplier,
        )
        triggered, reason_for_close, stop_price = (
            self._calculate_trailing_stop_and_reason(atr_calc_params)
        )

        if triggered:
            print(
                f"Strategy {strategy_name} for {market_data.symbol} ({option_type_held.upper()}): CLOSE signal. {reason_for_close}"
            )
            self.params["position_open"] = False

            base_details = {
                "action_type": ACTION_CLOSE_OPTION_POSITION,
                "symbol": market_data.symbol,
                "dte": self.params.get("option_dte", 0),
                "reason": reason_for_close,
            }
            action_specific_details = {
                "quantity": self.params["option_quantity"],
                "option_type_to_close": option_type_held.upper(),
                "strike_price_held": self.params.get("option_strike_held", "N/A"),
                "expiration_held": self.params.get("option_expiration_held", "N/A"),
                "extra_details": {
                    "stop_price_triggered_at": stop_price,
                    "underlying_price_at_stop": current_underlying_price,
                },
            }
            return self._create_option_action_details_dict(
                base_details, action_specific_details
            )

        if self.params.get("position_open"):
            action_details_result["details"][
                "reason"
            ] = f"Holding {option_type_held} option. Underlying price {current_underlying_price:.2f}. ATR stop ({stop_price:.2f} for {option_type_held}) not hit."
        return action_details_result

    def _validate_atr_inputs(
        self, current_atr: float | None, strategy_name: str
    ) -> tuple[bool, Dict[str, Any] | None]:
        """Validates the current ATR value. Returns (isValid, errorDetailsIfInvalid)."""
        if current_atr is None or not isinstance(current_atr, (int, float)):
            reason = f"Invalid ATR value (None or not a number) for {strategy_name}."
            error_details = {
                "action": ACTION_HOLD,
                "details": {"reason": reason, "strategy": strategy_name},
            }
            return False, error_details

        if current_atr <= 0:
            reason = f"Invalid ATR value ({current_atr} <= 0) for {strategy_name}."
            error_details = {
                "action": ACTION_HOLD,
                "details": {"reason": reason, "strategy": strategy_name},
            }
            return False, error_details

        return True, None

    def _perform_atr_pre_checks(self, strategy_name: str) -> ATRPreCheckResult:
        """Performs initial checks for ATR trailing stop readiness."""
        position_open = self.params.get("position_open", False)
        if not position_open:
            return ATRPreCheckResult(
                should_proceed=False,
                action_details={
                    "action": ACTION_HOLD,
                    "details": {
                        "reason": "No open option position to manage.",
                        "strategy": strategy_name,
                    },
                },
            )

        underlying_entry_price = self.params.get("entry_price", None)
        if underlying_entry_price is None:
            return ATRPreCheckResult(
                should_proceed=False,
                action_details={
                    "action": ACTION_HOLD,
                    "details": {
                        "reason": "Missing underlying entry price for ATR stop.",
                        "strategy": strategy_name,
                    },
                },
            )

        option_type_held = self.params.get("option_type_held", None)
        if option_type_held is None:
            return ATRPreCheckResult(
                should_proceed=False,
                action_details={
                    "action": ACTION_HOLD,
                    "details": {
                        "reason": "Missing option type for ATR stop.",
                        "strategy": strategy_name,
                    },
                },
            )

        return ATRPreCheckResult(
            should_proceed=True,
            position_open=position_open,
            underlying_entry_price=underlying_entry_price,
            option_type_held=option_type_held,
        )

    def _execute_atr_trailing_stop(self, market_data) -> Dict[str, Any]:
        strategy_name = STRATEGY_ATR_TRAILING_STOP

        pre_check_result = self._perform_atr_pre_checks(strategy_name)
        if not pre_check_result.should_proceed:
            return pre_check_result.action_details

        atr_series = market_data.calculate_atr()
        current_underlying_price = market_data.get_current_price()

        if _is_data_insufficient([atr_series, current_underlying_price]):
            print(
                f"Strategy {strategy_name} for {market_data.symbol}: Not enough data for ATR or underlying price."
            )
            return {
                "action": ACTION_HOLD,
                "details": {
                    "reason": REASON_INSUFFICIENT_DATA,
                    "strategy": strategy_name,
                },
            }

        if current_underlying_price is None or not isinstance(
            current_underlying_price, (int, float)
        ):
            return {
                "action": ACTION_HOLD,
                "details": {
                    "reason": f"Invalid current underlying price for {strategy_name}.",
                    "strategy": strategy_name,
                },
            }

        current_atr = atr_series.iloc[-1]
        is_valid_atr, atr_error_details = self._validate_atr_inputs(
            current_atr, strategy_name
        )
        if not is_valid_atr:
            return atr_error_details

        return self._get_atr_stop_details(
            market_data, current_underlying_price, current_atr
        )

    def _execute_simple_call_buyer(self, market_data) -> Dict[str, Any]:
        config = SimpleOptionStrategyConfig(
            strategy_name=STRATEGY_SIMPLE_CALL_BUYER,
            action_type=ACTION_BUY_CALL,
            otm_percentage_direction=1,
            rsi_threshold_key="rsi_oversold_threshold",
            rsi_operator=operator.lt,  # RSI < threshold
            ma_operator=operator.gt,  # MA > previous MA (trending up)
        )
        return self._evaluate_simple_option_strategy(market_data, config)

    def _execute_simple_put_buyer(self, market_data) -> Dict[str, Any]:
        config = SimpleOptionStrategyConfig(
            strategy_name=STRATEGY_SIMPLE_PUT_BUYER,
            action_type=ACTION_BUY_PUT,
            otm_percentage_direction=-1,
            rsi_threshold_key="rsi_overbought_threshold",
            rsi_operator=operator.gt,  # RSI > threshold
            ma_operator=operator.lt,  # MA < previous MA (trending down)
        )
        return self._evaluate_simple_option_strategy(market_data, config)

    def customize(self, params: Dict[str, Any]):
        self.params.update(params)
        print(f"Strategy {self.name} parameters updated: {self.params}")
