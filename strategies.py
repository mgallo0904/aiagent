import operator
from dataclasses import dataclass, field
from typing import Any, Callable, Dict
from datetime import datetime, timedelta # Added
import QuantLib as ql # Added

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
STRATEGY_ML_TERNARY = "ML Ternary Strategy" # New ML Strategy Name


# Added OptionPricer class
class OptionPricer:
    def __init__(self, risk_free_rate=0.02, dividend_yield=0.0):
        self.rf_rate = risk_free_rate
        self.div_yield = dividend_yield
        self.day_count = ql.Actual365Fixed()
        self.calendar = ql.TARGET() # Using TARGET calendar as a common choice

    def _to_ql_date(self, date_py: datetime) -> ql.Date:
        """Converts python datetime to QuantLib Date."""
        return ql.Date(date_py.day, date_py.month, date_py.year)

    def price_and_greeks(self, eval_date_py: datetime, spot: float, strike: float, vol: float, expiry_date_py: datetime, option_type: str = 'call'):
        """Return price, delta, vega for a European option."""
        eval_date_ql = self._to_ql_date(eval_date_py)
        ql.Settings.instance().evaluationDate = eval_date_ql
        
        maturity_ql = self._to_ql_date(expiry_date_py)

        # Market data
        spot_handle = ql.QuoteHandle(ql.SimpleQuote(spot))
        # For vol, using BlackConstantVol. A more advanced setup would use a BlackVolTermStructure.
        # The 'vol' passed here is assumed to be the annualized implied volatility for the specific option.
        # If using realized vol, it's a simplification.
        vol_handle = ql.BlackVolTermStructureHandle(
            ql.BlackConstantVol(eval_date_ql, self.calendar, vol, self.day_count)
        )
        rf_handle = ql.YieldTermStructureHandle(
            ql.FlatForward(eval_date_ql, self.calendar, self.rf_rate, self.day_count)
        )
        dv_handle = ql.YieldTermStructureHandle(
            ql.FlatForward(eval_date_ql, self.calendar, self.div_yield, self.day_count)
        )

        # Payoff & process
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

        # Ensure calculations are done if evaluation date changed
        option.recalculate()

        price = option.NPV()
        delta = option.delta()
        vega = option.vega() / 100 # Vega is typically per 1% change in vol, so divide by 100 for per 1.0 change

        # Handle cases where Greeks might not be calculated (e.g., deep OTM, near expiry)
        if price is None or price < 0: price = 0.0 # Price should not be negative
        if delta is None: delta = 0.0 if option_type.lower() == 'call' and spot < strike else (-1.0 if option_type.lower() == 'put' and spot > strike else 0.5) # Rough estimate
        if vega is None: vega = 0.0
        
        return price, delta, vega


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
        
        # New parameters for pricer and advanced backtesting considerations
        self.params.setdefault("risk_free_rate", 0.02)
        self.params.setdefault("dividend_yield", 0.00)
        self.params.setdefault("target_option_vega_exposure", None) # e.g., 100 means target $100 vega exposure
        self.params.setdefault("commission_per_contract", 0.65) # Example commission
        self.params.setdefault("slippage_per_contract", 0.01) # Example slippage per share/contract price

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
        
        # Initialize OptionPricer
        self.pricer = OptionPricer(
            risk_free_rate=self.params["risk_free_rate"],
            dividend_yield=self.params["dividend_yield"]
        )

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

        # Assume market_data can provide the current date for TTE and pricer eval_date
        # This is a new requirement for the market_data object.
        try:
            current_date = market_data.get_current_date()
            if not isinstance(current_date, datetime):
                # If it's a pandas Timestamp, convert to datetime
                current_date = current_date.to_pydatetime()
        except AttributeError:
            print(f"Warning: market_data object for {market_data.symbol} does not have get_current_date(). Using datetime.now(). This is not suitable for backtesting.")
            current_date = datetime.now()


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
            
            # Option Pricing & Greeks Integration
            estimated_option_price, estimated_delta, estimated_vega = None, None, None
            quantity = self.params["option_quantity"] # Default quantity

            try:
                # NOTE: market_data.calculate_volatility() likely gives historical/realized vol.
                # For option pricing, implied volatility for the specific option is preferred.
                # This is a simplification and a known area for improvement (use IV surface).
                current_volatility_series = market_data.calculate_volatility()
                if _is_data_insufficient([current_volatility_series]) or current_volatility_series.empty:
                    raise ValueError("Insufficient volatility data.")
                current_vol = current_volatility_series.iloc[-1]
                if current_vol <= 0: # Volatility must be positive
                    raise ValueError(f"Invalid volatility: {current_vol}")

                expiry_date = current_date + timedelta(days=self.params["option_dte"])
                
                opt_price, delta, vega = self.pricer.price_and_greeks(
                    eval_date_py=current_date,
                    spot=current_price,
                    strike=strike_price,
                    vol=current_vol, # Using realized vol as a placeholder for implied vol
                    expiry_date_py=expiry_date,
                    option_type=config.action_type.split('_')[-1].lower() # e.g. 'CALL' from 'BUY_CALL'
                )
                estimated_option_price = round(opt_price, 2)
                estimated_delta = round(delta, 4)
                estimated_vega = round(vega, 4)

                # Vega-based sizing
                target_vega_exposure = self.params.get("target_option_vega_exposure")
                if target_vega_exposure is not None and estimated_vega != 0:
                    calculated_qty = target_vega_exposure / abs(estimated_vega * 100) # Vega is per 1% move, exposure might be dollar based for 1%
                    # Or if vega from pricer is already per 1.0 change in vol, then abs(estimated_vega)
                    # The OptionPricer returns vega per 1.0 change (divided by 100 from QL's typical output)
                    # So, if target_option_vega_exposure is $ for a 1 point change in IV:
                    calculated_qty = target_vega_exposure / abs(estimated_vega)


                    quantity = max(1, round(calculated_qty)) # Ensure at least 1 contract, whole number
                
            except Exception as e:
                print(f"Strategy {config.strategy_name} for {market_data.symbol}: Error during option pricing/sizing: {e}")
                # Fallback to default quantity or could decide to hold if pricing fails
                # For now, we'll proceed with default quantity but without price/Greeks info in reason.

            reason_suffix = ""
            if estimated_option_price is not None:
                reason_suffix = f" Est. Option Price: ${estimated_option_price:.2f}, Delta: {estimated_delta:.4f}, Vega: {estimated_vega:.4f}. Quantity: {quantity}."

            reason = f"Conditions met for {config.strategy_name}: RSI {rsi_val:.2f} ({config.rsi_operator.__name__} {self.params[config.rsi_threshold_key]}), MA trend favorable (MA {ma_val:.2f} {config.ma_operator.__name__} previous MA), Current Price {current_price:.2f}.{reason_suffix}"

            base_details = {
                "action_type": config.action_type,
                "symbol": market_data.symbol,
                "dte": self.params["option_dte"],
                "reason": reason,
            }
            action_specific_details = {
                "strike_price": strike_price,
                "quantity": quantity, # Updated quantity
                "estimated_option_price": estimated_option_price,
                "estimated_delta": estimated_delta,
                "estimated_vega": estimated_vega,
                "commission_per_contract": self.params["commission_per_contract"],
                "slippage_per_contract": self.params["slippage_per_contract"]
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


# --- ML-Driven Trading Strategy ---
class MLTradingStrategy:
    """
    A trading strategy driven by predictions from a machine learning model.
    The ML model is expected to output ternary signals: 1 (buy), -1 (sell), 0 (hold).
    """

    def __init__(
        self,
        ml_model, # Type hint would be MLModel, but need to import it from ml_models.py
        market_data_provider, # Type hint: MarketData from market_data.py
        strategy_params: Optional[Dict[str, Any]] = None,
    ):
        self.ml_model = ml_model
        self.market_data_provider = market_data_provider
        self.name = STRATEGY_ML_TERNARY
        self.pricer = OptionPricer() # Initialize OptionPricer
        
        self.params = {
            "option_dte": 30,
            "option_quantity": 1,
            "action_mapping": { # Default mapping of ML signals to actions
                1: ACTION_BUY_CALL, # ML signal 1 -> Buy Call
                -1: ACTION_BUY_PUT, # ML signal -1 -> Buy Put
                0: ACTION_HOLD,     # ML signal 0 -> Hold
            },
            # Parameters for prepare_features_and_labels if needed by MarketData
            "prediction_horizon": 1, # Example, should match model training
            "dynamic_threshold_factor": 0.5, # Example
            "commission_per_contract": 0.65,
            "slippage_per_contract": 0.01
        }
        if strategy_params:
            self.params.update(strategy_params)
            
        # Ensure market_data_provider has the methods needed
        if not all(hasattr(self.market_data_provider, attr) for attr in ['fetch_historical', 'prepare_features_and_labels', 'get_current_price']):
            raise ValueError("market_data_provider is missing required methods (fetch_historical, prepare_features_and_labels, get_current_price).")
        
        # Ensure ml_model has predict method
        if not hasattr(self.ml_model, 'predict'):
            raise ValueError("ml_model is missing required 'predict' method.")


    def _get_latest_features(self) -> Optional[pd.DataFrame]:
        """
        Fetches the latest data, prepares features, and returns the latest feature row.
        """
        try:
            # Fetch historical data to ensure indicators are up-to-date
            # The window fetched should be large enough for all indicators used in prepare_features_and_labels
            self.market_data_provider.fetch_historical(period="1y", interval="1d") # Example period/interval
            
            if self.market_data_provider.data is None or self.market_data_provider.data.empty:
                print(f"{self.name}: No data after fetch_historical for {self.market_data_provider.symbol}.")
                return None

            # Prepare features and labels. We only need X.
            # The prepare_features_and_labels method in MarketData should be capable of returning
            # features for the entire dataset. We'll then take the last row.
            X, _ = self.market_data_provider.prepare_features_and_labels(
                prediction_horizon=self.params["prediction_horizon"],
                dynamic_threshold_factor=self.params["dynamic_threshold_factor"],
            )

            if X is None or X.empty:
                print(f"{self.name}: No features (X) generated by prepare_features_and_labels for {self.market_data_provider.symbol}.")
                return None
            
            # Return the latest row of features
            return X.iloc[[-1]] # Return as DataFrame to keep column names
        except Exception as e:
            print(f"{self.name}: Error getting latest features for {self.market_data_provider.symbol}: {e}")
            return None

    def generate_signal(self) -> Dict[str, Any]:
        """
        Generates a trading signal based on the ML model's prediction.
        """
        symbol = self.market_data_provider.symbol
        latest_X_row = self._get_latest_features()

        if latest_X_row is None:
            return {
                "action": ACTION_HOLD,
                "details": {
                    "reason": f"Could not get latest features for {symbol}.",
                    "strategy": self.name,
                    "symbol": symbol,
                },
            }

        try:
            # Get prediction from ML model (e.g., [1], [-1], or [0])
            prediction_array = self.ml_model.predict(latest_X_row)
            if prediction_array is None or len(prediction_array) == 0:
                raise ValueError("Model prediction is None or empty.")
            
            ml_signal = int(prediction_array[0]) # Get the first element as the signal
        except Exception as e:
            print(f"{self.name}: Error during model prediction for {symbol}: {e}")
            return {
                "action": ACTION_HOLD,
                "details": {
                    "reason": f"Error during model prediction for {symbol}: {e}",
                    "strategy": self.name,
                    "symbol": symbol,
                },
            }

        action_type = self.params["action_mapping"].get(ml_signal, ACTION_HOLD)
        
        reason = f"ML Model prediction: {ml_signal} for {symbol}."
        if action_type == ACTION_HOLD and ml_signal not in self.params["action_mapping"]:
            reason = f"ML Model prediction: {ml_signal} (unmapped) for {symbol}. Defaulting to HOLD."
        elif action_type == ACTION_HOLD:
             reason = f"ML Model prediction: {ml_signal} (mapped to HOLD) for {symbol}."


        # For BUY actions, determine strike price (e.g., ATM or slightly OTM)
        # This is a simplified placeholder; more sophisticated logic would be needed.
        strike_price = None
        option_specific_details = {}

        if action_type == ACTION_BUY_CALL or action_type == ACTION_BUY_PUT:
            current_price = self.market_data_provider.get_current_price()
            if current_price:
                strike_price = round(current_price, 0) # Example: ATM strike
                estimated_premium = 0.0 # Default if pricing fails

                # Attempt to get estimated premium using the pricer
                try:
                    # We need current date and volatility for the pricer
                    # Assuming market_data_provider can give current date and some form of volatility
                    current_date_for_pricer = self.market_data_provider.data.index[-1].to_pydatetime() # Get last date from data
                    
                    # Volatility: Using a placeholder or fetching from market_data_provider if available
                    # For simplicity, let's assume a fixed volatility or that market_data_provider has a method for it.
                    # This part is crucial for realistic premium estimation.
                    # If market_data_provider has 'annualized_volatility_20' or similar:
                    vol_series = self.market_data_provider.data.get("annualized_volatility_20")
                    if vol_series is not None and not vol_series.empty:
                        current_vol_for_pricer = vol_series.iloc[-1]
                        if pd.isna(current_vol_for_pricer) or current_vol_for_pricer <=0:
                            current_vol_for_pricer = 0.20 # Fallback vol
                    else:
                        current_vol_for_pricer = 0.20 # Default placeholder if not available

                    option_type_for_pricer = 'call' if action_type == ACTION_BUY_CALL else 'put'
                    expiry_date_for_pricer = current_date_for_pricer + timedelta(days=self.params["option_dte"])

                    opt_price, _, _ = self.pricer.price_and_greeks(
                        eval_date_py=current_date_for_pricer,
                        spot=current_price,
                        strike=strike_price,
                        vol=current_vol_for_pricer,
                        expiry_date_py=expiry_date_for_pricer,
                        option_type=option_type_for_pricer
                    )
                    estimated_premium = round(opt_price, 2)
                    reason += f" Est. Premium: ${estimated_premium:.2f}."

                except Exception as pricing_error:
                    reason += f" Pricing error: {pricing_error}. Using default premium 0."
                    # print(f"Debug: Pricing error for {symbol} on {current_date_for_pricer} with vol {current_vol_for_pricer}: {pricing_error}")


                option_specific_details = {
                    "strike_price": strike_price,
                    "estimated_premium": estimated_premium, # Added estimated premium
                    "quantity": self.params["option_quantity"],
                    "dte": self.params["option_dte"],
                    "commission_per_contract": self.params["commission_per_contract"],
                    "slippage_per_contract": self.params["slippage_per_contract"],
                    "optionType": 'call' if action_type == ACTION_BUY_CALL else 'put' # Needed by backtester
                }
            else:
                # Cannot determine strike, so cannot place option trade
                reason += " Could not get current price to determine strike, defaulting to HOLD."
                action_type = ACTION_HOLD


        base_details = {
            "action_type": action_type,
            "symbol": symbol,
            "reason": reason,
            "strategy": self.name,
        }
        # Using TradingStrategy's helper for consistent formatting, if accessible and desired.
        # Or replicate its logic here.
        # For now, direct dict construction:
        final_details = base_details["details"] = {
            "symbol": symbol,
            "reason": reason,
            "strategy": self.name,
            "ml_prediction_raw": ml_signal,
        }
        final_details.update(option_specific_details)
        
        return {"action": action_type, "details": final_details}

# Example Usage (Conceptual - requires MLModel and MarketData instances)
# from ml_models import MLModel # Assuming MLModel can be loaded or instantiated
# from market_data import MarketData # Assuming MarketData can be instantiated

# if __name__ == '__main__':
#     # This is placeholder code and won't run directly without proper setup
#     # 1. Load or train an MLModel
#     # model_path = "path/to/your/model.joblib"
#     # ml_model_instance = MLModel.load_model(model_path) # Fictional load
#     
#     # 2. Setup MarketDataProvider
#     # stock_symbol = "AAPL"
#     # market_data_prov = MarketData(stock_symbol)
#     
#     # 3. Instantiate the strategy
#     # if ml_model_instance and market_data_prov:
#     #     ml_strategy = MLTradingStrategy(ml_model=ml_model_instance, market_data_provider=market_data_prov)
#     #     signal_decision = ml_strategy.generate_signal()
#     #     print(f"Generated Signal for {stock_symbol}: {signal_decision}")
#     # else:
#     #     print("Could not instantiate MLModel or MarketDataProvider.")
#     pass
