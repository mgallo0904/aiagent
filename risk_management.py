import dataclasses
from typing import Optional

import numpy as np
import pandas as pd  # Assuming pandas is used for Series type hints


# Placeholder for risk management (stop-loss, take-profit, risk assessment)
class RiskManager:
    def __init__(
        self,
        total_capital,
        risk_per_trade_pct=0.02,
        default_stop_loss_pct=0.05,
        default_take_profit_pct=0.10,
    ):
        self.total_capital = total_capital
        self.risk_per_trade_pct = (
            risk_per_trade_pct  # Max percentage of capital to risk on a single trade
        )
        self.default_stop_loss_pct = default_stop_loss_pct
        self.default_take_profit_pct = default_take_profit_pct
        print(
            f"RiskManager initialized with Capital: ${self.total_capital:,.2f}, Risk/Trade: {self.risk_per_trade_pct*100}%"
        )

    def _get_volatility_risk_factor(self, volatility: Optional[pd.Series]) -> str:
        """Determines risk factor based on volatility."""
        if volatility is not None and not volatility.empty:
            current_vol = volatility.iloc[-1]
            if pd.isna(current_vol):  # Check for NaN
                return "Volatility data (NaN)"
            if current_vol > 0.05:  # Example threshold for high volatility
                return "High Market Volatility"
            elif current_vol < 0.01:
                return "Low Market Volatility"
            return "Moderate Market Volatility"
        return "Volatility data unavailable"

    def _get_atr_risk_factor(self, atr: Optional[pd.Series]) -> str:
        """Determines risk factor based on ATR."""
        if atr is not None and not atr.empty:
            current_atr = atr.iloc[-1]
            if pd.isna(current_atr):  # Check for NaN
                return "ATR data (NaN)"
            return f"ATR available: {current_atr:.2f}"
        return "ATR data unavailable"

    def _get_position_risk_factors(self, position_details: Optional[dict]) -> list[str]:
        """Placeholder for more detailed position-specific risk factors."""
        if position_details:
            pass  # Actual logic would go here
        return []

    def assess_risk(self, market_data, position_details: Optional[dict] = None) -> str:
        """
        Assesses overall market risk and specific position risk if details are provided.
        `market_data` is an instance of the MarketData class.
        `position_details` could be a dict with info like entry_price, quantity, etc.
        """
        volatility = market_data.calculate_volatility()
        atr = market_data.calculate_atr()  # Assuming MarketData has calculate_atr

        risk_factors = [
            self._get_volatility_risk_factor(volatility),
            self._get_atr_risk_factor(atr),
        ]

        if position_details:
            risk_factors.extend(
                self._get_position_risk_factors(position_details))

        meaningful_factors = [
            rf
            for rf in risk_factors
            if not rf.endswith("unavailable")
            and not rf.endswith("(NaN)")
            and rf != "Risk assessment inconclusive."
        ]

        if "High Market Volatility" in risk_factors:
            return f"High Risk: {risk_factors}"
        elif (
            len(meaningful_factors) > 1 or "Moderate Market Volatility" in risk_factors
        ):
            return f"Moderate Risk: {risk_factors}"
        elif meaningful_factors:
            return f"Low Risk: {risk_factors}"

        return "Risk assessment inconclusive: Insufficient or unclear data."

    @dataclasses.dataclass
    class RiskPerShareParams:
        current_price: float
        stop_loss_price: Optional[float] = None
        atr_value: Optional[float] = None
        atr_multiplier: float = 2.0
        order_type: str = "BUY"

    def _calculate_risk_for_buy_order(
        self, current_price: float, stop_loss_price: Optional[float]
    ) -> Optional[float]:
        """Calculates risk for a BUY order given a stop_loss_price."""
        if stop_loss_price is None or stop_loss_price <= 0:
            return None
        risk = current_price - stop_loss_price
        if risk <= 0:
            print("Warning: Stop loss for BUY is not below current price.")
            return None
        return risk

    def _calculate_risk_for_sell_order(
        self, current_price: float, stop_loss_price: Optional[float]
    ) -> Optional[float]:
        """Calculates risk for a SELL order given a stop_loss_price."""
        if stop_loss_price is None or stop_loss_price <= 0:
            return None
        risk = stop_loss_price - current_price
        if risk <= 0:
            print("Warning: Stop loss for SELL is not above current price.")
            return None
        return risk

    def _calculate_risk_per_share(self, params: RiskPerShareParams) -> Optional[float]:
        """Helper to determine risk per share based on available inputs using a dataclass."""
        if params.stop_loss_price is not None:
            if params.order_type == "BUY":
                risk = self._calculate_risk_for_buy_order(
                    params.current_price, params.stop_loss_price
                )
                if risk is not None:
                    return risk
            elif params.order_type == "SELL":
                risk = self._calculate_risk_for_sell_order(
                    params.current_price, params.stop_loss_price
                )
                if risk is not None:
                    return risk

        if params.atr_value is not None and params.atr_value > 0:
            return params.atr_value * params.atr_multiplier

        return params.current_price * self.default_stop_loss_pct

    @dataclasses.dataclass
    class PositionSizeParams:
        current_price: float
        order_type: str = "BUY"
        stop_loss_price: Optional[float] = None
        atr_value: Optional[float] = None
        atr_multiplier: float = 2.0
        model_confidence_scalar: Optional[float] = (
            None  # New field for ML model confidence
        )

    def calculate_position_size(self, params: PositionSizeParams) -> int:
        """
        Calculates position size based on max risk per trade and stop-loss distance.
        Uses dataclass for parameters.
        Incorporates model_confidence_scalar to adjust capital at risk.
        """
        if params.current_price is None or params.current_price <= 0:
            print("Invalid current price for position sizing.")
            return 0

        base_capital_at_risk_pct = self.risk_per_trade_pct
        effective_risk_pct = base_capital_at_risk_pct

        if params.model_confidence_scalar is not None:
            # Clamp the scalar to be between 0.0 and 1.0
            # This scalar directly multiplies the base risk percentage.
            # A scalar of 1.0 means full base risk, 0.5 means half base risk, 0 means no risk.
            confidence_scalar_clamped = np.clip(
                params.model_confidence_scalar, 0.0, 1.0
            )
            effective_risk_pct *= confidence_scalar_clamped
            print(
                f"Model confidence scalar provided: {params.model_confidence_scalar:.4f}, Clamped: {confidence_scalar_clamped:.4f}"
            )
            print(
                f"Base risk/trade pct: {base_capital_at_risk_pct*100:.2f}%, Effective risk/trade pct after confidence: {effective_risk_pct*100:.2f}%"
            )
        else:
            print(
                "No model confidence scalar provided, using full base risk percentage."
            )

        capital_at_risk = self.total_capital * effective_risk_pct

        print(f"Total Capital: ${self.total_capital:,.2f}")
        print(
            f"Calculated Capital at Risk for this trade: ${capital_at_risk:,.2f}")

        rps_params = self.RiskPerShareParams(
            current_price=params.current_price,
            stop_loss_price=params.stop_loss_price,
            atr_value=params.atr_value,
            atr_multiplier=params.atr_multiplier,
            order_type=params.order_type,
        )
        risk_per_share = self._calculate_risk_per_share(rps_params)

        if risk_per_share is None or risk_per_share <= 0:
            print(
                "Risk per share is zero, negative, or invalid. Cannot calculate position size."
            )
            if capital_at_risk == 0:
                print(
                    "This might be due to zero capital at risk (e.g. low model confidence or zero effective risk pct)."
                )
            return 0

        if (
            capital_at_risk == 0
        ):  # If confidence scalar or effective risk pct made it zero
            print("Capital at risk is zero. Position size will be 0.")
            return 0

        position_size = capital_at_risk / risk_per_share
        final_position_size = int(position_size)
        print(
            f"Calculated position size (shares/contracts): {final_position_size}")
        return final_position_size

    def _get_stop_offset(
        self, current_price: float, atr_value: Optional[float], atr_multiplier: float
    ) -> float:
        """Calculates the stop offset based on ATR or default percentage."""
        if atr_value is not None and atr_value > 0:
            return atr_value * atr_multiplier
        return current_price * self.default_stop_loss_pct

    def calculate_stop_loss(
        self,
        current_price: float,
        order_type: str = "BUY",
        atr_value: Optional[float] = None,
        atr_multiplier: float = 2.0,
    ) -> Optional[float]:
        """
        Calculates a stop-loss price.
        For a BUY order, stop-loss is below current_price.
        For a SELL order (short sell), stop-loss is above current_price.
        """
        if current_price is None or current_price <= 0:
            print("Error: Invalid current_price for stop-loss calculation.")
            return None

        stop_offset = self._get_stop_offset(
            current_price, atr_value, atr_multiplier)

        if stop_offset <= 0:
            print(
                "Warning: Stop offset is zero or negative. Cannot set a meaningful stop loss."
            )
            return None

        adjustment_factor = 0
        if order_type == "BUY":
            adjustment_factor = -1
        elif order_type == "SELL":
            adjustment_factor = 1
        else:
            print(
                f"Warning: Invalid order_type '{order_type}' for stop-loss calculation."
            )
            return None

        return current_price + (stop_offset * adjustment_factor)

    def calculate_take_profit(
        self, current_price, order_type="BUY", atr_value=None, risk_reward_ratio=2.0
    ):
        """
        Calculates a take-profit price.
        For a BUY order, take-profit is above current_price.
        For a SELL order (short sell), take-profit is below current_price.
        Uses a risk_reward_ratio relative to a potential stop-loss distance.
        The stop-loss distance for R:R is based on ATR (with a default 2.0 multiplier) or default_stop_loss_pct.
        """
        if current_price is None:
            return None

        stop_loss_distance = None
        if atr_value is not None and atr_value > 0:
            stop_loss_distance = atr_value * 2.0
        else:
            stop_loss_distance = current_price * self.default_stop_loss_pct

        if order_type == "BUY":
            return current_price + (stop_loss_distance * risk_reward_ratio)
        elif order_type == "SELL":  # Short sell
            return current_price - (stop_loss_distance * risk_reward_ratio)
        return None

    def check_trade_risk(self, proposed_trade_value):
        """
        Checks if a single trade's value exceeds a certain percentage of total capital.
        This is a simple check on notional value, not necessarily risk.
        A more accurate check would use the calculated stop-loss.
        """
        max_trade_value = (
            self.total_capital * 0.25
        )  # Example: no single trade > 25% of capital
        if proposed_trade_value > max_trade_value:
            print(
                f"Warning: Proposed trade value ${proposed_trade_value:,.2f} exceeds 25% of capital."
            )
            return False
        return True

    def update_capital(self, pnl):
        """Updates total capital after a trade is closed."""
        self.total_capital += pnl
        print(
            f"Capital updated by ${pnl:,.2f}. New total capital: ${self.total_capital:,.2f}"
        )
