# Handles real-time and historical market data retrieval and technical indicators

import numpy as np  # Added for numerical operations
import pandas as pd
import ta  # Import ta at the top level
import yfinance as yf


class MarketData:
    def __init__(self, symbol: str):  # Removed ui parameter
        self.symbol = symbol
        self.data = None
        self.ticker_info = yf.Ticker(self.symbol)  # Store Ticker object

    @property
    def _has_valid_data(self) -> bool:
        """Checks if self.data is populated and not empty."""
        return self.data is not None and not self.data.empty

    @property
    def _has_valid_close_data(self) -> bool:
        """Checks if data is valid for close-based indicator calculations."""
        return (
            self._has_valid_data
            and "Close" in self.data.columns
            and not self.data["Close"].empty
        )

    @property
    def _has_valid_ohlc_data(self) -> bool:
        """Checks if data is valid for OHLC-based indicator calculations."""
        return self._has_valid_data and all(
            col in self.data.columns for col in ["High", "Low", "Close"]
        )

    @property
    def _has_valid_volume_data(self) -> bool:
        """Checks if data is valid for Volume-based indicator calculations."""
        return (
            self._has_valid_data
            and "Volume" in self.data.columns
            and not self.data["Volume"].empty
        )

    def _get_price_from_ticker_info(self):
        """Attempts to get current price from ticker_info.info."""
        try:
            info = self.ticker_info.info
            if "currentPrice" in info and info["currentPrice"] is not None:
                return info["currentPrice"]
            if "regularMarketPrice" in info and info["regularMarketPrice"] is not None:
                return info["regularMarketPrice"]
        except Exception as e:
            print(f"Error accessing ticker_info.info for {self.symbol}: {e}")
        return None

    def _get_price_from_recent_history(self):
        """Attempts to get current price from recent ticker_info.history."""
        try:
            recent_data = self.ticker_info.history(period="5d", interval="1h")

            if recent_data is None or recent_data.empty:
                print(
                    f"No recent history data found for {self.symbol} (period=5d, interval=1h)."
                )
                return None

            if "Close" not in recent_data.columns or recent_data["Close"].empty:
                print(
                    f"Recent history for {self.symbol} fetched, but 'Close' column is missing or empty."
                )
                return None

            return recent_data["Close"].iloc[-1]

        except Exception as e:
            print(f"Error fetching recent history for {self.symbol}: {e}")
        return None

    def fetch_historical(self, period="1y", interval="1d", use_adj_close=True):
        # yfinance auto_adjust=True by default, which provides adjusted close prices
        # and also adjusts Open, High, Low. This is generally preferred.
        # Fetch raw OHLCV + Adj Close
        self.data = self.ticker_info.history(
            period=period,
            interval=interval,
            auto_adjust=False
        )
        # If requested, override Close with Adj Close to handle splits/dividends
        if use_adj_close and "Adj Close" in self.data.columns:
            self.data["Close"] = self.data["Adj Close"]
        if self.data.empty:
            print(
                f"No historical data found for {self.symbol} with period={period}, interval={interval}"
            )
        else:
            # Calculate all technical indicators after fetching historical data
            self._calculate_all_technical_indicators()
        return self.data

    def _calculate_all_technical_indicators(self):
        """Helper method to calculate and store all technical indicators."""
        if not self._has_valid_data:  # Use helper property
            print(
                f"Cannot calculate technical indicators for {self.symbol}: Data not fetched or empty."
            )
            return

        # Import specific components from 'ta' package
        from ta.trend import SMAIndicator, EMAIndicator, MACD
        from ta.momentum import RSIIndicator
        from ta.volatility import BollingerBands, AverageTrueRange

        close = self.data["Close"]
        high = self.data["High"]
        low = self.data["Low"]
        vol = self.data["Volume"]

        # SMA / EMA
        self.data["sma_20"] = SMAIndicator(close, window=20, fillna=False).sma_indicator()
        self.data["ema_20"] = EMAIndicator(close, window=20, fillna=False).ema_indicator()
        self.data["sma_50"] = SMAIndicator(close, window=50, fillna=False).sma_indicator()
        self.data["ema_50"] = EMAIndicator(close, window=50, fillna=False).ema_indicator()
        self.data["sma_delta"] = self.data["sma_20"] - self.data["sma_50"]
        self.data["ema_delta"] = self.data["ema_20"] - self.data["ema_50"]

        # RSI
        self.data["rsi_14"] = RSIIndicator(close, window=14, fillna=False).rsi()

        # MACD
        macd_ind = MACD(close, window_slow=26, window_fast=12, window_sign=9, fillna=False)
        self.data["macd_line"] = macd_ind.macd()
        self.data["macd_signal_line"] = macd_ind.macd_signal()
        self.data["macd_hist"] = macd_ind.macd_diff()
        self.data["macd_crossover_bullish"] = ((self.data["macd_line"] > self.data["macd_signal_line"]) 
                                               & (self.data["macd_line"].shift(1) <= self.data["macd_signal_line"].shift(1))).astype(int)
        self.data["macd_crossover_bearish"] = ((self.data["macd_line"] < self.data["macd_signal_line"]) 
                                               & (self.data["macd_line"].shift(1) >= self.data["macd_signal_line"].shift(1))).astype(int)

        # Bollinger Bands
        bb = BollingerBands(close, window=20, window_dev=2, fillna=False)
        self.data["bb_upper"] = bb.bollinger_hband()
        self.data["bb_lower"] = bb.bollinger_lband()
        self.data["bb_mavg"] = bb.bollinger_mavg()
        self.data["bb_pband"] = bb.bollinger_pband()

        # ATR & normalized volatility
        atr = AverageTrueRange(high, low, close, window=14, fillna=False).average_true_range()
        self.data["atr_14"] = atr
        self.data["normalized_volatility_14"] = atr / close

        # Volume features
        self.data["volume_sma_5"] = vol.rolling(5).mean()
        self.data["volume_ratio"] = vol / self.data["volume_sma_5"]

        # Drop rows with NaNs created by indicators with different window sizes
        self.data.dropna(inplace=True)  # Ensure data is clean after all indicators are calculated

    def fetch_realtime(self):
        # Fetches last day data with 1-minute interval
        # Note: yfinance might have limitations on true real-time data frequency
        realtime_data = self.ticker_info.history(period="1d", interval="1m")
        if realtime_data.empty:
            print(
                f"No real-time data found for {self.symbol} (last 1d, 1m interval)")
        return realtime_data

    def get_current_price(self):
        """Fetches the current price from available sources, prioritizing."""
        # 1. Prefer the latest close from existing self.data if available
        if self._has_valid_close_data:
            return self.data["Close"].iloc[-1]

        # 2. Try fetching from ticker_info.info
        price = self._get_price_from_ticker_info()
        if price is not None:
            return price

        # 3. Fallback to fetching minimal recent history
        price = self._get_price_from_recent_history()
        if price is not None:
            return price

        print(
            f"Could not fetch current price for {self.symbol} from any source.")
        return None

    def calculate_moving_average(self, window=20):
        if self._has_valid_close_data:
            close_prices = self.data["Close"]
            sma = close_prices.rolling(
                window=window, min_periods=window).mean()
            return sma
        print(
            f"Data not available or 'Close' column missing/empty for MA calculation for {self.symbol}."
        )
        return pd.Series(dtype="float64")

    def calculate_exponential_moving_average(self, window=20):
        if self._has_valid_close_data:
            close_prices = self.data["Close"]
            ema = close_prices.ewm(
                span=window, adjust=False, min_periods=window).mean()
            return ema
        print(
            f"Data not available or 'Close' column missing/empty for EMA calculation for {self.symbol}."
        )
        return pd.Series(dtype="float64")

    def calculate_rsi(self, window=14):
        if self._has_valid_close_data:
            close_prices = self.data["Close"]
            try:
                rsi_indicator = ta.momentum.RSIIndicator(
                    close=close_prices, window=window, fillna=False
                )
                rsi = rsi_indicator.rsi()
                return rsi
            except ImportError:
                print(
                    "Technical Analysis library 'ta' not installed. RSI cannot be calculated."
                )
            except Exception as e:
                print(
                    f"Error calculating RSI with ta library for {self.symbol}: {e}")
            return pd.Series(dtype="float64")
        print(
            f"Data not available or 'Close' column missing/empty for RSI calculation for {self.symbol}."
        )
        return pd.Series(dtype="float64")

    def calculate_bollinger_bands(self, window=20, n_std=2):
        if self._has_valid_close_data:
            close_prices = self.data["Close"]
            sma = close_prices.rolling(
                window=window, min_periods=window).mean()
            std_dev = close_prices.rolling(
                window=window, min_periods=window).std()
            upper_band = sma + (std_dev * n_std)
            lower_band = sma - (std_dev * n_std)
            return upper_band, lower_band, sma
        print(
            f"Data not available or 'Close' column missing/empty for Bollinger Bands calculation for {self.symbol}."
        )
        return (
            pd.Series(dtype="float64"),
            pd.Series(dtype="float64"),
            pd.Series(dtype="float64"),
        )

    def calculate_bollinger_bands_with_pband(self, window=20, n_std=2):
        if self._has_valid_close_data:
            close_prices = self.data["Close"]
            try:
                bb_indicator = ta.volatility.BollingerBands(
                    close=close_prices, window=window, window_dev=n_std, fillna=False
                )
                upper_band = bb_indicator.bollinger_hband()
                lower_band = bb_indicator.bollinger_lband()
                mavg_band = bb_indicator.bollinger_mavg()
                pband = bb_indicator.bollinger_pband()
                return upper_band, lower_band, mavg_band, pband
            except ImportError:
                print(
                    "Technical Analysis library 'ta' not installed. Bollinger Bands (%B) cannot be calculated."
                )
            except Exception as e:
                print(
                    f"Error calculating Bollinger Bands (%B) with ta library for {self.symbol}: {e}"
                )
        print(
            f"Data not available or 'Close' column missing/empty for Bollinger Bands (%B) calculation for {self.symbol}."
        )
        return (
            pd.Series(dtype="float64"),
            pd.Series(dtype="float64"),
            pd.Series(dtype="float64"),
            pd.Series(dtype="float64"),
        )

    def calculate_macd(self, window_slow=26, window_fast=12, window_sign=9):
        if self._has_valid_close_data:
            close_prices = self.data["Close"]
            try:
                macd_indicator = ta.trend.MACD(
                    close=close_prices,
                    window_slow=window_slow,
                    window_fast=window_fast,
                    window_sign=window_sign,
                    fillna=False,
                )
                macd_line = macd_indicator.macd()
                signal_line = macd_indicator.macd_signal()
                macd_hist = macd_indicator.macd_diff()
                return macd_line, signal_line, macd_hist
            except ImportError:
                print(
                    "Technical Analysis library 'ta' not installed. MACD cannot be calculated."
                )
            except Exception as e:
                print(
                    f"Error calculating MACD with ta library for {self.symbol}: {e}")
        print(
            f"Data not available or 'Close' column missing/empty for MACD calculation for {self.symbol}."
        )
        return (
            pd.Series(dtype="float64"),
            pd.Series(dtype="float64"),
            pd.Series(dtype="float64"),
        )

    def calculate_atr(self, window=14):
        if self._has_valid_ohlc_data:  # Use OHLC helper
            try:
                atr_indicator = ta.volatility.AverageTrueRange(
                    high=self.data["High"],
                    low=self.data["Low"],
                    close=self.data["Close"],
                    window=window,
                    fillna=False,
                )
                atr = atr_indicator.average_true_range()
                return atr
            except ImportError:
                print(
                    "Technical Analysis library 'ta' not installed. ATR cannot be calculated."
                )
            except Exception as e:
                print(
                    f"Error calculating ATR with ta library for {self.symbol}: {e}")
            return pd.Series(dtype="float64")
        print(
            f"Data not available or 'High', 'Low', 'Close' columns missing/empty for ATR calculation for {self.symbol}."
        )
        return pd.Series(dtype="float64")

    def calculate_volatility(self, window=20):
        if self._has_valid_close_data:
            close_prices = self.data["Close"]
            log_returns = np.log(close_prices / close_prices.shift(1))
            volatility = log_returns.rolling(
                window=window, min_periods=window
            ).std() * np.sqrt(window)
            volatility.name = f"volatility_{window}"
            return volatility
        print(
            f"Data not available or 'Close' column missing/empty for Volatility calculation for {self.symbol}."
        )
        return pd.Series(dtype="float64")
