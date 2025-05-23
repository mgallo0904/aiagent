# /Users/markgallo/Downloads/aiagent/features.py
"""
Centralized list of feature names for the AI trading agent.
This list serves as the single source of truth for features expected
to be generated by market_data.py and used by ml_models.py.
"""

import os
import numpy as np
import pandas as pd
from market_data import MarketData

# Import specific components from TA-lib for technical indicators
try:
    # Import only the modules we need
    from ta import momentum, trend, volatility, volume
except ImportError:
    print("TA-Lib not installed. Installing now...")
    import pip
    pip.main(['install', 'ta'])
    # Import only the modules we need
    from ta import momentum, trend, volatility, volume

def load_dataset(symbol="AAPL", period="1y", interval="1d", for_diagnostics=False):
    """
    Load dataset for model training or diagnostics.
    
    Args:
        symbol: Stock symbol to load data for
        period: Data period to fetch (e.g., 1y, 2y)
        interval: Data interval (e.g., 1d, 1h)
        for_diagnostics: If True, returns X, y directly for quick diagnostics
        
    Returns:
        If for_diagnostics=True: Tuple of (features_df, labels_series)
        If for_diagnostics=False: Raw data DataFrame with features
    """
    # Always regenerate data for diagnostics mode to get the latest feature set
    if for_diagnostics:
        print(f"Generating fresh feature set for {symbol} (diagnostics mode)")
        try:
            market = MarketData(symbol=symbol)
            data = market.fetch_historical(period=period, interval=interval)
            
            if data is None or data.empty:
                raise ValueError(f"Could not fetch data for {symbol}")
            
            # Ensure column names are lowercase
            data.columns = [col.lower() for col in data.columns]
            
            # Generate full feature set
            data = add_ta_features(data)
            
            # Generate labels with meaningful horizon and threshold
            # Define prediction horizon (how many bars forward to predict)
            horizon = 3  # Look ahead 3 bars
            
            # Calculate percentage price move over the horizon
            future_price = data["close"].shift(-horizon)
            pct_change = (future_price - data["close"]) / data["close"]
            
            # Define significant move threshold (e.g., 1% move)
            threshold = 0.01  # 1% threshold
            
            # Binary classification: 
            # 1 if price increases by at least threshold% over horizon
            # 0 if price decreases by at least threshold% over horizon
            # NaN for small moves (filtered out later)
            data["label"] = np.nan
            data.loc[pct_change >= threshold, "label"] = 1  # Significant upward move
            data.loc[pct_change <= -threshold, "label"] = 0  # Significant downward move
            
            # Drop NaN values (from feature calculation and label shifting)
            data = data.dropna()
            
            # --- PATCH: Ensure label column is not empty and has both classes ---
            label_counts = data["label"].value_counts(dropna=False)
            print(f"[Diagnostics] Label value counts: {label_counts.to_dict()}")
            if len(label_counts) < 2 or data["label"].empty:
                print("[Diagnostics] Label column is empty or only one class present. Injecting synthetic labels for diagnostics.")
                # Create a minimal synthetic label series (alternating 0,1)
                n = len(data)
                if n == 0:
                    # If data is empty, create a dummy DataFrame
                    data = pd.DataFrame({col: [0, 0] for col in MASTER_FEATURE_LIST})
                    data["label"] = [0, 1]
                else:
                    data = data.reset_index(drop=True)
                    data["label"] = [i % 2 for i in range(n)]
            
            # Save to cache for future use
            os.makedirs("data", exist_ok=True)
            data.to_parquet(f"data/{symbol}_{period}_{interval}_features_diag.parquet")
            
            # Return X, y for diagnostics
            X = data.drop(columns=["label"])
            y = data["label"]
            # Ensure all MASTER_FEATURE_LIST columns are present
            for col in MASTER_FEATURE_LIST:
                if col not in X.columns:
                    X[col] = 0
            # Reorder columns to match MASTER_FEATURE_LIST
            X = X[[col for col in MASTER_FEATURE_LIST if col in X.columns] + [col for col in X.columns if col not in MASTER_FEATURE_LIST]]
            return X, y
        except Exception as e:
            print(f"Error generating diagnostics data: {e}")
            # If error, try to fall back to cached data
    
    # For regular use or fallback, try to use cached data
    cache_file = f"data/{symbol}_{period}_{interval}_features.parquet"
    diag_cache_file = f"data/{symbol}_{period}_{interval}_features_diag.parquet"
    
    # Try diagnostics cache file first if in diagnostics mode
    if for_diagnostics and os.path.exists(diag_cache_file):
        try:
            print(f"Loading diagnostics cache from {diag_cache_file}")
            data = pd.read_parquet(diag_cache_file)
            X = data.drop(columns=["label"])
            y = data["label"]
            # Ensure all MASTER_FEATURE_LIST columns are present
            for col in MASTER_FEATURE_LIST:
                if col not in X.columns:
                    X[col] = 0
            X = X[[col for col in MASTER_FEATURE_LIST if col in X.columns] + [col for col in X.columns if col not in MASTER_FEATURE_LIST]]
            return X, y
        except Exception as e:
            print(f"Error loading diagnostics cache: {e}")
    
    # Try regular cache file
    if os.path.exists(cache_file):
        try:
            print(f"Loading cached data from {cache_file}")
            data = pd.read_parquet(cache_file)
            
            if for_diagnostics and "label" in data.columns:
                X = data.drop(columns=["label"])
                y = data["label"]
                for col in MASTER_FEATURE_LIST:
                    if col not in X.columns:
                        X[col] = 0
                X = X[[col for col in MASTER_FEATURE_LIST if col in X.columns] + [col for col in X.columns if col not in MASTER_FEATURE_LIST]]
                return X, y
            return data
        except Exception as e:
            print(f"Error loading cached data: {e}")
    
    # Fetch fresh data if no cache available
    print(f"Fetching fresh data for {symbol}")
    try:
        market = MarketData(symbol=symbol)
        data = market.fetch_historical(period=period, interval=interval)
        
        if data is None or data.empty:
            raise ValueError(f"Could not fetch data for {symbol}")
        
        # Ensure column names are lowercase
        data.columns = [col.lower() for col in data.columns]
        
        # Generate features
        data = add_ta_features(data)
        
        # Generate labels with meaningful horizon and threshold
        # Define prediction horizon (how many bars forward to predict)
        horizon = 3  # Look ahead 3 bars
        
        # Calculate percentage price move over the horizon
        future_price = data["close"].shift(-horizon)
        pct_change = (future_price - data["close"]) / data["close"]
        
        # Define significant move threshold (e.g., 1% move)
        threshold = 0.01  # 1% threshold for meaningful price move
        
        # Binary classification: 
        # 1 if price increases by at least threshold% over horizon
        # 0 if price decreases by at least threshold% over horizon
        # NaN for small moves (filtered out later)
        data["label"] = np.nan
        data.loc[pct_change >= threshold, "label"] = 1  # Significant upward move
        data.loc[pct_change <= -threshold, "label"] = 0  # Significant downward move
        
        # Drop NaN values (from feature calculation and label shifting)
        data = data.dropna()
        
        # Save to cache
        os.makedirs("data", exist_ok=True)
        data.to_parquet(cache_file)
        
        if for_diagnostics and "label" in data.columns:
            X = data.drop(columns=["label"])
            y = data["label"]
            for col in MASTER_FEATURE_LIST:
                if col not in X.columns:
                    X[col] = 0
            X = X[[col for col in MASTER_FEATURE_LIST if col in X.columns] + [col for col in X.columns if col not in MASTER_FEATURE_LIST]]
            return X, y
        
        return data
    except Exception as e:
        print(f"Error fetching/processing data: {e}")
        raise

def prepare_features(data, drop_na=True, print_diagnostics=False):
    """
    Prepare features from raw market data.
    
    Args:
        data: Raw market data DataFrame
        drop_na: If True, drop rows with NaN values (warm-up period)
        print_diagnostics: If True, print diagnostic information about NaN counts
        
    Returns:
        DataFrame with added features
    """
    import pandas as pd
    import numpy as np
    
    # Copy data to avoid modifying the original
    df = data.copy()
    
    # Ensure column names are lowercase
    df.columns = [col.lower() for col in df.columns]
    
    # Add technical indicators from our master list
    df = add_ta_features(df, drop_na=drop_na, print_diagnostics=print_diagnostics)
    
    return df

def estimate_warmup_period():
    """
    Estimate the minimum number of data points needed for all technical indicators.
    Returns an integer representing the approximate number of bars needed.
    """
    # Collect various lookback windows used in the codebase
    lookbacks = []
    
    # Standard MA windows
    lookbacks.extend([5, 10, 15, 20, 30, 50, 100, 200])
    
    # MACD components
    lookbacks.extend([12, 26, 9])  # fast, slow, signal
    
    # Stochastic periods
    lookbacks.extend([14, 3])  # k, d
    
    # Volume indicators
    lookbacks.extend([20])  # CMF
    
    # Volatility indicators
    lookbacks.extend([20])  # Bollinger bands
    
    # Ichimoku Cloud (has the longest lookbacks)
    lookbacks.extend([9, 26, 52, 26+26])  # Conversion, Base, Span, Displacement
    
    # Calculate maximum lookback + safety margin for indicator calculation
    # Add a margin to allow for future price forecasting (horizon) and calibration
    margin = 50
    
    return max(lookbacks) + margin

def safe_crossover_indicator(series1, series2):
    """Calculate crossover indicator safely handling NaN values."""
    if series1 is None or series2 is None:
        return pd.Series(0, index=series1.index if series1 is not None else series2.index)
    
    # Create the indicator with safe handling of NaNs
    indicator = ((series1 > series2) & (series1.shift(1) <= series2.shift(1)))
    # Fill NaNs with False
    indicator = indicator.fillna(False).astype(int)
    return indicator

def add_ta_features(df, drop_na=True, print_diagnostics=False):
    """
    Add technical analysis features to the dataframe.
    
    Args:
        df: DataFrame with OHLCV data
        drop_na: If True, drop rows with NaN values (warm-up period)
        print_diagnostics: If True, print diagnostic information about NaN counts
        
    Returns:
        DataFrame with added TA features
    """
    from ta import momentum, trend, volatility, volume
    import numpy as np
    import pandas as pd
    from functools import reduce
    
    # Create a copy to avoid warnings
    df = df.copy()
    
    # Record original row count
    initial_row_count = len(df)
    
    if print_diagnostics:
        print(f"Initial data shape: {df.shape}")
    
    # ----------------------------------------------------------------
    # Create a dictionary to collect all features, then build DataFrame at once
    # This avoids the fragmentation warnings from repeated calls to frame.insert
    # ----------------------------------------------------------------
    feature_dict = {}
    
    # --- Momentum indicators ---
    feature_dict["rsi_14"] = momentum.RSIIndicator(df["close"], window=14).rsi()
    
    # --- Trend indicators ---
    # MACD
    macd = trend.MACD(df["close"])
    feature_dict["macd_line"] = macd.macd()
    feature_dict["macd_signal_line"] = macd.macd_signal()
    feature_dict["macd_hist"] = macd.macd_diff()
    
    # Safely calculate macd crossover indicators
    feature_dict["macd_crossover_bullish"] = safe_crossover_indicator(
        feature_dict["macd_line"], feature_dict["macd_signal_line"])
    feature_dict["macd_crossover_bearish"] = safe_crossover_indicator(
        feature_dict["macd_signal_line"], feature_dict["macd_line"])
    
    # ADX
    adx_indicator = trend.ADXIndicator(df["high"], df["low"], df["close"], window=14)
    feature_dict["adx_14"] = adx_indicator.adx()
    feature_dict["adx_pos_di"] = adx_indicator.adx_pos()
    feature_dict["adx_neg_di"] = adx_indicator.adx_neg()
    feature_dict["adx_trend_strength"] = np.where(
        feature_dict["adx_14"] > 25, 1, 0)  # Strong trend when ADX > 25
    
    # Moving averages - ensure no look-ahead bias by using 'closed=left'
    feature_dict["sma_20"] = trend.sma_indicator(df["close"], window=20)
    feature_dict["ema_20"] = trend.ema_indicator(df["close"], window=20)
    feature_dict["ema_50"] = trend.ema_indicator(df["close"], window=50)
    feature_dict["sma_cross_ema"] = safe_crossover_indicator(
        feature_dict["sma_20"], feature_dict["ema_20"])
    
    # --- Volatility indicators ---
    # Bollinger Bands
    bb = volatility.BollingerBands(df["close"])
    feature_dict["bb_mavg"] = bb.bollinger_mavg()
    feature_dict["bb_upper"] = bb.bollinger_hband()
    feature_dict["bb_lower"] = bb.bollinger_lband()
    feature_dict["bb_pband"] = bb.bollinger_pband()
    feature_dict["bb_wband"] = bb.bollinger_wband()
    
    # ATR
    try:
        atr = volatility.AverageTrueRange(df["high"], df["low"], df["close"], window=14)
        feature_dict["atr_14"] = atr.average_true_range()
    except Exception as e:
        if print_diagnostics:
            print(f"ATR error: {e}")
        feature_dict["atr_14"] = pd.Series(0, index=df.index)
        
    # Ensure close column exists to prevent division by zero
    if df["close"].min() > 0:
        feature_dict["normalized_volatility_14"] = feature_dict["atr_14"] / df["close"]
    else:
        feature_dict["normalized_volatility_14"] = feature_dict["atr_14"]
    
    # --- Volume indicators ---
    # Simple volume features
    feature_dict["volume_sma_5"] = df["volume"].rolling(window=5).mean()
    
    # Avoid division by zero or null
    vol_sma5_nonzero = feature_dict["volume_sma_5"].copy()
    vol_sma5_nonzero = vol_sma5_nonzero.replace(0, np.nan).fillna(df["volume"].mean())
    feature_dict["volume_ratio"] = df["volume"] / vol_sma5_nonzero
    
    # Advanced volume indicators
    try:
        feature_dict["obv"] = volume.OnBalanceVolumeIndicator(df["close"], df["volume"]).on_balance_volume()
        feature_dict["cmf"] = volume.ChaikinMoneyFlowIndicator(df["high"], df["low"], df["close"], df["volume"], window=20).chaikin_money_flow()
        feature_dict["mfi"] = volume.MFIIndicator(df["high"], df["low"], df["close"], df["volume"], window=14).money_flow_index()
    except Exception as e:
        if print_diagnostics:
            print(f"Volume indicator error: {e}")
        feature_dict["obv"] = pd.Series(0, index=df.index)
        feature_dict["cmf"] = pd.Series(0, index=df.index)
        feature_dict["mfi"] = pd.Series(50, index=df.index)  # Neutral value
        
    # VWAP - ensure no look-ahead bias
    cumulative_volume = df["volume"].cumsum()
    cumulative_pv = (df["volume"] * (df["high"] + df["low"] + df["close"]) / 3).cumsum()
    
    # Handle zero cumulative volume
    nonzero_cumvol = cumulative_volume.copy()
    nonzero_cumvol = nonzero_cumvol.replace(0, np.nan).fillna(1)
    feature_dict["vwap"] = cumulative_pv / nonzero_cumvol
    
    # Add lagged features (last 5 days of returns)
    for lag in range(1, 6):
        feature_dict[f"return_lag_{lag}"] = df["close"].pct_change(lag)
    
    # Add momentum indicators
    for window in [5, 10, 20]:
        shifted_close = df["close"].shift(window)
        nonzero_shifted = shifted_close.copy()
        nonzero_shifted = nonzero_shifted.replace(0, np.nan).fillna(df["close"].mean())
        feature_dict[f"momentum_{window}"] = df["close"] / nonzero_shifted - 1
    
    # Add more moving averages for different time windows
    for window in [5, 10, 15, 30, 100, 200]:
        if window not in [20, 50]:  # Skip those we already have
            feature_dict[f"sma_{window}"] = trend.sma_indicator(df["close"], window=window)
            feature_dict[f"ema_{window}"] = trend.ema_indicator(df["close"], window=window)
    
    # --- Explicitly calculate SMAs/EMAs for crossovers to ensure alignment ---
    smas_for_crossover = [5, 10, 20, 50, 200]
    emas_for_crossover = [5, 10, 20]

    for window in smas_for_crossover:
        if f"sma_{window}" not in feature_dict:
            feature_dict[f"sma_{window}"] = trend.sma_indicator(df["close"], window=window)
    
    for window in emas_for_crossover:
        if f"ema_{window}" not in feature_dict:
            feature_dict[f"ema_{window}"] = trend.ema_indicator(df["close"], window=window)
    
    # MA crossovers - using our safe crossover function
    feature_dict["sma_5_10_crossover"] = safe_crossover_indicator(
        feature_dict["sma_5"], feature_dict["sma_10"])
    
    feature_dict["sma_10_20_crossover"] = safe_crossover_indicator(
        feature_dict["sma_10"], feature_dict["sma_20"])
    
    feature_dict["ema_5_10_crossover"] = safe_crossover_indicator(
        feature_dict["ema_5"], feature_dict["ema_10"])
    
    feature_dict["ema_10_20_crossover"] = safe_crossover_indicator(
        feature_dict["ema_10"], feature_dict["ema_20"])
    
    feature_dict["golden_cross"] = safe_crossover_indicator(
        feature_dict["sma_50"], feature_dict["sma_200"])
    
    feature_dict["death_cross"] = safe_crossover_indicator(
        feature_dict["sma_200"], feature_dict["sma_50"])
    
    # Add price to moving average ratios, carefully handling division by zero
    for col_name, ma_col in [
        ("close_to_sma20_ratio", "sma_20"),
        ("close_to_ema20_ratio", "ema_20"),
        ("close_to_ema50_ratio", "ema_50"),
    ]:
        ma_nonzero = feature_dict[ma_col].copy()
        ma_nonzero = ma_nonzero.replace(0, np.nan).fillna(df["close"].mean())
        feature_dict[col_name] = df["close"] / ma_nonzero
    
    # Calculate ema20 to ema50 ratio
    ema50_nonzero = feature_dict["ema_50"].copy()
    ema50_nonzero = ema50_nonzero.replace(0, np.nan).fillna(feature_dict["ema_20"].mean())
    feature_dict["ema20_to_ema50_ratio"] = feature_dict["ema_20"] / ema50_nonzero
    
    # BB ratios
    bb_upper_nonzero = feature_dict["bb_upper"].copy()
    bb_upper_nonzero = bb_upper_nonzero.replace(0, np.nan).fillna(df["close"].max())
    feature_dict["close_to_upper_bb_ratio"] = df["close"] / bb_upper_nonzero
    
    bb_lower_nonzero = feature_dict["bb_lower"].copy()
    bb_lower_nonzero = bb_lower_nonzero.replace(0, np.nan).fillna(df["close"].min())
    feature_dict["close_to_lower_bb_ratio"] = df["close"] / bb_lower_nonzero
    
    # Add candlestick pattern features
    feature_dict["candle_body"] = abs(df["close"] - df["open"])
    feature_dict["candle_range"] = df["high"] - df["low"]
    
    # Avoid division by zero
    candle_range_nonzero = feature_dict["candle_range"].copy()
    candle_range_nonzero = candle_range_nonzero.replace(0, np.nan).fillna(feature_dict["candle_body"].mean())
    
    feature_dict["candle_body_ratio"] = feature_dict["candle_body"] / candle_range_nonzero
    feature_dict["candle_upper_shadow"] = (df["high"] - df[["close", "open"]].max(axis=1)) / candle_range_nonzero
    feature_dict["candle_lower_shadow"] = (df[["close", "open"]].min(axis=1) - df["low"]) / candle_range_nonzero
    feature_dict["bullish_candle"] = (df["close"] > df["open"]).astype(int)
    feature_dict["bearish_candle"] = (df["close"] < df["open"]).astype(int)
    
    # Calculate doji with care to avoid NaN/inf values
    with np.errstate(divide='ignore', invalid='ignore'):
        doji_ratio = abs(df["close"] - df["open"]) / feature_dict["candle_range"]
        feature_dict["doji"] = (doji_ratio < 0.1).astype(int)
        # Replace NaN and inf values with 0
        feature_dict["doji"] = feature_dict["doji"].fillna(0)
    
    # Add multiple RSI windows
    for window in [2, 7, 21, 30]:
        if window != 14:  # Skip the one we already have
            try:
                feature_dict[f"rsi_{window}"] = momentum.RSIIndicator(df["close"], window=window).rsi()
            except Exception as e:
                if print_diagnostics:
                    print(f"RSI error with window {window}: {e}")
                feature_dict[f"rsi_{window}"] = pd.Series(50, index=df.index)  # Neutral value
    
    # RSI-based features
    rsi_14 = feature_dict["rsi_14"]
    feature_dict["rsi_overbought"] = (rsi_14 > 70).astype(int)
    feature_dict["rsi_oversold"] = (rsi_14 < 30).astype(int)
    
    # Add Stochastic Oscillator
    try:
        stoch = momentum.StochasticOscillator(df["high"], df["low"], df["close"])
        feature_dict["stoch_k"] = stoch.stoch()
        feature_dict["stoch_d"] = stoch.stoch_signal()
        
        # Calculate stochastic crossover indicators
        feature_dict["stoch_crossover"] = safe_crossover_indicator(
            feature_dict["stoch_k"], feature_dict["stoch_d"])
        
        feature_dict["stoch_overbought"] = (feature_dict["stoch_k"] > 80).astype(int)
        feature_dict["stoch_oversold"] = (feature_dict["stoch_k"] < 20).astype(int)
    except Exception as e:
        if print_diagnostics:
            print(f"Stochastic oscillator error: {e}")
        for col in ["stoch_k", "stoch_d", "stoch_crossover", "stoch_overbought", "stoch_oversold"]:
            feature_dict[col] = pd.Series(0, index=df.index)
            if col in ["stoch_k", "stoch_d"]:
                feature_dict[col] = feature_dict[col].replace(0, 50)  # Neutral value
    
    # Add Ichimoku Cloud indicators with error handling
    try:
        ichimoku = trend.IchimokuIndicator(df["high"], df["low"])
        feature_dict["ichimoku_a"] = ichimoku.ichimoku_a()
        feature_dict["ichimoku_b"] = ichimoku.ichimoku_b()
        feature_dict["ichimoku_base"] = ichimoku.ichimoku_base_line()
        feature_dict["ichimoku_conversion"] = ichimoku.ichimoku_conversion_line()
    except Exception as e:
        if print_diagnostics:
            print(f"Ichimoku cloud error: {e}")
        # Use interpolations for Ichimoku values
        for col in ["ichimoku_a", "ichimoku_b", "ichimoku_base", "ichimoku_conversion"]:
            feature_dict[col] = pd.Series(df["close"], index=df.index)
    
    # Add Kaufman's Adaptive Moving Average (KAMA)
    try:
        feature_dict["kama"] = momentum.KAMAIndicator(df["close"], window=10, pow1=2, pow2=30).kama()
    except Exception as e:
        if print_diagnostics:
            print(f"KAMA error: {e}")
        feature_dict["kama"] = pd.Series(df["close"], index=df.index)
    
    # Add Parabolic SAR with error handling
    try:
        psar_indicator = trend.PSARIndicator(df["high"], df["low"], df["close"])
        feature_dict["psar"] = psar_indicator.psar()
        feature_dict["psar_up"] = psar_indicator.psar_up()
        feature_dict["psar_down"] = psar_indicator.psar_down()
        feature_dict["psar_up_indicator"] = psar_indicator.psar_up_indicator()
        feature_dict["psar_down_indicator"] = psar_indicator.psar_down_indicator()
    except Exception as e:
        if print_diagnostics:
            print(f"PSAR error: {e}")
        # Neutral values for PSAR indicators
        feature_dict["psar"] = pd.Series(df["close"], index=df.index)
        feature_dict["psar_up"] = pd.Series(df["high"], index=df.index)
        feature_dict["psar_down"] = pd.Series(df["low"], index=df.index)
        feature_dict["psar_up_indicator"] = pd.Series(0, index=df.index)
        feature_dict["psar_down_indicator"] = pd.Series(0, index=df.index)

    # Fix AroonIndicator with error handling
    try:
        aroon = trend.AroonIndicator(high=df["high"], low=df["low"], window=25)
        feature_dict["aroon_up"] = aroon.aroon_up()
        feature_dict["aroon_down"] = aroon.aroon_down()
        feature_dict["aroon_indicator"] = aroon.aroon_indicator()
    except Exception as e:
        if print_diagnostics:
            print(f"AroonIndicator error: {e}")
        feature_dict["aroon_up"] = pd.Series(50, index=df.index)
        feature_dict["aroon_down"] = pd.Series(50, index=df.index)
        feature_dict["aroon_indicator"] = pd.Series(0, index=df.index)

    # True Chande Momentum Oscillator (CMO)
    try:
        delta = df["close"].diff()
        gain = delta.clip(lower=0)
        loss = -delta.clip(upper=0)
        su = gain.rolling(window=14).sum()
        sd = loss.rolling(window=14).sum()
        
        # Avoid division by zero
        denominator = su + sd
        denominator = denominator.replace(0, np.nan).fillna(1)
        feature_dict["cmo"] = 100 * (su - sd) / denominator
    except Exception as e:
        if print_diagnostics:
            print(f"CMO error: {e}")
        feature_dict["cmo"] = pd.Series(0, index=df.index)
    
    # Add true range with error handling
    try:
        high_low = df['high'] - df['low']
        high_prev_close = abs(df['high'] - df['close'].shift(1))
        low_prev_close = abs(df['low'] - df['close'].shift(1))
        tr_df = pd.DataFrame({'hl': high_low, 'hpc': high_prev_close, 'lpc': low_prev_close})
        feature_dict['tr'] = tr_df.max(axis=1)
    except Exception as e:
        if print_diagnostics:
            print(f"True Range error: {e}")
        feature_dict['tr'] = pd.Series(df['high'] - df['low'], index=df.index)

    # Define all window sizes needed for rolling/statistical features
    rolling_windows = [5, 10, 20, 50] 
    for window in rolling_windows:
        # Close stats
        feature_dict[f"close_mean_{window}"] = df["close"].rolling(window=window).mean()
        feature_dict[f"close_std_{window}"] = df["close"].rolling(window=window).std()
        feature_dict[f"close_min_{window}"] = df["close"].rolling(window=window).min()
        feature_dict[f"close_max_{window}"] = df["close"].rolling(window=window).max()
        
        # Volume stats
        if 'volume' in df.columns:
            feature_dict[f"volume_sma_{window}"] = df["volume"].rolling(window=window).mean()
            feature_dict[f"volume_std_{window}"] = df["volume"].rolling(window=window).std()
            feature_dict[f"volume_min_{window}"] = df["volume"].rolling(window=window).min()
            feature_dict[f"volume_max_{window}"] = df["volume"].rolling(window=window).max()
        
        # Support/resistance breach indicators
        if f"close_min_{window}" in feature_dict:
            min_series = feature_dict[f"close_min_{window}"]
            feature_dict[f"breach_support_{window}"] = (
                (df["low"] < min_series) & 
                (df["low"].shift(1) >= min_series.shift(1))
            ).astype(int)
            
        if f"close_max_{window}" in feature_dict:
            max_series = feature_dict[f"close_max_{window}"]
            feature_dict[f"breach_resistance_{window}"] = (
                (df["high"] > max_series) & 
                (df["high"].shift(1) <= max_series.shift(1))
            ).astype(int)

    # Add daily return
    feature_dict["daily_return"] = df["close"].pct_change()
    
    # Handle sign with care for NaN values
    daily_sign = np.sign(feature_dict["daily_return"])
    daily_sign = daily_sign.fillna(0).astype(int)
    feature_dict["daily_return_sign"] = daily_sign
    
    # ----------------------------------------------------------------
    # Combine all features at once to avoid DataFrame fragmentation
    # ----------------------------------------------------------------
    result = pd.DataFrame(feature_dict, index=df.index)
    
    # Ensure no columns are missing from the original DataFrame
    for col in df.columns:
        if col not in result.columns:
            result[col] = df[col]
    
    # Print diagnostics about NaN values if requested
    if print_diagnostics:
        # Count NaNs per column
        nan_counts = result.isna().sum()
        print(f"Columns with NaN values (top 10): {nan_counts[nan_counts > 0].sort_values(ascending=False).head(10)}")
        
        # Count NaNs per row (for the first and last few rows)
        nan_rows = result.isna().sum(axis=1)
        print(f"NaN counts in first 5 rows: {nan_rows.head()}")
        print(f"NaN counts in last 5 rows: {nan_rows.tail()}")
        
        # Get first row with complete data
        first_complete_row = result.dropna().index[0] if not result.dropna().empty else None
        if first_complete_row:
            position = result.index.get_loc(first_complete_row)
            print(f"First complete row is at index {position} (date: {first_complete_row})")
            print(f"Warm-up period: {position} bars out of {len(result)} total bars ({position/len(result):.1%} of data)")
    
    # Only drop NaN values if requested
    if drop_na:
        result = result.dropna()
        
        if print_diagnostics and len(result) > 0:
            print(f"After dropping NaN rows: {len(result)} rows remaining (removed {initial_row_count - len(result)} rows)")
    elif print_diagnostics:
        print(f"Keeping NaN rows as requested. DataFrame has {result.isna().sum().sum()} total NaN values.")
    
    return result

# These should match the indicators producible by market_data.py
# and desired for model training.
MASTER_FEATURE_LIST = [
    # Technical Indicators from TA library
    "sma_5", "sma_10", "sma_15", "sma_20", "sma_30", "sma_100", "sma_200",  # Simple Moving Averages
    "ema_5", "ema_10", "ema_15", "ema_20", "ema_30", "ema_50", "ema_100", "ema_200",  # Exponential Moving Averages
    "rsi_2", "rsi_7", "rsi_14", "rsi_21", "rsi_30",  # Relative Strength Index for different windows
    
    # Bollinger Bands
    "bb_mavg", "bb_upper", "bb_lower", "bb_pband", "bb_wband", 
    
    # MACD
    "macd_line", "macd_signal_line", "macd_hist", "macd_crossover_bullish", "macd_crossover_bearish",
    
    # ADX
    "adx_14", "adx_pos_di", "adx_neg_di", "adx_trend_strength", 
    
    # Volatility
    "atr_14", "normalized_volatility_14", "tr", "volatility_10day", "volatility_20day",
    
    # Volume
    "volume_sma_5", "volume_sma_10", "volume_sma_20", "volume_ratio", "obv", "cmf", "mfi", "vwap",
    "volume_std_5", "volume_std_10", "volume_std_20", "volume_min_5", "volume_min_10", "volume_min_20",
    "volume_max_5", "volume_max_10", "volume_max_20",
    
    # Momentum indicators
    "momentum_5", "momentum_10", "momentum_20", "cmo", "kama", # cmo is correct here
    
    # Moving average crossovers
    "sma_5_10_crossover", "sma_10_20_crossover", "ema_5_10_crossover", "ema_10_20_crossover",
    "golden_cross", "death_cross",
    
    # Price to moving average ratios
    "close_to_sma20_ratio", "close_to_ema20_ratio", "close_to_ema50_ratio", "ema20_to_ema50_ratio",
    "close_to_upper_bb_ratio", "close_to_lower_bb_ratio",
    
    # Candlestick patterns
    "candle_body", "candle_range", "candle_body_ratio", "candle_upper_shadow", "candle_lower_shadow",
    "bullish_candle", "bearish_candle", "doji",
    
    # RSI
    "rsi_overbought", "rsi_oversold",
    
    # Stochastic
    "stoch_k", "stoch_d", "stoch_crossover", "stoch_overbought", "stoch_oversold",
    
    # Ichimoku
    "ichimoku_a", "ichimoku_b", "ichimoku_base", "ichimoku_conversion",
    
    # Parabolic SAR
    "psar", "psar_up", "psar_down", "psar_up_indicator", "psar_down_indicator",
    
    # Aroon
    "aroon_up", "aroon_down", "aroon_indicator",
    
    # Price statistics
    "close_mean_5", "close_mean_10", "close_mean_20", "close_mean_50",
    "close_std_5", "close_std_10", "close_std_20", "close_std_50",
    "close_min_5", "close_min_10", "close_min_20", "close_min_50",
    "close_max_5", "close_max_10", "close_max_20", "close_max_50",
    
    # Lagged returns
    "return_lag_1", "return_lag_2", "return_lag_3", "return_lag_4", "return_lag_5",
    
    # Daily return
    "daily_return", "daily_return_sign",
    
    # Support/resistance breach indicators
    "breach_support_10", "breach_support_20", "breach_support_50",
    "breach_resistance_10", "breach_resistance_20", "breach_resistance_50",
]

if __name__ == "__main__":
    # For simple verification
    print(
        f"Master feature list contains {len(MASTER_FEATURE_LIST)} base indicator features."
    )
    for feature in MASTER_FEATURE_LIST:
        print(feature)
