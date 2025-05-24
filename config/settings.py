# AI Options Trading Agent Configuration

# API keys and sensitive data should be stored in environment variables
# or a secure external file, not here.

# General settings
DEFAULT_SYMBOL = "AAPL"
DEFAULT_CAPITAL = 100000
DEFAULT_STRATEGY = "Call"

# Data fetch settings
DEFAULT_DATA_PERIOD = "2y"  # 2 years of historical data
DEFAULT_DATA_INTERVAL = "1d"  # Daily intervals
# Available periods: 1d, 5d, 1mo, 3mo, 6mo, 1y, 2y, 5y, 10y, ytd, max
# Available intervals: 1m, 2m, 5m, 15m, 30m, 60m, 90m, 1h, 1d, 5d, 1wk, 1mo, 3mo

# ML Model settings
DEFAULT_MODEL_TYPE = "xgboost" # e.g., "xgboost", "lightgbm", "catboost"
# Parameters for the *base* ML model (before HPO or calibration)
DEFAULT_MODEL_PARAMS = {
    "n_estimators": 150,
    "max_depth": 4,
    "learning_rate": 0.05,
    "random_state": 42,
    "reg_alpha": 0.1,       # L1 regularization
    "reg_lambda": 1.0,      # L2 regularization
    "min_child_weight": 5,  # Prevents tiny leaves
    "gamma": 0.1,           # Min loss reduction to make a split
}
# Default probability threshold for binary classification decisions (if model outputs probabilities)
DEFAULT_PROBA_THRESHOLD = 0.6 

# --- Ternary Labeling Settings (for MarketData.prepare_features_and_labels) ---
# Horizon (in periods) for calculating future returns to determine labels
ML_LABEL_PREDICTION_HORIZON = 5 # e.g., 5 days ahead for daily data
# Factor for dynamic threshold calculation (threshold = factor * rolling_std_of_returns)
ML_LABEL_DYNAMIC_THRESHOLD_FACTOR = 0.75 

# --- Advanced ML Settings ---
USE_TIME_SERIES_CV = True # Whether to use time-series aware cross-validation in HPO
TIME_SERIES_SPLITS = 5    # Number of splits for time-series CV

# Probability Calibration Settings (for MLModel)
CALIBRATE_PROBABILITIES = True  # Enable/disable probability calibration for classifiers
CALIBRATION_METHOD = "isotonic" # "isotonic" or "sigmoid"
CALIBRATION_CV_FOLDS = 3        # Number of CV folds for CalibratedClassifierCV

# Ensemble methods (currently not implemented in MLModel, placeholder)
USE_ENSEMBLE = False  
ENSEMBLE_SIZE = 3     

# Hyperparameter Optimization (Optuna) Settings (for MLModel)
USE_OPTUNA = True               # Use Optuna for hyperparameter tuning
OPTUNA_N_TRIALS = 50            # Number of trials for Optuna (can be overridden in specific contexts like backtesting)
# Optuna study direction: 'minimize' (e.g. for RMSE) or 'maximize' (e.g. for ROC AUC, F1-score)
OPTUNA_STUDY_DIRECTION = "maximize" 

# --- MLTradingStrategy Settings ---
# Default DTE for options when MLTradingStrategy generates a signal
ML_STRATEGY_DEFAULT_DTE = 30 
# Default quantity for options (can be overridden by risk management)
ML_STRATEGY_DEFAULT_QUANTITY = 1 
# Mapping of ML model's ternary output (1, -1, 0) to trading actions.
# This allows flexibility if e.g., 1 should mean BUY_STOCK instead of BUY_CALL.
# For now, keeping this hardcoded in MLTradingStrategy, but could be moved here.
# ML_STRATEGY_ACTION_MAPPING = { 
#     1: "ACTION_BUY_CALL",
#     -1: "ACTION_BUY_PUT",
#     0: "ACTION_HOLD"
# }

# --- Technical Indicator Parameters (used in MarketData for feature generation) ---
# These are general defaults; specific features might use variations.
MOVING_AVERAGE_PERIODS = [20, 50, 200] # For SMA, EMA calculations
RSI_PERIOD = 14
BOLLINGER_PERIOD = 20
BOLLINGER_STD_DEV = 2
ATR_PERIOD = 14

# Risk management parameters
DEFAULT_RISK_PER_TRADE_PCT = 0.02  # 2% max risk per trade
DEFAULT_STOP_LOSS_PCT = 0.05  # 5% stop loss
DEFAULT_TAKE_PROFIT_PCT = 0.10  # 10% take profit

# Apple Silicon optimization
# Number of threads for ML operations (use fewer to avoid throttling)
ML_NUM_THREADS = 4
# Enable Apple Metal acceleration for TensorFlow models
USE_METAL_ACCELERATION = True
