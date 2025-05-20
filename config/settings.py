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
DEFAULT_MODEL_TYPE = "xgboost"
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
DEFAULT_PROBA_THRESHOLD = 0.6

# Advanced ML settings
USE_TIME_SERIES_CV = True
TIME_SERIES_SPLITS = 5
CALIBRATE_PROBABILITIES = True
CALIBRATION_METHOD = "isotonic"  # "sigmoid" or "isotonic"
USE_ENSEMBLE = False  # Whether to use ensemble methods
ENSEMBLE_SIZE = 3     # Number of models in ensemble
USE_OPTUNA = False    # Use Optuna for hyperparameter tuning
OPTUNA_N_TRIALS = 250  # Number of trials for Optuna (set to 250 for consistency)

# Technical indicator parameters
MOVING_AVERAGE_PERIODS = [20, 50, 200]
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
