"""
Configuration loader for AI Options Trading Agent.
Loads settings from config files and environment variables.
"""

import os
import sys
from typing import Any, Dict


def get_config() -> Dict[str, Any]:
    """
    Load configuration from settings.py and environment variables.
    Environment variables take precedence over configuration file settings.

    Returns:
        Dict containing all configuration settings
    """
    try:
        # Add the project root to sys.path if needed
        project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        if project_root not in sys.path:
            sys.path.append(project_root)

        # Import settings
        from config.settings import (
            DEFAULT_SYMBOL,
            DEFAULT_CAPITAL,
            DEFAULT_STRATEGY,
            DEFAULT_MODEL_TYPE,
            DEFAULT_MODEL_PARAMS,
            DEFAULT_PROBA_THRESHOLD,
            MOVING_AVERAGE_PERIODS,
            RSI_PERIOD,
            BOLLINGER_PERIOD,
            BOLLINGER_STD_DEV,
            ATR_PERIOD,
            DEFAULT_RISK_PER_TRADE_PCT,
            DEFAULT_STOP_LOSS_PCT,
            DEFAULT_TAKE_PROFIT_PCT,
            ML_NUM_THREADS,
            USE_METAL_ACCELERATION,
        )

        # Create configuration dictionary
        config = {
            "DEFAULT_SYMBOL": DEFAULT_SYMBOL,
            "DEFAULT_CAPITAL": DEFAULT_CAPITAL,
            "DEFAULT_STRATEGY": DEFAULT_STRATEGY,
            "DEFAULT_MODEL_TYPE": DEFAULT_MODEL_TYPE,
            "DEFAULT_MODEL_PARAMS": DEFAULT_MODEL_PARAMS,
            "DEFAULT_PROBA_THRESHOLD": DEFAULT_PROBA_THRESHOLD,
            "MOVING_AVERAGE_PERIODS": MOVING_AVERAGE_PERIODS,
            "RSI_PERIOD": RSI_PERIOD,
            "BOLLINGER_PERIOD": BOLLINGER_PERIOD,
            "BOLLINGER_STD_DEV": BOLLINGER_STD_DEV,
            "ATR_PERIOD": ATR_PERIOD,
            "DEFAULT_RISK_PER_TRADE_PCT": DEFAULT_RISK_PER_TRADE_PCT,
            "DEFAULT_STOP_LOSS_PCT": DEFAULT_STOP_LOSS_PCT,
            "DEFAULT_TAKE_PROFIT_PCT": DEFAULT_TAKE_PROFIT_PCT,
            "ML_NUM_THREADS": ML_NUM_THREADS,
            "USE_METAL_ACCELERATION": USE_METAL_ACCELERATION,
        }

        # Override with environment variables
        for key in config:
            env_val = os.environ.get(f"AIAGENT_{key}")
            if env_val is not None:
                # Convert the environment variable to the appropriate type
                if isinstance(config[key], bool):
                    config[key] = env_val.lower() in ("true", "yes", "1")
                elif isinstance(config[key], int):
                    config[key] = int(env_val)
                elif isinstance(config[key], float):
                    config[key] = float(env_val)
                else:
                    config[key] = env_val

        return config

    except ImportError as e:
        print(f"Error loading configuration: {e}")
        # Return default values as fallback
        return {
            "DEFAULT_SYMBOL": "AAPL",
            "DEFAULT_CAPITAL": 100000,
            "DEFAULT_STRATEGY": "Call",
            "DEFAULT_MODEL_TYPE": "xgboost",
            "DEFAULT_MODEL_PARAMS": {"n_estimators": 100, "random_state": 42},
            "DEFAULT_PROBA_THRESHOLD": 0.5,
        }
