"""
Common utilities for the AI Options Trading Agent.
This module provides shared functionality used across the application.
"""

import logging
import os
from datetime import datetime
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd


def setup_logger(name: str, level: int = logging.INFO) -> logging.Logger:
    """
    Configure and return a logger with consistent formatting.

    Args:
        name: Logger name (typically __name__ from the calling module)
        level: Logging level, defaults to INFO

    Returns:
        Configured logger instance
    """
    logger = logging.getLogger(name)

    # Only configure if handlers haven't been added yet
    if not logger.handlers:
        logger.setLevel(level)
        handler = logging.StreamHandler()
        formatter = logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)

    return logger


def safe_convert_to_numeric(series: pd.Series) -> pd.Series:
    """
    Safely convert a pandas Series to numeric format, handling errors.

    Args:
        series: Input pandas Series

    Returns:
        Numeric pandas Series
    """
    if series is None:
        return pd.Series([])

    try:
        return pd.to_numeric(series, errors="coerce").fillna(0)
    except Exception as e:
        logger = setup_logger(__name__)
        logger.warning(f"Error converting series to numeric: {e}")
        return series if isinstance(series, pd.Series) else pd.Series([])


def get_timestamp_string() -> str:
    """
    Return a formatted timestamp string for filenames.

    Returns:
        Formatted timestamp string (e.g., '20250515_123456')
    """
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def ensure_directory_exists(directory_path: str) -> None:
    """
    Ensure that the specified directory exists, create if it doesn't.

    Args:
        directory_path: Directory path to check/create
    """
    if not os.path.exists(directory_path):
        os.makedirs(directory_path, exist_ok=True)


def filter_dict(input_dict: Dict, keys_to_keep: List[str]) -> Dict:
    """
    Filter a dictionary to only keep specific keys.

    Args:
        input_dict: Input dictionary
        keys_to_keep: List of keys to keep

    Returns:
        Filtered dictionary
    """
    return {k: v for k, v in input_dict.items() if k in keys_to_keep}
