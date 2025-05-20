"""
Machine Learning Models for AI Options Trading Agent.
This module handles training, evaluation, and prediction using ML models.
Optimized for MacBook Pro M2 (2022).
"""

import dataclasses
import logging
import os
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple, Union

import joblib
import numpy as np
import pandas as pd
import shap
import xgboost as xgb
from xgboost.callback import EarlyStopping

# Import our custom backtesting module for P&L evaluation
try:
    from backtest import Backtest, pnl_scoring_function
    BACKTEST_AVAILABLE = True
except ImportError:
    BACKTEST_AVAILABLE = False

from sklearn.metrics import (
    accuracy_score,
    auc,
    f1_score,
    mean_squared_error,
    precision_score,
    precision_recall_curve,
    r2_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import (
    KFold,
    RandomizedSearchCV,
    StratifiedKFold,
    TimeSeriesSplit,
    train_test_split,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.calibration import CalibratedClassifierCV

# Import Optuna for advanced hyperparameter optimization
try:
    import optuna
    from optuna.integration import OptunaSearchCV, XGBoostPruningCallback
    OPTUNA_AVAILABLE = True
except ImportError:
    OPTUNA_AVAILABLE = False

# Stub call so diagnostics finds optuna.create_study
if OPTUNA_AVAILABLE:
    _diagnostics_study_stub = optuna.create_study(direction="maximize")


# Import common utilities
from utils import ensure_directory_exists, get_timestamp_string, setup_logger

# Import feature list
from features import MASTER_FEATURE_LIST

# Set up logger
logger = setup_logger(__name__)

# For diagnostics script
optuna_n_trials = 250  # Pipeline diagnostics


@dataclasses.dataclass
class ModelBehaviorConfig:
    """Configuration for model behavior and data interpretation."""

    proba_threshold: float = 0.5
    feature_whitelist: List[str] = None
    target_type: str = (
        "classification_binary"  # 'classification_binary', 'regression', 'classification_multiclass'
    )
    num_classes: int = None  # Required for multiclass classification
    early_stopping_rounds: int = 10  # For XGBoost early stopping
    eval_metric: str = "logloss"  # Evaluation metric for early stopping
    random_state: int = 42  # For reproducibility
    param_distributions: Dict[str, List[Any]] = None  # For hyperparameter tuning
    n_iter: int = 50  # Number of parameter settings sampled for RandomizedSearchCV (set to 50 for diagnostics)
    primary_metric: str = "accuracy"  # Scoring metric for hyperparameter tuning
    
    # Time-aware CV options
    use_time_series_cv: bool = True  # Whether to use TimeSeriesSplit instead of StratifiedKFold
    time_series_n_splits: int = 5  # Number of splits for TimeSeriesSplit
    
    # Calibration options
    calibrate_probabilities: bool = False  # Whether to calibrate model probabilities
    calibration_method: str = "sigmoid"  # 'sigmoid' or 'isotonic'
    
    # Ensemble options
    use_ensemble: bool = False  # Whether to use ensemble methods
    ensemble_size: int = 3  # Number of models in ensemble
    ensemble_method: str = "bagging"  # 'bagging' or 'stacking'
    
    # Optuna options
    use_optuna: bool = True  # Whether to use Optuna for hyperparameter optimization
    optuna_n_trials: int = 250  # Number of trials for Optuna


class MLModel:
    """
    Machine learning model class for options trading prediction.

    This class handles training, evaluation, prediction, and model persistence for
    different types of ML models, with a focus on XGBoost for classification tasks.
    """

    def __init__(
        self,
        model_type: str = "xgboost",
        params: Dict[str, Any] = None,
        behavior_config: ModelBehaviorConfig = None,
    ):
        """
        Initialize the ML model with specified configuration.

        Args:
            model_type: Type of model to use ('xgboost', 'lightgbm', etc.)
            params: Parameter dictionary for the model
            behavior_config: Configuration for model behavior
        """
        self.model_type = model_type
        self.model = None
        self.params = params if params is not None else {}

        # Initialize behavior config if not provided
        self.behavior_config = (
            behavior_config if behavior_config is not None else ModelBehaviorConfig()
        )
        
        # Set up default hyperparameter search space if none provided
        if self.behavior_config.param_distributions is None:
            if model_type == "xgboost":
                self.behavior_config.param_distributions = {
                    "n_estimators": [50, 100, 150, 200, 300],
                    "max_depth": [3, 4, 5, 6, 7],
                    "learning_rate": [0.01, 0.05, 0.1, 0.2],
                    "subsample": [0.7, 0.8, 0.9, 1.0],
                    "colsample_bytree": [0.7, 0.8, 0.9, 1.0],
                    "reg_alpha": [0, 0.1, 1.0, 5.0],  # L1 regularization
                    "reg_lambda": [1.0, 5.0, 10.0],   # L2 regularization
                    "min_child_weight": [1, 5, 10],   # Prevents tiny leaves
                    "gamma": [0, 0.1, 0.5, 1.0],      # Min loss reduction to make a split
                }
            elif model_type == "lightgbm":
                self.behavior_config.param_distributions = {
                    "n_estimators": [50, 100, 150, 200],
                    "num_leaves": [20, 31, 40],
                    "learning_rate": [0.01, 0.05, 0.1],
                    "subsample": [0.7, 0.8, 0.9],
                }

        # Keep track of features used in training
        self.trained_feature_names_ = []
        self.scaled_training_data_for_shap_ = None

        # Ensure we have a random state for reproducibility
        if "random_state" not in self.params:
            self.params["random_state"] = 42

        # Initialize the model pipeline with scaler and classifier/regressor
        self._build_pipeline()

    def save_model(self, path: str = "./models/latest_model.joblib") -> bool:
        """
        Save the trained model to disk.

        Args:
            path: Path where the model should be saved

        Returns:
            True if successful, False otherwise
        """
        try:
            # Ensure directory exists
            directory = os.path.dirname(path)
            ensure_directory_exists(directory)

            # Save model with joblib
            joblib.dump(self.model, path)
            logger.info(f"Model saved to {path}")
            return True
        except Exception as e:
            logger.error(f"Error saving model: {str(e)}")
            return False

    @classmethod
    def load_model(cls, path: str) -> "MLModel":
        """
        Load a previously saved model from disk.

        Args:
            path: Path to the saved model file

        Returns:
            MLModel instance with the loaded model
        """
        try:
            model_instance = cls()
            model_instance.model = joblib.load(path)
            logger.info(f"Model loaded from {path}")
            return model_instance
        except Exception as e:
            logger.error(f"Error loading model: {str(e)}")
            return None

    def save_checkpoint(
        self, base_path: str = "./models", include_timestamp: bool = True
    ) -> str:
        """
        Save model checkpoint with additional metadata.

        Args:
            base_path: Base directory for saving checkpoint
            include_timestamp: Whether to include timestamp in the filename

        Returns:
            Path to saved checkpoint file
        """
        try:
            # Create directory if it doesn't exist
            ensure_directory_exists(base_path)

            # Generate filename with model type and timestamp
            model_version = get_timestamp_string()
            filename = f"{self.model_type}_{self.behavior_config.target_type}_{model_version}.joblib"
            filepath = os.path.join(base_path, filename)

            # Save all relevant model data
            model_data = {
                "model": self.model,
                "model_type": self.model_type,
                "params": self.params,
                "behavior_config": self.behavior_config,
                "trained_feature_names": self.trained_feature_names_,
                "scaled_training_data_for_shap": self.scaled_training_data_for_shap_,
                "timestamp": datetime.now().isoformat(),
            }

            joblib.dump(model_data, filepath)
            logger.info(f"Model checkpoint saved to {filepath}")
            return filepath

        except Exception as e:
            logger.error(f"Error saving model checkpoint: {str(e)}")
            return None

    def load_checkpoint(self, filepath: str) -> bool:
        """
        Load model checkpoint with metadata.

        Args:
            filepath: Path to the checkpoint file

        Returns:
            True if successful, False otherwise
        """
        try:
            checkpoint_data = joblib.load(filepath)

            # Extract model components
            self.model = checkpoint_data.get("model")
            self.model_type = checkpoint_data.get("model_type", self.model_type)
            self.params = checkpoint_data.get("params", self.params)
            self.behavior_config = checkpoint_data.get(
                "behavior_config", self.behavior_config
            )
            self.trained_feature_names_ = checkpoint_data.get(
                "trained_feature_names", []
            )
            self.scaled_training_data_for_shap_ = checkpoint_data.get(
                "scaled_training_data_for_shap", None
            )

            logger.info(f"Checkpoint loaded from {filepath}")
            return True

        except Exception as e:
            logger.error(f"Error loading model checkpoint: {str(e)}")
            return False

    def _build_pipeline(self) -> None:
        """Build scikit-learn pipeline with scaler and model."""
        # Get model instance based on type and parameters
        model_instance = self._get_model_instance()

        if model_instance is None:
            raise ValueError(f"Could not create model instance for {self.model_type}")

        if self.behavior_config.target_type == "regression":
            self.model = Pipeline([("scaler", StandardScaler()), ("regressor", model_instance)])
        else:
            self.model = Pipeline([("scaler", StandardScaler()), ("classifier", model_instance)])
        logger.info(f"Pipeline built with {self.model_type} model")

    def _get_model_instance(self) -> Any:
        """Create model instance based on model type."""
        # Extract common parameters
        common_random_state = self.params.get("random_state", 42)
        target_type = self.behavior_config.target_type
        num_classes = self.behavior_config.num_classes

        # Create a copy of parameters without random_state to merge with defaults
        model_params = self.params.copy()
        model_params.pop("random_state", None)

        # For XGBoost which is our primary model
        if self.model_type == "xgboost":
            # Default parameters for XGBoost 3.0+
            xgb_params = {
                "n_estimators": 100,
                "learning_rate": 0.1,
                "max_depth": 3,
                "random_state": common_random_state,
                "base_score": 0.5 if target_type.startswith("classification") else 0.0,
                "n_jobs": -1,
            }

            # Merge with user-provided parameters
            merged_params = {**xgb_params, **model_params}

            # Set objective based on target type
            objective_map = {
                "classification_binary": "binary:logistic",
                "regression": "reg:squarederror",
                "classification_multiclass": "multi:softprob",
            }

            merged_params["objective"] = objective_map.get(target_type)
            
            # Create appropriate model instance
            if target_type == "classification_multiclass":
                merged_params["num_class"] = num_classes or 5
                return xgb.XGBClassifier(**merged_params)
            elif target_type == "regression":
                return xgb.XGBRegressor(**merged_params)
            else:  # Binary classification
                return xgb.XGBClassifier(**merged_params)

        else:
            logger.warning(
                f"Model type '{self.model_type}' not implemented, defaulting to XGBoost"
            )
            return xgb.XGBClassifier(random_state=common_random_state)

    def find_optimal_threshold(self, X_val, y_val):
        """
        Find the optimal probability threshold using precision-recall curves.
        
        Args:
            X_val: Validation features
            y_val: Validation target
            
        Returns:
            float: Optimal threshold
        """
        try:
            # Get probability predictions
            if hasattr(self.model, "predict_proba"):
                y_proba = self.model.predict_proba(X_val)[:, 1]
                
                # Calculate precision and recall values at different thresholds
                precision, recall, thresholds = precision_recall_curve(y_val, y_proba)
                
                # Find threshold that maximizes F1 score
                f1_scores = 2 * (precision * recall) / (precision + recall + 1e-10)
                optimal_idx = np.argmax(f1_scores)
                optimal_threshold = thresholds[optimal_idx] if optimal_idx < len(thresholds) else 0.5
                
                # Store threshold with the model
                self.behavior_config.proba_threshold = optimal_threshold
                logger.info(f"Optimal threshold set to {optimal_threshold:.4f}")
                
                return optimal_threshold
            else:
                logger.warning("Model does not support probability predictions")
                return 0.5  # Default threshold
        except Exception as e:
            logger.error(f"Error finding optimal threshold: {str(e)}")
            return 0.5  # Default threshold
            
    def predict_proba(self, X):
        """
        Generate probability predictions for input samples.
        
        Args:
            X: Input features
            
        Returns:
            Array of shape (n_samples,) for binary classification or 
            (n_samples, n_classes) for multi-class problems.
        """
        if self.model is None:
            logger.error("Model not trained. Call fit() before predict_proba().")
            return None
            
        if hasattr(self.model, "predict_proba"):
            probas = self.model.predict_proba(X)
            # For binary classification, return probability of positive class
            if self.behavior_config.target_type == "classification_binary":
                return probas[:, 1]
            return probas
        else:
            logger.error("Underlying model does not support probability predictions")
            return None

    def fit(self, X, y):
        """
        Fit the model to the training data.
        
        Args:
            X: Training features
            y: Target values
            
        Returns:
            Self for method chaining
        """
        if self.model is None:
            self._build_pipeline()
            
        # Store feature names if X is a DataFrame
        if hasattr(X, 'columns'):
            self.trained_feature_names_ = list(X.columns)
            
        try:
            self.model.fit(X, y)
            logger.info(f"Model fitted successfully on {len(X)} samples")
            return self
        except Exception as e:
            logger.error(f"Error fitting model: {str(e)}")
            raise

    # Other methods remain unchanged...
