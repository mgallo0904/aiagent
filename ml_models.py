"""
Machine Learning Models for AI Options Trading Agent.
This module handles training, evaluation, and prediction using ML models.
Optimized for MacBook Pro M2 (2022).
"""

import dataclasses
import logging
import os
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple, Union, Callable

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score, f1_score, mean_squared_error, mean_absolute_error,
    precision_score, recall_score, roc_auc_score, precision_recall_curve
)
from sklearn.model_selection import KFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.calibration import CalibratedClassifierCV

import optuna

# Model imports
from xgboost import XGBClassifier, XGBRegressor
import lightgbm as lgb
from catboost import CatBoostClassifier, CatBoostRegressor
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.svm import SVC, SVR
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.neural_network import MLPClassifier, MLPRegressor

# Local imports
from .utils import ensure_directory_exists, get_timestamp_string, get_time_series_cv_splits # Removed setup_logger as logger is configured per module
from .features import MASTER_FEATURE_LIST
from .config import ModelConfig, ModelBehaviorConfig, ModelBehaviorType
from backtest import run_backtest  # Use the unified backtest engine

logger = logging.getLogger(__name__)

class MLModel:
    """
    Machine learning model class for options trading prediction.
    Handles training, evaluation, prediction, and model persistence.
    Uses scikit-learn pipelines and Optuna for hyperparameter optimization.
    """

    def __init__(
        self,
        model_type: str,
        config: ModelConfig,
        behavior_config: ModelBehaviorConfig,
        params: Dict[str, Any] = None, # For base model parameters before HPO
    ):
        self.model_type = model_type
        self.config = config
        self.behavior_config = behavior_config
        self.initial_params = params if params is not None else {}
        self.best_params = None # Store best params after HPO
        self.model_instance = self._init_model_instance() # Raw model instance
        self.pipeline = self._build_pipeline(self.model_instance)
        
        self.label_encoder = None
        if self.behavior_config.behavior_type in [
            ModelBehaviorType.CLASSIFICATION_BINARY,
            ModelBehaviorType.CLASSIFICATION_TERNARY,
        ]:
            self.label_encoder = LabelEncoder()
        
        self.trained_feature_names_ = []
        # self.scaled_training_data_for_shap_ = None # SHAP not used for now

    def _init_model_instance(self) -> Any:
        """Initialize the underlying ML model instance with initial_params."""
        # Use random_seed from ModelConfig for all models
        common_params = {'random_state': self.config.random_seed}
        merged_params = {**common_params, **self.initial_params}

        # XGBoost specific handling for use_label_encoder if it's in params
        if self.model_type == 'xgboost' and self.behavior_config.behavior_type != ModelBehaviorType.REGRESSION:
            merged_params.setdefault('use_label_encoder', False) # For XGBoost > 1.3
            merged_params.setdefault('eval_metric', 'logloss' if self.behavior_config.behavior_type == ModelBehaviorType.CLASSIFICATION_BINARY else 'mlogloss')


        logger.info(f"Initializing model: {self.model_type} with base params: {merged_params}")

        if self.model_type == 'xgboost':
            if self.behavior_config.behavior_type == ModelBehaviorType.REGRESSION:
                return XGBRegressor(**merged_params)
            return XGBClassifier(**merged_params)
        elif self.model_type == 'lightgbm':
            # LightGBM specific: n_jobs, verbosity
            lgbm_specific_params = {'n_jobs': -1, 'verbose': -1}
            final_params = {**merged_params, **lgbm_specific_params}
            if self.behavior_config.behavior_type == ModelBehaviorType.REGRESSION:
                return lgb.LGBMRegressor(**final_params)
            return lgb.LGBMClassifier(**final_params)
        elif self.model_type == 'catboost':
            # CatBoost specific: verbose
            catboost_specific_params = {'verbose': 0}
            final_params = {**merged_params, **catboost_specific_params}
            if self.behavior_config.behavior_type == ModelBehaviorType.REGRESSION:
                return CatBoostRegressor(**final_params)
            return CatBoostClassifier(**final_params)
        elif self.model_type == 'logistic_regression':
            if self.behavior_config.behavior_type == ModelBehaviorType.REGRESSION:
                logger.warning("Logistic Regression for regression task. Using LinearRegression.")
                return LinearRegression(n_jobs=-1) # n_jobs for LinearRegression
            return LogisticRegression(solver='liblinear', **merged_params)
        elif self.model_type == 'svm':
            if self.behavior_config.behavior_type == ModelBehaviorType.REGRESSION:
                return SVR() # Params for SVR can be added to merged_params
            return SVC(probability=True, **merged_params) # probability for predict_proba
        elif self.model_type == 'random_forest':
            rf_specific_params = {'n_jobs': -1}
            final_params = {**merged_params, **rf_specific_params}
            if self.behavior_config.behavior_type == ModelBehaviorType.REGRESSION:
                return RandomForestRegressor(**final_params)
            return RandomForestClassifier(**final_params)
        elif self.model_type == 'mlp':
            mlp_specific_params = {'max_iter': 500} # Default max_iter
            final_params = {**merged_params, **mlp_specific_params}
            if self.behavior_config.behavior_type == ModelBehaviorType.REGRESSION:
                return MLPRegressor(**final_params)
            return MLPClassifier(**final_params)
        else:
            logger.error(f"Unsupported model type: {self.model_type}")
            raise ValueError(f"Unsupported model type: {self.model_type}")

    def _build_pipeline(self, model_instance: Any) -> Pipeline:
        """Build scikit-learn pipeline with scaler and model."""
        steps = [('scaler', StandardScaler())]
        if self.behavior_config.behavior_type == ModelBehaviorType.REGRESSION:
            steps.append(('regressor', model_instance))
        else:
            steps.append(('classifier', model_instance))
        return Pipeline(steps)

    def _optuna_objective_factory(self, model_name: str):
        """Factory to create Optuna objective functions for different models."""
        def objective(trial: optuna.Trial, X: pd.DataFrame, y: pd.Series) -> float:
            # Common CV setup
            cv = KFold(n_splits=self.config.cv_folds, shuffle=True, random_state=self.config.random_seed)
            scores = []

            if model_name == 'xgboost':
                param_grid = {
                    'n_estimators': trial.suggest_int('n_estimators', 50, 400, step=50),
                    'learning_rate': trial.suggest_float('learning_rate', 0.005, 0.2, log=True),
                    'max_depth': trial.suggest_int('max_depth', 3, 10),
                    'subsample': trial.suggest_float('subsample', 0.6, 1.0),
                    'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
                    'gamma': trial.suggest_float('gamma', 0, 0.5),
                    'reg_alpha': trial.suggest_float('reg_alpha', 1e-3, 1.0, log=True),
                    'reg_lambda': trial.suggest_float('reg_lambda', 1e-3, 1.0, log=True),
                    'random_state': self.config.random_seed,
                    'n_jobs': -1
                }
                if self.behavior_config.behavior_type != ModelBehaviorType.REGRESSION:
                    param_grid['use_label_encoder'] = False
                    param_grid['eval_metric'] = 'logloss' if self.behavior_config.behavior_type == ModelBehaviorType.CLASSIFICATION_BINARY else 'mlogloss'
                
                model_class = XGBRegressor if self.behavior_config.behavior_type == ModelBehaviorType.REGRESSION else XGBClassifier
            
            elif model_name == 'lightgbm':
                param_grid = {
                    'n_estimators': trial.suggest_int('n_estimators', 50, 400, step=50),
                    'learning_rate': trial.suggest_float('learning_rate', 0.005, 0.2, log=True),
                    'num_leaves': trial.suggest_int('num_leaves', 20, 150),
                    'max_depth': trial.suggest_int('max_depth', 3, 12),
                    'subsample': trial.suggest_float('subsample', 0.6, 1.0),
                    'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
                    'reg_alpha': trial.suggest_float('reg_alpha', 1e-3, 1.0, log=True),
                    'reg_lambda': trial.suggest_float('reg_lambda', 1e-3, 1.0, log=True),
                    'random_state': self.config.random_seed, 'n_jobs': -1, 'verbose': -1
                }
                model_class = lgb.LGBMRegressor if self.behavior_config.behavior_type == ModelBehaviorType.REGRESSION else lgb.LGBMClassifier

            elif model_name == 'catboost':
                param_grid = {
                    'iterations': trial.suggest_int('iterations', 50, 400, step=50),
                    'learning_rate': trial.suggest_float('learning_rate', 0.005, 0.2, log=True),
                    'depth': trial.suggest_int('depth', 3, 10),
                    'l2_leaf_reg': trial.suggest_float('l2_leaf_reg', 1e-3, 10.0, log=True),
                    'random_state': self.config.random_seed, 'verbose': 0
                }
                model_class = CatBoostRegressor if self.behavior_config.behavior_type == ModelBehaviorType.REGRESSION else CatBoostClassifier
            else:
                logger.warning(f"Optuna objective not defined for {model_name}. Skipping HPO for this model in trial.")
                # Fallback: evaluate default model or raise error
                # For now, let's assume this path means we don't run HPO for this specific model type
                # and it should have been caught before calling _fit_with_search with an unsupported model.
                # Or, the _fit_with_search should handle this.
                # Returning a very bad score to make Optuna avoid this.
                return -float('inf') if self.config.optuna_study_direction == 'maximize' else float('inf')


            for train_idx, val_idx in cv.split(X, y):
                X_train_fold, X_val_fold = X.iloc[train_idx], X.iloc[val_idx]
                y_train_fold, y_val_fold = y.iloc[train_idx], y.iloc[val_idx]

                # Pipeline for each fold: scaler + model with trial params
                fold_model_instance = model_class(**param_grid)
                fold_pipeline = Pipeline([
                    ('scaler', StandardScaler()),
                    ('model', fold_model_instance)
                ])
                
                fold_pipeline.fit(X_train_fold, y_train_fold)
                
                if self.behavior_config.behavior_type == ModelBehaviorType.REGRESSION:
                    preds = fold_pipeline.predict(X_val_fold)
                    # Example: RMSE. Optuna direction should be 'minimize' for RMSE.
                    score = mean_squared_error(y_val_fold, preds, squared=False) 
                else: # Classification
                    if self.behavior_config.behavior_type == ModelBehaviorType.CLASSIFICATION_BINARY and hasattr(fold_pipeline, "predict_proba"):
                        preds_proba = fold_pipeline.predict_proba(X_val_fold)[:, 1]
                        # Example: ROC AUC. Optuna direction 'maximize'.
                        try:
                            score = roc_auc_score(y_val_fold, preds_proba)
                        except ValueError: # Handle cases with only one class in y_val_fold
                            score = 0.5 # Neutral score
                    else: # Multiclass or models without predict_proba for binary
                        preds = fold_pipeline.predict(X_val_fold)
                        # Example: F1 weighted. Optuna direction 'maximize'.
                        score = f1_score(y_val_fold, preds, average='weighted', zero_division=0)
                scores.append(score)
            
            return np.mean(scores)

        return objective

    def _fit_with_search(self, X: pd.DataFrame, y: pd.Series):
        """Fits the model using Optuna hyperparameter search."""
        logger.info(f"Starting Optuna hyperparameter search for {self.model_type} ({self.config.hyperparameter_search_budget} trials).")

        objective_func = self._optuna_objective_factory(self.model_type)
        
        if not objective_func: # Should be caught by factory returning bad score
            logger.warning(f"Optuna objective not available for {self.model_type}. Fitting with initial parameters.")
            self.pipeline.fit(X, y)
            self.best_params = self.initial_params
            return

        study = optuna.create_study(direction=self.config.optuna_study_direction)
        study.optimize(lambda trial: objective_func(trial, X, y), 
                       n_trials=self.config.hyperparameter_search_budget,
                       callbacks=self.config.optuna_callbacks)

        self.best_params = study.best_params
        logger.info(f"Best Optuna params for {self.model_type}: {self.best_params}")

        # Update the main model instance and pipeline with best params and refit on full data
        self.model_instance.set_params(**self.best_params)
        # self.pipeline already has the model_instance reference, so it's updated.
        self.pipeline.fit(X, y)
        logger.info(f"Model {self.model_type} refitted with best Optuna parameters.")

    def train(self, X: pd.DataFrame, y: pd.Series, test_size: float = 0.2):
        """
        Train the model on a training set and evaluate on a test set.
        Data is split chronologically (no shuffle).
        """
        if not isinstance(X, pd.DataFrame) or not isinstance(y, pd.Series):
            logger.error("X must be a pandas DataFrame and y a pandas Series.")
            return self

        if X.empty or y.empty:
            logger.error("Input data X or y is empty. Skipping training.")
            return self

        # Split data into training and testing sets (chronological split)
        # Ensure test_size is valid
        if not (0 < test_size < 1):
            logger.warning(f"Invalid test_size {test_size}. Defaulting to 0.2.")
            test_size = 0.2
        
        # Calculate split index for chronological split
        split_idx = int(len(X) * (1 - test_size))
        
        if split_idx <= 0 or split_idx >= len(X):
            logger.error(f"Cannot create a valid train/test split with test_size {test_size} for data of length {len(X)}. Training on full dataset instead.")
            X_train, y_train = X, y
            X_test, y_test = X.iloc[0:0], y.iloc[0:0] # Empty test set
        else:
            X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
            y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]

        logger.info(f"Data split: {len(X_train)} training samples, {len(X_test)} testing samples.")

        if hasattr(X_train, 'columns'):
            self.trained_feature_names_ = list(X_train.columns)
            logger.info(f"Training with features: {self.trained_feature_names_}")

        y_train_processed = y_train
        if self.label_encoder and self.behavior_config.behavior_type != ModelBehaviorType.REGRESSION:
            y_train_processed = self.label_encoder.fit_transform(y_train) # Fit encoder ONLY on training data
            logger.info(f"LabelEncoder classes: {self.label_encoder.classes_} for target: {self.config.target_variable_name}")

        if self.config.use_hyperparameter_search and self.config.hyperparameter_search_budget > 0:
            self._fit_with_search(X_train, y_train_processed)
        else:
            logger.info(f"Fitting {self.model_type} with initial parameters (no hyperparameter search).")
            self.pipeline.fit(X_train, y_train_processed)
        self.best_params = self.initial_params
        
        # --- Probability Calibration Step ---
        if self.behavior_config.behavior_type != ModelBehaviorType.REGRESSION and self.config.use_probability_calibration:
            logger.info(f"Applying probability calibration for {self.model_type}.")
            # Extract the base model from the pipeline
            # The pipeline is currently: [('scaler', scaler_instance), ('classifier', model_instance)]
            # We need to calibrate the 'classifier' (which is self.model_instance)
            # self.model_instance should already be configured with best_params if HPO was run,
            # or initial_params otherwise.
            
            # Note: _fit_with_search already calls pipeline.fit() on X_train, y_train_processed.
            # If HPO is used, the model_instance inside the pipeline is already trained.
            # CalibratedClassifierCV will refit this base estimator on its internal CV folds.
            
            base_classifier_for_calibration = self.pipeline.named_steps['classifier']
            
            # Some models like SVM need probability=True. Ensured during _init_model_instance for SVC.
            # For CatBoost, if it's part of a pipeline, CalibratedClassifierCV might have issues
            # if the base estimator isn't a standard scikit-learn one.
            # However, CatBoostClassifier has its own calibration options too.
            # For now, proceeding with general CalibratedClassifierCV wrapper.

            calibration_method = self.config.calibration_method # e.g., 'isotonic' or 'sigmoid'
            calibration_cv_folds = self.config.calibration_cv_folds # e.g., 3 or 5

            calibrated_model = CalibratedClassifierCV(
                base_classifier_for_calibration,
                method=calibration_method,
                cv=calibration_cv_folds
                # ensemble=True, # For sklearn >= 1.4, default is True. For older, cv implies ensemble.
            )
            
            # Replace the classifier in the pipeline with the calibrated one
            # self.pipeline.steps[-1] = ('classifier', calibrated_model) # Less robust
            self.pipeline.set_params(classifier=calibrated_model) # More robust
            
            logger.info(f"Fitting pipeline with {self.model_type} wrapped by CalibratedClassifierCV (method: {calibration_method}, cv: {calibration_cv_folds}).")
            # Fit the pipeline again, this time CalibratedClassifierCV will handle its process
            self.pipeline.fit(X_train, y_train_processed)
            logger.info("Probability calibration completed.")
        else:
            logger.info(f"{self.model_type} training completed on the training set (no calibration applied or not applicable).")

        # Evaluate on the test set
        if not X_test.empty:
            logger.info("Evaluating model performance on the test set...")
            test_metrics = self.evaluate(X_test, y_test) # y_test is raw, evaluate handles encoding
            logger.info(f"Test Set Performance Metrics for {self.model_type}:")
            for metric_name, value in test_metrics.items():
                if isinstance(value, float):
                    logger.info(f"  {metric_name}: {value:.4f}")
                else:
                    logger.info(f"  {metric_name}: {value}")
        else:
            logger.info("Test set is empty. Skipping evaluation on test set.")
            
        return self

    def predict(self, X_test: pd.DataFrame) -> np.ndarray:
        """Generate predictions. For classifiers, returns encoded labels."""
        if not self.trained_feature_names_:
             logger.warning("Model predicting without stored trained_feature_names_.")
        elif list(X_test.columns) != self.trained_feature_names_:
            logger.warning("Feature mismatch between training and testing. Ensure order and names are consistent.")
            # Consider reordering X_test columns to match self.trained_feature_names_ if robust handling is needed.
            # X_test = X_test[self.trained_feature_names_] 
        return self.pipeline.predict(X_test)

    def predict_proba(self, X_test: pd.DataFrame) -> Optional[np.ndarray]:
        """Generate probability predictions."""
        if self.behavior_config.behavior_type == ModelBehaviorType.REGRESSION:
            logger.warning("predict_proba called for a regression model. Returning None.")
            return None
        
        if hasattr(self.pipeline, "predict_proba"):
            return self.pipeline.predict_proba(X_test)
        else:
            logger.warning(f"{self.model_type} via pipeline does not support predict_proba.")
            return None

    def evaluate(self, X_test: pd.DataFrame, y_test_raw: pd.Series) -> Dict[str, float]:
        """Evaluate the model on test data."""
        y_pred_encoded = self.predict(X_test) # Predictions are already encoded by the model
        
        y_test_encoded = y_test_raw
        if self.label_encoder and self.behavior_config.behavior_type != ModelBehaviorType.REGRESSION:
            try:
                y_test_encoded = self.label_encoder.transform(y_test_raw)
            except ValueError as e:
                logger.error(f"Error transforming y_test_raw with label encoder: {e}. Unique values in y_test_raw: {y_test_raw.unique()}")
                # Fallback or re-raise, for now, try to proceed if possible or return empty metrics
                return {"error": f"Label encoding failed for test set: {e}"}


        metrics = {}
        if self.behavior_config.behavior_type == ModelBehaviorType.REGRESSION:
            metrics['mse'] = mean_squared_error(y_test_encoded, y_pred_encoded)
            metrics['mae'] = mean_absolute_error(y_test_encoded, y_pred_encoded)
            metrics['rmse'] = np.sqrt(metrics['mse'])
            logger.info(f"Regression Metrics - RMSE: {metrics['rmse']:.4f}, MAE: {metrics['mae']:.4f}")
        else: # Classification
            metrics['accuracy'] = accuracy_score(y_test_encoded, y_pred_encoded)
            avg_method = 'binary' if self.behavior_config.behavior_type == ModelBehaviorType.CLASSIFICATION_BINARY else 'weighted'
            if self.behavior_config.behavior_type == ModelBehaviorType.CLASSIFICATION_BINARY and self.label_encoder and len(self.label_encoder.classes_)==2:
                 # Ensure pos_label is correctly identified for binary precision/recall/f1
                 # Assuming the positive class is the one with label 1 after encoding
                 try:
                     pos_label_index = list(self.label_encoder.classes_).index(self.behavior_config.positive_class_label)
                     pos_label_encoded = self.label_encoder.transform([self.behavior_config.positive_class_label])[0]
                 except (ValueError, IndexError):
                     logger.warning(f"Positive class label '{self.behavior_config.positive_class_label}' not found in encoder classes. Defaulting to 1 for pos_label in metrics.")
                     pos_label_encoded = 1 # Default if not found or binary without explicit positive label set
            else: # For weighted average in multiclass, or if binary setup is ambiguous
                pos_label_encoded = 1 # Default, 'binary' average will use this if applicable

            metrics['precision'] = precision_score(y_test_encoded, y_pred_encoded, average=avg_method, pos_label=pos_label_encoded if avg_method=='binary' else None, zero_division=0)
            metrics['recall'] = recall_score(y_test_encoded, y_pred_encoded, average=avg_method, pos_label=pos_label_encoded if avg_method=='binary' else None, zero_division=0)
            metrics['f1_score'] = f1_score(y_test_encoded, y_pred_encoded, average=avg_method, pos_label=pos_label_encoded if avg_method=='binary' else None, zero_division=0)
            logger.info(f"Classification Metrics - Accuracy: {metrics['accuracy']:.4f}, F1 ({avg_method}): {metrics['f1_score']:.4f}")

            y_pred_proba = self.predict_proba(X_test)
            if y_pred_proba is not None:
                if self.behavior_config.behavior_type == ModelBehaviorType.CLASSIFICATION_BINARY and y_pred_proba.shape[1] >= 2:
                    try:
                        # Assuming positive class probability is in the column corresponding to its encoded label
                        # If label_encoder.classes_ = [neg_label, pos_label], then proba for pos_label is proba[:,1]
                        # Find index of positive class
                        idx_pos = list(self.label_encoder.classes_).index(self.behavior_config.positive_class_label)
                        metrics['roc_auc'] = roc_auc_score(y_test_encoded, y_pred_proba[:, idx_pos])
                        logger.info(f"ROC AUC: {metrics['roc_auc']:.4f}")
                    except ValueError as e:
                        logger.warning(f"Could not compute ROC AUC for binary: {e}. y_test_encoded unique: {np.unique(y_test_encoded)}. Proba shape: {y_pred_proba.shape}")
                elif self.behavior_config.behavior_type == ModelBehaviorType.CLASSIFICATION_TERNARY and y_pred_proba.shape[1] == len(self.label_encoder.classes_):
                    try:
                        metrics['roc_auc_ovr'] = roc_auc_score(y_test_encoded, y_pred_proba, multi_class='ovr', average='weighted', labels=self.label_encoder.transform(self.label_encoder.classes_))
                        logger.info(f"ROC AUC (OVR weighted): {metrics['roc_auc_ovr']:.4f}")
                    except ValueError as e:
                        logger.warning(f"Could not compute ROC AUC for ternary: {e}. y_test_encoded unique: {np.unique(y_test_encoded)}. Proba shape: {y_pred_proba.shape}")
        return metrics

    def find_optimal_threshold(self, X_val: pd.DataFrame, y_val_raw: pd.Series) -> float:
        """Find optimal probability threshold for binary classification using precision-recall curve (maximizes F1)."""
        if self.behavior_config.behavior_type != ModelBehaviorType.CLASSIFICATION_BINARY:
            logger.warning("Optimal threshold finding is typically for binary classification. Using default 0.5.")
            return 0.5

        y_val_encoded = y_val_raw
        if self.label_encoder:
            y_val_encoded = self.label_encoder.transform(y_val_raw)

        y_proba = self.predict_proba(X_val)
        if y_proba is None or y_proba.ndim == 1: # Ensure y_proba is 2D for [:, idx_pos]
             # If predict_proba returns 1D array (e.g. from CalibratedClassifierCV or some models)
             # and it's for the positive class, this is fine. Otherwise, need to adjust.
             # For now, assume if 1D, it's proba of positive class.
             pass # Keep as is
        elif y_proba.ndim == 2 and y_proba.shape[1] >=2 :
            try:
                idx_pos = list(self.label_encoder.classes_).index(self.behavior_config.positive_class_label)
                y_proba = y_proba[:, idx_pos]
            except (ValueError, AttributeError) as e:
                logger.warning(f"Could not get positive class probabilities for threshold finding: {e}. Defaulting to column 1.")
                y_proba = y_proba[:, 1] # Default assumption
        else:
            logger.warning("Probability predictions not suitable for threshold finding. Using default 0.5.")
            return 0.5
            
        precision, recall, thresholds = precision_recall_curve(y_val_encoded, y_proba)
        f1_scores = 2 * (precision * recall) / (precision + recall + 1e-10) # Add epsilon to avoid division by zero
        
        # Handle cases where thresholds might not align perfectly with f1_scores (e.g. precision_recall_curve behavior)
        # thresholds array can be one element shorter than precision and recall.
        # We want the threshold that corresponds to the max F1.
        # If len(thresholds) == len(f1_scores) -1, then f1_scores[0] is for threshold=0 (approx) and f1_scores[-1] for threshold=1 (approx)
        # and thresholds[i] is the threshold for f1_scores[i+1]
        
        # A common approach:
        # Skip the last precision/recall pair as it corresponds to threshold 1 and recall 0.
        if len(f1_scores) > len(thresholds):
             f1_scores = f1_scores[:-1]

        optimal_idx = np.argmax(f1_scores)
        optimal_threshold = thresholds[optimal_idx]
        
        # Update the behavior_config if this threshold is to be used globally for this model instance
        # self.behavior_config.proba_threshold_binary = optimal_threshold # If ModelBehaviorConfig is updated
        logger.info(f"Optimal threshold found: {optimal_threshold:.4f} (F1: {f1_scores[optimal_idx]:.4f})")
        return optimal_threshold

    # --- Persistence Methods ---
    def save_model(self, path: str = "./models/latest_model_pipeline.joblib") -> bool:
        try:
            ensure_directory_exists(os.path.dirname(path))
            model_data_to_save = {
                'pipeline': self.pipeline,
                'model_type': self.model_type,
                'config': self.config, # Save ModelConfig
                'behavior_config': self.behavior_config, # Save ModelBehaviorConfig
                'initial_params': self.initial_params,
                'best_params': self.best_params,
                'label_encoder': self.label_encoder,
                'trained_feature_names': self.trained_feature_names_,
                'timestamp': datetime.now().isoformat(),
            }
            joblib.dump(model_data_to_save, path)
            logger.info(f"Model pipeline and metadata saved to {path}")
            return True
        except Exception as e:
            logger.error(f"Error saving model: {str(e)}", exc_info=True)
            return False

    @classmethod
    def load_model(cls, path: str) -> Optional["MLModel"]:
        try:
            loaded_data = joblib.load(path)
            if not isinstance(loaded_data, dict) or 'pipeline' not in loaded_data:
                 logger.error(f"Loaded file {path} is not a valid model save format (expected dict with 'pipeline').")
                 # Try to load as old format (just the pipeline) for backward compatibility
                 pipeline_obj = joblib.load(path) # This might be the raw pipeline
                 if isinstance(pipeline_obj, Pipeline):
                     logger.warning("Loading model from an old format (raw pipeline). Metadata will be missing.")
                     # Cannot fully reconstruct MLModel instance without metadata.
                     # This path needs careful consideration on how to handle.
                     # For now, returning None as full state cannot be restored.
                     logger.error("Cannot fully reconstruct MLModel from old format. Please re-save model with new format.")
                     return None
                 return None


            # Reconstruct MLModel instance
            # Need to pass dummy config if not saved, or make them part of saved object
            instance = cls(
                model_type=loaded_data['model_type'],
                config=loaded_data['config'],
                behavior_config=loaded_data['behavior_config'],
                params=loaded_data.get('initial_params') 
            )
            instance.pipeline = loaded_data['pipeline']
            instance.best_params = loaded_data.get('best_params')
            instance.label_encoder = loaded_data.get('label_encoder')
            instance.trained_feature_names_ = loaded_data.get('trained_feature_names_', [])
            
            logger.info(f"Model pipeline and metadata loaded from {path}")
            return instance
        except Exception as e:
            logger.error(f"Error loading model: {str(e)}", exc_info=True)
            return None

def perform_walk_forward_backtest(
    price_df: pd.DataFrame,  # DataFrame with at least a 'close' column (or as specified in model_config.price_column_name)
    feature_df: pd.DataFrame,
    target_series: pd.Series,
    model_config: ModelConfig,
    model_behavior_config: ModelBehaviorConfig,
    initial_capital: float = 100000.0,
    transaction_cost_pct: float = 0.001,
    master_feature_list: List[str] = MASTER_FEATURE_LIST,  # Default to MASTER_FEATURE_LIST
):
    """
    Perform walk-forward backtesting on the given price and feature data.

    Args:
        price_df (pd.DataFrame): DataFrame with price data, must include a 'close' column.
        feature_df (pd.DataFrame): DataFrame with feature data.
        target_series (pd.Series): Series with target variable for model training.
        model_config (ModelConfig): Configuration object for the model.
        model_behavior_config (ModelBehaviorConfig): Behavior configuration for the model.
        initial_capital (float): Initial capital for backtesting.
        transaction_cost_pct (float): Transaction cost as a percentage.
        master_feature_list (List[str]): List of features to be used from feature_df. Defaults to MASTER_FEATURE_LIST.

    Returns:
        pd.DataFrame: DataFrame containing the equity curve over the backtest period.
    """
    logger.info("Starting walk-forward backtest.")

    # Ensure the necessary columns are in the price DataFrame
    required_price_columns = ['close']  # Add other required columns if needed
    for col in required_price_columns:
        if col not in price_df.columns:
            logger.error(f"Missing required column '{col}' in price DataFrame.")
            return pd.DataFrame()  # Return empty DataFrame on error

    # Ensure the necessary columns are in the feature DataFrame
    if master_feature_list is None:
        logger.error("master_feature_list cannot be None. Ensure features are specified.")
        return pd.DataFrame()  # Return empty DataFrame on error

    missing_features = [f for f in master_feature_list if f not in feature_df.columns]
    if missing_features:
        logger.error(f"Missing features in feature DataFrame: {missing_features}")
        return pd.DataFrame()  # Return empty DataFrame on error

    # Align price and feature data on the index (assuming datetime index)
    combined_df = price_df.join(feature_df, how='inner', rsuffix='_feature')
    if combined_df.empty:
        logger.error("No matching data between price and feature DataFrames after join.")
        return pd.DataFrame()  # Return empty DataFrame on error

    # Walk-forward setup
    n_splits = model_config.cv_folds
    fold_indices = get_time_series_cv_splits(
        feature_df,
        n_splits=model_config.cv_folds,
        train_period_length=model_config.train_length,
        test_period_length=model_config.test_length,
        gap=model_config.gap_length
    )

    equity_curves = []
    all_trades_list = []  # Collect trades from all folds

    # Walk-forward loop
    for i, (train_indices, test_indices) in enumerate(fold_indices):
        logger.info(f"  Cycle {i+1}/{n_splits} - Train indices: {train_indices}, Test indices: {test_indices}")

        # Split data
        train_df = combined_df.iloc[train_indices]
        test_df = combined_df.iloc[test_indices]

        # Feature selection
        X_train, y_train = train_df[master_feature_list], target_series.iloc[train_indices]
        X_test, y_test = test_df[master_feature_list], target_series.iloc[test_indices]

        # Model initialization
        model = MLModel(
            model_type=model_config.model_type,
            config=model_config,
            behavior_config=model_behavior_config,
            params=model_config.get_params_for_fold(i)  # Get fold-specific params if available
        )

        # Train the model
        model.fit(X_train, y_train)

        # Generate signals (1, 0, -1) for backtest: 1 for buy, -1 for sell, 0 for hold
        # Placeholder: implement your signal generation logic based on model predictions
        signals = model.predict(X_test)  # Assuming model outputs signals directly

        # Current prices for this fold
        current_prices = price_df.iloc[test_indices]

        if not signals.empty and not current_prices.empty:
            # Unified backtest engine returns a DataFrame with equity column
            bt_results = run_backtest(
                price=current_prices,
                signals=signals.to_numpy(),
                initial_capital=current_fold_capital,
                transaction_cost_pct=transaction_cost_pct
            )
            equity_curve_fold = bt_results['equity']
            trades_fold_df = pd.DataFrame()  # Detailed trades not available in unified engine

            equity_curves.append(equity_curve_fold)

            if not equity_curve_fold.empty:
                # Trades list is empty in unified engine
                current_fold_capital = equity_curve_fold.iloc[-1] # Update capital for next fold
                fold_return = (equity_curve_fold.iloc[-1] / equity_curve_fold.iloc[0] - 1) * 100 if len(equity_curve_fold) > 1 else 0
                logger.info(f"  Cycle {i+1} Backtest: Return {fold_return:.2f}%, End Capital: {current_fold_capital:.2f}")

    # Combine equity curves from all folds
    if equity_curves:
        combined_equity_curve = pd.concat(equity_curves)
        logger.info(f"Walk-forward backtest completed. Total return: {(combined_equity_curve.iloc[-1] / combined_equity_curve.iloc[0] - 1) * 100:.2f}%")
        return combined_equity_curve
    else:
        logger.warning("No equity curves to combine. Backtest may not have run successfully.")
        return pd.DataFrame()  # Return empty DataFrame if no curves to combine
