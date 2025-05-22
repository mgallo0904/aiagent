#!/usr/bin/env python3
"""
TFT-driven options backtesting implementation.
This script extends the equity backtester to trade a single-leg ATM option
(using QuantLib for pricing/Greeks), including chain-fetching,
pricing, Greek-based sizing, and P&L calculation.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import logging
import torch
import torch.nn as nn
from datetime import datetime, timedelta
from typing import List

# QuantLib imports
import QuantLib as ql

from market_data import MarketData      # for underlying price history
from features import prepare_features   # TFT features
# # from risk_management import RiskManager # Not used in this version
# # from performance_metrics import summarize_performance # Not used in this version
from lightning.pytorch import Trainer 
from pytorch_forecasting import TimeSeriesDataSet, TemporalFusionTransformer
from pytorch_forecasting.metrics import MultiHorizonMetric
from pytorch_forecasting.data import GroupNormalizer # Added for consistency

# ------------------------------------------------------
# Custom Metric for BCEWithLogitsLoss (Copied from tft_backtest.py)
# ------------------------------------------------------
class BCEWithLogitsMetric(MultiHorizonMetric):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.loss_fn = nn.BCEWithLogitsLoss()

    def loss(self, y_pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        if y_pred.ndim == target.ndim + 1 and y_pred.shape[-1] == 1:
            y_pred_squeezed = y_pred.squeeze(-1)
        else:
            y_pred_squeezed = y_pred
        loss_values = self.loss_fn(y_pred_squeezed, target.float())
        return loss_values

    def to_prediction(self, y_pred: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(y_pred)

    def to_quantiles(self, y_pred: torch.Tensor, quantiles: List[float] = None) -> torch.Tensor:
        if quantiles is None:
            quantiles = self.quantiles
        return torch.sigmoid(y_pred).expand(-1, -1, len(quantiles))

# ------------------------------------------------------
# TFT Training Function (Copied and adapted from tft_backtest.py)
# ------------------------------------------------------
def train_tft(data: pd.DataFrame,
              max_encoder_length=252,
              max_prediction_length=21,
              learning_rate=3e-3,
              hidden_size=16,
              attention_head_size=4,
              dropout=0.1,
              epochs=10,
              gpus=1, 
              time_idx_col="time_idx",
              target_col="target",
              group_id_col="group_id",
              time_varying_known_reals_cols=None,
              time_varying_unknown_reals_cols=None
              ) -> TemporalFusionTransformer:
    '''
    Builds TimeSeriesDataSet, trains a TFT, and returns the fitted model.
    Adapted for options backtester.
    '''
    if target_col not in data.columns:
        raise ValueError(f"DataFrame must contain a '{target_col}' column for TFT training.")
    if time_idx_col not in data.columns:
        raise ValueError(f"DataFrame must contain a '{time_idx_col}' column.")
    if group_id_col not in data.columns:
        raise ValueError(f"DataFrame must contain a '{group_id_col}' column.")

    data[target_col] = data[target_col].astype(float)

    # Default known reals if not provided
    if time_varying_known_reals_cols is None:
        time_varying_known_reals_cols = [time_idx_col]
    
    # Default unknown reals if not provided (all other feature columns)
    if time_varying_unknown_reals_cols is None:
        all_cols = set(data.columns)
        reserved_cols = {time_idx_col, group_id_col, target_col}
        known_reals_set = set(time_varying_known_reals_cols)
        
        # Start with all columns not in reserved or known reals
        potential_unknown_reals = list(all_cols - reserved_cols - known_reals_set)
        
        # Filter out categorical columns if any were accidentally included
        time_varying_unknown_reals_cols = [
            col for col in potential_unknown_reals 
            if data[col].dtype != 'category' and data[col].dtype.name != 'category'
        ]


    dataset = TimeSeriesDataSet(
        data,
        time_idx=time_idx_col,
        target=target_col,
        group_ids=[group_id_col],
        static_categoricals=[], # Assuming no static categoricals for now
        static_reals=[],        # Assuming no static reals for now
        time_varying_known_reals=time_varying_known_reals_cols,
        time_varying_unknown_reals=time_varying_unknown_reals_cols,
        target_normalizer=None, 
        max_encoder_length=max_encoder_length,
        max_prediction_length=max_prediction_length,
        allow_missing_timesteps=True 
    )

    if len(dataset) == 0:
        raise ValueError(
            f"TimeSeriesDataSet is empty. Data length: {len(data)}, "
            f"max_encoder_length: {max_encoder_length}, max_prediction_length: {max_prediction_length}. "
            "Ensure data is long enough to form at least one sample."
        )

    train_loader = dataset.to_dataloader(train=True, batch_size=64, num_workers=0)

    tft = TemporalFusionTransformer.from_dataset(
        dataset,
        learning_rate=learning_rate,
        hidden_size=hidden_size,
        attention_head_size=attention_head_size,
        dropout=dropout,
        loss=BCEWithLogitsMetric(), 
        log_interval=10, # Reduce logging verbosity
        reduce_on_plateau_patience=3,
    )

    trainer_kwargs = {
        "max_epochs": epochs,
        "gradient_clip_val": 0.1,
        "logger": False, 
        "enable_checkpointing": False,
    }
    if torch.backends.mps.is_available() and torch.backends.mps.is_built() and gpus > 0: # Check for MPS
        trainer_kwargs["accelerator"] = "mps"
        trainer_kwargs["devices"] = 1 # MPS typically uses 1 device
        logging.info("Using MPS (Apple Silicon GPU) for training.")
    elif torch.cuda.is_available() and gpus > 0:
        trainer_kwargs["accelerator"] = "cuda" # Corrected from "gpu" for PyTorch Lightning
        trainer_kwargs["devices"] = gpus
        logging.info(f"Using CUDA GPU(s): {gpus} for training.")
    else:
        trainer_kwargs["accelerator"] = "cpu"
        trainer_kwargs["devices"] = 1 # Keeping consistent with existing CPU path
        logging.info("Using CPU for training.")


    trainer = Trainer(**trainer_kwargs)
    trainer.fit(tft, train_loader)
    return tft

# ------------------------------------------------------
# --- OPTION PRICER & GREEKS
# ------------------------------------------------------
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

        spot_handle = ql.QuoteHandle(ql.SimpleQuote(spot))
        vol_handle  = ql.BlackVolTermStructureHandle(
            ql.BlackConstantVol(eval_date_ql, self.calendar, vol, self.day_count)
        )
        rf_handle  = ql.YieldTermStructureHandle(
            ql.FlatForward(eval_date_ql, self.rf_rate, self.day_count)
        )
        dv_handle  = ql.YieldTermStructureHandle(
            ql.FlatForward(eval_date_ql, self.div_yield, self.day_count)
        )

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
        option.recalculate()

        price = option.NPV()
        delta = option.delta()
        vega  = option.vega() / 100 # Assuming QL vega is per 100% change, convert to per 1% or ensure consistency

        if price is None or price < 0: price = 0.0
        if delta is None: delta = 0.0
        if vega is None: vega = 0.0
        
        return price, delta, vega

# ------------------------------------------------------
# --- Walk-forward backtest for options
# ------------------------------------------------------
def walkforward_options_backtest(
    df: pd.DataFrame,
    features: pd.DataFrame, 
    labels_series_full: pd.Series, 
    train_window=252, # Number of days in the training set for EACH walk-forward cycle
    test_window=21,   # Number of days in the test set for EACH walk-forward cycle (also TFT prediction length)
    risk_vega_fraction=0.1,
    rfr=0.02,
    commission_per_contract=2.0,
    ticker_symbol="STOCK",
    tft_max_encoder_length=60, 
    # tft_max_prediction_length is implicitly test_window
    tft_epochs=10 
):
    '''
    Walk-forward backtest through ATM calls/puts using TFT signals.
    Positions are sized by vega_at_risk (fraction of account vega).
    TFT is trained in each walk-forward cycle.
    '''
    dates = df.index
    if dates.empty:
        logging.error("Input DataFrame 'df' has no dates. Exiting backtest.")
        return pd.Series(dtype=float)
        
    cash = 10000.0
    equity = pd.Series(index=dates, dtype=float)
    if not dates.empty: 
        equity.iloc[0] = cash

    pricer = OptionPricer(risk_free_rate=rfr)
    
    tft_params = {
        "learning_rate": 3e-3, "hidden_size": 16, "attention_head_size": 4,
        "dropout": 0.1, "epochs": tft_epochs,
        "gpus": 1 if torch.cuda.is_available() else 0
    }
    # tft_max_prediction_length for TFT training call is distinct from test_window.
    # For TFT training, max_prediction_length is how far into the future the model *can* learn to predict.
    # For actual prediction, we predict `test_window` steps.
    # Let's set training max_prediction_length to be related to test_window.
    tft_training_max_prediction_length = test_window 


    logging.info(f"Starting TFT options backtest for {ticker_symbol} from {dates[0].strftime('%Y-%m-%d')} to {dates[-1].strftime('%Y-%m-%d')}")
    logging.info(f"TFT params: Encoder={tft_max_encoder_length}, TrainMaxPredLen={tft_training_max_prediction_length}, Epochs={tft_params['epochs']}")
    logging.info(f"Walk-forward: TrainWindow={train_window}, TestWindow={test_window}")


    for start_idx in range(0, len(dates) - train_window - test_window + 1, test_window):
        train_period_end_idx = start_idx + train_window
        test_period_start_idx = train_period_end_idx
        test_period_end_idx = test_period_start_idx + test_window

        if test_period_end_idx > len(dates): # Adjust if last window is partial
            test_period_end_idx = len(dates)
            if test_period_start_idx >= test_period_end_idx:
                logging.info("Skipping final incomplete test window.")
                continue
        
        # Data for the current walk-forward cycle
        current_train_features = features.iloc[start_idx:train_period_end_idx].copy() 
        current_train_labels = labels_series_full.iloc[start_idx:train_period_end_idx].copy()
        
        current_test_features_orig = features.iloc[test_period_start_idx:test_period_end_idx].copy()
        price_te_series = df['Close'].iloc[test_period_start_idx:test_period_end_idx]

        if current_train_features.empty or current_test_features_orig.empty or price_te_series.empty:
            logging.warning(f"Empty data for a period. Train features: {len(current_train_features)}, Test features: {len(current_test_features_orig)}. Skipping.")
            if test_period_start_idx > 0 and test_period_start_idx < len(equity):
                 equity.iloc[test_period_start_idx:test_period_end_idx] = equity.iloc[test_period_start_idx-1]
            cash = equity.iloc[test_period_end_idx-1] if test_period_end_idx <= len(equity) and test_period_end_idx > 0 else cash
            continue

        logging.info(f"Cycle: Train {dates[start_idx].strftime('%Y-%m-%d')} to {dates[train_period_end_idx-1].strftime('%Y-%m-%d')}, Test {dates[test_period_start_idx].strftime('%Y-%m-%d')} to {dates[test_period_end_idx-1].strftime('%Y-%m-%d')}")

        # --- TFT Training for this fold ---
        tft_train_data_fold = current_train_features.copy() 
        tft_train_data_fold['target'] = current_train_labels.values
        tft_train_data_fold['time_idx'] = np.arange(len(tft_train_data_fold)) 
        tft_train_data_fold['group_id'] = ticker_symbol
        
        # Identify feature columns for TFT (unknown reals)
        # Exclude 'time_idx', 'group_id', 'target'. All others are unknown reals by default.
        time_varying_unknown_reals_cols = [
            col for col in tft_train_data_fold.columns 
            if col not in {'time_idx', 'group_id', 'target'}
        ]
        # Ensure 'normalized_volatility_14' is in this list if it's a feature
        if 'normalized_volatility_14' not in time_varying_unknown_reals_cols and 'normalized_volatility_14' in tft_train_data_fold.columns:
             # This case should not happen if it's a feature and not target/id/time_idx
             pass


        if len(tft_train_data_fold) < tft_max_encoder_length + tft_training_max_prediction_length:
            logging.warning(f"Skipping fold. Not enough data for TFT training ({len(tft_train_data_fold)}) for encoder ({tft_max_encoder_length}) + train_pred_len ({tft_training_max_prediction_length}).")
            if test_period_start_idx > 0 and test_period_start_idx < len(equity):
                 equity.iloc[test_period_start_idx:test_period_end_idx] = equity.iloc[test_period_start_idx-1]
            cash = equity.iloc[test_period_end_idx-1] if test_period_end_idx <= len(equity) and test_period_end_idx > 0 else cash
            continue
        
        tft_model = None
        try:
            logging.info(f"Training TFT model for cycle. Data shape: {tft_train_data_fold.shape}")
            tft_model = train_tft(
                data=tft_train_data_fold,
                max_encoder_length=tft_max_encoder_length,
                max_prediction_length=tft_training_max_prediction_length, 
                learning_rate=tft_params["learning_rate"],
                hidden_size=tft_params["hidden_size"],
                attention_head_size=tft_params["attention_head_size"],
                dropout=tft_params["dropout"],
                epochs=tft_params["epochs"],
                gpus=tft_params["gpus"],
                time_idx_col="time_idx",
                target_col="target",
                group_id_col="group_id",
                time_varying_unknown_reals_cols=time_varying_unknown_reals_cols
            )
        except Exception as e:
            logging.error(f"Error during TFT training: {e}. Skipping fold and using random signals.")
            # Fallback to random signals if training fails
            probs = np.random.rand(len(current_test_features_orig)) 
            signals = (probs > 0.5).astype(int)
        
        # --- TFT Prediction for the test window ---
        if tft_model: # Only proceed if model training was successful
            try:
                # For prediction, TFT needs encoder data leading up to the prediction period,
                # and known future inputs for the prediction period itself.
                # Encoder data: last `tft_max_encoder_length` from the training set.
                encoder_data_for_pred = current_train_features.iloc[-tft_max_encoder_length:].copy()
                
                # Decoder data: features for the test period.
                # Ensure it has the same columns as training (excluding target, time_idx, group_id)
                decoder_data_for_pred = current_test_features_orig[time_varying_unknown_reals_cols].copy()

                # Combine for prediction input DataFrame
                # The `predict` method internally handles constructing the sequences.
                # We need to pass a DataFrame that contains data up to the end of the prediction period.
                # The model will use `max_encoder_length` before each prediction point.
                
                # The `data_for_prediction` should span from the start of necessary encoder data
                # up to the end of the decoder data (test window).
                # It needs 'time_idx' and 'group_id' and all feature columns.
                # 'target' can be dummy for the prediction part.

                # Take last `tft_max_encoder_length` of full training data (features + labels)
                # and append test features.
                
                # Data that tft_model was trained on (for dataset_parameters)
                # This is `tft_train_data_fold`

                # Data to make predictions from:
                # Needs to be a continuous series including encoder data and test period features
                # `tft_model.predict` can take a DataFrame.
                # It needs the historical data (encoder) and future known inputs (decoder).
                
                # Let's use the `TimeSeriesDataSet.from_parameters` method with the test data.
                # The `data` argument for `predict` should be a DataFrame that includes
                # the encoder sequence and the decoder sequence for which predictions are desired.
                
                # Construct the DataFrame for the `predict` call:
                # It must contain data for `tft_max_encoder_length` (history) + `test_window` (future)
                # History comes from `current_train_features`
                # Future comes from `current_test_features_orig`
                
                # History part for prediction:
                history_for_prediction = current_train_features.iloc[-tft_max_encoder_length:].copy()
                history_for_prediction['group_id'] = ticker_symbol
                # `time_idx` should be continuous if model expects it.
                # For simplicity, let `predict` handle `time_idx` generation based on its internal dataset.

                # The `tft_model.predict()` method can take a dataloader or a dataframe.
                # If dataframe, it should be the part of the data for which to make predictions.
                # It will internally find the encoder data based on `max_encoder_length`.
                
                # So, we need a dataframe that covers the test period, plus enough history for the first prediction.
                # `current_test_features_orig` is just the test period.
                # We need to give it `current_train_features` (or relevant part) + `current_test_features_orig`
                
                # Create a combined DataFrame for prediction context
                full_context_for_prediction_df = pd.concat([
                    current_train_features, # Full training history for this fold
                    current_test_features_orig # Features for the period to predict
                ]).copy()
                full_context_for_prediction_df['time_idx'] = np.arange(len(full_context_for_prediction_df))
                full_context_for_prediction_df['group_id'] = ticker_symbol
                # Add dummy target, Pytorch Forecasting might need it for dataset creation, even for prediction
                full_context_for_prediction_df['target'] = 0.0 
                # Fill actual targets for the known part (training part)
                known_targets = labels_series_full.iloc[start_idx : test_period_start_idx].values # up to end of train
                if len(known_targets) <= len(full_context_for_prediction_df):
                     full_context_for_prediction_df.iloc[:len(known_targets), full_context_for_prediction_df.columns.get_loc('target')] = known_targets


                # Predict for the test window.
                # The `data` for predict should be the `full_context_for_prediction_df`.
                # The model will pick the last `test_window` predictions.
                logging.info(f"Generating predictions for test window. Input context shape: {full_context_for_prediction_df.shape}")
                
                # Ensure all columns expected by the model are present
                for col in tft_model.dataset_parameters['time_varying_unknown_reals']:
                    if col not in full_context_for_prediction_df.columns:
                        logging.warning(f"Feature {col} expected by model not in prediction data. Adding dummy 0s.")
                        full_context_for_prediction_df[col] = 0.0
                for col in tft_model.dataset_parameters['time_varying_known_reals']:
                     if col not in full_context_for_prediction_df.columns and col != 'time_idx': # time_idx is special
                        logging.warning(f"Known real feature {col} expected by model not in prediction data. Adding dummy 0s.")
                        full_context_for_prediction_df[col] = 0.0


                raw_preds_output_all = tft_model.predict(
                    full_context_for_prediction_df,
                    mode="prediction", # Ensure it's logits
                    return_index=False # Get raw tensor output
                )

                if isinstance(raw_preds_output_all, tuple): 
                    raw_preds_tensor_all = raw_preds_output_all[0]
                else:
                    raw_preds_tensor_all = raw_preds_output_all
                
                # Predictions are for each time step in `full_context_for_prediction_df`
                # starting from `max_encoder_length`.
                # We need the predictions that correspond to `current_test_features_orig`.
                # The output tensor shape is (n_samples_in_full_context, n_quantiles_or_outputs)
                # or (n_sequences, n_timesteps_ahead, n_outputs)
                # For `mode="prediction"`, it should be (len(full_context_for_prediction_df) - max_encoder_length, 1)
                # if predicting one step ahead for each possible start.
                # Or, if it predicts a sequence: (num_sequences, max_prediction_length, 1)
                # The `predict` method with a DataFrame typically returns predictions for each row in the DataFrame
                # for which a full encoder sequence is available.

                # Let's get predictions for the last `test_window` period.
                # The output of `tft_model.predict(dataframe)` is a tensor of predictions
                # for each time step in the dataframe that can be predicted.
                # starting from the first row for which `max_encoder_length` is met.
                # We are interested in the predictions corresponding to the `test_window`.
                
                # The shape of raw_preds_tensor_all is likely (total_predictable_points, num_outputs)
                # num_outputs is 1 for our BCEWithLogitsMetric.
                # total_predictable_points = len(full_context_for_prediction_df) - tft_max_encoder_length (approximately)
                
                # We need the last `test_window` predictions from this tensor.
                if raw_preds_tensor_all.ndim > 1 and raw_preds_tensor_all.shape[-1] == 1:
                    raw_preds_flat = raw_preds_tensor_all.squeeze(-1).cpu().numpy()
                else:
                    raw_preds_flat = raw_preds_tensor_all.cpu().numpy().flatten()

                # The number of predictions should correspond to the number of rows in `full_context_for_prediction_df`
                # for which a prediction could be made.
                # `predict` method when given a dataframe, will return predictions for each row of the dataframe,
                # starting from the first row for which `max_encoder_length` is available.
                # So, the number of predictions is `len(full_context_for_prediction_df) - tft_max_encoder_length`.
                # We want the last `test_window` of these.
                
                num_predictions_made = len(raw_preds_flat)
                expected_predictions_for_test_window = len(current_test_features_orig) # which is test_window

                if num_predictions_made >= expected_predictions_for_test_window:
                    # Take the last `expected_predictions_for_test_window` predictions
                    raw_preds = raw_preds_flat[-expected_predictions_for_test_window:]
                else:
                    logging.warning(f"Not enough predictions made ({num_predictions_made}) for the test window ({expected_predictions_for_test_window}). Padding.")
                    raw_preds = np.pad(raw_preds_flat, (0, expected_predictions_for_test_window - num_predictions_made), 'constant', constant_values=0.0)


                if len(raw_preds) != len(current_test_features_orig):
                    logging.warning(f"Prediction length mismatch after selection: Got {len(raw_preds)}, expected {len(current_test_features_orig)}. Adjusting.")
                    if len(raw_preds) < len(current_test_features_orig):
                        raw_preds = np.pad(raw_preds, (0, len(current_test_features_orig) - len(raw_preds)), 'constant', constant_values=0.0)
                    else:
                        raw_preds = raw_preds[:len(current_test_features_orig)]
                
                probs = torch.sigmoid(torch.tensor(raw_preds, dtype=torch.float32)).cpu().numpy()
                signals = (probs > 0.5).astype(int)
                logging.info(f"TFT predictions generated. Probabilities (first 5): {probs[:5]}")

            except Exception as e:
                logging.error(f"Error during TFT prediction: {e}. Using random signals for this fold.")
                probs = np.random.rand(len(current_test_features_orig)) 
                signals = (probs > 0.5).astype(int)
        else: # tft_model was None due to training error
             logging.info("Using random signals as TFT model training failed.")
             # probs and signals are already set from the training exception block


        # --- Options Trading Logic ---
        current_option_position = {} 

        for i, current_date_pd in enumerate(price_te_series.index):
            current_date_dt = current_date_pd.to_pydatetime() 
            spot_price = price_te_series.iloc[i]
            signal_today = signals[i]
            
            # Prioritize 'implied_volatility', then 'normalized_volatility_14', then fallback
            current_vol = np.nan
            vol_source = "fallback"

            if 'implied_volatility' in features.columns:
                try:
                    current_vol = features.loc[current_date_pd, 'implied_volatility']
                    if not pd.isna(current_vol) and current_vol > 0:
                        vol_source = "implied_volatility"
                    else:
                        # Log if found but invalid
                        if not pd.isna(current_vol) and current_vol <= 0:
                            logging.debug(f"Found 'implied_volatility' for {current_date_pd} but it was invalid ({current_vol}). Will try fallback.")
                        current_vol = np.nan # Ensure it's NaN if invalid or not found by .loc
                except KeyError:
                    logging.debug(f"KeyError accessing 'implied_volatility' for {current_date_pd} in features index. Will try fallback.")
                    current_vol = np.nan
            
            if pd.isna(current_vol): # If implied_volatility was not found, NaN, or <=0
                if 'normalized_volatility_14' in features.columns:
                    try:
                        current_vol = features.loc[current_date_pd, 'normalized_volatility_14']
                        if not pd.isna(current_vol) and current_vol > 0:
                            vol_source = "normalized_volatility_14"
                        else:
                            if not pd.isna(current_vol) and current_vol <= 0:
                                logging.debug(f"Found 'normalized_volatility_14' for {current_date_pd} but it was invalid ({current_vol}). Will use hardcoded fallback.")
                            current_vol = np.nan
                    except KeyError:
                        logging.debug(f"KeyError accessing 'normalized_volatility_14' for {current_date_pd} in features index. Will use hardcoded fallback.")
                        current_vol = np.nan
                else: 
                    logging.debug(f"'normalized_volatility_14' column not found. Will use hardcoded fallback for {current_date_pd}.")
                    current_vol = np.nan

            if pd.isna(current_vol) or current_vol <= 0: 
                current_vol = 0.2 # Hardcoded fallback
                vol_source = "hardcoded_fallback (0.2)"
                logging.warning(f"Invalid or missing volatility for {current_date_pd}. Using {vol_source}.")
            # else: # Removed redundant logging, covered by debug log below if needed
                # logging.debug(f"Using {vol_source} ({current_vol:.4f}) for {current_date_pd} for option pricing.")


            # --- Position Management & P&L ---
            # P&L Calculation for existing position
            if current_option_position:
                held_expiry_dt = current_option_position['expiry_date']
                if isinstance(held_expiry_dt, pd.Timestamp): held_expiry_dt = held_expiry_dt.to_pydatetime()

                current_opt_price, _, _ = pricer.price_and_greeks(
                    eval_date_py=current_date_dt, spot=spot_price, strike=current_option_position['strike'],
                    vol=current_vol, expiry_date_py=held_expiry_dt, option_type=current_option_position['type']
                )
                
                # Close position logic
                # Signal changes: if today's signal is different from the option type (1 for call, 0 for put)
                signal_implies_call = (signal_today == 1)
                position_is_call = (current_option_position['type'] == 'call')
                
                if current_date_dt >= held_expiry_dt or (signal_implies_call != position_is_call):
                    pnl_on_close = (current_opt_price - current_option_position['entry_price_per_contract']) * current_option_position['quantity']
                    
                    # Add back initial cost (premium) that was subtracted at entry
                    cash += (current_option_position['entry_price_per_contract'] * current_option_position['quantity'])
                    cash += pnl_on_close # Add P&L (which is (exit_value - entry_value))
                    cash -= abs(current_option_position['quantity']) * commission_per_contract 
                    
                    logging.info(f"  {current_date_dt.strftime('%Y-%m-%d')}: Closing {current_option_position['quantity']} {current_option_position['type']} S={spot_price:.2f}, K={current_option_position['strike']:.2f}. OptPrice: {current_opt_price:.2f}. P&L on trade: {pnl_on_close:.2f}")
                    current_option_position = {}
            
            # New Trade Decision
            if not current_option_position and (signal_today == 1 or signal_today == 0): # If signal to trade and no position
                option_to_trade = 'call' if signal_today == 1 else 'put'
                strike_price = round(spot_price) 
                
                # Ensure expiry is within data range or 30 days, and at least 1 day in future
                max_days_to_expiry = (dates[-1] - current_date_dt).days - 1 if dates[-1] > current_date_dt else 30
                days_to_expiry = min(30, max_days_to_expiry)
                if days_to_expiry < 1: days_to_expiry = 1 # Ensure at least 1 day to expiry
                expiry_date_dt = current_date_dt + timedelta(days=days_to_expiry)


                entry_opt_price, _, entry_vega = pricer.price_and_greeks(
                    eval_date_py=current_date_dt, spot=spot_price, strike=strike_price,
                    vol=current_vol, expiry_date_py=expiry_date_dt, option_type=option_to_trade
                )

                quantity_to_trade = 0
                if entry_vega != 0 and entry_opt_price > 0.01: 
                    # Base risk capital on current cash.
                    target_portfolio_vega_risk = risk_vega_fraction * cash 
                    quantity_to_trade = abs(target_portfolio_vega_risk / entry_vega)
                    quantity_to_trade = round(quantity_to_trade)
                    
                    cost_of_trade_premium = quantity_to_trade * entry_opt_price
                    cost_of_trade_commission = quantity_to_trade * commission_per_contract
                    total_cost = cost_of_trade_premium + cost_of_trade_commission

                    if total_cost > cash * 0.95 : # Don't use all cash
                        # Reduce quantity if cannot afford (leave 5% cash buffer)
                        affordable_premium = cash * 0.95 - cost_of_trade_commission
                        if entry_opt_price > 0:
                             quantity_to_trade = round(max(0, affordable_premium / entry_opt_price))
                        else:
                             quantity_to_trade = 0
                else:
                    logging.warning(f"  {current_date_dt.strftime('%Y-%m-%d')}: Vega is zero ({entry_vega:.4f}) or opt price too low ({entry_opt_price:.2f}) for {option_to_trade} K={strike_price}. Skipping trade.")

                if quantity_to_trade > 0:
                    current_option_position = {
                        'type': option_to_trade, 'strike': strike_price, 'expiry_date': expiry_date_dt,
                        'entry_date_dt': current_date_dt, 'entry_price_per_contract': entry_opt_price,
                        'quantity': quantity_to_trade, 'entry_spot': spot_price,
                        'entry_vol': current_vol, 'entry_vega_per_contract': entry_vega
                    }
                    # Subtract premium and commission from cash at entry
                    cash -= (quantity_to_trade * entry_opt_price) 
                    cash -= abs(quantity_to_trade) * commission_per_contract
                    logging.info(f"  {current_date_dt.strftime('%Y-%m-%d')}: Opened {quantity_to_trade} {option_to_trade} S={spot_price:.2f}, K={strike_price:.2f}, Exp={expiry_date_dt.strftime('%Y-%m-%d')}, OptPrice={entry_opt_price:.2f}, Cost={quantity_to_trade * entry_opt_price:.2f}")
                # else: (No trade executed message removed for brevity unless it's a warning condition)


            # Update equity for the day: cash + market value of open positions
            current_position_value = 0
            if current_option_position:
                # Re-price to get current market value
                opt_val_price, _, _ = pricer.price_and_greeks(
                    eval_date_py=current_date_dt, spot=spot_price,
                    strike=current_option_position['strike'], vol=current_vol, 
                    expiry_date_py=current_option_position['expiry_date'], # Already datetime
                    option_type=current_option_position['type']
                )
                current_position_value = opt_val_price * current_option_position['quantity']
            
            if current_date_pd in equity.index:
                equity[current_date_pd] = cash + current_position_value
            else: # Should not happen if equity series is pre-created with all dates
                logging.warning(f"Date {current_date_pd} not in equity index. Equity update skipped.")

        # End of test window daily loop
        if not price_te_series.empty:
            last_equity_date_in_cycle = price_te_series.index[-1]
            if last_equity_date_in_cycle in equity.index:
                 cash = equity[last_equity_date_in_cycle] # Cash for next cycle is the final equity of this one
            else: # Fallback
                 cash = equity.iloc[test_period_end_idx-1] if test_period_end_idx > 0 and test_period_end_idx <= len(equity) else cash
                 logging.warning(f"Last date {last_equity_date_in_cycle} of cycle not in equity index. Using iloc fallback for cash.")


    return equity.ffill()

# ------------------------------------------------------
# Main
# ------------------------------------------------------
if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    
    TICKER = 'AAPL'
    md = MarketData(TICKER)
    
    # Fetch data (e.g., 2 years for enough training cycles)
    price_df_orig = md.fetch_historical(period='2y', interval='1d')
    if price_df_orig.empty:
        logging.error(f"Failed to fetch historical data for {TICKER}. Exiting.")
        exit()

    # Prepare features
    # Ensure prepare_features does not drop rows with NaNs if drop_na=False,
    # as we need to align with price_df_orig first.
    features_df_prepared = prepare_features(price_df_orig.copy(), drop_na=False, print_diagnostics=True)
    
    # Align price_df and features_df
    common_index = price_df_orig.index.intersection(features_df_prepared.index)
    price_df = price_df_orig.loc[common_index].copy()
    features_df = features_df_prepared.loc[common_index].copy()

    # NaN Handling for features (critical for TFT)
    # 1. Forward fill, then backward fill
    features_df.fillna(method='ffill', inplace=True)
    features_df.fillna(method='bfill', inplace=True)
    
    # 2. Drop columns that are still entirely NaN (if any)
    features_df.dropna(axis=1, how='all', inplace=True)
    
    # 3. For any remaining NaNs in feature columns (e.g., at the very start if bfill couldn't cover)
    #    fill with 0 or a more sophisticated imputation. For TFT, NaNs in unknown_reals are problematic.
    cols_with_nans = features_df.columns[features_df.isnull().any()].tolist()
    if cols_with_nans:
        logging.warning(f"NaNs still present in columns: {cols_with_nans} after ffill/bfill. Filling with 0.")
        for col in cols_with_nans:
            features_df[col].fillna(0, inplace=True)

    # Ensure 'normalized_volatility_14' (or other vol proxy) exists for option pricer
    # This vol is NOT for TFT directly, but for the QuantLib pricer.
    # TFT will use its own features which might include various vol measures.
    if 'normalized_volatility_14' not in features_df.columns:
        logging.warning("'normalized_volatility_14' is missing. Adding dummy column with 0.2 for pricer.")
        features_df['normalized_volatility_14'] = 0.2 

    # Convert object/category columns to numeric if possible (excluding IDs, target for TFT later)
    for col in features_df.columns:
        if features_df[col].dtype == 'object' or pd.api.types.is_categorical_dtype(features_df[col].dtype):
            try:
                features_df[col] = pd.to_numeric(features_df[col])
                logging.info(f"Converted column {col} to numeric.")
            except ValueError:
                logging.warning(f"Could not convert column {col} to numeric. If it's a feature for TFT, this might be an issue.")


    # Define labels for TFT (e.g., next day's direction: 1 if up, 0 if down/flat)
    # Labels should be aligned with the features_df and price_df
    labels_series = (price_df['Close'].pct_change().shift(-1) > 0).astype(int)
    labels_series = labels_series.loc[features_df.index] # Align with features_df index
    labels_series.fillna(0, inplace=True) # Fill last NaN label

    # Final alignment after all processing, drop any rows that might have become all NaN
    # or where labels could not be generated.
    final_common_index = price_df.index.intersection(features_df.index).intersection(labels_series.index)
    price_df = price_df.loc[final_common_index]
    features_df = features_df.loc[final_common_index]
    labels_series = labels_series.loc[final_common_index]
    
    # Drop any rows where features might still be NaN (should not happen if above fillna(0) worked)
    # features_df.dropna(inplace=True) # This could de-align again
    # price_df = price_df.loc(features_df.index]
    # labels_series = labels_series.loc(features_df.index]


    if price_df.empty or features_df.empty or len(price_df) < 100:
        logging.error(f"DataFrames are empty or too short after all cleaning. Price: {len(price_df)}, Features: {len(features_df)}. Exiting.")
        exit()
    
    logging.info(f"Final data shapes for backtest: price_df {price_df.shape}, features_df {features_df.shape}, labels_series {labels_series.shape}")
    
    # Parameters for the backtest
    # These define the walk-forward structure
    TRAIN_WINDOW_MAIN = 120  # e.g., ~6 months for the outer loop's training set
    TEST_WINDOW_MAIN = 21    # e.g., ~1 month for the outer loop's test set (also TFT prediction length)
    
    # These define TFT's internal sequence lengths
    TFT_ENCODER_HISTORY = 60 # How much past data TFT model sees for each prediction point within its training
    TFT_TRAINING_EPOCHS = 10 # Epochs for each TFT model retrain

    if len(price_df) < TRAIN_WINDOW_MAIN + TEST_WINDOW_MAIN:
        logging.error(f"Not enough data ({len(price_df)} rows) for one full walk-forward cycle (train: {TRAIN_WINDOW_MAIN}, test: {TEST_WINDOW_MAIN}). Exiting.")
        exit()
    # Check for TFT's own data needs within the training slice
    if TRAIN_WINDOW_MAIN < TFT_ENCODER_HISTORY + TEST_WINDOW_MAIN: # test_window is used as tft_training_max_prediction_length
         logging.warning(f"Main training window ({TRAIN_WINDOW_MAIN}) might be too short for TFT encoder ({TFT_ENCODER_HISTORY}) + TFT pred length ({TEST_WINDOW_MAIN}).")


    eq_curve = walkforward_options_backtest(
        df=price_df, 
        features=features_df, 
        labels_series_full=labels_series,
        train_window=TRAIN_WINDOW_MAIN, 
        test_window=TEST_WINDOW_MAIN,  
        rfr=0.01,
        commission_per_contract=1.0, # Example commission
        risk_vega_fraction=0.05,     # Target 5% of cash in vega risk
        ticker_symbol=TICKER,
        tft_max_encoder_length=TFT_ENCODER_HISTORY,
        # tft_max_prediction_length for training is implicitly test_window inside walkforward
        tft_epochs=TFT_TRAINING_EPOCHS
    )

    if not eq_curve.empty:
        plt.figure(figsize=(12, 7))
        eq_curve.plot(title=f'TFT Options Walk-Forward Equity Curve ({TICKER})')
        plt.ylabel('Portfolio Value ($)')
        plt.xlabel('Date')
        plt.grid(True)
        plt.tight_layout()
        filename = f"tft_options_backtest_results_{TICKER}_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.png"
        try:
            plt.savefig(filename)
            logging.info(f"Equity curve plotted and saved to {filename}")
        except Exception as e:
            logging.error(f"Error saving plot: {e}")
        # plt.show() # Comment out for non-GUI environments
    else:
        logging.error("Equity curve is empty. No plot generated.")

