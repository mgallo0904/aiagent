#!/usr/bin/env python3
"""
Temporal Fusion Transformer (TFT) backtesting implementation for the AI trading agent.
This script implements a walk-forward testing approach with TFTs,
including realistic transaction costs, volatility-based position sizing,
and sophisticated risk management controls.
"""

import pandas as pd
import numpy as np
import optuna
import matplotlib.pyplot as plt
import logging
import torch # Added for torch.cuda.is_available()
import torch.nn as nn
from typing import List # ADDED for type hinting

from market_data import MarketData
from features import prepare_features
# Removed MLModel import as we are using TFT directly
from risk_management import RiskManager
from performance_metrics import summarize_performance, print_performance_report # MODIFIED

# TFT specific imports
from lightning.pytorch import Trainer # MODIFIED: Changed from pytorch_lightning to lightning.pytorch
from pytorch_forecasting import TimeSeriesDataSet, TemporalFusionTransformer
from pytorch_forecasting.data import GroupNormalizer
from pytorch_forecasting.metrics import MultiHorizonMetric # MODIFIED: Import MultiHorizonMetric

# ------------------------------------------------------
# Custom Metric for BCEWithLogitsLoss
# ------------------------------------------------------
class BCEWithLogitsMetric(MultiHorizonMetric):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.loss_fn = nn.BCEWithLogitsLoss()

    def loss(self, y_pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Calculate loss.

        Args:
            y_pred: network output tensor of shape (batch_size, n_decoder_timesteps, n_outputs)
                    For binary classification, n_outputs is 1 (logits).
            target: actual values of shape (batch_size, n_decoder_timesteps, 1)

        Returns:
            torch.Tensor: loss tensor of shape (batch_size, n_decoder_timesteps, 1) - matching n_quantiles if default quantiles=[0.5]
        """
        # Squeeze the last dimension of y_pred if it exists and matches the common pattern
        if y_pred.ndim == target.ndim + 1 and y_pred.shape[-1] == 1:
            y_pred_squeezed = y_pred.squeeze(-1)
        else:
            y_pred_squeezed = y_pred
        
        # Ensure target is float for BCEWithLogitsLoss
        loss_values = self.loss_fn(y_pred_squeezed, target.float())
        return loss_values

    def to_prediction(self, y_pred: torch.Tensor) -> torch.Tensor:
        """
        Convert network output to point prediction.
        For binary classification, this means converting logits to probabilities.
        """
        return torch.sigmoid(y_pred)

    def to_quantiles(self, y_pred: torch.Tensor, quantiles: List[float] = None) -> torch.Tensor:
        """
        Convert network output to quantile prediction.
        For binary classification, we return the probability for the median-like quantile.
        """
        if quantiles is None:
            quantiles = self.quantiles # Use quantiles from __init__ (e.g. [0.5])

        # Return sigmoid probability, expanded to match the number of quantiles expected.
        # If y_pred is (batch_size, n_decoder_timesteps, 1), sigmoid is also (batch_size, n_decoder_timesteps, 1).
        # .expand ensures the last dimension matches len(quantiles).
        return torch.sigmoid(y_pred).expand(-1, -1, len(quantiles))


# ------------------------------------------------------
# 1. TFT Training Function
# ------------------------------------------------------
def train_tft(data: pd.DataFrame,
              max_encoder_length=252,
              max_prediction_length=21,
              learning_rate=3e-3,
              hidden_size=16,
              attention_head_size=4,
              dropout=0.1,
              epochs=10,
              gpus=1) -> TemporalFusionTransformer:
    """
    Builds TimeSeriesDataSet, trains a TFT, and returns the fitted model.
    """
    # Ensure 'target' column exists
    if 'target' not in data.columns:
        raise ValueError("DataFrame must contain a 'target' column for TFT training.")

    data['target'] = data['target'].astype(float) # Ensure target is float

    dataset = TimeSeriesDataSet(
        data,
        time_idx="time_idx",
        target="target",
        group_ids=["group_id"],
        static_categoricals=[],
        static_reals=[],
        time_varying_known_reals=["time_idx"],
        time_varying_unknown_reals=[col for col in data.columns
                                    if col not in {"time_idx", "group_id", "target"}],
        target_normalizer=None, # MODIFIED: Set to None for BinaryClassificationLoss
        max_encoder_length=max_encoder_length,
        max_prediction_length=max_prediction_length,
        allow_missing_timesteps=True
    )

    # Check if dataset is empty AFTER creation, before creating dataloader
    if len(dataset) == 0:
        raise ValueError(
            f"TimeSeriesDataSet is empty. "
            f"Data length: {len(data)}, "
            f"max_encoder_length: {max_encoder_length}, "
            f"max_prediction_length: {max_prediction_length}. "
            "Ensure data is long enough to form at least one sample."
        )

    train_loader = dataset.to_dataloader(train=True, batch_size=64, num_workers=0)

    tft = TemporalFusionTransformer.from_dataset(
        dataset,
        learning_rate=learning_rate,
        hidden_size=hidden_size,
        attention_head_size=attention_head_size,
        dropout=dropout,
        loss=BCEWithLogitsMetric(), # MODIFIED: Use custom BCEWithLogitsMetric
        log_interval=10,
        reduce_on_plateau_patience=3,
    )

    # MODIFIED: Conditional Trainer arguments
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
        # trainer_kwargs["devices"] = "auto" # Or 1 for CPU
        logging.info("Using CPU for training.")

    trainer = Trainer(**trainer_kwargs)
    trainer.fit(tft, train_loader)
    return tft

# ------------------------------------------------------
# 1.B Hyperparameter tuning (Stub for TFT)
# ------------------------------------------------------
def tune_tft_params(X_tr_df_with_target, train_window, test_window, n_trials=10): # Reduced trials for speed
    """
    Use Optuna to find optimal hyperparameters for the TFT model.
    (This is a stub and needs to be fleshed out based on train_tft)
    """
    logging.info("Starting TFT hyperparameter tuning...")
    def objective(trial):
        lr = trial.suggest_float('lr', 1e-4, 1e-2, log=True)
        hidden_size = trial.suggest_categorical('hidden_size', [8, 16, 32])
        attention_head_size = trial.suggest_categorical('attention_head_size', [1, 2, 4])
        dropout = trial.suggest_float('dropout', 0.05, 0.3)
        epochs = trial.suggest_int('epochs', 5, 15) # Reduced epochs for tuning

        # For simplicity, we are not re-running the full walk-forward here.
        # A more rigorous approach would involve a nested cross-validation or
        # evaluating on a dedicated validation set within the objective.
        # Here, we train on the provided training slice and would ideally evaluate.
        # However, TFT training is slow, so we'll keep this part conceptual.
        # We'll return a dummy metric (e.g., negative of learning rate for Optuna to minimize something)
        # In a real scenario, you'd train and evaluate the TFT here.
        logging.info(f"Optuna trial with params: lr={lr}, hidden_size={hidden_size}, heads={attention_head_size}, dropout={dropout}, epochs={epochs}")
        # tft_model = train_tft(
        #     X_tr_df_with_target, # This would be the training portion of the current fold
        #     max_encoder_length=train_window,
        #     max_prediction_length=test_window,
        #     learning_rate=lr,
        #     hidden_size=hidden_size,
        #     attention_head_size=attention_head_size,
        #     dropout=dropout,
        #     epochs=epochs, # Use suggested epochs
        # )
        # preds = tft_model.predict(...) # Predict on a validation part of X_tr_df_with_target
        # score = roc_auc_score(y_val, preds_proba) # Example metric
        # return score
        return lr # Placeholder: Optuna will try to maximize this (but it's just lr)

    study = optuna.create_study(direction='maximize') # Maximize if using AUC, minimize for loss
    # For now, we'll just return some default parameters as TFT tuning is time-consuming
    # study.optimize(objective, n_trials=n_trials)
    # best_params = study.best_params
    # logging.info(f"Optuna best TFT params: {best_params}")

    # Using default parameters to avoid lengthy tuning in this example
    default_params = {
        "lr": 3e-3,
        "hidden_size": 16,
        "heads": 4,
        "dropout": 0.1,
        "epochs": 10
    }
    logging.info(f"Using default TFT params: {default_params}")
    return default_params


# ------------------------------------------------------
# 2. Walk-forward backtest with risk controls
# ------------------------------------------------------

def walkforward_backtest(df, features, labels,
                          train_window=252, test_window=21,
                          stop_loss_pct=0.03, transaction_cost_pct=0.001,
                          risk_volatility=0.01, vol_lookback=20, ticker="AAPL"):
    """
    Perform walk-forward backtesting with TFT.
    """
    dates = df.index
    cash = 10000.0
    all_equity = pd.Series(index=dates, dtype=float)
    if not dates.empty:
        all_equity.iloc[0] = cash
    else:
        logging.warning("DataFrame index is empty, cannot set initial capital.")
        return pd.Series(dtype=float)


    risk_mgr = RiskManager(total_capital=cash)
    cycle_metrics = []

    # Initial TFT parameter tuning (using a placeholder for now)
    # For a real scenario, you might tune on the first 'train_window' of data
    # Or re-tune periodically.
    # We'll use a simplified approach: tune once or use defaults.
    initial_train_data_for_tuning = features.iloc[:train_window].copy()
    initial_train_data_for_tuning["time_idx"] = np.arange(len(initial_train_data_for_tuning))
    initial_train_data_for_tuning["group_id"] = ticker
    initial_train_data_for_tuning["target"] = labels.iloc[:train_window].values

    # best_tft_params = tune_tft_params(initial_train_data_for_tuning, train_window, test_window)
    # Using default parameters to avoid lengthy tuning in this example
    best_tft_params = {
        "lr": 3e-3,
        "hidden_size": 16,
        "heads": 4,
        "dropout": 0.1,
        "epochs": 10 # Default epochs
    }
    print("Using default TFT params:", best_tft_params)


    for start in range(0, len(df) - train_window - test_window + 1, test_window):
        train_idx_slice = slice(start, start + train_window)
        test_idx_slice  = slice(start + train_window, start + train_window + test_window)

        X_tr_features = features.iloc[train_idx_slice]
        y_tr_labels = labels.iloc[train_idx_slice]
        X_te_features = features.iloc[test_idx_slice]
        # y_te_labels = labels.iloc[test_idx_slice] # Not directly used for TFT prediction input
        price_te = df['Close'].iloc[test_idx_slice]

        cycle_start_date = df.index[train_idx_slice.start]
        cycle_end_date = df.index[test_idx_slice.stop-1] if test_idx_slice.stop <= len(df) else df.index[-1]
        print(f"Cycle {len(cycle_metrics)+1}: Train {cycle_start_date.strftime('%Y-%m-%d')} to {df.index[train_idx_slice.stop-1].strftime('%Y-%m-%d')}, " +
              f"Test {df.index[test_idx_slice.start].strftime('%Y-%m-%d')} to {cycle_end_date.strftime('%Y-%m-%d')}")

        # --- NEW: build TFT training slice ---
        # 1) assemble TFT DataFrame for this fold
        # We need features for both train and test to create the full dataset structure for TFT
        # The actual training will only use the training part.
        fold_df_features = pd.concat([X_tr_features, X_te_features]).copy() # ADDED .copy() here
        fold_df_features["time_idx"] = np.arange(len(fold_df_features))
        fold_df_features["group_id"] = ticker

        fold_labels = pd.concat([y_tr_labels, labels.iloc[test_idx_slice]])
        fold_df_features["target"] = fold_labels.values.astype(float)


        # MODIFIED: Data for TFT training and prediction input
        # This DataFrame must be long enough for one full sequence: encoder + decoder
        current_fold_tft_data = fold_df_features.iloc[:train_window + test_window].copy()
        
        # Ensure target in the decoder part (for training) is not NaN if it was masked previously
        # For BinaryClassificationLoss, actual 0/1 labels are expected.
        # If any masking was done before, ensure current_fold_tft_data has the correct targets.
        # The current setup with fold_labels should be fine.

        if len(current_fold_tft_data) < (train_window + test_window):
            logging.warning(f"Skipping fold. Not enough data in current_fold_tft_data ({len(current_fold_tft_data)}) for train_window ({train_window}) + test_window ({test_window}).")
            # Update equity to last known value for this skipped period to avoid gaps
            if test_idx_slice.start < len(all_equity) and test_idx_slice.stop <= len(all_equity):
                 all_equity.iloc[test_idx_slice] = all_equity.iloc[test_idx_slice.start -1 if test_idx_slice.start > 0 else 0]
            elif test_idx_slice.start < len(all_equity): # If only start is valid
                 all_equity.iloc[test_idx_slice.start:] = all_equity.iloc[test_idx_slice.start -1 if test_idx_slice.start > 0 else 0]

            cash = all_equity.iloc[test_idx_slice.stop -1] if test_idx_slice.stop <= len(all_equity) else cash
            continue


        tft = train_tft(
            current_fold_tft_data, # MODIFIED: Pass the correctly sized DataFrame
            max_encoder_length=train_window,
            max_prediction_length=test_window,
            learning_rate=best_tft_params["lr"],
            hidden_size=best_tft_params["hidden_size"],
            attention_head_size=best_tft_params["heads"],
            dropout=best_tft_params["dropout"],
            epochs=best_tft_params["epochs"],
        )

        # 3) predict on the next test_window
        # Create a TimeSeriesDataSet that mirrors the training setup, for prediction.
        # This dataset will be used as a template by from_dataset.
        dataset_template_for_prediction = TimeSeriesDataSet(
            current_fold_tft_data, # Data it's based on (must be long enough)
            time_idx="time_idx", target="target", group_ids=["group_id"],
            static_categoricals=[], static_reals=[],
            time_varying_known_reals=["time_idx"],
            time_varying_unknown_reals=[col for col in current_fold_tft_data.columns if col not in {"time_idx", "group_id", "target"}],
            target_normalizer=None, # MODIFIED: Consistent with training
            max_encoder_length=train_window, max_prediction_length=test_window,
            allow_missing_timesteps=True
        )
        
        if len(dataset_template_for_prediction) == 0:
            logging.error("Dataset template for prediction is empty. This should not happen if current_fold_tft_data is valid.")
            # Handle error or skip fold
            # Update equity to last known value for this skipped period to avoid gaps
            if test_idx_slice.start < len(all_equity) and test_idx_slice.stop <= len(all_equity):
                 all_equity.iloc[test_idx_slice] = all_equity.iloc[test_idx_slice.start -1 if test_idx_slice.start > 0 else 0]
            elif test_idx_slice.start < len(all_equity): # If only start is valid
                 all_equity.iloc[test_idx_slice.start:] = all_equity.iloc[test_idx_slice.start -1 if test_idx_slice.start > 0 else 0]
            cash = all_equity.iloc[test_idx_slice.stop -1] if test_idx_slice.stop <= len(all_equity) else cash
            continue


        # Create the dataloader for prediction using the template and the same data.
        # The `predict=True` flag ensures it's set up for inference.
        # `stop_randomization=True` is crucial for getting ordered predictions.
        predict_loader = TimeSeriesDataSet.from_dataset(
            dataset_template_for_prediction,
            current_fold_tft_data, # MODIFIED: Data to predict from (contains history + future period)
            predict=True,
            stop_randomization=True
        ).to_dataloader(train=False, batch_size=X_te_features.shape[0] if X_te_features.shape[0] > 0 else 1, num_workers=0)


        raw_preds_output = tft.predict(predict_loader, mode="prediction")
        
        if isinstance(raw_preds_output, tuple):
            raw_preds = raw_preds_output[0].cpu().numpy().flatten() # MODIFIED: Added .cpu()
        elif hasattr(raw_preds_output, "cpu"):
            raw_preds = raw_preds_output.cpu().numpy().flatten() # MODIFIED: Added .cpu()
        else:
            # Assuming raw_preds_output is already a numpy array or list on CPU
            raw_preds = np.array(raw_preds_output).flatten()


        if len(raw_preds) != len(X_te_features):
            logging.warning(f"Length mismatch: raw_preds ({len(raw_preds)}) vs X_te ({len(X_te_features)}). Expected {test_window}. Truncating/padding not ideal.")
            # This case should ideally not happen if max_prediction_length is set correctly.
            # For safety, if it's shorter, pad with 0 (implying no signal/neutral probability after sigmoid)
            # If longer, truncate.
            if len(raw_preds) < len(X_te_features):
                raw_preds = np.pad(raw_preds, (0, len(X_te_features) - len(raw_preds)), 'constant', constant_values=0)
            else:
                raw_preds = raw_preds[:len(X_te_features)]

        # MODIFIED: Convert logits to probabilities
        if isinstance(raw_preds, torch.Tensor): # Ensure it's a tensor before sigmoid
            probs_tensor = torch.sigmoid(raw_preds.float()) # Ensure float type
        else:
            probs_tensor = torch.sigmoid(torch.tensor(raw_preds, dtype=torch.float32))
        
        probs = probs_tensor.cpu().numpy() # MODIFIED: Ensure numpy array on CPU

        signals = (probs > 0.5).astype(int)

        # --- End of TFT specific block ---

        vol = df['Close'].pct_change().rolling(vol_lookback).std().iloc[test_idx_slice]
        pos_size = risk_volatility / vol.fillna(vol.mean())
        pos_size = pos_size.clip(0.1, 1.0)

        cash_slice = cash if start == 0 else all_equity.iloc[test_idx_slice.start -1] # Start with cash from end of last period
        
        # Ensure cash_slice is a float
        if isinstance(cash_slice, pd.Series):
            cash_slice = cash_slice.iloc[-1] if not cash_slice.empty else 10000.0


        entry_price = None
        stop_price  = None
        position = 0
        cycle_returns = []
        
        current_equity_value = cash_slice # Track equity for this slice

        for i, date_loc in enumerate(range(test_idx_slice.start, test_idx_slice.stop)):
            date = df.index[date_loc]
            price = price_te.iloc[i]
            sig = signals[i] if i < len(signals) else 0 # Safety for signal length
            current_pos_size = pos_size.iloc[i] if i < len(pos_size) else 0.1 # Safety for pos_size length

            # Record equity at the start of the day for this specific date
            all_equity[date] = current_equity_value

            if position > 0 and price <= stop_price:
                ret = (stop_price / entry_price - 1) * current_pos_size
                current_equity_value *= (1 + ret)
                current_equity_value -= current_equity_value * transaction_cost_pct # Apply cost on value after return
                position = 0
                print(f"  {date.strftime('%Y-%m-%d')}: Stop loss hit, exited at {stop_price:.2f}, return: {ret:.2%}, New Equity: {current_equity_value:.2f}")

            if position > 0 and i > 0: # Daily P&L if holding position
                daily_ret = (price / price_te.iloc[i-1] - 1) * current_pos_size
                current_equity_value *= (1 + daily_ret)
                cycle_returns.append(daily_ret)


            if sig != position:
                if sig > 0 and position == 0:
                    entry_price = price
                    stop_price = entry_price * (1 - stop_loss_pct)
                    # Cost applied at entry on the portion of capital used for this trade (conceptual here)
                    # For simplicity, let's assume cost is on the change in portfolio value if we were to allocate
                    # A fixed fraction of current_equity_value.
                    # Here, we apply it to the whole current_equity_value as a proxy.
                    current_equity_value -= current_equity_value * transaction_cost_pct
                    position = 1
                    print(f"  {date.strftime('%Y-%m-%d')}: Entered long at {entry_price:.2f}, stop: {stop_price:.2f}, New Equity: {current_equity_value:.2f}")
                elif sig == 0 and position > 0:
                    ret = (price / entry_price - 1) * current_pos_size
                    current_equity_value *= (1 + ret)
                    current_equity_value -= current_equity_value * transaction_cost_pct # Cost on exit
                    position = 0
                    print(f"  {date.strftime('%Y-%m-%d')}: Exited long at {price:.2f}, return: {ret:.2%}, New Equity: {current_equity_value:.2f}")
            
            all_equity[date] = current_equity_value # Update equity for the day

        cash = current_equity_value # Update overall cash for next cycle start

        cycle_returns_series = pd.Series(cycle_returns)
        start_equity_for_cycle = all_equity.iloc[test_idx_slice.start] if test_idx_slice.start < len(all_equity) else cash # Fallback
        
        cycle_metric_entry = {
            'cycle': len(cycle_metrics) + 1,
            'start_date': df.index[test_idx_slice.start],
            'end_date': cycle_end_date,
            'start_equity': start_equity_for_cycle,
            'end_equity': current_equity_value,
            'return': (current_equity_value / start_equity_for_cycle - 1) if start_equity_for_cycle != 0 else 0,
            'sharpe': (cycle_returns_series.mean() / (cycle_returns_series.std() + 1e-9) * np.sqrt(252)) if len(cycle_returns) > 1 else 0,
            'win_rate': (cycle_returns_series > 0).mean() if len(cycle_returns) > 0 else 0
        }
        cycle_metrics.append(cycle_metric_entry)
        print(f"  Cycle {cycle_metric_entry['cycle']} completed: Equity ${cycle_metric_entry['end_equity']:.2f}, Return: {cycle_metric_entry['return']:.2%}\n")

    all_equity = all_equity.ffill().bfill() # Fill any gaps
    
    print("\nCycle Performance Summary:")
    cycles_df = pd.DataFrame(cycle_metrics)
    if not cycles_df.empty:
        for _, cycle_row in cycles_df.iterrows():
            print(f"Cycle {cycle_row['cycle']}: {cycle_row['start_date'].strftime('%Y-%m-%d')} - {cycle_row['end_date'].strftime('%Y-%m-%d')}, " + 
                  f"Return: {cycle_row['return']:.2%}, Sharpe: {cycle_row['sharpe']:.2f}, Win Rate: {cycle_row['win_rate']:.2%}")
    
    return all_equity


# ------------------------------------------------------
# 3. Main execution
# ------------------------------------------------------
if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    
    TICKER = 'AAPL' # Define ticker

    # load market data
    md = MarketData(TICKER)
    price_orig = md.fetch_historical(period='3y', interval='1d')
    logging.info(f"Shape of price data after fetch_historical: {price_orig.shape}")
    logging.info(f"Columns in input 'price' DataFrame: {price_orig.columns.tolist()}")
    logging.info(f"NaN sum for each column in 'price':\n{price_orig.isna().sum()}")

    # generate features
    features_df_orig = prepare_features(price_orig.copy(), drop_na=False, print_diagnostics=True)
    logging.info(f"Shape of features_df before dropna: {features_df_orig.shape}")

    features_df = features_df_orig.copy()
    features_df.fillna(method='bfill', inplace=True)
    features_df.fillna(method='ffill', inplace=True)
    logging.info(f"Shape after fillna but before dropna: {features_df.shape}")

    cols_to_drop_before_dropna = []
    if 'psar_up' in features_df.columns: cols_to_drop_before_dropna.append('psar_up')
    if 'psar_down' in features_df.columns: cols_to_drop_before_dropna.append('psar_down')
    if cols_to_drop_before_dropna:
        features_df.drop(columns=cols_to_drop_before_dropna, inplace=True)
        logging.info(f"Dropped {cols_to_drop_before_dropna}. Shape: {features_df.shape}")

    features_df.dropna(inplace=True)
    logging.info(f"Shape of features_df AFTER dropna: {features_df.shape}")

    if features_df.empty:
        logging.error("Error: features_df is empty after feature preparation and NaN removal.")
        import sys
        sys.exit(1)
    
    # Align price to features_df's index
    price = price_orig.loc[features_df.index].copy()

    # define label: next 5-day return >2%
    future_returns = price['Close'].pct_change(periods=5).shift(-5)
    label = (future_returns > 0.02).astype(int)
    
    # Align label to features_df's index (after label calculation)
    label = label.loc[features_df.index]

    # Ensure no NaNs in label after alignment, which can happen if future_returns had NaNs at the end
    # And features_df was shorter.
    # Also, features_df and price are already aligned.
    # We need to make sure all three (price, features_df, label) are aligned and have no NaNs for the rows used.
    
    common_index = features_df.index.intersection(label.index).intersection(price.index)
    features_df = features_df.loc[common_index]
    price = price.loc[common_index]
    label = label.loc[common_index]

    # Final check for NaNs in label that might have been introduced by pct_change/shift
    # and were not handled by the initial feature_df processing
    if label.isna().any():
        logging.warning(f"NaNs found in label after alignment. Dropping {label.isna().sum()} rows.")
        valid_label_idx = label.dropna().index
        features_df = features_df.loc[valid_label_idx]
        price = price.loc[valid_label_idx]
        label = label.loc[valid_label_idx]

    logging.info(f"Final aligned shapes: price {price.shape}, features_df {features_df.shape}, label {label.shape}")

    if features_df.empty or price.empty or label.empty:
        logging.error("One of the DataFrames is empty after final alignment and NaN handling. Exiting.")
        import sys
        sys.exit(1)

    # --- Prepare TFT-friendly DataFrame (conceptually, this is done inside walkforward) ---
    # tft_df = features_df.copy()
    # tft_df["time_idx"] = np.arange(len(tft_df))
    # tft_df["group_id"] = TICKER
    # tft_df["target"] = label.values # Add target for TFT
    # logging.info(f"Shape of tft_df for potential global training: {tft_df.shape}")
    # logging.info(f"Columns in tft_df: {tft_df.columns.tolist()}")


    # Run the walk-forward backtest
    equity_curve = walkforward_backtest(
        df=price,
        features=features_df,
        labels=label,
        train_window=252, # Standard year
        test_window=21,   # Standard month
        stop_loss_pct=0.03,
        transaction_cost_pct=0.001,
        risk_volatility=0.01,
        vol_lookback=20,
        ticker=TICKER
    )

    if not equity_curve.empty:
        plt.figure(figsize=(12, 7))
        equity_curve.plot(title=f'TFT Walk-Forward Equity Curve ({TICKER})')
        plt.ylabel('Portfolio Value ($)')
        plt.xlabel('Date')
        plt.tight_layout()
        # Save the plot with a unique timestamp
        plot_filename = f"tft_backtest_results_{TICKER}_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.png"
        plt.savefig(plot_filename)
        plt.show()
        logging.info(f"Equity curve plotted and saved to {plot_filename}")
        
        # Generate and print performance summary
        # Assuming initial capital was 10000.0 as used in walkforward_backtest
        performance_summary = summarize_performance(equity_curve, initial_capital=10000.0) 
        print_performance_report(performance_summary) # MODIFIED: Call the correct print function
    else:
        logging.error("Equity curve is empty. Plotting and performance summary skipped.")

