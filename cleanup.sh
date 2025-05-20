#!/bin/bash
# Enhanced Cleanup script for AI Agent project

echo "Starting enhanced cleanup..."

# --- Configuration ---
# Number of recent files to keep for each category
# For example, if NUM_WALKFORWARD_TO_KEEP=2, the 2 newest files will be kept.
NUM_WALKFORWARD_TO_KEEP=2
NUM_BACKTEST_TO_KEEP=2
NUM_CYCLES_TO_KEEP=2
NUM_DRAWDOWNS_TO_KEEP=2
NUM_PERFORMANCE_PLOTS_TO_KEEP=2
NUM_MODELS_TO_KEEP=2 # For XGBoost models

# Calculate the value for tail (number to keep + 1)
TAIL_N_WALKFORWARD=$((NUM_WALKFORWARD_TO_KEEP + 1))
TAIL_N_BACKTEST=$((NUM_BACKTEST_TO_KEEP + 1))
TAIL_N_CYCLES=$((NUM_CYCLES_TO_KEEP + 1))
TAIL_N_DRAWDOWNS=$((NUM_DRAWDOWNS_TO_KEEP + 1))
TAIL_N_PERFORMANCE_PLOTS=$((NUM_PERFORMANCE_PLOTS_TO_KEEP + 1))
TAIL_N_MODELS=$((NUM_MODELS_TO_KEEP + 1))

# --- Clean Python Cache Files ---
echo "Removing Python cache files (__pycache__, *.pyc, *.pyo, *.pyd)..."
find . -path "*/__pycache__" -type d -exec rm -rf {} +
find . -name "*.pyc" -type f -delete
find . -name "*.pyo" -type f -delete
find . -name "*.pyd" -type f -delete

# --- Clean Data Files in 'data/' Directory ---
echo "Cleaning up data files in 'data/' directory..."
if [ -d "data" ]; then
    cd data

    # Keep only the most recent walkforward files (e.g., walkforward_AAPL_*.csv)
    echo "Cleaning up walkforward_AAPL_*.csv files (keeping ${NUM_WALKFORWARD_TO_KEEP})..."
    # Check if files exist before trying to list and delete
    if ls walkforward_AAPL_* 1> /dev/null 2>&1; then
        ls -t walkforward_AAPL_* | tail -n "+${TAIL_N_WALKFORWARD}" | xargs -I {} rm -f {}
    else
        echo "No walkforward_AAPL_*.csv files found to clean."
    fi

    # Keep only the most recent backtest files (e.g., backtest_AAPL_*.csv)
    echo "Cleaning up backtest_AAPL_*.csv files (keeping ${NUM_BACKTEST_TO_KEEP})..."
    if ls backtest_AAPL_* 1> /dev/null 2>&1; then
        ls -t backtest_AAPL_* | tail -n "+${TAIL_N_BACKTEST}" | xargs -I {} rm -f {}
    else
        echo "No backtest_AAPL_*.csv files found to clean."
    fi

    # Keep only the most recent cycles files (e.g., cycles_AAPL_*.csv)
    echo "Cleaning up cycles_AAPL_*.csv files (keeping ${NUM_CYCLES_TO_KEEP})..."
    if ls cycles_AAPL_* 1> /dev/null 2>&1; then
        ls -t cycles_AAPL_* | tail -n "+${TAIL_N_CYCLES}" | xargs -I {} rm -f {}
    else
        echo "No cycles_AAPL_*.csv files found to clean."
    fi

    # Keep only the most recent drawdowns files (e.g., drawdowns_AAPL_*.csv)
    echo "Cleaning up drawdowns_AAPL_*.csv files (keeping ${NUM_DRAWDOWNS_TO_KEEP})..."
    if ls drawdowns_AAPL_* 1> /dev/null 2>&1; then
        ls -t drawdowns_AAPL_* | tail -n "+${TAIL_N_DRAWDOWNS}" | xargs -I {} rm -f {}
    else
        echo "No drawdowns_AAPL_*.csv files found to clean."
    fi

    # Keep only the most recent performance plot files (e.g., performance_AAPL_*.png)
    echo "Cleaning up performance_AAPL_*.png files (keeping ${NUM_PERFORMANCE_PLOTS_TO_KEEP})..."
    if ls performance_AAPL_* 1> /dev/null 2>&1; then
        ls -t performance_AAPL_*.png | tail -n "+${TAIL_N_PERFORMANCE_PLOTS}" | xargs -I {} rm -f {}
    else
        echo "No performance_AAPL_*.png files found to clean."
    fi
    
    # Note: equity_{symbol}.png and label_comparison_{symbol}.png are not timestamped
    # in your current scripts, so they will be overwritten. No cleanup rule needed for them
    # unless their naming convention changes to include timestamps.

    cd ..
else
    echo "Directory 'data' not found. Skipping data file cleanup."
fi

# --- Clean Model Files in 'models/' Directory ---
echo "Cleaning up model files in 'models/' directory..."
if [ -d "models" ]; then
    cd models

    # Remove older model files (keeping most recent specified by NUM_MODELS_TO_KEEP)
    # Assuming a common model naming pattern like 'xgboost_classification_binary_*'
    MODEL_PATTERN="xgboost_classification_binary_*" 
    echo "Cleaning up ${MODEL_PATTERN} files (keeping ${NUM_MODELS_TO_KEEP})..."
    if ls ${MODEL_PATTERN} 1> /dev/null 2>&1; then
        ls -t ${MODEL_PATTERN} | tail -n "+${TAIL_N_MODELS}" | xargs -I {} rm -f {}
    else
        echo "No ${MODEL_PATTERN} files found to clean."
    fi
    
    # Add rules for other model patterns if you have them, e.g.:
    # MODEL_PATTERN_LGBM="lightgbm_*"
    # if ls ${MODEL_PATTERN_LGBM} 1> /dev/null 2>&1; then
    #     ls -t ${MODEL_PATTERN_LGBM} | tail -n "+${TAIL_N_MODELS}" | xargs -I {} rm -f {}
    # else
    #     echo "No ${MODEL_PATTERN_LGBM} files found."
    # fi

    cd ..
else
    echo "Directory 'models' not found. Skipping model file cleanup."
fi

# --- Optional: Remove Diagnostic Scripts ---
# Pass --remove-diagnostics as the first argument to the script to trigger this
if [ "$1" == "--remove-diagnostics" ]; then
    echo "Removing specified diagnostic scripts..."
    rm -f quick_diagnostics.py
    # Add other diagnostic scripts here if needed, e.g.:
    # rm -f test_label_improvement.py 
    echo "Diagnostic scripts removed."
fi

# --- Clean Temporary and Editor Backup Files ---
echo "Removing temporary files (*~, *.bak, *.swp, *.swo, .DS_Store)..."
find . -name "*~" -type f -delete
find . -name "*.bak" -type f -delete
find . -name "*.swp" -type f -delete
find . -name "*.swo" -type f -delete
find . -name ".DS_Store" -type f -delete

# --- Clean Log Files (Optional - Example: remove logs older than 7 days) ---
LOG_DIR="logs" # Assuming your logs are in a 'logs' directory
DAYS_TO_KEEP_LOGS=7
if [ -d "$LOG_DIR" ]; then
    echo "Cleaning up log files older than ${DAYS_TO_KEEP_LOGS} days in '${LOG_DIR}'..."
    find "${LOG_DIR}" -name "*.log" -type f -mtime "+${DAYS_TO_KEEP_LOGS}" -exec rm -f {} \;
else
    echo "Directory '${LOG_DIR}' not found. Skipping log cleanup."
fi


echo "Cleanup completed successfully!"