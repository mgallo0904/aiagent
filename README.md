# AI Options Trading Agent

An AI-powered options trading system optimized for MacBook Pro M2 (2022).

## Features

- Machine learning models for market prediction (XGBoost, TensorFlow)
- Technical indicator analysis
- Risk management and position sizing
- Options strategy execution
- Backtesting capabilities
- User-friendly macOS UI with PyQt6

## Installation

1. Clone this repository:

   ```bash
   git clone https://github.com/yourusername/ai-options-trader.git
   cd ai-options-trader
   ```

2. Create and activate a virtual environment:

   ```bash
   python -m venv .venv
   source .venv/bin/activate
   ```

3. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

## Usage

### Running the application

Use the `run_agent.py` script to start the application with optimal settings for Apple Silicon:

```bash
./run_agent.py
```

### Running with the helper script

For convenience, you can use the `run_trader.sh` helper script:

```bash
# Run in backtest mode with default settings
./run_trader.sh

# Run with custom settings
./run_trader.sh --symbol=MSFT --capital=50000 --threshold=0.7

# Run in live mode
./run_trader.sh --live
```

### Command line options

```
usage: run_agent.py [-h] [--symbol SYMBOL] [--capital CAPITAL] [--strategy STRATEGY] 
                   [--mode {live,backtest}] [--prob-threshold PROBA_THRESHOLD]

options:
  -h, --help            show this help message and exit
  --symbol SYMBOL       Stock symbol to trade
  --capital CAPITAL     Initial capital amount
  --strategy STRATEGY   Trading strategy name
  --mode {live,backtest}
                        Operational mode (live or backtest)
  --prob-threshold PROBA_THRESHOLD
                        Probability threshold for ML model (0.0-1.0)
```

Example:

```bash
./run_agent.py --symbol AAPL --capital 50000 --strategy "Covered Call" --mode backtest
```

### Project maintenance

Use the cleanup script to keep the project organized:

```bash
./cleanup.sh --keep-models 5 --remove-pyc --clean-logs
```

## Project Structure

```text
├── config/               # Configuration settings
├── data/                 # Data storage
├── logs/                 # Application logs
├── models/               # Saved ML models
│   └── archive/          # Archived model versions
├── tests/                # Test modules
├── archive/              # Archived legacy files
├── features.py           # Feature definitions
├── main.py               # Main application
├── market_data.py        # Market data retrieval
├── ml_models.py          # ML model implementations
├── risk_management.py    # Risk management system
├── run_agent.py          # Launcher script
├── security.py           # Security utilities
├── strategies.py         # Trading strategies
├── trade_executor.py     # Trade execution
├── ui.py                 # User interface
├── utils.py              # Common utilities
└── cleanup.sh            # Maintenance script
```

## Project Maintenance

The project includes a comprehensive maintenance script (`cleanup.sh`) to keep your workspace clean and efficient:

```bash
# Basic cleanup: Archive models keeping only 5 most recent
./cleanup.sh --keep-models 5

# Remove Python cache files and temporary files
./cleanup.sh --remove-pyc

# Clean up log files older than 7 days
./cleanup.sh --clean-logs

# Perform a deep clean (all of the above plus temporary files)
./cleanup.sh --deep-clean
```

Run the maintenance script periodically to keep the project organized, especially after long development sessions.

## Configuration

Edit `config/settings.py` to adjust default settings, or override them with environment variables.

You can also copy `config/example.env` to `.env` in the project root to set environment variables:

```bash
cp config/example.env .env
# Edit .env with your preferred values
```

Example environment variables:

```bash
export AIAGENT_DEFAULT_SYMBOL=TSLA
export AIAGENT_DEFAULT_PROBA_THRESHOLD=0.7
./run_agent.py
```

## Notes

- This project is optimized for MacBook Pro M2 (2022) with Apple Silicon.
- TensorFlow is configured to use Metal acceleration for improved performance.
