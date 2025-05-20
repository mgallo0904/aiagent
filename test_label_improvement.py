#!/usr/bin/env python3
"""
Demonstrate the improved label definition approach for the AI trading agent.
This script compares the original next-bar direction approach with the 
improved significant-move threshold approach.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from market_data import MarketData
from features import add_ta_features

def compare_label_approaches(symbol="AAPL", period="1y", interval="1d"):
    """
    Compare different label definition approaches and their impact on signal quality.
    
    Args:
        symbol: Stock ticker symbol
        period: Data period to fetch
        interval: Data interval
    """
    print(f"Comparing label approaches for {symbol} ({period}, {interval})")
    
    # Fetch market data
    market = MarketData(symbol=symbol)
    data = market.fetch_historical(period=period, interval=interval)
    
    if data is None or data.empty:
        print("Error: Could not fetch data")
        return
    
    # Ensure lowercase column names
    data.columns = [col.lower() for col in data.columns]
    
    # Add features
    data = add_ta_features(data)
    
    # Original approach: next-bar direction
    data["label_original"] = (data["close"].shift(-1) > data["close"]).astype(int)
    
    # Improved approach: significant moves with threshold and horizon
    horizon = 3  # Look ahead 3 bars
    threshold = 0.01  # 1% threshold
    
    # Calculate future price and percent change
    future_price = data["close"].shift(-horizon)
    pct_change = (future_price - data["close"]) / data["close"]
    
    # Create labels for significant moves
    data["label_improved"] = np.nan
    data.loc[pct_change >= threshold, "label_improved"] = 1  # Significant upward move
    data.loc[pct_change <= -threshold, "label_improved"] = 0  # Significant downward move
    
    # Drop NaN values
    data = data.dropna(subset=["label_improved"])
    
    # Calculate statistics
    original_counts = data["label_original"].value_counts(normalize=True)
    improved_counts = data["label_improved"].value_counts(normalize=True)
    
    # Calculate signal balance
    original_balance = 1 - abs(original_counts[1] - original_counts[0])
    improved_balance = 1 - abs(improved_counts[1] - improved_counts[0])
    
    # Print statistics
    print("\nLabel Distribution Comparison:")
    print("Original approach (next-bar direction):")
    print(f"  Up: {original_counts[1]:.2%}, Down: {original_counts[0]:.2%}")
    print(f"  Balance: {original_balance:.2%}")
    
    print("\nImproved approach (significant moves):")
    print(f"  Up: {improved_counts[1]:.2%}, Down: {improved_counts[0]:.2%}")
    print(f"  Balance: {improved_balance:.2%}")
    
    # Visualize the difference
    fig, axes = plt.subplots(3, 1, figsize=(10, 12), sharex=True)
    
    # Price chart
    axes[0].plot(data.index, data["close"])
    axes[0].set_title(f"{symbol} Price")
    axes[0].set_ylabel("Price")
    axes[0].grid(True)
    
    # Original labels
    axes[1].scatter(data.index, data["close"], c=data["label_original"], 
                   cmap="coolwarm", alpha=0.8)
    axes[1].set_title("Original Labels (Next-Bar Direction)")
    axes[1].set_ylabel("Price")
    axes[1].grid(True)
    
    # Improved labels
    axes[2].scatter(data.index, data["close"], c=data["label_improved"], 
                   cmap="coolwarm", alpha=0.8)
    axes[2].set_title(f"Improved Labels (Significant {threshold*100:.1f}% Moves over {horizon} Bars)")
    axes[2].set_ylabel("Price")
    axes[2].set_xlabel("Date")
    axes[2].grid(True)
    
    plt.tight_layout()
    plt.savefig(f"data/label_comparison_{symbol}.png")
    print(f"Visualization saved to data/label_comparison_{symbol}.png")
    
    # Return statistics for further analysis
    return {
        "original_balance": original_balance,
        "improved_balance": improved_balance,
        "original_counts": original_counts,
        "improved_counts": improved_counts
    }

if __name__ == "__main__":
    # Compare labels for different symbols
    compare_label_approaches(symbol="AAPL", period="1y", interval="1d")
    # Uncomment to test with other symbols
    # compare_label_approaches(symbol="MSFT", period="1y", interval="1d")
    # compare_label_approaches(symbol="AMZN", period="1y", interval="1d")
