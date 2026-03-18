"""
Example: How to Export Backtest Results
========================================

This script demonstrates how to use the backtest export functionality
to save results in different formats (CSV, JSON, TXT).
"""

import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from data.mock_data import generate_mock_minute_bars
from features.indicators import compute_all_macro_indicators
from backtest.backtest_engine import BacktestEngine


def main():
    print("=" * 60)
    print("BACKTEST EXPORT EXAMPLE")
    print("=" * 60)

    # 1. Generate mock data
    print("\n1. Generating mock data...")
    symbol = "NIFTY"
    minute_df = generate_mock_minute_bars(symbol, trading_days=125)
    print(f"   Generated {len(minute_df)} minute bars")

    # 2. Compute features
    print("\n2. Computing technical indicators...")
    featured_df = compute_all_macro_indicators(minute_df)
    print(f"   Computed features for {len(featured_df)} candles")

    # 3. Run backtest
    print("\n3. Running backtest...")
    engine = BacktestEngine()
    result = engine.run(featured_df, symbol=symbol, predictor=None)

    # 4. Export results in different formats
    print("\n4. Exporting results...")

    # Option A: Export all formats at once (recommended)
    print("\n   Option A: Export all formats to backtest_results/")
    result.export_all(base_name="example_backtest", output_dir="backtest_results")

    # Option B: Export individual formats
    print("\n   Option B: Export individual formats")
    result.export_to_csv("example_trades.csv")
    result.export_to_json("example_results.json")
    result.export_to_txt("example_report.txt")

    print("\n" + "=" * 60)
    print("EXPORT COMPLETE")
    print("=" * 60)
    print("\nFiles created:")
    print("  - backtest_results/example_backtest_trades.csv")
    print("  - backtest_results/example_backtest_results.json")
    print("  - backtest_results/example_backtest_report.txt")
    print("  - example_trades.csv")
    print("  - example_results.json")
    print("  - example_report.txt")
    print("\nYou can now:")
    print("  - Open CSV in Excel/Google Sheets for analysis")
    print("  - Parse JSON programmatically")
    print("  - Read TXT report for human-readable summary")


if __name__ == "__main__":
    main()
