"""Generate plots from backtest_results.json and save them into the logs directory.

This module is used after a backtest to create summary charts.
"""
import json
import os
from typing import Optional

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import config


def generate_backtest_plots(
    results_path: Optional[str] = None,
    output_dir: Optional[str] = None,
    prefix: str = "backtest",
) -> None:
    """Generate and save plots based on the backtest results.

    Parameters
    ----------
    results_path: Path to the backtest_results.json file. If None, uses
        config.BACKTEST_RESULTS_FILE.
    output_dir: Directory where the images will be written. If None, uses
        config.LOGS_DIR.
    prefix: Filename prefix for the generated images.
    """
    if results_path is None:
        results_path = config.BACKTEST_RESULTS_FILE
    if output_dir is None:
        output_dir = config.LOGS_DIR

    if not os.path.exists(results_path):
        print(f"[WARN] Backtest results file not found: {results_path}")
        return

    with open(results_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    trades = data.get("trades", [])
    if not trades:
        print("[WARN] No trades found in backtest results; skipping plot generation.")
        return

    df = pd.DataFrame(trades)

    # Convert timestamps to datetime where present
    for col in ("entry_time", "exit_time"):
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors="coerce")

    os.makedirs(output_dir, exist_ok=True)
    sns.set_style("darkgrid")

    # 1) Equity Curve – Kapitalverlauf
    if "exit_time" in df.columns and "capital_after_exit" in df.columns:
        fig, ax = plt.subplots(figsize=(12, 5))
        ax.plot(df["exit_time"], df["capital_after_exit"], color="blue")
        ax.set_title("Equity Curve – Kapitalverlauf")
        ax.set_xlabel("Zeit")
        ax.set_ylabel("Kontostand")
        fig.autofmt_xdate()
        fig.tight_layout()
        path = os.path.join(output_dir, f"{prefix}_equity_curve.png")
        fig.savefig(path, dpi=150)
        plt.close(fig)
        print(f"[SAVE] {path}")

    # 2) PnL pro Trade
    if "net_pnl" in df.columns:
        fig, ax = plt.subplots(figsize=(12, 5))
        colors = df["net_pnl"].apply(lambda x: "green" if x > 0 else "red")
        ax.bar(df.index, df["net_pnl"], color=colors)
        ax.set_title("PnL pro Trade")
        ax.set_xlabel("Trade Nummer")
        ax.set_ylabel("Profit / Verlust")
        fig.tight_layout()
        path = os.path.join(output_dir, f"{prefix}_pnl_per_trade.png")
        fig.savefig(path, dpi=150)
        plt.close(fig)
        print(f"[SAVE] {path}")

    # 3) Scatterplot: Exit Time vs Net PnL
    if "exit_time" in df.columns and "net_pnl" in df.columns:
        fig, ax = plt.subplots(figsize=(12, 5))
        if "direction" in df.columns:
            sns.scatterplot(
                x=df["exit_time"],
                y=df["net_pnl"],
                hue=df["direction"],
                palette={"LONG": "blue", "SHORT": "orange"},
                ax=ax,
            )
        else:
            sns.scatterplot(x=df["exit_time"], y=df["net_pnl"], ax=ax)
        ax.axhline(0, color="black", linestyle="--")
        ax.set_title("Trade PnL über die Zeit")
        ax.set_xlabel("Exit Time")
        ax.set_ylabel("Net PnL")
        fig.autofmt_xdate()
        fig.tight_layout()
        path = os.path.join(output_dir, f"{prefix}_pnl_over_time.png")
        fig.savefig(path, dpi=150)
        plt.close(fig)
        print(f"[SAVE] {path}")

    # 4) Verteilung der Gewinne/Verluste
    if "net_pnl" in df.columns:
        fig, ax = plt.subplots(figsize=(10, 4))
        sns.histplot(df["net_pnl"], bins=40, kde=True, color="purple", ax=ax)
        ax.set_title("Distribution of Net PnL")
        ax.set_xlabel("Net PnL")
        ax.set_ylabel("Häufigkeit")
        fig.tight_layout()
        path = os.path.join(output_dir, f"{prefix}_pnl_distribution.png")
        fig.savefig(path, dpi=150)
        plt.close(fig)
        print(f"[SAVE] {path}")


if __name__ == "__main__":
    generate_backtest_plots()
