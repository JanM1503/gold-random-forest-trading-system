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

    # Sortiere nach Exit-Zeit, damit alle Zeitachsen konsistent sind
    if "exit_time" in df.columns:
        df = df.sort_values("exit_time").reset_index(drop=True)

    # Hypothetische Equity-Kurve ohne Spread-Kosten (Slippage bleibt erhalten)
    no_spread_cols = {
        "mid_price_entry",
        "mid_price_exit",
        "position_size",
        "direction",
        "slippage_entry",
        "slippage_exit",
    }
    if no_spread_cols.issubset(df.columns):
        dir_sign = df["direction"].map({"LONG": 1.0, "SHORT": -1.0}).fillna(0.0)
        slip_entry = df["slippage_entry"].fillna(0.0)
        slip_exit = df["slippage_exit"].fillna(0.0)

        # Entry/Exit ohne Spread (aber mit Slippage)
        entry_no_spread = df["mid_price_entry"] + dir_sign * slip_entry
        exit_no_spread = df["mid_price_exit"] - dir_sign * slip_exit

        df["pnl_no_spread"] = (exit_no_spread - entry_no_spread) * dir_sign * df["position_size"]
        df["capital_no_spread"] = config.INITIAL_CAPITAL + df["pnl_no_spread"].cumsum()

    os.makedirs(output_dir, exist_ok=True)
    sns.set_style("darkgrid")

    # 1) Equity Curve  Kapitalverlauf
    if "exit_time" in df.columns and "capital_after_exit" in df.columns:
        fig, ax = plt.subplots(figsize=(12, 5))
        ax.plot(
            df["exit_time"],
            df["capital_after_exit"],
            label="Mit Spread",
            color="blue",
        )

        # Zweite Kurve: gleiche Trades, aber ohne Spread-Kosten (Slippage bleibt)
        if "capital_no_spread" in df.columns:
            ax.plot(
                df["exit_time"],
                df["capital_no_spread"],
                label="Ohne Spread",
                color="orange",
            )
            ax.legend()

        ax.set_title("Equity Curve - Kapitalverlauf")
        ax.set_xlabel("Zeit")
        ax.set_ylabel("Kontostand")
        fig.autofmt_xdate()
        fig.tight_layout()
        path = os.path.join(output_dir, f"{prefix}_equity_curve.png")
        fig.savefig(path, dpi=150)
        plt.close(fig)
        print(f"[SAVE] {path}")

    # 1b) Vergleich: Strategie vs. Buy & Hold
    # Buy & Hold: Am ersten Tag mit gesamtem Startkapital Gold kaufen und bis zum Ende halten.
    if (
        "exit_time" in df.columns
        and "capital_after_exit" in df.columns
        and "mid_price_exit" in df.columns
    ):
        # Preis des Underlyings am ersten Exit-Punkt als Referenz
        base_price = df["mid_price_exit"].iloc[0]
        if pd.notna(base_price) and base_price != 0:
            buy_hold_capital = config.INITIAL_CAPITAL * (df["mid_price_exit"] / base_price)

            fig, ax = plt.subplots(figsize=(12, 5))
            ax.plot(df["exit_time"], df["capital_after_exit"], label="Strategie", color="blue")
            ax.plot(df["exit_time"], buy_hold_capital, label="Buy & Hold", color="orange")
            ax.set_title("Portfolio-Entwicklung: Strategie vs. Buy & Hold")
            ax.set_xlabel("Zeit")
            ax.set_ylabel("Kontostand")
            ax.legend()
            fig.autofmt_xdate()
            fig.tight_layout()
            path = os.path.join(output_dir, f"{prefix}_equity_vs_buyhold.png")
            fig.savefig(path, dpi=150)
            plt.close(fig)
            print(f"[SAVE] {path}")

            # Zweite Version mit getrennten Y-Achsen, um Bewegungen besser zu sehen
            # Wir verwenden wie im Original absolute Kapitalwerte, sorgen aber dafür,
            # dass beide Kurven bei 10'000 exakt auf demselben Punkt starten.
            strat_cap = df["capital_after_exit"]
            bh_cap = buy_hold_capital

            fig2, ax1 = plt.subplots(figsize=(12, 5))
            ax1.plot(df["exit_time"], strat_cap, color="blue", label="Strategie")
            ax1.set_xlabel("Zeit")
            ax1.set_ylabel("Strategie-Kapital", color="blue")
            ax1.tick_params(axis="y", labelcolor="blue")

            ax2 = ax1.twinx()
            ax2.plot(df["exit_time"], bh_cap, color="orange", label="Buy & Hold")
            ax2.set_ylabel("Buy & Hold Kapital", color="orange")
            ax2.tick_params(axis="y", labelcolor="orange")

            # Y-Skalen so setzen, dass 10'000 auf gleicher Höhe liegt,
            # aber Buy & Hold (größere Bewegung) eine "gestrecktere" Skala bekommt
            center = config.INITIAL_CAPITAL
            strat_down = max(center - strat_cap.min(), 0)
            strat_up = max(strat_cap.max() - center, 0)
            bh_down = max(center - bh_cap.min(), 0)
            bh_up = max(bh_cap.max() - center, 0)

            strat_range = max(strat_down, strat_up)
            bh_range = max(bh_down, bh_up)

            # Verhindere Null-Range
            strat_range = max(strat_range, 1.0)
            bh_range = max(bh_range, 1.0)

            # Skalenfaktor: Buy & Hold Achse soll "lockerer" sein
            scale_factor = 2.0

            if bh_range >= strat_range:
                # Strategie eng, Buy&Hold weiter
                r1 = strat_range
                r2 = max(bh_range, r1 * scale_factor)
            else:
                # Falls ausnahmsweise Strategie volatiler ist, umgekehrt
                r2 = bh_range
                r1 = max(strat_range, r2 * scale_factor)

            ax1.set_ylim(center - r1, center + r1)
            ax2.set_ylim(center - r2, center + r2)
            ax1.axhline(center, color="gray", linestyle="--", linewidth=0.8)

            fig2.suptitle("Strategie vs. Buy & Hold (separate Y-Achsen, beide starten bei 10'000)")
            fig2.autofmt_xdate()

            # kombinierte Legende
            h1, l1 = ax1.get_legend_handles_labels()
            h2, l2 = ax2.get_legend_handles_labels()
            ax1.legend(h1 + h2, l1 + l2, loc="upper left")

            fig2.tight_layout()
            path2 = os.path.join(output_dir, f"{prefix}_equity_vs_buyhold_y_adjusted.png")
            fig2.savefig(path2, dpi=150)
            plt.close(fig2)
            print(f"[SAVE] {path2}")

    # 2) PnL pro Trade
    if "net_pnl" in df.columns:
        fig, ax = plt.subplots(figsize=(12, 5))

        # Dunklere Grün/Rot-Töne und schwarze Kontur für bessere Lesbarkeit auf weißem Papier
        colors = df["net_pnl"].apply(lambda x: "#006400" if x > 0 else "#8B0000")
        ax.bar(
            df.index,
            df["net_pnl"],
            color=colors,
            edgecolor="black",
            linewidth=0.6,
        )

        ax.axhline(0, color="black", linewidth=1.0)
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

    # 5) Zusammenhang VIX und PnL
    # Nutzt tägliche VIX-Daten (data/fred_vix.json) und mapped sie auf die Exit-Zeitpunkte der Trades
    vix_file = os.path.join(config.DATA_DIR, "fred_vix.json")
    if os.path.exists(vix_file) and "exit_time" in df.columns and "net_pnl" in df.columns:
        try:
            vix_df = pd.read_json(vix_file)
        except Exception as e:  # pragma: no cover - nur Logging
            print(f"[WARN] Konnte VIX-Datei nicht lesen: {vix_file} ({e})")
            vix_df = None

        if vix_df is not None and not vix_df.empty and "date" in vix_df.columns and "VIXCLS" in vix_df.columns:
            vix_df = vix_df.copy()
            vix_df["date"] = pd.to_datetime(vix_df["date"], errors="coerce")
            vix_df["VIXCLS"] = pd.to_numeric(vix_df["VIXCLS"], errors="coerce")
            vix_df = vix_df.dropna(subset=["date", "VIXCLS"]).sort_values("date").reset_index(drop=True)

            if not vix_df.empty:
                # Merge-asof: ordnet jedem Trade den zuletzt verfügbaren VIX-Wert zu
                trades_for_merge = df.dropna(subset=["exit_time"]).sort_values("exit_time").reset_index()
                vix_merge = vix_df.rename(columns={"date": "exit_time"})[["exit_time", "VIXCLS"]]

                merged = pd.merge_asof(
                    trades_for_merge,
                    vix_merge,
                    on="exit_time",
                    direction="backward",
                )

                merged = merged.dropna(subset=["VIXCLS", "net_pnl"])

                if len(merged) > 0:
                    # Scatter: VIX-Level vs. absolute PnL-Größe pro Trade
                    merged["is_win"] = merged["net_pnl"] > 0

                    fig, ax = plt.subplots(figsize=(10, 5))
                    sns.scatterplot(
                        data=merged,
                        x="VIXCLS",
                        y=merged["net_pnl"].abs(),
                        hue="is_win",
                        palette={True: "#006400", False: "#8B0000"},
                        alpha=0.6,
                        ax=ax,
                    )
                    ax.set_xlabel("VIX (CBOE Volatility Index)")
                    ax.set_ylabel("|Net PnL| pro Trade")
                    ax.set_title("Zusammenhang: VIX vs. Größe der PnL-Bewegungen")

                    # Optional: Korrelationskoeffizient anzeigen
                    corr_df = merged[["VIXCLS", "net_pnl"]].dropna()
                    if len(corr_df) > 2:
                        r = corr_df["VIXCLS"].corr(corr_df["net_pnl"].abs())
                        if pd.notna(r):
                            ax.text(
                                0.02,
                                0.98,
                                f"Korrelationskoeffizient (|PnL|, VIX): {r:.2f}",
                                transform=ax.transAxes,
                                va="top",
                                ha="left",
                                fontsize=9,
                            )

                    fig.tight_layout()
                    path = os.path.join(output_dir, f"{prefix}_vix_vs_pnl.png")
                    fig.savefig(path, dpi=150)
                    plt.close(fig)
                    print(f"[SAVE] {path}")

                    # 5a) Gewinne / Verluste nach VIX-Regime (Buckets)
                    # VIX in Bereiche einteilen: niedrig, normal, hoch, extrem
                    bins = [0, 15, 25, 40, float("inf")]
                    labels = [
                        "niedrig (<15)",
                        "normal (15-25)",
                        "hoch (25-40)",
                        "extrem (>40)",
                    ]
                    merged["vix_bin"] = pd.cut(
                        merged["VIXCLS"],
                        bins=bins,
                        labels=labels,
                        right=False,
                    )

                    bucket_stats = (
                        merged.dropna(subset=["vix_bin"])
                        .groupby("vix_bin")
                        .apply(
                            lambda g: pd.Series(
                                {
                                    "count": len(g),
                                    "winrate_pct": 100.0 * g["is_win"].mean() if len(g) > 0 else 0.0,
                                    "avg_pnl": g["net_pnl"].mean() if len(g) > 0 else 0.0,
                                    "avg_win": g.loc[g["net_pnl"] > 0, "net_pnl"].mean()
                                    if (g["net_pnl"] > 0).any()
                                    else 0.0,
                                    "avg_loss": g.loc[g["net_pnl"] <= 0, "net_pnl"].mean()
                                    if (g["net_pnl"] <= 0).any()
                                    else 0.0,
                                }
                            )
                        )
                        .reset_index()
                    )

                    if len(bucket_stats) > 0:
                        # 5a) Balkendiagramm: Ø Gewinn/Verlust + Winrate je VIX-Bereich
                        fig, ax1 = plt.subplots(figsize=(10, 5))
                        x_pos = list(range(len(bucket_stats)))
                        width = 0.35

                        ax1.bar(
                            [x - width / 2 for x in x_pos],
                            bucket_stats["avg_win"],
                            width,
                            label="Ø Gewinn",
                            color="#006400",
                        )
                        ax1.bar(
                            [x + width / 2 for x in x_pos],
                            bucket_stats["avg_loss"],
                            width,
                            label="Ø Verlust",
                            color="#8B0000",
                        )
                        ax1.set_xticks(x_pos)
                        ax1.set_xticklabels(bucket_stats["vix_bin"].astype(str))
                        ax1.set_ylabel("Durchschn. PnL pro Trade")
                        ax1.set_xlabel("VIX-Bereich")
                        ax1.set_title("PnL-Statistik nach VIX-Regime")

                        ax2 = ax1.twinx()
                        ax2.plot(
                            x_pos,
                            bucket_stats["winrate_pct"],
                            color="black",
                            marker="o",
                            label="Winrate (%)",
                        )
                        ax2.set_ylabel("Winrate (%)")
                        ax2.set_ylim(0, max(100, bucket_stats["winrate_pct"].max() * 1.1))

                        # Legenden kombinieren
                        handles1, labels1 = ax1.get_legend_handles_labels()
                        handles2, labels2 = ax2.get_legend_handles_labels()
                        ax1.legend(
                            handles1 + handles2,
                            labels1 + labels2,
                            loc="upper right",
                        )

                        fig.tight_layout()
                        path = os.path.join(output_dir, f"{prefix}_vix_bucket_stats.png")
                        fig.savefig(path, dpi=150)
                        plt.close(fig)
                        print(f"[SAVE] {path}")

                        # 5b) Einfacher Plot: Winrate (%) pro VIX-Bereich
                        fig, ax = plt.subplots(figsize=(8, 4))
                        ax.bar(
                            x_pos,
                            bucket_stats["winrate_pct"],
                            color="steelblue",
                        )
                        ax.set_xticks(x_pos)
                        ax.set_xticklabels(bucket_stats["vix_bin"].astype(str))
                        ax.set_xlabel("VIX-Bereich")
                        ax.set_ylabel("Winrate (%)")
                        ax.set_title("Winrate nach VIX-Regime")

                        for x, wr in zip(x_pos, bucket_stats["winrate_pct"]):
                            ax.text(
                                x,
                                wr + 1,
                                f"{wr:.0f}%",
                                ha="center",
                                va="bottom",
                                fontsize=9,
                            )

                        ax.set_ylim(0, max(100, bucket_stats["winrate_pct"].max() * 1.1))
                        fig.tight_layout()
                        path = os.path.join(output_dir, f"{prefix}_vix_winrate_by_bucket.png")
                        fig.savefig(path, dpi=150)
                        plt.close(fig)
                        print(f"[SAVE] {path}")


if __name__ == "__main__":
    generate_backtest_plots()
