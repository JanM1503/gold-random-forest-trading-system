"""
Haupteinstiegspunkt für das Gold Trading Framework
"""
import sys
import argparse
from data_loader import DataManager
from sentiment import SentimentAnalyzer
from feature_engineering import FeatureEngineer
from models import TradingModels
from explainability import ModelExplainer
# from backtest_new import Backtest  # OLD: Slow version
from backtest_optimized import BacktestOptimized as Backtest  # NEW: 50-100x faster!
from live_trading import LiveTrader
from fred_data import load_fred_data


def load_data(force_reload=False):
    """Lädt alle Daten"""
    print("\n[LOAD] LADE DATEN")
    dm = DataManager()
    all_data = dm.load_all_data(force_reload=force_reload)
    return all_data


def load_sentiment(force_reload=False):
    """Lädt Sentiment-Daten"""
    print("\n[LOAD] LADE SENTIMENT-DATEN")
    analyzer = SentimentAnalyzer()
    sentiment_data = analyzer.load_and_analyze_sentiment(force_reload=force_reload)
    return sentiment_data


def create_features(all_data, sentiment_data, fred_data):
    """Erstellt Feature-Matrix"""
    print("\n[LOAD] ERSTELLE FEATURES")
    fe = FeatureEngineer()
    
    df = fe.create_feature_matrix(
        candle_data=all_data['gold_candles'],
        macro_data={
            'oil_candles': all_data['oil_candles'],
            'silver_candles': all_data['silver_candles'],
            'vix': all_data['vix'],
            'yields': all_data['yields'],
            'fear_greed': all_data['fear_greed'],
            'macro': all_data['macro']
        },
        sentiment_data=sentiment_data,
        fred_data=fred_data,
        generate_labels=True
    )
    
    feature_columns = fe.get_feature_columns(df)
    
    return df, feature_columns


def train_models(df, feature_columns):
    """Trainiert ML Models"""
    print("\n[LOAD] TRAINIERE MODELS")
    models = TradingModels()
    models.train_and_save_models(df, feature_columns)
    return models


def run_backtest(df):
    """Führt Backtest durch - NUR auf Out-of-Sample Periode"""
    import config
    import pandas as pd
    print("\n[LOAD] STARTE BACKTEST")
    
    test_start = pd.to_datetime(config.TEST_START_DATE)
    test_df = df[df['timestamp'] >= test_start].copy()
    print(f"[INFO] Backtest period: {test_df['timestamp'].min()} to {test_df['timestamp'].max()}")
    print(f"[INFO] Backtest samples: {len(test_df):,} candles")
    
    models = TradingModels()
    if not models.load_models():
        print("[ERROR] Models nicht gefunden! Trainiere zuerst mit --train")
        return
    
    explainer = ModelExplainer(models)
    explainer.print_top_features(top_n=20)
    
    backtest = Backtest(models, explainer)
    performance = backtest.run(test_df)
    backtest.save_results()

    # Optional: Backtest-Grafiken erzeugen und in logs/ speichern
    try:
        from maturagraph import generate_backtest_plots
        generate_backtest_plots()
    except Exception as e:
        print(f"[WARN] Konnte Backtest-Grafiken nicht erzeugen: {e}")
    
    return performance


def run_live_trading(loop=False):
    """Startet Live Trading"""
    trader = LiveTrader()
    
    if loop:
        trader.run_live_loop()
    else:
        trader.run_once()


def main():
    parser = argparse.ArgumentParser(
        description="Gold Trading Framework - Backtest & Live Trading"
    )
    
    parser.add_argument(
        '--mode',
        choices=['data', 'train', 'backtest', 'live', 'full'],
        default='full',
        help='Ausführungsmodus'
    )
    
    parser.add_argument(
        '--force-reload',
        action='store_true',
        help='Lädt alle Daten komplett neu'
    )
    
    parser.add_argument(
        '--live-loop',
        action='store_true',
        help='Live Trading in Endlosschleife (nur mit --mode live)'
    )
    
    parser.add_argument(
        '--test-signal',
        choices=['LONG', 'SHORT'],
        help='Testet Trading-Pipeline mit simuliertem Signal (LONG oder SHORT)'
    )
    
    args = parser.parse_args()
    
    print("="*70)
    print("[GOLD] GOLD TRADING FRAMEWORK")
    print("="*70)
    print(f"Modus: {args.mode}")
    print("="*70)
    
    if args.mode == 'data':
        # Nur Daten laden
        all_data = load_data(force_reload=args.force_reload)
        sentiment_data = load_sentiment(force_reload=args.force_reload)
        fred_data = load_fred_data(force_reload=args.force_reload)
        print("\n[OK] Daten geladen")
    
    elif args.mode == 'train':
        # Daten laden + Models trainieren
        all_data = load_data(force_reload=args.force_reload)
        sentiment_data = load_sentiment(force_reload=args.force_reload)
        fred_data = load_fred_data(force_reload=args.force_reload)
        df, feature_columns = create_features(all_data, sentiment_data, fred_data)
        models = train_models(df, feature_columns)
        print("\n[OK] Training abgeschlossen")
    
    elif args.mode == 'backtest':
        # Daten laden + Backtest
        all_data = load_data(force_reload=args.force_reload)
        sentiment_data = load_sentiment(force_reload=args.force_reload)
        fred_data = load_fred_data(force_reload=args.force_reload)
        df, feature_columns = create_features(all_data, sentiment_data, fred_data)
        performance = run_backtest(df)
        print("\n[OK] Backtest abgeschlossen")
    
    elif args.mode == 'live':
        # Live Trading oder Test
        if args.test_signal:
            # Test-Modus mit simuliertem Signal
            trader = LiveTrader()
            trader.test_trade_signal(signal_type=args.test_signal)
        else:
            # Normales Live Trading
            run_live_trading(loop=args.live_loop)
        print("\n[OK] Live Trading abgeschlossen")
    
    elif args.mode == 'full':
        # Kompletter Workflow
        print("\n[PLAN] FULL WORKFLOW: Daten → Training → Backtest")
        
        # 1. Daten laden (inkl. FRED macro data)
        all_data = load_data(force_reload=args.force_reload)
        sentiment_data = load_sentiment(force_reload=args.force_reload)
        fred_data = load_fred_data(force_reload=args.force_reload)
        
        # 2. Features erstellen (mit FRED data)
        df, feature_columns = create_features(all_data, sentiment_data, fred_data)
        
        # 3. Models trainieren
        models = train_models(df, feature_columns)
        
        # 4. Backtest
        performance = run_backtest(df)
        
        print("\n[OK] FULL WORKFLOW ABGESCHLOSSEN")
        print("\n[TARGET] NÄCHSTE SCHRITTE:")
        print("  1. Prüfe Backtest-Ergebnisse in logs/backtest_results.json")
        print("  2. Für Live-Demo: python run.py --mode live")
        print("  3. Für kontinuierliches Trading: python run.py --mode live --live-loop")


if __name__ == "__main__":
    main()
