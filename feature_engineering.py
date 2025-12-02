"""
Feature Engineering für Gold Trading
Berechnet technische Indikatoren, Makro-Features und kombiniert alle Datenquellen
"""
import pandas as pd
import numpy as np
from datetime import datetime
from typing import Dict, List, Tuple
import config


class FeatureEngineer:
    """Feature Engineering Pipeline"""
    
    def __init__(self):
        pass
    
    # ====================================
    # TECHNISCHE INDIKATOREN
    # ====================================
    
    def calculate_sma(self, df: pd.DataFrame, periods: List[int]) -> pd.DataFrame:
        """Berechnet Simple Moving Averages"""
        for period in periods:
            df[f'sma_{period}'] = df['close'].rolling(window=period).mean()
            # Distanz zum SMA (normalisiert)
            df[f'dist_to_sma_{period}'] = (df['close'] - df[f'sma_{period}']) / df[f'sma_{period}']
        return df
    
    def calculate_rsi(self, df: pd.DataFrame, periods: List[int] = None) -> pd.DataFrame:
        """Berechnet Relative Strength Index für mehrere Perioden"""
        if periods is None:
            periods = [14, 28]
        
        for period in periods:
            delta = df['close'].diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
            
            rs = gain / loss
            df[f'rsi_{period}'] = 100 - (100 / (1 + rs))
        
        return df
    
    def calculate_bollinger_bands(
        self, 
        df: pd.DataFrame, 
        period: int = 20, 
        std_dev: float = 2.0
    ) -> pd.DataFrame:
        """Berechnet Bollinger Bands"""
        df['bb_middle'] = df['close'].rolling(window=period).mean()
        std = df['close'].rolling(window=period).std()
        
        df['bb_upper'] = df['bb_middle'] + (std * std_dev)
        df['bb_lower'] = df['bb_middle'] - (std * std_dev)
        df['bb_bandwidth'] = (df['bb_upper'] - df['bb_lower']) / df['bb_middle']
        
        # Position im Band (0 = am unteren Band, 1 = am oberen Band)
        df['bb_position'] = (df['close'] - df['bb_lower']) / (df['bb_upper'] - df['bb_lower'])
        
        return df
    
    def calculate_atr(self, df: pd.DataFrame, period: int = 14) -> pd.DataFrame:
        """Berechnet Average True Range"""
        high_low = df['high'] - df['low']
        high_close = np.abs(df['high'] - df['close'].shift())
        low_close = np.abs(df['low'] - df['close'].shift())
        
        true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        df['atr'] = true_range.rolling(window=period).mean()
        
        # ATR als % des Preises
        df['atr_pct'] = (df['atr'] / df['close']) * 100
        
        return df
    
    def calculate_returns(self, df: pd.DataFrame) -> pd.DataFrame:
        """Berechnet Returns und Volatilität"""
        # Einfache Returns
        df['return_1'] = df['close'].pct_change(1)
        df['return_4'] = df['close'].pct_change(4)  # 1 Stunde
        df['return_16'] = df['close'].pct_change(16)  # 4 Stunden
        
        # Log Returns
        df['log_return'] = np.log(df['close'] / df['close'].shift(1))
        
        # Rolling Volatilität
        df['volatility_16'] = df['return_1'].rolling(window=16).std()
        df['volatility_96'] = df['return_1'].rolling(window=96).std()  # 1 Tag
        
        return df
    
    def calculate_momentum(self, df: pd.DataFrame) -> pd.DataFrame:
        """Berechnet Momentum-Indikatoren"""
        # Momentum
        df['momentum_4'] = df['close'] - df['close'].shift(4)
        df['momentum_16'] = df['close'] - df['close'].shift(16)
        
        # Rate of Change
        df['roc_4'] = (df['close'] - df['close'].shift(4)) / df['close'].shift(4) * 100
        df['roc_16'] = (df['close'] - df['close'].shift(16)) / df['close'].shift(16) * 100
        
        return df
    
    def calculate_technical_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Berechnet ALLE technischen Indikatoren"""
        print("  [TOOL] Berechne technische Indikatoren...")
        
        df = self.calculate_sma(df, config.SMA_PERIODS)
        df = self.calculate_rsi(df, config.RSI_PERIODS)
        df = self.calculate_bollinger_bands(df, config.BOLLINGER_PERIOD, config.BOLLINGER_STD)
        df = self.calculate_atr(df, config.ATR_PERIOD)
        df = self.calculate_returns(df)
        df = self.calculate_momentum(df)
        
        print(f"  [OK] {len([c for c in df.columns if c not in ['timestamp', 'open', 'high', 'low', 'close', 'volume']])} technische Features berechnet")
        
        return df
    
    # ====================================
    # MAKRO-FEATURES
    # ====================================
    
    def merge_fred_data(
        self,
        df: pd.DataFrame,
        fred_data: Dict[str, pd.DataFrame]
    ) -> pd.DataFrame:
        """Merge FRED macroeconomic data with candle data"""
        if not fred_data:
            return df
        
        df['timestamp'] = pd.to_datetime(df['timestamp']).dt.tz_localize(None)
        df = df.sort_values('timestamp').reset_index(drop=True)
        
        for series_name, series_df in fred_data.items():
            if len(series_df) == 0:
                continue
            series_df = series_df.copy()
            series_df['date'] = pd.to_datetime(series_df['date']).dt.tz_localize(None)
            series_df = series_df.rename(columns={'date': 'timestamp'})
            value_col = [c for c in series_df.columns if c != 'timestamp'][0]
            feature_name = series_name.lower().replace('_', '')
            series_df = series_df.rename(columns={value_col: feature_name})
            df = pd.merge_asof(df, series_df[['timestamp', feature_name]], on='timestamp', direction='backward')
            df[feature_name] = df[feature_name].ffill()
        
        # Yield curve features
        if 'us2y' in df.columns and 'us10y' in df.columns:
            df['yield_curve_2y10y'] = df['us10y'] - df['us2y']
        if 'us5y' in df.columns and 'us30y' in df.columns:
            df['yield_curve_5y30y'] = df['us30y'] - df['us5y']
        
        # VIX features
        if 'vix' in df.columns:
            df['vix_change'] = df['vix'].diff()
            df['vix_spike'] = (df['vix'] > df['vix'].rolling(20).mean() + df['vix'].rolling(20).std()).astype(int)
        
        # CPI/inflation features
        if 'cpi' in df.columns:
            df['cpi_change'] = df['cpi'].pct_change() * 100
        if 'corecpi' in df.columns:
            df['corecpi_change'] = df['corecpi'].pct_change() * 100
        
        return df
    
    def merge_macro_data(
        self, 
        df: pd.DataFrame,
        macro_data: Dict[str, List[Dict]]
    ) -> pd.DataFrame:
        """Merged Makro-Daten mit Candle-Daten"""
        df['timestamp'] = pd.to_datetime(df['timestamp']).dt.tz_localize(None)
        df = df.sort_values('timestamp').reset_index(drop=True)
        
        # Oil
        if macro_data.get('oil_candles'):
            oil_df = pd.DataFrame(macro_data['oil_candles'])
            oil_df['timestamp'] = pd.to_datetime(oil_df['timestamp']).dt.tz_localize(None)
            oil_df = oil_df[['timestamp', 'close']].rename(columns={'close': 'oil_price'})
            df = pd.merge_asof(df, oil_df, on='timestamp', direction='backward')
            df['oil_return'] = df['oil_price'].pct_change()
        
        # Silver
        if macro_data.get('silver_candles'):
            silver_df = pd.DataFrame(macro_data['silver_candles'])
            silver_df['timestamp'] = pd.to_datetime(silver_df['timestamp']).dt.tz_localize(None)
            silver_df = silver_df[['timestamp', 'close']].rename(columns={'close': 'silver_price'})
            df = pd.merge_asof(df, silver_df, on='timestamp', direction='backward')
            df['silver_return'] = df['silver_price'].pct_change()
            df['gold_silver_ratio'] = df['close'] / df['silver_price']
        
        # Fear & Greed
        if macro_data.get('fear_greed'):
            fg_df = pd.DataFrame(macro_data['fear_greed'])
            fg_df['timestamp'] = pd.to_datetime(fg_df['timestamp'], format='ISO8601').dt.tz_localize(None)
            fg_df = fg_df[['timestamp', 'fear_greed']]
            df = pd.merge_asof(df, fg_df, on='timestamp', direction='backward')
        
        # Makro-Daten (CPI, NFP, etc.)
        if macro_data.get('macro'):
            macro_df = pd.DataFrame(macro_data['macro'])
            macro_df['timestamp'] = pd.to_datetime(macro_df['timestamp']).dt.tz_localize(None)
            numeric_cols = ['timestamp', 'cpi', 'inflation', 'unemployment', 'gdp_growth']
            available_cols = [c for c in numeric_cols if c in macro_df.columns]
            if available_cols:
                df = pd.merge_asof(df, macro_df[available_cols], on='timestamp', direction='backward')
        
        return df
    
    # ====================================
    # SENTIMENT FEATURES
    # ====================================
    
    def merge_sentiment_data(
        self, 
        df: pd.DataFrame,
        sentiment_articles: List[Dict]
    ) -> pd.DataFrame:
        """Merged Sentiment-Daten mit Candle-Daten"""
        print("  [TOOL] Merge Sentiment-Daten...")
        
        if not sentiment_articles:
            print("  [WARN] Keine Sentiment-Daten verfügbar - setze default Werte")
            df['sentiment_score'] = 0.0
            # WICHTIG: Auch rolling features erstellen, sonst fehlen Features!
            for window in config.SENTIMENT_ROLLING_WINDOWS:
                df[f'sentiment_roll_{window}'] = 0.0
            return df
        
        # Aggregiere Sentiment pro 15-Minuten-Periode
        from sentiment import SentimentAnalyzer
        analyzer = SentimentAnalyzer()
        sentiment_agg = analyzer.aggregate_sentiment_by_period(sentiment_articles, period_minutes=15)
        
        # Erstelle Sentiment DataFrame
        sentiment_df = pd.DataFrame([
            {'timestamp': k, 'sentiment_score': v} 
            for k, v in sentiment_agg.items()
        ])
        sentiment_df['timestamp'] = pd.to_datetime(sentiment_df['timestamp']).dt.tz_localize(None)
        sentiment_df = sentiment_df.sort_values('timestamp').reset_index(drop=True)
        
        # Merge
        df = pd.merge_asof(df, sentiment_df, on='timestamp', direction='backward')
        df['sentiment_score'] = df['sentiment_score'].fillna(0.0)
        
        # Rolling Sentiment
        for window in config.SENTIMENT_ROLLING_WINDOWS:
            df[f'sentiment_roll_{window}'] = df['sentiment_score'].rolling(window=window).mean()
        
        print(f"  [OK] Sentiment-Features gemerged")
        
        return df
    
    # ====================================
    # EVENT DUMMIES
    # ====================================
    
    def create_event_dummies(self, df: pd.DataFrame) -> pd.DataFrame:
        """Erstellt Dummy-Variablen für wichtige Events"""
        print("  [TOOL] Erstelle Event-Dummies...")
        
        # Extrahiere Zeitkomponenten
        df['hour'] = df['timestamp'].dt.hour
        df['day_of_week'] = df['timestamp'].dt.dayofweek
        df['day_of_month'] = df['timestamp'].dt.day
        
        # NFP Day (erster Freitag im Monat)
        df['is_nfp_day'] = ((df['day_of_week'] == 4) & (df['day_of_month'] <= 7)).astype(int)
        
        # CPI Day (ca. 13. des Monats)
        df['is_cpi_day'] = ((df['day_of_month'] >= 12) & (df['day_of_month'] <= 14)).astype(int)
        
        # FOMC Day (8 Mal pro Jahr, ca. alle 6 Wochen)
        # Vereinfachung: Mittwoch in bestimmten Wochen
        df['is_fomc_week'] = 0  # TODO: Tatsächliche FOMC-Termine einpflegen
        
        # Market Hours (US-Handelszeiten)
        df['is_us_trading_hours'] = ((df['hour'] >= 14) & (df['hour'] <= 21)).astype(int)
        
        print(f"  [OK] Event-Dummies erstellt")
        
        return df
    
    # ====================================
    # LABEL-GENERIERUNG
    # ====================================
    
    def generate_labels(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Generiert Labels für Entry-Signal
        
        Labels:
            - LONG (1): Future Return >= ENTRY_THRESHOLD_PCT
            - SHORT (-1): Future Return <= -ENTRY_THRESHOLD_PCT
            - FLAT (0): Sonst
        
        KORRIGIERT: Future Returns werden jetzt korrekt berechnet
        """
        print("  [TOOL] Generiere Labels...")
        
        # KORRIGIERT: Future Returns richtig berechnen
        # ALT (FALSCH): df['close'].shift(-period).pct_change(period)
        # NEU (KORREKT): (df['close'].shift(-period) - df['close']) / df['close']
        for period in config.FUTURE_RETURN_PERIODS:
            df[f'future_return_{period}'] = (df['close'].shift(-period) - df['close']) / df['close']
        
        # Entry-Label (basierend auf kürzestem Future Return)
        future_return = df[f'future_return_{config.FUTURE_RETURN_PERIODS[0]}']
        
        df['label_entry'] = 0  # FLAT
        df.loc[future_return >= (config.ENTRY_THRESHOLD_PCT / 100), 'label_entry'] = 1  # LONG
        df.loc[future_return <= -(config.ENTRY_THRESHOLD_PCT / 100), 'label_entry'] = -1  # SHORT
        
        # Risk Labels: ML-Model sagt Risk-% (0.5-1.5% vom Kapital) vorher
        # FIXED: Reduced from 1-3% to 0.5-1.5% for more conservative, realistic trading
        # Gold typically has ATR_pct of 0.1-0.2%, so these thresholds are appropriate
        df['atr_pct'] = df['atr'] / df['close']
        
        # Conservative risk sizing: 0.5-1.5% per trade
        # Higher volatility -> lower risk to avoid large drawdowns
        df['label_risk_pct'] = 1.0  # Default: 1% vom Kapital
        df.loc[df['atr_pct'] < 0.001, 'label_risk_pct'] = 1.5  # Very low vol -> 1.5% (rare)
        df.loc[df['atr_pct'] > 0.003, 'label_risk_pct'] = 0.5  # High vol -> 0.5%
        
        # SL/TP ATR-Multiplikatoren (ML lernt optimale Werte)
        # Default-Werte basierend auf historischer Performance
        df['label_sl_atr'] = config.DEFAULT_SL_ATR_MULTIPLIER
        df['label_tp_atr'] = config.DEFAULT_TP_ATR_MULTIPLIER
        
        print(f"  [OK] Labels generiert")
        print(f"     LONG: {(df['label_entry'] == 1).sum()} ({(df['label_entry'] == 1).sum() / len(df) * 100:.1f}%)")
        print(f"     SHORT: {(df['label_entry'] == -1).sum()} ({(df['label_entry'] == -1).sum() / len(df) * 100:.1f}%)")
        print(f"     FLAT: {(df['label_entry'] == 0).sum()} ({(df['label_entry'] == 0).sum() / len(df) * 100:.1f}%)")
        
        # Warnung bei extremem Ungleichgewicht
        long_pct = (df['label_entry'] == 1).sum() / len(df) * 100
        short_pct = (df['label_entry'] == -1).sum() / len(df) * 100
        if abs(long_pct - short_pct) > 30:
            print(f"  [WARN] Starkes Klassenungleichgewicht: LONG {long_pct:.1f}% vs SHORT {short_pct:.1f}%")
            print(f"         class_weight='balanced' im RandomForest wird dies ausgleichen")
        
        return df
    
    # ====================================
    # MASTER PIPELINE
    # ====================================
    
    def create_feature_matrix(
        self, 
        candle_data: List[Dict],
        macro_data: Dict[str, List[Dict]],
        sentiment_data: List[Dict],
        fred_data: Dict[str, pd.DataFrame] = None,
        generate_labels: bool = True
    ) -> pd.DataFrame:
        """
        Erstellt vollständige Feature-Matrix
        
        Args:
            candle_data: Gold Candles
            macro_data: Dict mit allen Makro-Datenquellen
            sentiment_data: Sentiment-Artikel
            fred_data: Dict mit FRED macro series (VIX, yields, CPI, etc.)
            generate_labels: Ob Labels generiert werden sollen
        
        Returns:
            DataFrame mit allen Features
        """
        print("\n[TOOL] ERSTELLE FEATURE-MATRIX")
        print("="*60)
        
        # Basis-DataFrame aus Candles
        df = pd.DataFrame(candle_data)
        print(f"[*] {len(df)} Candles geladen")
        
        # Technische Features
        df = self.calculate_technical_features(df)
        
        # FRED Macro Data (VIX, Yields, CPI, NFP, Unemployment, GDP)
        if fred_data:
            df = self.merge_fred_data(df, fred_data)
        
        # Makro-Features (Oil, Silver, Fear & Greed)
        df = self.merge_macro_data(df, macro_data)
        
        # Sentiment-Features
        df = self.merge_sentiment_data(df, sentiment_data)
        
        # Event-Dummies
        df = self.create_event_dummies(df)
        
        # Labels generieren (nur für Training)
        if generate_labels:
            df = self.generate_labels(df)
        
        # NaN-Handling (durch rolling windows entstanden)
        initial_rows = len(df)
        df = df.dropna(subset=['close', 'atr'])  # Mindest-Requirements
        print(f"\n[*] {initial_rows - len(df)} Zeilen wegen NaN entfernt")
        
        print(f"\n[OK] FEATURE-MATRIX ERSTELLT")
        print(f"   Zeilen: {len(df)}")
        print(f"   Spalten: {len(df.columns)}")
        print("="*60)
        
        return df
    
    def get_feature_columns(self, df: pd.DataFrame) -> List[str]:
        """
        Gibt Liste der Feature-Spalten zurück (ohne Labels und Timestamp)
        
        Returns:
            Liste von Feature-Namen
        """
        exclude_cols = [
            'timestamp', 'open', 'high', 'low', 'close', 'volume',
            'label_entry', 'label_risk_pct', 'label_sl_atr', 'label_tp_atr'
        ]
        # Schließe alle future_return Spalten aus (sind Labels, keine Features)
        exclude_cols += [c for c in df.columns if c.startswith('future_return_')]
        
        feature_cols = [c for c in df.columns if c not in exclude_cols]
        
        return feature_cols


if __name__ == "__main__":
    # Test Feature Engineering
    from data_loader import DataManager
    
    dm = DataManager()
    all_data = dm.load_all_data(force_reload=False)
    
    fe = FeatureEngineer()
    
    # Erstelle Feature-Matrix
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
        sentiment_data=[],  # Placeholder
        generate_labels=True
    )
    
    print(f"\n[DATA] Feature-Matrix:")
    print(df.head())
    print(f"\n[DATA] Features: {len(fe.get_feature_columns(df))}")
