"""
Explainability: Erklärt Random Forest Entscheidungen
Feature Importances + lokale Erklärungen
"""
import pandas as pd
import numpy as np
from typing import Dict, List
from models import TradingModels


class ModelExplainer:
    """Erklärt Model-Entscheidungen"""
    
    def __init__(self, models: TradingModels):
        self.models = models
    
    def get_feature_importances(self, top_n: int = 20) -> Dict[str, float]:
        """
        Holt Feature Importances vom Entry-Model
        
        Args:
            top_n: Anzahl Top-Features
        
        Returns:
            Dict mit Feature -> Importance
        """
        if not self.models.entry_model:
            return {}
        
        importances = self.models.entry_model.feature_importances_
        feature_names = self.models.feature_columns
        
        # Sortiere nach Importance
        sorted_idx = np.argsort(importances)[::-1][:top_n]
        
        result = {}
        for idx in sorted_idx:
            result[feature_names[idx]] = float(importances[idx])
        
        return result
    
    def explain_prediction(
        self, 
        features: pd.DataFrame,
        prediction: Dict,
        top_n: int = 10
    ) -> str:
        """
        Erstellt textuelle Erklärung für eine Prediction
        
        Args:
            features: Feature-DataFrame (1 Zeile)
            prediction: Prediction-Dict von models.predict()
            top_n: Anzahl Top-Features zu zeigen
        
        Returns:
            Erklärungstext
        """
        # Hole Feature-Importances
        importances = self.get_feature_importances(top_n=top_n)
        
        # Aktueller Preis und ATR
        current_price = features['close'].values[0] if 'close' in features.columns else 0
        atr = features['atr'].values[0] if 'atr' in features.columns else 0
        
        # Baue Erklärung
        explanation = []
        explanation.append("=" * 70)
        explanation.append("[*] RANDOM FOREST ENTSCHEIDUNG")
        explanation.append("=" * 70)
        
        # Entry-Signal
        signal_name = prediction['entry_signal_name']
        confidence = prediction['confidence'] * 100
        
        explanation.append(f"\n[DATA] SIGNAL: {signal_name}")
        explanation.append(f"   Confidence: {confidence:.1f}%")
        explanation.append(f"   Proba: LONG={prediction['entry_proba']['LONG']*100:.1f}% | "
                         f"FLAT={prediction['entry_proba']['FLAT']*100:.1f}% | "
                         f"SHORT={prediction['entry_proba']['SHORT']*100:.1f}%")
        
        # Risk Management (nur wenn nicht FLAT)
        if prediction['entry_signal'] != 0:
            explanation.append(f"\n[*] RISK MANAGEMENT:")

            # Position Size nur anzeigen, wenn vorhanden (kommt aus RiskEngine)
            position_size = prediction.get('position_size')
            if position_size is not None:
                explanation.append(f"   Position Size: {position_size} units")

            # Berechne SL/TP Preise nur, wenn die Multiplikatoren vorhanden sind
            sl_mult = prediction.get('sl_atr_multiplier')
            tp_mult = prediction.get('tp_atr_multiplier')
            if sl_mult is not None and tp_mult is not None and atr > 0:
                sl_distance = atr * sl_mult
                tp_distance = atr * tp_mult

                if prediction['entry_signal'] == 1:  # LONG
                    sl_price = current_price - sl_distance
                    tp_price = current_price + tp_distance
                else:  # SHORT
                    sl_price = current_price + sl_distance
                    tp_price = current_price - tp_distance

                explanation.append(f"   Stop Loss: {sl_price:.2f} (ATR Mult: {sl_mult:.2f})")
                explanation.append(f"   Take Profit: {tp_price:.2f} (ATR Mult: {tp_mult:.2f})")
                if sl_mult != 0:
                    explanation.append(f"   Risk/Reward: 1:{tp_mult/sl_mult:.2f}")
        
        # Feature-Begründung
        explanation.append(f"\n[TARGET] WICHTIGSTE EINFLUSSFAKTOREN (Top {top_n}):")
        explanation.append("")
        
        for i, (feature_name, importance) in enumerate(importances.items(), 1):
            # Hole aktuellen Wert des Features
            if feature_name in features.columns:
                value = features[feature_name].values[0]
                
                # Formatierung je nach Feature
                if 'pct' in feature_name or 'return' in feature_name:
                    value_str = f"{value*100:.2f}%"
                elif 'sentiment' in feature_name:
                    value_str = f"{value:.3f}"
                elif feature_name in ['rsi']:
                    value_str = f"{value:.1f}"
                else:
                    value_str = f"{value:.4f}"
                
                explanation.append(f"   {i}. {feature_name}")
                explanation.append(f"      Importance: {importance:.4f} | Aktueller Wert: {value_str}")
        
        # Marktkontext
        explanation.append(f"\n[*] MARKTKONTEXT:")
        
        # Technische Indikatoren
        if 'rsi' in features.columns:
            rsi = features['rsi'].values[0]
            rsi_status = "überkauft" if rsi > 70 else "überverkauft" if rsi < 30 else "neutral"
            explanation.append(f"   RSI: {rsi:.1f} ({rsi_status})")
        
        if 'dist_to_sma_200' in features.columns:
            dist_sma = features['dist_to_sma_200'].values[0] * 100
            sma_status = "über" if dist_sma > 0 else "unter"
            explanation.append(f"   Preis {sma_status} 200-SMA: {abs(dist_sma):.2f}%")
        
        if 'bb_position' in features.columns:
            bb_pos = features['bb_position'].values[0]
            if bb_pos > 0.8:
                bb_status = "nahe oberem Band"
            elif bb_pos < 0.2:
                bb_status = "nahe unterem Band"
            else:
                bb_status = "im mittleren Bereich"
            explanation.append(f"   Bollinger Band Position: {bb_pos:.2f} ({bb_status})")
        
        # Volatilität
        if 'atr_pct' in features.columns:
            atr_pct = features['atr_pct'].values[0]
            explanation.append(f"   Volatilität (ATR): {atr_pct:.2f}%")
        
        # Sentiment
        if 'sentiment_score' in features.columns:
            sent = features['sentiment_score'].values[0]
            sent_status = "bullish" if sent > 0.1 else "bearish" if sent < -0.1 else "neutral"
            explanation.append(f"   Sentiment: {sent:.3f} ({sent_status})")
        
        # Makro
        if 'vix' in features.columns:
            vix = features['vix'].values[0]
            vix_status = "hoch" if vix > 20 else "niedrig"
            explanation.append(f"   VIX: {vix:.1f} ({vix_status})")
        
        explanation.append("\n" + "=" * 70)
        
        return "\n".join(explanation)
    
    def create_feature_importance_summary(self) -> pd.DataFrame:
        """
        Erstellt DataFrame mit Feature Importances
        
        Returns:
            DataFrame mit allen Features und Importances
        """
        if not self.models.entry_model:
            return pd.DataFrame()
        
        importances = self.models.entry_model.feature_importances_
        feature_names = self.models.feature_columns
        
        df = pd.DataFrame({
            'feature': feature_names,
            'importance': importances
        })
        
        df = df.sort_values('importance', ascending=False).reset_index(drop=True)
        
        return df
    
    def print_top_features(self, top_n: int = 20):
        """Gibt Top-Features aus"""
        importances = self.get_feature_importances(top_n=top_n)
        
        print(f"\n[TARGET] TOP {top_n} FEATURES (Feature Importance)")
        print("=" * 60)
        
        for i, (feature, importance) in enumerate(importances.items(), 1):
            print(f"{i:2d}. {feature:40s} {importance:.6f}")
        
        print("=" * 60)


if __name__ == "__main__":
    print("Explainability-Modul")
    print("Wird von backtest.py und live_trading.py verwendet")
