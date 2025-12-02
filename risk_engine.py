"""
Risk Engine: Kapitalgebundenes Risikomanagement
Berechnet Position Sizes basierend auf verfügbarem Kapital und Risk-per-Trade
"""
import numpy as np
from typing import Dict, Optional
import config


class RiskEngine:
    """
    Kapitalgebundene Risk Engine mit Spread + Slippage
    
    Berechnet realistische Position Sizes basierend auf:
    - Verfügbarem Kapital
    - Risiko pro Trade (% des Kapitals)
    - Stop-Loss-Distanz
    - Maximum Leverage
    - Spread and Slippage Costs (no commissions)
    """
    
    def __init__(
        self,
        risk_per_trade_pct: float = None,
        max_leverage: float = None,
        min_capital: float = None
    ):
        """
        Args:
            risk_per_trade_pct: Risiko pro Trade in % (default: aus config)
            max_leverage: Maximales Leverage (default: aus config)
            min_capital: Minimum Kapital für Trading (default: aus config)
        """
        self.risk_per_trade_pct = risk_per_trade_pct or config.RISK_PER_TRADE_PCT
        self.max_leverage = max_leverage or config.MAX_LEVERAGE
        self.min_capital = min_capital or config.MIN_CAPITAL_FOR_TRADING
        
        # Risk Multipliers für verschiedene Kategorien
        self.risk_multipliers = {
            'conservative': 0.5,  # 50% des Standard-Risikos
            'normal': 1.0,        # 100% des Standard-Risikos
            'aggressive': 1.5     # 150% des Standard-Risikos
        }
        
        # SL/TP ATR-Multiplikatoren pro Kategorie
        self.sl_atr_multipliers = {
            'conservative': 2.5,
            'normal': 2.0,
            'aggressive': 1.5
        }
        
        self.tp_atr_multipliers = {
            'conservative': 5.0,
            'normal': 3.0,
            'aggressive': 2.5
        }
    
    def calculate_effective_entry_price(
        self,
        mid_price: float,
        direction: str,
        atr: float
    ) -> tuple:
        """
        Calculate effective entry price including spread and slippage
        
        LONG:  entry = mid + spread/2 + slippage
        SHORT: entry = mid - spread/2 - slippage
        
        Args:
            mid_price: Mid price (typically close price)
            direction: 'LONG' or 'SHORT'
            atr: Average True Range for slippage calculation
        
        Returns:
            tuple: (effective_entry_price, spread_half, slippage)
        """
        spread_half = config.SPREAD_PER_OUNCE / 2.0
        slippage = atr * config.ATR_SLIPPAGE_MULT
        
        if direction == 'LONG':
            # LONG pays spread and slippage on entry
            effective_entry = mid_price + spread_half + slippage
        else:  # SHORT
            # SHORT receives less on entry due to spread and slippage
            effective_entry = mid_price - spread_half - slippage
        
        return effective_entry, spread_half, slippage
    
    def calculate_effective_exit_price(
        self,
        mid_price: float,
        direction: str,
        atr: float
    ) -> tuple:
        """
        Calculate effective exit price including spread and slippage
        
        LONG:  exit = mid - spread/2 - slippage
        SHORT: exit = mid + spread/2 + slippage
        
        Args:
            mid_price: Mid price (typically close price)
            direction: 'LONG' or 'SHORT'
            atr: Average True Range for slippage calculation
        
        Returns:
            tuple: (effective_exit_price, spread_half, slippage)
        """
        spread_half = config.SPREAD_PER_OUNCE / 2.0
        slippage = atr * config.ATR_SLIPPAGE_MULT
        
        if direction == 'LONG':
            # LONG pays spread and slippage on exit
            effective_exit = mid_price - spread_half - slippage
        else:  # SHORT
            # SHORT pays spread and slippage on exit
            effective_exit = mid_price + spread_half + slippage
        
        return effective_exit, spread_half, slippage
    
    def calculate_position(
        self,
        capital: float,
        entry_price: float,
        atr: float,
        direction: str,  # 'LONG' or 'SHORT'
        risk_category: str = 'normal',
        min_rr_ratio: float = None,
        ml_risk_pct: Optional[float] = None,
        ml_sl_atr_mult: Optional[float] = None,
        ml_tp_atr_mult: Optional[float] = None
    ) -> Optional[Dict]:
        """
        Berechnet Position mit realistischer Kapitalanbindung + Spread/Slippage
        
        Args:
            capital: Verfügbares Kapital
            entry_price: Mid-Preis (candle close) - NOT including spread/slippage yet
            atr: Average True Range (für SL/TP und slippage)
            direction: 'LONG' oder 'SHORT'
            risk_category: 'conservative', 'normal', oder 'aggressive'
            min_rr_ratio: Minimum Risk-Reward Ratio (default: aus config)
            ml_risk_pct: ML-vorhergesagtes Risiko in % (1-3%), überschreibt risk_category
            ml_sl_atr_mult: ML-vorhergesagter SL ATR-Multiplikator
            ml_tp_atr_mult: ML-vorhergesagter TP ATR-Multiplikator
        
        Returns:
            Dict mit Position-Details oder None wenn Trade nicht möglich
        """
        min_rr_ratio = min_rr_ratio or config.MIN_RISK_REWARD
        
        # Prüfe Minimum-Kapital
        if capital < self.min_capital:
            return None
        
        # STEP 1: Calculate effective entry price (mid + spread/2 ± slippage)
        effective_entry, spread_half, slippage = self.calculate_effective_entry_price(
            entry_price, direction, atr
        )
        
        # Verwende ML-Predictions wenn vorhanden, sonst Defaults
        if ml_risk_pct is not None:
            # ML-Model gibt direktes Risiko-% (1-3%)
            risk_mult = ml_risk_pct / self.risk_per_trade_pct  # Normalisiere auf Standard
            risk_mult = np.clip(risk_mult, 0.5, 3.0)  # Begrenze auf 0.5-3x Standard
        else:
            # Fallback auf risk_category
            risk_mult = self.risk_multipliers.get(risk_category, 1.0)
        
        if ml_sl_atr_mult is not None:
            sl_mult = np.clip(ml_sl_atr_mult, 0.5, 5.0)  # Begrenze SL
        else:
            sl_mult = self.sl_atr_multipliers.get(risk_category, 2.0)
        
        if ml_tp_atr_mult is not None:
            tp_mult = np.clip(ml_tp_atr_mult, 1.0, 10.0)  # Begrenze TP
        else:
            tp_mult = self.tp_atr_multipliers.get(risk_category, 3.0)
        
        # Prüfe Risk-Reward Ratio
        rr_ratio = tp_mult / sl_mult
        if rr_ratio < min_rr_ratio:
            # Adjustiere TP um Minimum RR zu erreichen
            tp_mult = sl_mult * min_rr_ratio
        
        # STEP 2: Berechne SL/TP Preise (von EFFECTIVE entry price)
        sl_distance = atr * sl_mult
        tp_distance = atr * tp_mult
        
        if direction == 'LONG':
            stop_loss = effective_entry - sl_distance
            take_profit = effective_entry + tp_distance
        else:  # SHORT
            stop_loss = effective_entry + sl_distance
            take_profit = effective_entry - tp_distance
        
        # STEP 3: KAPITALGEBUNDENE POSITION SIZE BERECHNUNG
        # Berechne erlaubtes Risiko
        allowed_risk = capital * (self.risk_per_trade_pct / 100) * risk_mult
        
        # SAFETY CAP: Never risk more than 2% of capital, even if ML says so
        # This prevents exponential growth bugs
        max_risk_cap = capital * 0.02  # Hard cap at 2%
        allowed_risk = min(allowed_risk, max_risk_cap)
        
        # STEP 4: Position Size basierend auf Risiko und SL-Distanz
        # Formel: position_size = allowed_risk / sl_distance_per_unit
        position_size = allowed_risk / sl_distance
        
        # STEP 5: Position Size Limits
        # CRITICAL FIX: Apply THREE caps to prevent exponential growth
        
        # Cap 1: Leverage limit (traditional) - use EFFECTIVE entry price
        max_notional = capital * self.max_leverage
        max_position_by_leverage = max_notional / effective_entry
        
        # Cap 2: MAX_POSITION_PCT limit (NEW - prevents using entire capital)
        max_position_pct = getattr(config, 'MAX_POSITION_PCT', 0.15)  # Default 15%
        max_position_by_pct = (capital * max_position_pct) / effective_entry
        
        # Take the SMALLEST of: risk-based, leverage-based, OR percentage-based
        position_size = min(position_size, max_position_by_leverage, max_position_by_pct)
        
        # Recalculate actual risk based on final position size
        actual_risk = position_size * sl_distance
        
        # Final notional value check - use EFFECTIVE entry price
        notional_value = abs(position_size) * effective_entry
        
        # STEP 6: Minimum Position Size Check
        if position_size < config.MIN_POSITION_SIZE:
            return None
        
        # STEP 7: Position Size für SHORT negativ machen
        units = position_size if direction == 'LONG' else -position_size
        
        # Berechne Metriken
        actual_risk_pct = (actual_risk / capital) * 100
        potential_reward = abs(take_profit - entry_price) * position_size
        potential_reward_pct = (potential_reward / capital) * 100
        actual_leverage = notional_value / capital
        
        # SANITY CHECKS - reject unrealistic positions
        if actual_risk_pct > 5.0:
            # Risk exceeds 5% - something is wrong
            print(f"[RISK ENGINE WARN] Risk {actual_risk_pct:.1f}% exceeds 5% cap - rejecting trade")
            return None
        
        if actual_leverage > 1.2:
            # Leverage exceeds configured max (with 20% buffer)
            print(f"[RISK ENGINE WARN] Leverage {actual_leverage:.2f}x exceeds 1.2x cap - rejecting trade")
            return None
        
        if position_size * entry_price > capital * 1.5:
            # Notional exceeds 150% of capital - unrealistic
            print(f"[RISK ENGINE WARN] Notional {notional_value:.0f} > 150% of capital {capital:.0f} - rejecting trade")
            return None
        
        return {
            'direction': direction,
            'mid_price': float(entry_price),  # Original mid price (close)
            'entry_price': float(effective_entry),  # EFFECTIVE entry (mid + spread/2 ± slippage)
            'spread_half': float(spread_half),
            'slippage': float(slippage),
            'stop_loss': float(stop_loss),
            'take_profit': float(take_profit),
            'position_size': float(position_size),
            'units': float(units),
            'notional_value': float(notional_value),
            'actual_leverage': float(actual_leverage),
            'sl_distance': float(sl_distance),
            'tp_distance': float(tp_distance),
            'sl_atr_multiplier': float(sl_mult),
            'tp_atr_multiplier': float(tp_mult),
            'risk_category': risk_category,
            'allowed_risk': float(allowed_risk),
            'actual_risk': float(actual_risk),
            'actual_risk_pct': float(actual_risk_pct),
            'potential_reward': float(potential_reward),
            'potential_reward_pct': float(potential_reward_pct),
            'risk_reward_ratio': float(tp_mult / sl_mult)
        }
    
    def validate_position(self, position: Dict, capital: float, debug=False) -> bool:
        """
        Validiert ob Position mit verfügbarem Kapital handelbar ist
        
        Args:
            position: Position Dict von calculate_position()
            capital: Verfügbares Kapital
            debug: If True, print rejection reason
        
        Returns:
            True wenn Position valide, sonst False
        """
        if not position:
            if debug: print(f"[REJECT] position=None")
            return False
        
        # Check 1: Minimum Kapital
        if capital < self.min_capital:
            if debug: print(f"[REJECT] capital {capital:.2f} < min_capital {self.min_capital}")
            return False
        
        # Check 2: Leverage Limit
        max_notional = capital * self.max_leverage
        if position['notional_value'] > max_notional:
            if debug: print(f"[REJECT] notional {position['notional_value']:.2f} > max {max_notional:.2f}")
            return False
        
        # Check 3: Risiko nicht größer als Kapital
        if position['actual_risk'] > capital:
            if debug: print(f"[REJECT] risk {position['actual_risk']:.2f} > capital {capital:.2f}")
            return False
        
        # Check 4: Minimum Position Size
        if position['position_size'] < config.MIN_POSITION_SIZE:
            if debug: print(f"[REJECT] position_size {position['position_size']:.4f} < min {config.MIN_POSITION_SIZE}")
            return False
        
        return True
    
    def calculate_pnl(
        self,
        position: Dict,
        exit_price: float,
        commission: float = 0.0
    ) -> Dict:
        """
        Berechnet P&L für geschlossene Position
        
        Args:
            position: Position Dict
            exit_price: Exit-Preis
            commission: Kommission/Gebühren
        
        Returns:
            Dict mit P&L Details
        """
        entry_price = position['entry_price']
        units = position['units']
        position_size = position['position_size']
        
        # P&L berechnen (für LONG und SHORT)
        if position['direction'] == 'LONG':
            price_change = exit_price - entry_price
        else:  # SHORT
            price_change = entry_price - exit_price
        
        gross_pnl = price_change * position_size
        net_pnl = gross_pnl - commission
        
        # Return on Risk
        ror = (net_pnl / position['actual_risk']) * 100 if position['actual_risk'] > 0 else 0
        
        return {
            'gross_pnl': float(gross_pnl),
            'commission': float(commission),
            'net_pnl': float(net_pnl),
            'return_on_risk_pct': float(ror),
            'price_change': float(price_change),
            'price_change_pct': float((price_change / entry_price) * 100)
        }
    
    def get_risk_category_from_volatility(
        self,
        atr: float,
        price: float,
        low_threshold: float = 0.005,  # 0.5%
        high_threshold: float = 0.015   # 1.5%
    ) -> str:
        """
        Bestimmt Risk Category basierend auf Volatilität
        
        Args:
            atr: Average True Range
            price: Aktueller Preis
            low_threshold: Threshold für niedrige Volatilität
            high_threshold: Threshold für hohe Volatilität
        
        Returns:
            'conservative', 'normal', oder 'aggressive'
        """
        atr_pct = atr / price
        
        if atr_pct < low_threshold:
            # Niedrige Volatilität -> aggressiver handeln
            return 'aggressive'
        elif atr_pct > high_threshold:
            # Hohe Volatilität -> konservativer handeln
            return 'conservative'
        else:
            return 'normal'


if __name__ == "__main__":
    # Test der Risk Engine
    print("="*70)
    print("[TEST] RISK ENGINE")
    print("="*70)
    
    engine = RiskEngine()
    
    # Test-Szenarien
    test_scenarios = [
        {
            'name': 'Normal LONG Trade',
            'capital': 10000,
            'entry_price': 2000,
            'atr': 10,
            'direction': 'LONG',
            'risk_category': 'normal'
        },
        {
            'name': 'Conservative SHORT Trade',
            'capital': 10000,
            'entry_price': 2000,
            'atr': 10,
            'direction': 'SHORT',
            'risk_category': 'conservative'
        },
        {
            'name': 'Aggressive LONG Trade',
            'capital': 10000,
            'entry_price': 2000,
            'atr': 10,
            'direction': 'LONG',
            'risk_category': 'aggressive'
        },
        {
            'name': 'Nach Verlust (5k Capital)',
            'capital': 5000,
            'entry_price': 2000,
            'atr': 10,
            'direction': 'LONG',
            'risk_category': 'normal'
        },
        {
            'name': 'Hohe Volatilität',
            'capital': 10000,
            'entry_price': 2000,
            'atr': 30,  # 3x normale Volatilität
            'direction': 'LONG',
            'risk_category': 'normal'
        }
    ]
    
    for scenario in test_scenarios:
        print(f"\n[TEST] {scenario['name']}")
        print("-"*70)
        
        position = engine.calculate_position(
            capital=scenario['capital'],
            entry_price=scenario['entry_price'],
            atr=scenario['atr'],
            direction=scenario['direction'],
            risk_category=scenario['risk_category']
        )
        
        if position:
            print(f"✓ Position erfolgreich berechnet")
            print(f"  Position Size: {position['position_size']:.2f} units")
            print(f"  Notional Value: ${position['notional_value']:,.2f}")
            print(f"  Leverage: {position['actual_leverage']:.2f}x")
            print(f"  Stop Loss: ${position['stop_loss']:.2f}")
            print(f"  Take Profit: ${position['take_profit']:.2f}")
            print(f"  Risk: ${position['actual_risk']:.2f} ({position['actual_risk_pct']:.2f}%)")
            print(f"  Potential Reward: ${position['potential_reward']:.2f} ({position['potential_reward_pct']:.2f}%)")
            print(f"  R:R Ratio: 1:{position['risk_reward_ratio']:.2f}")
            
            # Validierung
            is_valid = engine.validate_position(position, scenario['capital'])
            print(f"  Valid: {'✓' if is_valid else '✗'}")
        else:
            print(f"✗ Keine Position möglich (zu wenig Kapital oder andere Constraints)")
    
    print("\n" + "="*70)
    print("[OK] RISK ENGINE TESTS ABGESCHLOSSEN")
    print("="*70)
