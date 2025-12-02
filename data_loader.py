"""
Data Loader mit inkrementellem Update-Support
Speichert alle Daten in JSON und lädt nur neue Daten nach
"""
import json
import os
from datetime import datetime, timedelta
from typing import List, Dict, Optional
import requests
import config
from oanda_client import OandaClient


class DataManager:
    """Verwaltet alle Datenquellen mit inkrementellem Update"""
    
    def __init__(self):
        self.oanda_client = OandaClient()
        self._ensure_directories()
    
    def _ensure_directories(self):
        """Stellt sicher, dass alle Verzeichnisse existieren"""
        os.makedirs(config.DATA_DIR, exist_ok=True)
        os.makedirs(config.MODELS_DIR, exist_ok=True)
        os.makedirs(config.LOGS_DIR, exist_ok=True)
    
    def _load_json(self, filepath: str) -> Dict:
        """
        Lädt JSON-Datei
        
        Returns:
            Dict mit "data"-Key oder leeres Dict
        """
        if os.path.exists(filepath):
            try:
                with open(filepath, 'r', encoding='utf-8') as f:
                    return json.load(f)
            except Exception as e:
                print(f"[WARN] Fehler beim Laden von {filepath}: {e}")
                return {"data": []}
        return {"data": []}
    
    def _save_json(self, filepath: str, data: Dict):
        """Speichert Daten als JSON"""
        try:
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
            print(f"[SAVE] Gespeichert: {filepath} ({len(data.get('data', []))} Einträge)")
        except Exception as e:
            print(f"[ERROR] Fehler beim Speichern von {filepath}: {e}")
    
    def _get_last_timestamp(self, data_list: List[Dict]) -> Optional[datetime]:
        """
        Findet den letzten Timestamp in einer Datenliste
        
        Args:
            data_list: Liste mit Dicts, die "timestamp"-Key enthalten
        
        Returns:
            Letzter Timestamp als datetime oder None
        """
        if not data_list:
            return None
        
        try:
            last_entry = data_list[-1]
            timestamp_str = last_entry.get("timestamp", "")
            # Versuche verschiedene Timestamp-Formate
            for fmt in ["%Y-%m-%dT%H:%M:%S.%fZ", "%Y-%m-%d %H:%M:%S", "%Y-%m-%d"]:
                try:
                    return datetime.strptime(timestamp_str, fmt)
                except ValueError:
                    continue
            return None
        except Exception:
            return None
    
    # ====================================
    # OANDA CANDLES
    # ====================================
    
    def load_oanda_candles(
        self, 
        instrument: str, 
        filepath: str,
        force_reload: bool = False
    ) -> List[Dict]:
        """
        Lädt Oanda Candles mit inkrementellem Update
        
        Args:
            instrument: z.B. "XAU_USD"
            filepath: Pfad zur JSON-Datei
            force_reload: Wenn True, lädt alle Daten neu
        
        Returns:
            Liste aller Candles
        """
        print(f"\n[DATA] Lade {instrument} Candles...")
        
        # Existierende Daten laden
        existing_data = self._load_json(filepath)
        existing_candles = existing_data.get("data", [])
        
        if force_reload or not existing_candles:
            print(f"  [RELOAD] Lade ALLE Daten ab {config.DATA_START_DATE}...")
            start_datetime = datetime.strptime(config.DATA_START_DATE, "%Y-%m-%d")
            all_candles = self.oanda_client.fetch_all_candles(
                instrument=instrument,
                start_date=start_datetime,
                granularity=config.GRANULARITY
            )
            self._save_json(filepath, {"data": all_candles})
            return all_candles
        
        else:
            # Inkrementelles Update
            last_timestamp = self._get_last_timestamp(existing_candles)
            if not last_timestamp:
                print("  [WARN] Kann letzten Timestamp nicht parsen, lade alle Daten neu...")
                return self.load_oanda_candles(instrument, filepath, force_reload=True)
            
            print(f"  [RELOAD] Letzter Timestamp: {last_timestamp.strftime('%Y-%m-%d %H:%M')}")
            print(f"  [RELOAD] Lade nur neue Daten bis zum letzten verfügbaren Candle...")
            
            new_candles = self.oanda_client.fetch_candles(
                instrument=instrument,
                granularity=config.GRANULARITY,
                from_time=last_timestamp + timedelta(minutes=15),
                to_time=datetime.utcnow()
            )
            
            if new_candles:
                print(f"  [OK] {len(new_candles)} neue Candles gefunden")
                all_candles = existing_candles + new_candles
                self._save_json(filepath, {"data": all_candles})
                return all_candles
            else:
                print(f"  [*] Keine neuen Daten verfügbar")
                return existing_candles
    
    def load_gold_candles(self, force_reload: bool = False) -> List[Dict]:
        """Lädt Gold (XAUUSD) Candles"""
        return self.load_oanda_candles(
            config.GOLD_INSTRUMENT, 
            config.GOLD_CANDLES_FILE,
            force_reload
        )
    
    def load_oil_candles(self, force_reload: bool = False) -> List[Dict]:
        """Lädt Oil (WTI/Brent) Candles"""
        return self.load_oanda_candles(
            config.OIL_INSTRUMENT, 
            config.OIL_CANDLES_FILE,
            force_reload
        )
    
    def load_silver_candles(self, force_reload: bool = False) -> List[Dict]:
        """Lädt Silver (XAGUSD) Candles"""
        return self.load_oanda_candles(
            config.SILVER_INSTRUMENT, 
            config.SILVER_CANDLES_FILE,
            force_reload
        )
    # ====================================
    # VIX, YIELDS, MACRO DATA
    # ====================================
    
    def load_vix_data(self, force_reload: bool = False) -> List[Dict]:
        """Lädt VIX-Daten (silent loading)"""
        existing_data = self._load_json(config.VIX_FILE)
        if not existing_data.get("data"):
            placeholder_data = {
                "data": [{"timestamp": "2005-01-01", "vix": 12.5, "source": "placeholder"}],
                "info": "VIX data requires FRED API or manual data import"
            }
            self._save_json(config.VIX_FILE, placeholder_data)
            return placeholder_data["data"]
        return existing_data.get("data", [])
    
    def load_yields_data(self, force_reload: bool = False) -> List[Dict]:
        """Lädt US 10Y Treasury Yields (silent loading)"""
        existing_data = self._load_json(config.YIELDS_FILE)
        if not existing_data.get("data"):
            placeholder_data = {
                "data": [{"timestamp": "2005-01-01", "yield_10y": 4.2, "source": "placeholder"}],
                "info": "Yields data requires FRED API or manual data import"
            }
            self._save_json(config.YIELDS_FILE, placeholder_data)
            return placeholder_data["data"]
        return existing_data.get("data", [])
    
    def load_fear_greed_data(self, force_reload: bool = False) -> List[Dict]:
        """Lädt Fear & Greed Index (ONLY for historical training data)"""
        existing_data = self._load_json(config.FEAR_GREED_FILE)
        if not existing_data.get("data") or len(existing_data.get("data", [])) <= 1:
            placeholder_data = {
                "data": [{"timestamp": "2005-01-01", "fear_greed": 50, "classification": "Neutral"}],
                "info": "Historical Fear & Greed - For live trading use fear_greed.py scraper"
            }
            self._save_json(config.FEAR_GREED_FILE, placeholder_data)
            return placeholder_data["data"]
        return existing_data.get("data", [])
    
    def load_macro_data(self, force_reload: bool = False) -> List[Dict]:
        """Lädt Makro-Daten (silent loading)"""
        existing_data = self._load_json(config.MACRO_FILE)
        if not existing_data.get("data"):
            placeholder_data = {
                "data": [{
                    "timestamp": "2005-01-01",
                    "cpi": 100.0,
                    "inflation": 2.5,
                    "nfp": 150000,
                    "unemployment": 5.1,
                    "gdp_growth": 3.2,
                    "source": "placeholder"
                }],
                "info": "Macro data requires FRED API or manual import"
            }
            self._save_json(config.MACRO_FILE, placeholder_data)
            return placeholder_data["data"]
        return existing_data.get("data", [])
    
    def load_pizza_index(self, force_reload: bool = False) -> List[Dict]:
        """Lädt Pentagon Pizza Index (silent loading)"""
        existing_data = self._load_json(config.PIZZA_FILE)
        if not existing_data.get("data"):
            placeholder_data = {
                "data": [{"timestamp": "2005-01-01", "pizza_orders": 0, "event": "None"}],
                "info": "Pentagon Pizza Index must be manually maintained"
            }
            self._save_json(config.PIZZA_FILE, placeholder_data)
            return placeholder_data["data"]
        return existing_data.get("data", [])
    
    
    # ====================================
    # MASTER LOAD FUNCTION
    # ====================================
    
    def load_all_data(self, force_reload: bool = False) -> Dict[str, List[Dict]]:
        """
        Lädt ALLE Daten mit inkrementellem Update
        
        Args:
            force_reload: Wenn True, lädt alle Daten komplett neu
        
        Returns:
            Dict mit allen Datenquellen
        """
        print("\n" + "="*60)
        print("[*] LADE ALLE DATEN")
        print("="*60)
        
        # Load main candle data (with logging)
        data = {
            "gold_candles": self.load_gold_candles(force_reload),
            "oil_candles": self.load_oil_candles(force_reload),
            "silver_candles": self.load_silver_candles(force_reload)
        }
        
        # Load additional data sources silently (no print spam)
        data["vix"] = self.load_vix_data(force_reload)
        data["yields"] = self.load_yields_data(force_reload)
        data["fear_greed"] = self.load_fear_greed_data(force_reload)
        data["macro"] = self.load_macro_data(force_reload)
        data["pizza"] = self.load_pizza_index(force_reload)
        
        print("\n" + "="*60)
        print("[OK] ALLE DATEN GELADEN")
        print("="*60)
        # Only print main candle data counts (not the placeholder data)
        for key in ["gold_candles", "oil_candles", "silver_candles"]:
            print(f"  {key}: {len(data[key])} Einträge")
        
        return data


if __name__ == "__main__":
    # Test des Data Managers
    dm = DataManager()
    
    # Lade alle Daten (inkrementell)
    all_data = dm.load_all_data(force_reload=False)
    
    print("\n[DATA] Beispiel Gold Candle:")
    if all_data["gold_candles"]:
        print(json.dumps(all_data["gold_candles"][0], indent=2))
