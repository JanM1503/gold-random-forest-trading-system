"""
Oanda API Client für Candle-Fetching und Order-Platzierung
Verwendet Oanda v20 REST API
"""
import requests
import json
from datetime import datetime, timedelta
from typing import List, Dict, Optional
import config


class OandaClient:
    """Client für Oanda v20 API"""
    
    def __init__(self):
        self.api_key = config.OANDA_API_KEY
        self.account_id = config.OANDA_ACCOUNT_ID
        self.api_url = config.OANDA_API_URL
        self.headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
    
    def fetch_candles(
        self, 
        instrument: str, 
        granularity: str = "M15",
        from_time: Optional[datetime] = None,
        to_time: Optional[datetime] = None,
        count: int = 5000
    ) -> List[Dict]:
        """
        Lädt Candles von Oanda API
        
        Args:
            instrument: z.B. "XAU_USD"
            granularity: z.B. "M15" für 15-Minuten
            from_time: Start-Zeitpunkt
            to_time: End-Zeitpunkt
            count: Max Anzahl Candles (max 5000 pro Request)
        
        Returns:
            Liste von Candle-Dictionaries
        """
        url = f"{self.api_url}/instruments/{instrument}/candles"
        
        params = {
            "granularity": granularity,
            "price": "M",  # Mid prices
            "count": count  # Immer count setzen
        }
        
        if from_time:
            params["from"] = from_time.strftime("%Y-%m-%dT%H:%M:%S.000000000Z")
        if to_time:
            params["to"] = to_time.strftime("%Y-%m-%dT%H:%M:%S.000000000Z")
        
        try:
            response = requests.get(url, headers=self.headers, params=params)
            response.raise_for_status()
            data = response.json()
            
            candles = []
            for candle in data.get("candles", []):
                if candle["complete"]:  # Nur komplette Candles
                    candles.append({
                        "timestamp": candle["time"],
                        "open": float(candle["mid"]["o"]),
                        "high": float(candle["mid"]["h"]),
                        "low": float(candle["mid"]["l"]),
                        "close": float(candle["mid"]["c"]),
                        "volume": int(candle["volume"])
                    })
            
            return candles
        
        except requests.exceptions.RequestException as e:
            print(f"[ERROR] Fehler beim Laden von Candles für {instrument}: {e}")
            return []
    
    def fetch_all_candles(
        self, 
        instrument: str, 
        start_date: datetime,
        granularity: str = "M15"
    ) -> List[Dict]:
        """
        Lädt ALLE Candles ab einem Startdatum
        Oanda erlaubt max 5000 Candles pro Request, daher in Batches laden
        
        Args:
            instrument: z.B. "XAU_USD"
            start_date: Ab diesem Datum laden
            granularity: z.B. "M15"
        
        Returns:
            Liste aller Candles
        """
        print(f"[DOWNLOAD] Lade alle {instrument} Candles ab {start_date.strftime('%Y-%m-%d')}...")
        
        all_candles = []
        current_from = start_date
        now = datetime.utcnow()
        
        # Batch-Weise laden (5000 Candles pro Request)
        batch_num = 0
        while current_from < now:
            batch_num += 1
            print(f"  Batch {batch_num}: ab {current_from.strftime('%Y-%m-%d %H:%M')}...", end="")
            
            candles = self.fetch_candles(
                instrument=instrument,
                granularity=granularity,
                from_time=current_from,
                count=5000
            )
            
            if not candles:
                print(" keine weiteren Daten")
                break
            
            all_candles.extend(candles)
            print(f" {len(candles)} Candles geladen")
            
            # Nächster Batch startet nach dem letzten Candle
            # Oanda liefert Nanosekunden (9 Stellen), Python unterstützt nur Mikrosekunden (6 Stellen)
            timestamp_str = candles[-1]["timestamp"]
            # Kürze auf Mikrosekunden wenn nötig
            if "." in timestamp_str:
                parts = timestamp_str.split(".")
                microseconds = parts[1].rstrip("Z")[:6]  # Max 6 Stellen
                timestamp_str = f"{parts[0]}.{microseconds}Z"
            last_timestamp = datetime.strptime(timestamp_str, "%Y-%m-%dT%H:%M:%S.%fZ")
            current_from = last_timestamp + timedelta(seconds=1)
            
            # Prevent zu viele Requests
            if len(candles) < 5000:
                break
        
        print(f"[OK] {len(all_candles)} Candles für {instrument} geladen")
        return all_candles
    
    def get_latest_candle(self, instrument: str, granularity: str = "M15") -> Optional[Dict]:
        """
        Holt den neuesten abgeschlossenen Candle
        
        Args:
            instrument: z.B. "XAU_USD"
            granularity: z.B. "M15"
        
        Returns:
            Neuester Candle oder None
        """
        candles = self.fetch_candles(instrument, granularity, count=1)
        return candles[0] if candles else None
    
    def place_market_order(
        self,
        instrument: str,
        units: int,
        stop_loss: Optional[float] = None,
        take_profit: Optional[float] = None
    ) -> Optional[Dict]:
        """
        Platziert eine Market Order über Oanda Demo Account
        
        Args:
            instrument: z.B. "XAU_USD"
            units: Positionsgröße (positiv=LONG, negativ=SHORT)
            stop_loss: Stop Loss Preis
            take_profit: Take Profit Preis
        
        Returns:
            Order-Response oder None bei Fehler
        """
        url = f"{self.api_url}/accounts/{self.account_id}/orders"
        
        order_data = {
            "order": {
                "type": "MARKET",
                "instrument": instrument,
                "units": str(units),
                "timeInForce": "FOK",
                "positionFill": "DEFAULT"
            }
        }
        
        # Stop Loss hinzufügen
        if stop_loss:
            order_data["order"]["stopLossOnFill"] = {
                "price": str(stop_loss)
            }
        
        # Take Profit hinzufügen
        if take_profit:
            order_data["order"]["takeProfitOnFill"] = {
                "price": str(take_profit)
            }
        
        try:
            response = requests.post(url, headers=self.headers, json=order_data)
            response.raise_for_status()
            result = response.json()
            
            print(f"[OK] Order platziert: {instrument} | Units: {units} | SL: {stop_loss} | TP: {take_profit}")
            return result
        
        except requests.exceptions.RequestException as e:
            print(f"[ERROR] Fehler beim Platzieren der Order: {e}")
            if hasattr(e.response, 'text'):
                print(f"   Response: {e.response.text}")
            return None
    
    def get_account_summary(self) -> Optional[Dict]:
        """
        Holt Account-Informationen (Balance, P&L, etc.)
        
        Returns:
            Account-Summary oder None
        """
        url = f"{self.api_url}/accounts/{self.account_id}/summary"
        
        try:
            response = requests.get(url, headers=self.headers)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            print(f"[ERROR] Fehler beim Laden der Account-Daten: {e}")
            return None
    
    def get_open_positions(self) -> List[Dict]:
        """
        Holt alle offenen Positionen
        
        Returns:
            Liste offener Positionen
        """
        url = f"{self.api_url}/accounts/{self.account_id}/openPositions"
        
        try:
            response = requests.get(url, headers=self.headers)
            response.raise_for_status()
            data = response.json()
            return data.get("positions", [])
        except requests.exceptions.RequestException as e:
            print(f"[ERROR] Fehler beim Laden offener Positionen: {e}")
            return []
    
    def get_account_info(self) -> Dict:
        """
        Holt Account-Informationen (Balance, etc.)
        Alias für get_account_summary mit vereinfachtem Return
        
        Returns:
            Dict mit Account-Informationen
        """
        summary = self.get_account_summary()
        if summary and 'account' in summary:
            account = summary['account']
            return {
                'balance': account.get('balance', '10000'),
                'currency': account.get('currency', 'USD'),
                'unrealizedPL': account.get('unrealizedPL', '0'),
                'nav': account.get('NAV', account.get('balance', '10000'))
            }
        # Fallback wenn Account nicht geladen werden kann
        return {'balance': '10000', 'currency': 'USD', 'unrealizedPL': '0', 'nav': '10000'}


if __name__ == "__main__":
    # Test der Oanda-Verbindung
    client = OandaClient()
    
    print("\n[*] Test: Lade neuesten Gold-Candle...")
    latest = client.get_latest_candle(config.GOLD_INSTRUMENT)
    if latest:
        print(f"[OK] Neuester Candle: {latest['timestamp']} | Close: {latest['close']}")
    
    print("\n[*] Test: Account-Summary...")
    summary = client.get_account_summary()
    if summary:
        account = summary.get("account", {})
        print(f"[OK] Balance: {account.get('balance')} {account.get('currency')}")
