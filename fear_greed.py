"""
Fear & Greed Index Fetcher - RapidAPI
Fetches the current Fear & Greed Index value via RapidAPI for live trading
"""
from datetime import datetime
from typing import Optional, Dict
import requests
import json
import os
import config


class FearGreedScraper:
    """Fetches Fear & Greed Index from RapidAPI"""
    
    def __init__(self):
        self.api_url = "https://fear-and-greed-index.p.rapidapi.com/v1/fgi"
        self.api_key = config.RAPIDAPI_KEY
        self.api_host = "fear-and-greed-index.p.rapidapi.com"
    
    def scrape_current_value(self) -> Optional[Dict]:
        """
        Fetches the current Fear & Greed Index from RapidAPI
        
        Returns:
            Dict with fear_greed value (0-100) and timestamp
        """
        try:
            headers = {
                "x-rapidapi-key": self.api_key,
                "x-rapidapi-host": self.api_host
            }
            
            response = requests.get(self.api_url, headers=headers, timeout=10)
            response.raise_for_status()
            
            data = response.json()
            
            if 'fgi' in data and 'now' in data['fgi']:
                now_data = data['fgi']['now']
                fear_greed_value = float(now_data.get('value', 50))
                timestamp = now_data.get('timestamp', datetime.utcnow().isoformat())
                
                print(f"[OK] Fear & Greed fetched: {fear_greed_value}")
                
                return {
                    "fear_greed": fear_greed_value,
                    "timestamp": timestamp
                }
            
            print("[WARN] Unexpected API response format")
            return {
                "fear_greed": 50.0,
                "timestamp": datetime.utcnow().isoformat(),
                "note": "Unexpected API format"
            }
            
        except requests.exceptions.RequestException as e:
            print(f"[ERROR] Failed to fetch Fear & Greed Index: {e}")
            return {
                "fear_greed": 50.0,
                "timestamp": datetime.utcnow().isoformat(),
                "note": "Network error"
            }
        except Exception as e:
            print(f"[ERROR] Unexpected error fetching Fear & Greed: {e}")
            return {
                "fear_greed": 50.0,
                "timestamp": datetime.utcnow().isoformat(),
                "note": str(e)
            }
    
    def get_classification(self, value: float) -> str:
        """
        Classifies Fear & Greed value
        
        Args:
            value: Fear & Greed value (0-100)
        
        Returns:
            Classification string
        """
        if value <= 25:
            return "Extreme Fear"
        elif value <= 45:
            return "Fear"
        elif value <= 55:
            return "Neutral"
        elif value <= 75:
            return "Greed"
        else:
            return "Extreme Greed"
    
    def scrape_with_classification(self) -> Optional[Dict]:
        """
        Fetches Fear & Greed with classification
        
        Returns:
            Dict with value, classification, and timestamp
        """
        result = self.scrape_current_value()
        if result:
            result["classification"] = self.get_classification(result["fear_greed"])
        return result
    
    def save_to_file(self, data: Dict):
        """
        Saves Fear & Greed data to JSON file with historical tracking
        
        Args:
            data: Fear & Greed data to save
        """
        filepath = config.FEAR_GREED_FILE
        
        existing_data = {"data": [], "info": "Fear & Greed Index historical data"}
        if os.path.exists(filepath):
            try:
                with open(filepath, 'r', encoding='utf-8') as f:
                    existing_data = json.load(f)
            except Exception as e:
                print(f"[WARN] Could not read existing data: {e}")
        
        if "data" not in existing_data:
            existing_data["data"] = []
        
        entry = {
            "timestamp": data.get("timestamp"),
            "fear_greed": data.get("fear_greed"),
            "classification": data.get("classification")
        }
        
        existing_data["data"].append(entry)
        existing_data["last_updated"] = datetime.utcnow().isoformat()
        existing_data["total_entries"] = len(existing_data["data"])
        
        try:
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(existing_data, f, indent=2, ensure_ascii=False)
            print(f"[SAVE] Fear & Greed saved to {filepath}")
        except Exception as e:
            print(f"[ERROR] Could not save to file: {e}")
    
    def fetch_and_save(self) -> Optional[Dict]:
        """
        Fetches Fear & Greed and saves to file
        
        Returns:
            Dict with value, classification, and timestamp
        """
        result = self.scrape_with_classification()
        if result:
            self.save_to_file(result)
        return result


if __name__ == "__main__":
    scraper = FearGreedScraper()
    
    print("[*] Fetching Fear & Greed Index from RapidAPI...")
    result = scraper.fetch_and_save()
    
    if result:
        print(f"[OK] Fear & Greed Index: {result['fear_greed']}")
        print(f"     Classification: {result['classification']}")
        print(f"     Timestamp: {result['timestamp']}")
    else:
        print("[ERROR] Failed to fetch Fear & Greed Index")
