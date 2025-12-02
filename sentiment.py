"""
Sentiment-Analyse mit NewsAPI und FinBERT
Lädt News-Headlines und analysiert Sentiment
"""
import json
import os
import requests
from datetime import datetime, timedelta
from typing import List, Dict, Optional
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import config


class SentimentAnalyzer:
    """Sentiment-Analyse mit FinBERT"""
    
    def __init__(self):
        self.news_api_key = config.NEWS_API_KEY
        self.finbert_model_name = config.FINBERT_MODEL
        self.tokenizer = None
        self.model = None
        self._load_finbert()
    
    def _load_finbert(self):
        """Lädt FinBERT-Modell"""
        try:
            print("[*] Lade FinBERT-Modell...")
            self.tokenizer = AutoTokenizer.from_pretrained(self.finbert_model_name)
            self.model = AutoModelForSequenceClassification.from_pretrained(self.finbert_model_name)
            self.model.eval()
            print("[OK] FinBERT-Modell geladen")
        except Exception as e:
            print(f"[ERROR] Fehler beim Laden von FinBERT: {e}")
            print("   --> Sentiment-Analyse wird uebersprungen")
    
    def fetch_news(
        self, 
        query_terms: List[str],
        from_date: datetime,
        to_date: datetime,
        max_results: int = 100
    ) -> List[Dict]:
        """
        Lädt News-Headlines von NewsAPI
        
        Args:
            query_terms: Suchbegriffe
            from_date: Start-Datum
            to_date: End-Datum
            max_results: Max. Anzahl Ergebnisse
        
        Returns:
            Liste von News-Artikeln
        """
        if self.news_api_key == "YOUR_NEWS_API_KEY":
            print("[WARN] NewsAPI Key nicht konfiguriert - überspringe News-Fetching")
            return []
        
        query = " OR ".join(query_terms)
        url = config.NEWS_API_URL
        
        params = {
            "q": query,
            "from": from_date.strftime("%Y-%m-%d"),
            "to": to_date.strftime("%Y-%m-%d"),
            "sortBy": "publishedAt",
            "language": "en",
            "apiKey": self.news_api_key,
            "pageSize": min(max_results, 100)  # API Limit
        }
        
        try:
            response = requests.get(url, params=params)
            response.raise_for_status()
            data = response.json()
            
            articles = []
            for article in data.get("articles", []):
                articles.append({
                    "timestamp": article.get("publishedAt", ""),
                    "title": article.get("title", ""),
                    "description": article.get("description", ""),
                    "source": article.get("source", {}).get("name", ""),
                    "url": article.get("url", "")
                })
            
            return articles
        
        except requests.exceptions.RequestException as e:
            print(f"[ERROR] Fehler beim Laden von News: {e}")
            return []
    
    def analyze_sentiment(self, text: str) -> Dict:
        """
        Analysiert Sentiment eines Texts mit FinBERT (misraanay/finbert-tone-gold-lora-final)
        
        Args:
            text: Text zur Analyse
        
        Returns:
            Dict mit sentiment_label und sentiment_score
        """
        if not self.model or not self.tokenizer or not text:
            return {
                "sentiment_label": 0,
                "sentiment_score": 0.0,
                "confidence": 0.0
            }
        
        try:
            inputs = self.tokenizer(
                text, 
                return_tensors="pt", 
                truncation=True, 
                max_length=512,
                padding=True
            )
            
            with torch.no_grad():
                outputs = self.model(**inputs)
                predictions = torch.nn.functional.softmax(outputs.logits, dim=-1)
            
            probs = predictions[0].tolist()
            
            if len(probs) == 3:
                negative_prob = probs[0]
                neutral_prob = probs[1]
                positive_prob = probs[2]
            else:
                negative_prob = probs[0] if len(probs) > 0 else 0.33
                neutral_prob = probs[1] if len(probs) > 1 else 0.34
                positive_prob = probs[2] if len(probs) > 2 else 0.33
            
            if positive_prob > negative_prob and positive_prob > neutral_prob:
                sentiment_label = 1
                confidence = positive_prob
            elif negative_prob > positive_prob and negative_prob > neutral_prob:
                sentiment_label = -1
                confidence = negative_prob
            else:
                sentiment_label = 0
                confidence = neutral_prob
            
            sentiment_score = positive_prob - negative_prob
            
            return {
                "sentiment_label": sentiment_label,
                "sentiment_score": sentiment_score,
                "confidence": confidence,
                "probs": {
                    "negative": negative_prob,
                    "neutral": neutral_prob,
                    "positive": positive_prob
                }
            }
        
        except Exception as e:
            print(f"[ERROR] Fehler bei Sentiment-Analyse: {e}")
            return {
                "sentiment_label": 0,
                "sentiment_score": 0.0,
                "confidence": 0.0
            }
    
    def analyze_news_batch(self, articles: List[Dict]) -> List[Dict]:
        """
        Analysiert Sentiment für mehrere News-Artikel
        
        Args:
            articles: Liste von News-Artikeln
        
        Returns:
            Articles mit hinzugefügtem Sentiment
        """
        print(f"[*] Analysiere Sentiment für {len(articles)} Artikel...")
        
        for i, article in enumerate(articles):
            if i % 20 == 0 and i > 0:
                print(f"  ...{i}/{len(articles)} analysiert")
            
            # Kombiniere Titel und Beschreibung
            text = f"{article.get('title', '')} {article.get('description', '')}"
            sentiment = self.analyze_sentiment(text)
            
            article.update(sentiment)
        
        print(f"[OK] Sentiment-Analyse abgeschlossen")
        return articles
    
    def load_and_analyze_sentiment(
        self,
        force_reload: bool = False,
        lookback_days: int = None
    ) -> List[Dict]:
        """
        Lädt News und analysiert Sentiment mit inkrementellem Update
        
        Args:
            force_reload: Wenn True, lädt alle News neu
            lookback_days: Anzahl Tage zurück (default: config)
        
        Returns:
            Liste von News mit Sentiment
        """
        if lookback_days is None:
            lookback_days = config.NEWS_LOOKBACK_DAYS
        
        print("\n[DATA] Lade Sentiment-Daten...")
        
        # Lade existierende Daten
        if os.path.exists(config.SENTIMENT_FILE) and not force_reload:
            with open(config.SENTIMENT_FILE, 'r', encoding='utf-8') as f:
                existing_data = json.load(f)
                existing_articles = existing_data.get("data", [])
            
            if existing_articles:
                # Finde letztes Datum
                try:
                    last_date = datetime.strptime(
                        existing_articles[-1]["timestamp"][:10], 
                        "%Y-%m-%d"
                    )
                    from_date = last_date + timedelta(days=1)
                    print(f"  [RELOAD] Letztes Datum: {last_date.strftime('%Y-%m-%d')}")
                except:
                    from_date = datetime.utcnow() - timedelta(days=lookback_days)
            else:
                from_date = datetime.utcnow() - timedelta(days=lookback_days)
        else:
            existing_articles = []
            from_date = datetime.utcnow() - timedelta(days=lookback_days)
        
        to_date = datetime.utcnow()
        
        # Neue News laden
        print(f"  [RELOAD] Lade News von {from_date.strftime('%Y-%m-%d')} bis {to_date.strftime('%Y-%m-%d')}...")
        new_articles = self.fetch_news(
            query_terms=config.NEWS_QUERY_TERMS,
            from_date=from_date,
            to_date=to_date,
            max_results=config.NEWS_MAX_RESULTS
        )
        
        if new_articles:
            print(f"  [OK] {len(new_articles)} neue Artikel gefunden")
            # Analysiere Sentiment
            new_articles = self.analyze_news_batch(new_articles)
            
            # Kombiniere mit bestehenden Daten
            all_articles = existing_articles + new_articles
        else:
            print(f"  [*] Keine neuen Artikel gefunden")
            all_articles = existing_articles
        
        # Speichern
        sentiment_data = {
            "data": all_articles,
            "last_updated": datetime.utcnow().isoformat(),
            "total_articles": len(all_articles)
        }
        
        with open(config.SENTIMENT_FILE, 'w', encoding='utf-8') as f:
            json.dump(sentiment_data, f, indent=2, ensure_ascii=False)
        
        print(f"[SAVE] Gespeichert: {config.SENTIMENT_FILE} ({len(all_articles)} Artikel)")
        
        return all_articles
    
    def aggregate_sentiment_by_period(
        self, 
        articles: List[Dict],
        period_minutes: int = 15
    ) -> Dict[str, float]:
        """
        Aggregiert Sentiment-Scores nach Zeitperioden (z.B. 15min)
        
        Args:
            articles: Liste von Artikeln mit Sentiment
            period_minutes: Periode in Minuten
        
        Returns:
            Dict mit timestamp -> avg_sentiment_score
        """
        from collections import defaultdict
        
        period_sentiments = defaultdict(list)
        
        for article in articles:
            try:
                timestamp = datetime.strptime(
                    article["timestamp"][:19], 
                    "%Y-%m-%dT%H:%M:%S"
                )
                # Runde auf Periode
                rounded = timestamp.replace(
                    minute=(timestamp.minute // period_minutes) * period_minutes,
                    second=0,
                    microsecond=0
                )
                period_key = rounded.strftime("%Y-%m-%dT%H:%M:%S")
                
                period_sentiments[period_key].append(
                    article.get("sentiment_score", 0.0)
                )
            except:
                continue
        
        # Berechne Durchschnitt pro Periode
        aggregated = {}
        for period, scores in period_sentiments.items():
            aggregated[period] = sum(scores) / len(scores) if scores else 0.0
        
        return aggregated


if __name__ == "__main__":
    # Test der Sentiment-Analyse
    analyzer = SentimentAnalyzer()
    
    # Test mit Beispieltext
    test_text = "Gold prices surge to new highs amid market uncertainty"
    sentiment = analyzer.analyze_sentiment(test_text)
    print(f"\n[*] Test Sentiment-Analyse:")
    print(f"   Text: {test_text}")
    print(f"   Sentiment: {sentiment}")
    
    # Lade und analysiere News
    articles = analyzer.load_and_analyze_sentiment(force_reload=False)
    print(f"\n[DATA] {len(articles)} Artikel mit Sentiment verfügbar")
