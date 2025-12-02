"""
FRED API Integration
Fetches macroeconomic data from Federal Reserve Economic Data (FRED)
"""
import requests
import json
import os
from datetime import datetime, timedelta
from typing import Dict, List, Optional
import pandas as pd
import config


class FREDDataFetcher:
    """
    Fetches macro data from FRED API with incremental loading
    """
    
    FRED_SERIES = {
        'VIX': 'VIXCLS',
        'US_10Y': 'DGS10',
        'US_2Y': 'DGS2',
        'US_5Y': 'DGS5',
        'US_30Y': 'DGS30',
        'CPI': 'CPIAUCSL',
        'NFP': 'PAYEMS',
        'UNEMPLOYMENT': 'UNRATE',
        'GDP': 'GDP',
        'CORE_CPI': 'CPILFESL'
    }
    
    def __init__(self, api_key: str):
        """
        Args:
            api_key: FRED API key
        """
        self.api_key = api_key
        self.base_url = "https://api.stlouisfed.org/fred/series/observations"
        self.data_dir = "data"
        
        # Create data directory if not exists
        os.makedirs(self.data_dir, exist_ok=True)
    
    def fetch_series(
        self,
        series_id: str,
        start_date: datetime,
        end_date: Optional[datetime] = None
    ) -> pd.DataFrame:
        """
        Fetch a single FRED series
        
        Args:
            series_id: FRED series ID
            start_date: Start date
            end_date: End date (default: today)
        
        Returns:
            DataFrame with date and value columns
        """
        if end_date is None:
            end_date = datetime.now()
        
        params = {
            'series_id': series_id,
            'api_key': self.api_key,
            'file_type': 'json',
            'observation_start': start_date.strftime('%Y-%m-%d'),
            'observation_end': end_date.strftime('%Y-%m-%d')
        }
        
        try:
            response = requests.get(self.base_url, params=params)
            response.raise_for_status()
            data = response.json()
            
            if 'observations' not in data:
                print(f"[WARN] No observations for {series_id}")
                return pd.DataFrame()
            
            # Convert to DataFrame
            df = pd.DataFrame(data['observations'])
            
            if len(df) == 0:
                return pd.DataFrame()
            
            # Convert date and value
            df['date'] = pd.to_datetime(df['date'])
            df['value'] = pd.to_numeric(df['value'], errors='coerce')
            
            # Keep only date and value
            df = df[['date', 'value']].copy()
            df.columns = ['date', series_id]
            
            return df
        
        except requests.exceptions.RequestException as e:
            print(f"[ERROR] Failed to fetch {series_id}: {e}")
            return pd.DataFrame()
    
    def load_existing_data(self, series_name: str) -> Optional[pd.DataFrame]:
        """
        Load existing data from JSON file
        
        Args:
            series_name: Series name (e.g., 'VIX')
        
        Returns:
            DataFrame or None if file doesn't exist
        """
        filepath = os.path.join(self.data_dir, f"fred_{series_name.lower()}.json")
        
        if not os.path.exists(filepath):
            return None
        
        try:
            df = pd.read_json(filepath)
            if len(df) > 0:
                df['date'] = pd.to_datetime(df['date'])
                return df
        except Exception as e:
            print(f"[WARN] Could not load {filepath}: {e}")
            return None
    
    def save_data(self, df: pd.DataFrame, series_name: str):
        """
        Save data to JSON file
        
        Args:
            df: DataFrame to save
            series_name: Series name (e.g., 'VIX')
        """
        filepath = os.path.join(self.data_dir, f"fred_{series_name.lower()}.json")
        
        try:
            # Convert to JSON-friendly format
            df_save = df.copy()
            df_save['date'] = df_save['date'].dt.strftime('%Y-%m-%d')
            df_save.to_json(filepath, orient='records', indent=2)
            print(f"[SAVE] Saved {len(df)} records to {filepath}")
        except Exception as e:
            print(f"[ERROR] Failed to save {filepath}: {e}")
    
    def fetch_series_incremental(
        self,
        series_name: str,
        start_date: datetime,
        force_reload: bool = False
    ) -> pd.DataFrame:
        """
        Fetch FRED series with incremental loading
        
        Args:
            series_name: Series name (e.g., 'VIX')
            start_date: Start date for data
            force_reload: If True, reload all data
        
        Returns:
            DataFrame with date and value columns
        """
        series_id = self.FRED_SERIES.get(series_name)
        if not series_id:
            print(f"[ERROR] Unknown series: {series_name}")
            return pd.DataFrame()
        
        # Load existing data
        existing_df = None if force_reload else self.load_existing_data(series_name)
        
        if existing_df is not None and len(existing_df) > 0:
            # Get last date
            last_date = existing_df['date'].max()
            
            # Check if we need to fetch new data
            if last_date >= datetime.now() - timedelta(days=1):
                print(f"[FRED] {series_name}: Up to date ({len(existing_df)} records)")
                return existing_df
            
            # Fetch only new data
            print(f"[FRED] {series_name}: Fetching data from {last_date.date()} onwards...")
            new_df = self.fetch_series(series_id, last_date + timedelta(days=1))
            
            if len(new_df) > 0:
                # Merge with existing data
                df = pd.concat([existing_df, new_df], ignore_index=True)
                df = df.drop_duplicates(subset=['date'], keep='last')
                df = df.sort_values('date').reset_index(drop=True)
                print(f"[FRED] {series_name}: Added {len(new_df)} new records")
            else:
                df = existing_df
                print(f"[FRED] {series_name}: No new data available")
        else:
            # Fetch all data
            print(f"[FRED] {series_name}: Fetching all data from {start_date.date()}...")
            df = self.fetch_series(series_id, start_date)
            
            if len(df) == 0:
                print(f"[WARN] {series_name}: No data fetched")
                return pd.DataFrame()
            
            print(f"[FRED] {series_name}: Fetched {len(df)} records")
        
        # Save data
        self.save_data(df, series_name)
        
        return df
    
    def fetch_all_series(
        self,
        start_date: datetime,
        force_reload: bool = False
    ) -> Dict[str, pd.DataFrame]:
        """
        Fetch all FRED series
        
        Args:
            start_date: Start date for all series
            force_reload: If True, reload all data
        
        Returns:
            Dictionary of series_name -> DataFrame
        """
        print("\n" + "="*70)
        print("[FRED] FETCHING MACROECONOMIC DATA")
        print("="*70)
        
        results = {}
        
        for series_name in self.FRED_SERIES.keys():
            df = self.fetch_series_incremental(series_name, start_date, force_reload)
            if len(df) > 0:
                results[series_name] = df
        
        print("="*70)
        print(f"[OK] FRED data fetched: {len(results)}/{len(self.FRED_SERIES)} series")
        print("="*70)
        
        return results


def load_fred_data(force_reload: bool = False) -> Dict[str, pd.DataFrame]:
    """
    Load FRED macroeconomic data
    
    Args:
        force_reload: If True, reload all data from FRED API
    
    Returns:
        Dictionary of series_name -> DataFrame
    """
    fetcher = FREDDataFetcher(api_key=config.FRED_API_KEY)
    start_date = datetime.strptime(config.START_DATE, '%Y-%m-%d')
    
    return fetcher.fetch_all_series(start_date, force_reload)


if __name__ == "__main__":
    # Test FRED data fetching
    print("[TEST] FRED Data Fetcher")
    print("="*70)
    
    data = load_fred_data(force_reload=False)
    
    print("\n[RESULTS]")
    for series_name, df in data.items():
        if len(df) > 0:
            print(f"{series_name:15s}: {len(df):5d} records | {df['date'].min().date()} to {df['date'].max().date()}")
