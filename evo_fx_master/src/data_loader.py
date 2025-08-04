#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
EVO_FX_MASTER - Veri Çekme ve Hazırlama Modülü
==============================================

OANDA REST API ve MT5 CSV fallback ile EURUSD & XAUUSD verilerini çeker.
Eksik mumları forward-fill eder, tüm veri UTC'ye normalize eder.
"""

import pandas as pd
import numpy as np
import logging
import requests
import json
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Optional, Dict, List, Tuple
import time
import os

# MT5 desteği (opsiyonel)
try:
    import MetaTrader5 as mt5
    MT5_AVAILABLE = True
except ImportError:
    MT5_AVAILABLE = False
    print("⚠️  MT5 kütüphanesi bulunamadı. Sadece OANDA modu aktif.")

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DataLoader:
    """Ana veri çekme sınıfı."""
    
    def __init__(self, config: Dict):
        """
        Args:
            config: Veri kaynağı ayarları içeren sözlük
        """
        self.config = config
        self.oanda_token = config.get('oanda_token', '')
        self.oanda_account = config.get('oanda_account', '')
        self.base_url = config.get('oanda_base_url', 'https://api-fxpractice.oanda.com')
        
        # Desteklenen semboller
        self.symbols = ['EUR_USD', 'XAU_USD']  # OANDA formatı
        self.mt5_symbols = ['EURUSD', 'XAUUSD']  # MT5 formatı
        
        # Veri dizini
        self.data_dir = Path("data")
        self.raw_dir = self.data_dir / "raw"
        self.processed_dir = self.data_dir / "processed"
        
        # Dizinleri oluştur
        self.raw_dir.mkdir(parents=True, exist_ok=True)
        self.processed_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"📊 DataLoader hazırlandı. MT5: {MT5_AVAILABLE}")
    
    def fetch_oanda_data(self, symbol: str, start_date: str, end_date: str, 
                        granularity: str = "M1") -> Optional[pd.DataFrame]:
        """
        OANDA REST API'den veri çeker.
        
        Args:
            symbol: OANDA formatında sembol (EUR_USD, XAU_USD)
            start_date: Başlangıç tarihi (YYYY-MM-DD)
            end_date: Bitiş tarihi (YYYY-MM-DD)
            granularity: Zaman dilimi (M1=1dk, M5=5dk, H1=1sa)
            
        Returns:
            DataFrame veya None (hata durumunda)
        """
        if not self.oanda_token:
            logger.warning("OANDA token bulunamadı!")
            return None
            
        headers = {
            'Authorization': f'Bearer {self.oanda_token}',
            'Content-Type': 'application/json'
        }
        
        # API endpoint
        url = f"{self.base_url}/v3/instruments/{symbol}/candles"
        
        params = {
            'from': f"{start_date}T00:00:00Z",
            'to': f"{end_date}T23:59:59Z", 
            'granularity': granularity,
            'price': 'M'  # Mid prices
        }
        
        try:
            logger.info(f"📡 OANDA veri çekiliyor: {symbol} ({start_date} - {end_date})")
            response = requests.get(url, headers=headers, params=params, timeout=30)
            
            if response.status_code != 200:
                logger.error(f"OANDA API hatası: {response.status_code}")
                return None
                
            data = response.json()
            candles = data.get('candles', [])
            
            if not candles:
                logger.warning(f"Veri bulunamadı: {symbol}")
                return None
            
            # DataFrame'e dönüştür
            records = []
            for candle in candles:
                if candle['complete']:  # Sadece tamamlanmış mumlar
                    mid = candle['mid']
                    records.append({
                        'timestamp': pd.to_datetime(candle['time']),
                        'open': float(mid['o']),
                        'high': float(mid['h']),
                        'low': float(mid['l']),
                        'close': float(mid['c']),
                        'volume': int(candle['volume'])
                    })
            
            if not records:
                return None
                
            df = pd.DataFrame(records)
            df.set_index('timestamp', inplace=True)
            df.index = df.index.tz_convert('UTC')  # UTC'ye dönüştür
            
            logger.info(f"✓ OANDA veri alındı: {len(df)} mum, {symbol}")
            return df
            
        except Exception as e:
            logger.error(f"OANDA veri çekme hatası: {e}")
            return None
    
    def fetch_mt5_data(self, symbol: str, start_date: str, end_date: str) -> Optional[pd.DataFrame]:
        """
        MT5 terminalinden veri çeker (fallback).
        
        Args:
            symbol: MT5 formatında sembol (EURUSD, XAUUSD)
            start_date: Başlangıç tarihi
            end_date: Bitiş tarihi
            
        Returns:
            DataFrame veya None
        """
        if not MT5_AVAILABLE:
            logger.warning("MT5 kütüphanesi yok!")
            return None
            
        try:
            # MT5 bağlantısı
            if not mt5.initialize():
                logger.error("MT5 başlatılamadı!")
                return None
            
            # Tarih dönüşümü
            start_dt = datetime.strptime(start_date, "%Y-%m-%d").replace(tzinfo=timezone.utc)
            end_dt = datetime.strptime(end_date, "%Y-%m-%d").replace(tzinfo=timezone.utc)
            
            logger.info(f"📊 MT5 veri çekiliyor: {symbol}")
            
            # 1 dakikalık veriler
            rates = mt5.copy_rates_range(symbol, mt5.TIMEFRAME_M1, start_dt, end_dt)
            
            if rates is None or len(rates) == 0:
                logger.warning(f"MT5'ten veri alınamadı: {symbol}")
                return None
            
            # DataFrame'e dönüştür
            df = pd.DataFrame(rates)
            df['timestamp'] = pd.to_datetime(df['time'], unit='s', utc=True)
            df.set_index('timestamp', inplace=True)
            
            # Gereksiz sütunları kaldır
            df = df[['open', 'high', 'low', 'close', 'tick_volume']].copy()
            df.rename(columns={'tick_volume': 'volume'}, inplace=True)
            
            logger.info(f"✓ MT5 veri alındı: {len(df)} mum, {symbol}")
            return df
            
        except Exception as e:
            logger.error(f"MT5 veri çekme hatası: {e}")
            return None
        finally:
            try:
                mt5.shutdown()
            except:
                pass
    
    def load_csv_fallback(self, symbol: str) -> Optional[pd.DataFrame]:
        """
        CSV dosyasından veri yükler (son çare).
        
        Args:
            symbol: Sembol adı
            
        Returns:
            DataFrame veya None
        """
        csv_path = self.raw_dir / f"{symbol}_fallback.csv"
        
        if not csv_path.exists():
            logger.warning(f"CSV fallback bulunamadı: {csv_path}")
            return None
            
        try:
            df = pd.read_csv(csv_path, parse_dates=['timestamp'], index_col='timestamp')
            df.index = pd.to_datetime(df.index, utc=True)
            
            logger.info(f"📁 CSV fallback yüklendi: {len(df)} mum, {symbol}")
            return df
            
        except Exception as e:
            logger.error(f"CSV yükleme hatası: {e}")
            return None
    
    def clean_and_fill(self, df: pd.DataFrame, symbol: str) -> pd.DataFrame:
        """
        Veriyi temizler ve eksik mumları doldurur.
        
        Args:
            df: Ham veri DataFrame'i
            symbol: Sembol adı
            
        Returns:
            Temizlenmiş DataFrame
        """
        if df is None or df.empty:
            return df
            
        logger.info(f"🧹 Veri temizleniyor: {symbol}")
        
        # Duplikatları kaldır
        df = df[~df.index.duplicated(keep='first')]
        
        # Sırala
        df = df.sort_index()
        
        # Negatif fiyatları kaldır
        price_cols = ['open', 'high', 'low', 'close']
        for col in price_cols:
            if col in df.columns:
                df = df[df[col] > 0]
        
        # OHLC mantık kontrolü
        df = df[
            (df['high'] >= df['low']) & 
            (df['high'] >= df['open']) & 
            (df['high'] >= df['close']) &
            (df['low'] <= df['open']) & 
            (df['low'] <= df['close'])
        ]
        
        # Eksik değerleri forward fill
        df = df.fillna(method='ffill')
        
        # İlk satırdaki NaN'ları kaldır
        df = df.dropna()
        
        logger.info(f"✓ Temizleme tamamlandı: {len(df)} geçerli mum")
        return df
    
    def fetch_symbol_data(self, symbol: str, start_date: str, end_date: str) -> Optional[pd.DataFrame]:
        """
        Bir sembol için veri çeker (önce OANDA, sonra MT5, son olarak CSV).
        
        Args:
            symbol: Sembol adı (EURUSD formatı)
            start_date: Başlangıç tarihi
            end_date: Bitiş tarihi
            
        Returns:
            DataFrame veya None
        """
        df = None
        
        # 1. OANDA dene
        oanda_symbol = symbol.replace('USD', '_USD') if symbol != 'XAUUSD' else 'XAU_USD'
        if symbol == 'EURUSD':
            oanda_symbol = 'EUR_USD'
            
        df = self.fetch_oanda_data(oanda_symbol, start_date, end_date)
        
        if df is not None:
            logger.info(f"✓ OANDA'dan alındı: {symbol}")
        else:
            # 2. MT5 dene
            logger.info(f"🔄 MT5 deneniyor: {symbol}")
            df = self.fetch_mt5_data(symbol, start_date, end_date)
            
            if df is not None:
                logger.info(f"✓ MT5'ten alındı: {symbol}")
            else:
                # 3. CSV fallback
                logger.info(f"🔄 CSV fallback deneniyor: {symbol}")
                df = self.load_csv_fallback(symbol)
        
        # Veri temizleme
        if df is not None:
            df = self.clean_and_fill(df, symbol)
            
        return df
    
    def write_raw_data(self, symbol: str, df: pd.DataFrame) -> bool:
        """
        Ham veriyi parquet formatında kaydet.
        
        Args:
            symbol: Sembol adı  
            df: Veri DataFrame'i
            
        Returns:
            Başarı durumu
        """
        if df is None or df.empty:
            return False
            
        try:
            output_path = self.raw_dir / f"{symbol}_raw.parquet"
            df.to_parquet(output_path, compression='snappy')
            
            logger.info(f"💾 Ham veri kaydedildi: {output_path} ({len(df)} satır)")
            return True
            
        except Exception as e:
            logger.error(f"Kaydetme hatası: {e}")
            return False
    
    def fetch_all_data(self, start_date: str, end_date: str) -> Dict[str, pd.DataFrame]:
        """
        Tüm semboller için veri çeker.
        
        Args:
            start_date: Başlangıç tarihi (YYYY-MM-DD)
            end_date: Bitiş tarihi (YYYY-MM-DD)
            
        Returns:
            Sembol -> DataFrame sözlüğü
        """
        results = {}
        
        for symbol in self.mt5_symbols:
            logger.info(f"🎯 İşleniyor: {symbol}")
            
            df = self.fetch_symbol_data(symbol, start_date, end_date)
            
            if df is not None and not df.empty:
                # Ham veriyi kaydet
                if self.write_raw_data(symbol, df):
                    results[symbol] = df
                    logger.info(f"✅ {symbol} tamamlandı: {len(df)} mum")
                else:
                    logger.error(f"❌ {symbol} kaydedilemedi!")
            else:
                logger.error(f"❌ {symbol} için veri alınamadı!")
            
            # API rate limit için kısa bekleme
            time.sleep(0.5)
        
        logger.info(f"🏁 Veri çekme tamamlandı. Başarılı: {len(results)}/{len(self.mt5_symbols)}")
        return results


def main():
    """Test fonksiyonu."""
    logger.info("📊 DataLoader modülü test ediliyor...")
    
    # Test config
    config = {
        'oanda_token': '',  # Buraya gerçek token koy
        'oanda_account': '',
        'oanda_base_url': 'https://api-fxpractice.oanda.com'
    }
    
    # DataLoader oluştur
    loader = DataLoader(config)
    
    # Test tarihleri (son 7 gün)
    end_date = datetime.now().strftime("%Y-%m-%d")
    start_date = (datetime.now() - timedelta(days=7)).strftime("%Y-%m-%d")
    
    logger.info(f"📅 Test tarihleri: {start_date} - {end_date}")
    
    # Tek sembol test
    logger.info("🧪 EURUSD test ediliyor...")
    df = loader.fetch_symbol_data('EURUSD', start_date, end_date)
    
    if df is not None:
        logger.info(f"✓ Test başarılı: {len(df)} mum alındı")
        logger.info(f"📊 Veri özeti:\n{df.head()}")
        logger.info(f"📈 Fiyat aralığı: {df['close'].min():.5f} - {df['close'].max():.5f}")
    else:
        logger.warning("⚠️  Test verisi alınamadı (normal - token gerekli)")
    
    # Dizin kontrolü
    raw_files = list(loader.raw_dir.glob("*.parquet"))
    logger.info(f"📁 Raw dizinde {len(raw_files)} dosya var")
    
    logger.info("✅ DataLoader modülü test tamamlandı!")


if __name__ == "__main__":
    main()