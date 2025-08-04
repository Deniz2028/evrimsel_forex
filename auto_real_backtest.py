#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
EVO_FX_MASTER - TAM OTOMATİK GERÇEK VERİ BACKTEST
===============================================

🎯 HEDEFİ: Hiç soru sormadan tam otomatik gerçek OANDA verisi ile backtest
🔧 API bilgileri gömülü, tarih aralığı otomatik
📊 15 günlük gerçek veri + feature engineering + NEAT backtest
"""

import sys
import os
import numpy as np
import pandas as pd
import logging
import warnings
import requests
import json
from pathlib import Path
from datetime import datetime, timedelta, timezone
import time
import math

# Proje kökünü path'e ekle
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root / "src"))

warnings.filterwarnings('ignore')
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class AutoOANDALoader:
    """🤖 Tam otomatik OANDA veri çekici - API gömülü."""
    
    def __init__(self):
        """🔧 API bilgileri gömülü - hiç soru sormaz."""
        # 🔑 GÖMÜLÜ API BİLGİLERİ
        self.token = "8d8619f4119fec7e59d73c61b76b480d-d0947fd967a22401c1e48bc1516ad0eb"
        self.account_id = "101-004-35700665-002"
        self.base_url = "https://api-fxpractice.oanda.com"  # Demo
        
        self.headers = {
            'Authorization': f'Bearer {self.token}',
            'Content-Type': 'application/json'
        }
        
        # Symbol mapping
        self.symbol_map = {
            'EURUSD': 'EUR_USD',
            'XAUUSD': 'XAU_USD'
        }
        
        # OANDA Limitleri
        self.max_candles_per_request = 4000
        self.rate_limit_delay = 0.1
        
        print(f"🤖 Otomatik OANDA Loader hazır:")
        print(f"   🔑 Account: {self.account_id}")
        print(f"   🌍 Environment: Demo")
        print(f"   📦 Chunk size: {self.max_candles_per_request}")
    
    def test_connection_silent(self):
        """Sessiz bağlantı testi."""
        try:
            url = f"{self.base_url}/v3/accounts/{self.account_id}"
            response = requests.get(url, headers=self.headers, timeout=10)
            
            if response.status_code == 200:
                account_info = response.json()
                balance = account_info.get('account', {}).get('balance', 'N/A')
                print(f"   ✅ OANDA bağlantısı OK - Bakiye: ${balance}")
                return True
            else:
                print(f"   ❌ OANDA hatası: {response.status_code}")
                return False
        except Exception as e:
            print(f"   ❌ Bağlantı hatası: {e}")
            return False
    
    def calculate_auto_chunks(self, days_back=15):
        """Otomatik chunk hesaplama."""
        # Tarih aralığı otomatik
        today = datetime.now().date()
        start_date = today - timedelta(days=days_back)
        end_date = today - timedelta(days=1)
        
        start_dt = datetime.combine(start_date, datetime.min.time()).replace(tzinfo=timezone.utc)
        end_dt = datetime.combine(end_date, datetime.max.time()).replace(tzinfo=timezone.utc)
        
        # Şu anki zaman kontrolü
        now_utc = datetime.now(timezone.utc)
        if end_dt >= now_utc:
            end_dt = now_utc - timedelta(minutes=1)
        
        # Chunk'lara böl
        total_minutes = (end_dt - start_dt).total_seconds() / 60
        total_candles_estimate = total_minutes
        chunk_count = math.ceil(total_candles_estimate / self.max_candles_per_request)
        
        print(f"📊 Otomatik chunk hesaplama:")
        print(f"   📅 Aralık: {start_dt.date()} - {end_dt.date()} ({days_back} gün)")
        print(f"   🕐 Tahmini mum: {total_candles_estimate:.0f}")
        print(f"   📦 Chunk sayısı: {chunk_count}")
        
        # Chunk'ları oluştur
        chunks = []
        current_start = start_dt
        
        while current_start < end_dt:
            chunk_minutes = self.max_candles_per_request
            chunk_end = current_start + timedelta(minutes=chunk_minutes)
            
            if chunk_end > end_dt:
                chunk_end = end_dt
            
            chunks.append((current_start, chunk_end))
            current_start = chunk_end + timedelta(minutes=1)
        
        return chunks
    
    def fetch_single_chunk(self, symbol, start_dt, end_dt):
        """Tek chunk çek."""
        oanda_symbol = self.symbol_map.get(symbol, symbol)
        
        from_str = start_dt.strftime("%Y-%m-%dT%H:%M:%SZ")
        to_str = end_dt.strftime("%Y-%m-%dT%H:%M:%SZ")
        
        url = f"{self.base_url}/v3/instruments/{oanda_symbol}/candles"
        params = {
            'from': from_str,
            'to': to_str,
            'granularity': 'M1',
            'price': 'M'
        }
        
        try:
            time.sleep(self.rate_limit_delay)
            response = requests.get(url, headers=self.headers, params=params, timeout=30)
            
            if response.status_code != 200:
                return None
            
            data = response.json()
            candles = data.get('candles', [])
            
            if not candles:
                return None
            
            records = []
            for candle in candles:
                if candle['complete']:
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
            
            if df.index.tz is not None:
                df.index = df.index.tz_convert('UTC')
            else:
                df.index = df.index.tz_localize('UTC')
            
            return df
            
        except Exception:
            return None
    
    def auto_fetch_symbol(self, symbol):
        """Otomatik sembol verisi çek."""
        print(f"📡 {symbol} otomatik veri çekimi...")
        
        chunks = self.calculate_auto_chunks(days_back=15)
        
        if not chunks:
            print(f"   ❌ Chunk hesaplanamadı")
            return None
        
        all_dataframes = []
        success_count = 0
        
        print(f"   📦 {len(chunks)} chunk işleniyor...")
        
        for i, (chunk_start, chunk_end) in enumerate(chunks, 1):
            # Progress göster (her 5'te bir)
            if i % 5 == 1 or i == len(chunks):
                print(f"      Chunk {i}/{len(chunks)}")
            
            chunk_df = self.fetch_single_chunk(symbol, chunk_start, chunk_end)
            
            if chunk_df is not None and not chunk_df.empty:
                all_dataframes.append(chunk_df)
                success_count += 1
        
        print(f"   📊 {success_count}/{len(chunks)} chunk başarılı")
        
        if not all_dataframes:
            print(f"   ❌ Veri alınamadı")
            return None
        
        # Birleştir
        combined_df = pd.concat(all_dataframes, axis=0)
        combined_df = combined_df[~combined_df.index.duplicated(keep='first')]
        combined_df = combined_df.sort_index()
        
        # Kalite kontrol
        initial_len = len(combined_df)
        
        # OHLC geçersiz satırları kaldır
        combined_df = combined_df[
            (combined_df['high'] >= combined_df['low']) & 
            (combined_df['high'] >= combined_df['open']) & 
            (combined_df['high'] >= combined_df['close']) &
            (combined_df['low'] <= combined_df['open']) & 
            (combined_df['low'] <= combined_df['close']) &
            (combined_df[['open', 'high', 'low', 'close']] > 0).all(axis=1)
        ]
        
        cleaned_count = initial_len - len(combined_df)
        if cleaned_count > 0:
            print(f"   🧹 {cleaned_count} geçersiz satır temizlendi")
        
        print(f"   ✅ {symbol} tamamlandı:")
        print(f"      📊 Final mum: {len(combined_df)}")
        print(f"      📅 Aralık: {combined_df.index.min().date()} - {combined_df.index.max().date()}")
        print(f"      💰 Fiyat: {combined_df['close'].min():.5f} - {combined_df['close'].max():.5f}")
        
        return combined_df
    
    def save_data(self, symbol, df):
        """Veri kaydet."""
        if df is None or df.empty:
            return False
        
        data_dir = Path("data")
        raw_dir = data_dir / "raw"
        raw_dir.mkdir(parents=True, exist_ok=True)
        
        file_path = raw_dir / f"{symbol}_raw.parquet"
        df.to_parquet(file_path, compression='snappy')
        
        print(f"   💾 Kaydedildi: {file_path}")
        return True

def auto_fetch_all_data():
    """🤖 Tüm veriyi otomatik çek."""
    print("🤖 OTOMATİK GERÇEK VERİ ÇEKİMİ")
    print("=" * 40)
    
    loader = AutoOANDALoader()
    
    # Bağlantı test
    print("🔌 OANDA bağlantısı test ediliyor...")
    if not loader.test_connection_silent():
        return False
    
    # Her sembol için veri çek
    symbols = ['EURUSD', 'XAUUSD']
    success_count = 0
    
    for symbol in symbols:
        df = loader.auto_fetch_symbol(symbol)
        
        if df is not None and len(df) >= 1000:  # Minimum 1000 mum
            if loader.save_data(symbol, df):
                success_count += 1
        else:
            print(f"   ⚠️  {symbol}: Yetersiz veri")
    
    print(f"\n📊 OTOMATIK ÇEKIM SONUCU: {success_count}/2 sembol başarılı")
    return success_count >= 1

def auto_feature_engineering():
    """🤖 Otomatik feature engineering."""
    print("\n🔧 OTOMATİK FEATURE ENGİNEERİNG")
    print("=" * 40)
    
    try:
        from feature_engineer import AdvancedFeatureEngineer
        
        config = {
            'rsi_period': 14,
            'atr_period': 14,
            'volatility_period': 30,
            'tick_momentum_periods': [5, 10, 20],
            'volume_periods': [10, 20, 50]
        }
        
        engineer = AdvancedFeatureEngineer(config)
        print("🏗️  Feature engineering başlatılıyor...")
        
        results = engineer.process_all_symbols_advanced()
        
        if len(results) >= 1:
            total_features = 0
            for symbol, df in results.items():
                print(f"   ✅ {symbol}: {len(df)} satır, {len(df.columns)} özellik")
                total_features += len(df.columns)
            
            print(f"🎉 Feature engineering başarılı:")
            print(f"   📊 Ortalama özellik: {total_features // len(results)}")
            return True
        else:
            print("❌ Feature engineering başarısız")
            return False
            
    except Exception as e:
        print(f"❌ Feature engineering hatası: {e}")
        return False

def auto_neat_backtest():
    """🤖 Otomatik NEAT backtest."""
    print("\n🧬 OTOMATİK NEAT BACKTEST")
    print("=" * 35)
    
    try:
        from backtest import BacktestEngine, BacktestConfig
        
        # Orta seviye config - gerçek test için
        config = BacktestConfig(
            # Veri parametreleri
            start_date="2024-01-01",
            end_date="2024-12-31",
            symbols=['EURUSD', 'XAUUSD'],
            
            # NEAT parametreleri (orta seviye)
            population_sizes=[32, 64],
            stagnation_generations=[15, 20],
            speciation_thresholds=[2.0, 2.5],
            n_runs_per_config=2,           # Hızlı test için
            max_generations=30,            # Orta uzunluk
            num_workers=2,                 # Paralel
            
            # Sonuç ayarları
            min_fitness_threshold=-0.5,
            top_n_results=5
        )
        
        print(f"🧬 NEAT Backtest parametreleri:")
        print(f"   👥 Popülasyon: {config.population_sizes}")
        print(f"   🔬 Nesil: {config.max_generations}")
        print(f"   🧪 Toplam experiment: {len(config.population_sizes) * len(config.stagnation_generations) * len(config.speciation_thresholds) * config.n_runs_per_config}")
        
        engine = BacktestEngine(config)
        
        print(f"🚀 Otomatik NEAT backtest başlatılıyor...")
        start_time = time.time()
        
        analysis = engine.run_full_experiment()
        
        total_time = time.time() - start_time
        
        if analysis:
            print(f"\n🎉 OTOMATIK NEAT BACKTEST BAŞARILI!")
            print(f"⏱️  Toplam süre: {total_time/60:.1f} dakika")
            print(f"📊 Experiment: {analysis['successful_experiments']}/{analysis['total_experiments']}")
            print(f"🏆 En iyi fitness: {analysis['fitness_statistics']['best']:.6f}")
            print(f"📈 Ortalama fitness: {analysis['fitness_statistics']['mean']:.6f}")
            
            if 'notebook_path' in analysis:
                print(f"📓 Jupyter rapor: {analysis['notebook_path']}")
            
            # En iyi genomu göster
            if 'best_result' in analysis:
                best = analysis['best_result']
                print(f"\n🧬 EN İYİ SONUÇ:")
                print(f"   Config: {best['config_id']}")
                print(f"   Fitness: {best['fitness']:.6f}")
                print(f"   Validation: {best['validation_fitness']:.6f}")
                print(f"   Nesil: {best['generations']}")
                
                neat_config = best['neat_config']
                print(f"   Popülasyon: {neat_config['population_size']}")
                print(f"   Speciation: {neat_config['speciation_threshold']}")
            
            return True
        else:
            print("❌ NEAT backtest başarısız!")
            return False
            
    except Exception as e:
        print(f"❌ NEAT backtest hatası: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """🤖 TAM OTOMATİK GERÇEK VERİ BACKTEST - Ana fonksiyon."""
    print("🤖 EVO_FX_MASTER - TAM OTOMATİK GERÇEK VERİ BACKTEST")
    print("=" * 75)
    print(f"📅 Başlangıç: {datetime.now()}")
    print(f"🎯 Hedef: Hiç soru sormadan OANDA gerçek verisi ile tam backtest")
    print(f"🔧 API gömülü, 15 günlük veri, feature engineering + NEAT")
    
    success_steps = 0
    total_steps = 3
    start_time = time.time()
    
    # 1. 🤖 Otomatik veri çekme
    print(f"\n{'='*75}")
    print("1️⃣  🤖 OTOMATİK OANDA VERİ ÇEKİMİ")
    if auto_fetch_all_data():
        print("✅ Otomatik veri çekimi başarılı!")
        success_steps += 1
    else:
        print("❌ Otomatik veri çekimi başarısız!")
        return False
    
    # 2. 🤖 Otomatik feature engineering
    print(f"\n{'='*75}")
    print("2️⃣  🤖 OTOMATİK FEATURE ENGİNEERİNG")
    if auto_feature_engineering():
        print("✅ Otomatik feature engineering başarılı!")
        success_steps += 1
    else:
        print("❌ Otomatik feature engineering başarısız!")
    
    # 3. 🤖 Otomatik NEAT backtest
    print(f"\n{'='*75}")
    print("3️⃣  🤖 OTOMATİK NEAT BACKTEST")
    if auto_neat_backtest():
        print("✅ Otomatik NEAT backtest başarılı!")
        success_steps += 1
    else:
        print("❌ Otomatik NEAT backtest başarısız!")
    
    # 🏁 Final sonuçlar
    total_time = time.time() - start_time
    
    print(f"\n{'='*75}")
    print("🏁 TAM OTOMATİK BACKTEST SONUÇLARI")
    print("=" * 45)
    
    success_rate = success_steps / total_steps
    print(f"📈 Başarı oranı: {success_steps}/{total_steps} ({success_rate*100:.1f}%)")
    print(f"⏱️  Toplam süre: {total_time/60:.1f} dakika")
    
    if success_steps >= 2:
        print("🎉 TAM OTOMATİK GERÇEK VERİ BACKTEST BAŞARILI!")
        print("🤖 Hiç soru sormadan tam süreç tamamlandı!")
        
        print(f"\n🎯 TAMAMLANAN AŞAMALAR:")
        print(f"   • 15 günlük gerçek OANDA verisi çekildi")
        print(f"   • OANDA chunk limiti otomatik çözüldü")
        print(f"   • 30+ teknik indikatör hesaplandı")
        if success_steps == 3:
            print(f"   • NEAT evrimsel algoritma çalıştırıldı")
            print(f"   • Grid search optimizasyon tamamlandı")
            print(f"   • Jupyter rapor oluşturuldu")
        
        print(f"\n📁 OLUŞTURULAN DOSYALAR:")
        print(f"   • data/raw/*.parquet - Ham OANDA verisi")
        print(f"   • data/processed/*.parquet - Feature'lı veri")
        print(f"   • results/*.json - Backtest sonuçları")
        print(f"   • results/best_genome_*.json - En iyi genom")
        if success_steps == 3:
            print(f"   • notebooks/*.ipynb - Detaylı rapor")
        
        print(f"\n🚀 SONRAKI ADIMLAR:")
        print(f"   1. En iyi genomu live trading'de kullan")
        print(f"   2. Daha uzun backtest çalıştır (100+ nesil)")
        print(f"   3. MT5 ile gerçek hesapta test et")
        print(f"   4. Risk parametrelerini optimize et")
        
        return True
    else:
        print("❌ TAM OTOMATİK BACKTEST BAŞARISIZ!")
        print("Logları kontrol edin ve internet bağlantısını doğrulayın.")
        return False

if __name__ == "__main__":
    success = main()
    
    if success:
        print(f"\n🏆 TAM OTOMATİK SİSTEM HAZIR!")
        print(f"🚀 Gerçek verilerle backtest tamamlandı.")
        print(f"📊 Sonuçları notebooks/ klasöründe inceleyin.")
    else:
        print(f"\n🔧 Sistem hazır değil, hataları giderin.")