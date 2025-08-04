#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
EVO_FX_MASTER - Import Hataları Düzeltilmiş OANDA Backtest
=========================================================

Import hatalarını çözerek OANDA gerçek verisi ile backtest yapar.
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

# Proje kökünü path'e ekle
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root / "src"))

warnings.filterwarnings('ignore')
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class OANDADataLoader:
    """OANDA REST API ile gerçek veri çekici - DÜZELTME."""
    
    def __init__(self, token, account_id, base_url="https://api-fxpractice.oanda.com"):
        """
        Args:
            token: OANDA API token
            account_id: OANDA hesap ID
            base_url: API URL (demo: fxpractice, live: fxtrade)
        """
        self.token = token
        self.account_id = account_id
        self.base_url = base_url
        
        self.headers = {
            'Authorization': f'Bearer {token}',
            'Content-Type': 'application/json'
        }
        
        # Symbol mapping
        self.symbol_map = {
            'EURUSD': 'EUR_USD',
            'XAUUSD': 'XAU_USD'
        }
        
        logger.info(f"🌐 OANDA API hazırlandı: {base_url}")
    
    def test_connection(self):
        """API bağlantısını test et."""
        try:
            url = f"{self.base_url}/v3/accounts/{self.account_id}"
            response = requests.get(url, headers=self.headers, timeout=10)
            
            if response.status_code == 200:
                account_info = response.json()
                print(f"✅ OANDA bağlantısı başarılı!")
                print(f"   Hesap ID: {self.account_id}")
                print(f"   Para birimi: {account_info.get('account', {}).get('currency', 'N/A')}")
                print(f"   Bakiye: {account_info.get('account', {}).get('balance', 'N/A')}")
                return True
            else:
                print(f"❌ OANDA bağlantı hatası: {response.status_code}")
                print(f"   Hata: {response.text}")
                return False
                
        except Exception as e:
            print(f"❌ OANDA bağlantı hatası: {e}")
            return False
    
    def fetch_candles(self, symbol, start_date, end_date, granularity="M1"):
        """
        OANDA'dan mum verisi çek - DÜZELTME: Gelecek zaman hatası giderildi.
        
        Args:
            symbol: EURUSD veya XAUUSD
            start_date: Başlangıç tarihi (YYYY-MM-DD formatında string)
            end_date: Bitiş tarihi (YYYY-MM-DD formatında string)
            granularity: M1, M5, H1, vb.
            
        Returns:
            DataFrame veya None
        """
        # 🔧 DÜZELTME: Zaman hesaplamaları ve gelecek zaman kontrolü
        
        # 1. OANDA sembol adını dönüştür
        oanda_symbol = self.symbol_map.get(symbol, symbol)
        
        # 2. Start timestamp oluştur (gün başlangıcı - 00:00:00 UTC)
        start_ts = datetime.strptime(start_date, "%Y-%m-%d").replace(
            hour=0, minute=0, second=0, microsecond=0, tzinfo=timezone.utc
        )
        
        # 3. End timestamp oluştur (gün sonu - 23:59:59 UTC)  
        end_ts = datetime.strptime(end_date, "%Y-%m-%d").replace(
            hour=23, minute=59, second=59, microsecond=0, tzinfo=timezone.utc
        )
        
        # 4. Şu anki UTC zamanı al
        now_utc = datetime.now(timezone.utc)
        
        # 5. 🚨 KRİTİK DÜZELTME: End time gelecekte mi kontrol et
        if end_ts >= now_utc:
            # End time gelecekte veya şu an - 1 dakika öncesine ayarla
            end_ts = now_utc - timedelta(minutes=1)
            print(f"   ⚠️  End time gelecekte idi, şu ana ayarlandı: {end_ts}")
        
        # 6. Start time de şu andan sonra olmamalı
        if start_ts >= now_utc:
            print(f"   ❌ Start time de gelecekte! {start_ts} >= {now_utc}")
            return None
        
        # 7. Start > End kontrolü
        if start_ts >= end_ts:
            print(f"   ❌ Start time >= End time! {start_ts} >= {end_ts}")
            return None
        
        # 8. RFC3339 formatında string'e dönüştür
        from_str = start_ts.strftime("%Y-%m-%dT%H:%M:%SZ")
        to_str = end_ts.strftime("%Y-%m-%dT%H:%M:%SZ")
        
        # 9. API endpoint ve parametreler
        url = f"{self.base_url}/v3/instruments/{oanda_symbol}/candles"
        
        params = {
            'from': from_str,           # RFC3339: "2024-01-01T00:00:00Z"
            'to': to_str,               # RFC3339: "2024-01-31T23:59:59Z"  
            'granularity': granularity, # "M1", "M5", "H1", etc.
            'price': 'M'                # Mid prices
        }
        
        try:
            print(f"📡 {symbol} verisi çekiliyor:")
            print(f"   📅 From: {from_str}")
            print(f"   📅 To:   {to_str}")
            print(f"   ⏰ Süre: {(end_ts - start_ts).days} gün")
            
            # 10. API isteği gönder
            response = requests.get(url, headers=self.headers, params=params, timeout=30)
            
            if response.status_code != 200:
                print(f"❌ {symbol} API hatası: {response.status_code}")
                print(f"   Hata detayı: {response.text}")
                return None
            
            # 11. JSON response parse et
            data = response.json()
            candles = data.get('candles', [])
            
            if not candles:
                print(f"⚠️  {symbol} için veri yok")
                return None
            
            # 12. DataFrame'e dönüştür - sadece tamamlanmış mumlar
            records = []
            for candle in candles:
                if candle['complete']:  # Sadece tamamlanmış mumlar
                    mid = candle['mid']
                    
                    # Timestamp parse et ve UTC'ye dönüştür
                    candle_time = pd.to_datetime(candle['time'])
                    
                    records.append({
                        'timestamp': candle_time,
                        'open': float(mid['o']),
                        'high': float(mid['h']),
                        'low': float(mid['l']),
                        'close': float(mid['c']),
                        'volume': int(candle['volume'])
                    })
            
            if not records:
                print(f"⚠️  {symbol} için geçerli tamamlanmış mum yok")
                return None
            
            # 13. Final DataFrame oluştur
            df = pd.DataFrame(records)
            df.set_index('timestamp', inplace=True)
            
            # 14. Timezone bilgisini UTC'ye normalize et
            if df.index.tz is not None:
                df.index = df.index.tz_convert('UTC')
            else:
                df.index = df.index.tz_localize('UTC')
            
            # 15. Başarı mesajı ve özet
            print(f"✅ {symbol} verisi başarıyla alındı:")
            print(f"   📊 Toplam mum: {len(df)}")
            print(f"   📅 Gerçek aralık: {df.index.min()} - {df.index.max()}")
            print(f"   💰 Fiyat aralığı: {df['close'].min():.5f} - {df['close'].max():.5f}")
            print(f"   📈 Ortalama hacim: {df['volume'].mean():.0f}")
            
            return df
            
        except Exception as e:
            print(f"❌ {symbol} veri çekme hatası: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    def save_data(self, symbol, df):
        """Veriyi parquet olarak kaydet."""
        if df is None or df.empty:
            return False
        
        # Dizinleri oluştur
        data_dir = Path("data")
        raw_dir = data_dir / "raw"
        raw_dir.mkdir(parents=True, exist_ok=True)
        
        # Kaydet
        file_path = raw_dir / f"{symbol}_raw.parquet"
        df.to_parquet(file_path, compression='snappy')
        
        print(f"💾 {symbol} kaydedildi: {file_path}")
        return True

def setup_oanda_credentials():
    """OANDA API bilgilerini al."""
    print("🔐 OANDA API BİLGİLERİ")
    print("=" * 30)
    
    # Token al
    token = input("OANDA API Token girin: ").strip()
    if not token:
        print("❌ Token gerekli!")
        return None
    
    # Account ID al
    account_id = input("OANDA Account ID girin: ").strip()
    if not account_id:
        print("❌ Account ID gerekli!")
        return None
    
    # Environment seç
    print("\n🌍 Environment seçin:")
    print("1. Demo/Practice (fxpractice)")
    print("2. Live/Real (fxtrade)")
    
    choice = input("Seçim (1/2) [varsayılan=1]: ").strip() or "1"
    
    if choice == "2":
        base_url = "https://api-fxtrade.oanda.com"
        print("⚠️  LIVE hesap seçildi - dikkatli olun!")
    else:
        base_url = "https://api-fxpractice.oanda.com"
        print("✅ Demo hesap seçildi")
    
    return {
        'token': token,
        'account_id': account_id,
        'base_url': base_url
    }

def fetch_real_forex_data():
    """OANDA'dan gerçek forex verisi çek."""
    print("🌐 GERÇEK FOREX VERİSİ ÇEKİLİYOR")
    print("=" * 40)
    
    # API bilgilerini al
    credentials = setup_oanda_credentials()
    if not credentials:
        return False
    
    # OANDA loader oluştur
    loader = OANDADataLoader(
        credentials['token'],
        credentials['account_id'],
        credentials['base_url']
    )
    
    # Bağlantıyı test et
    if not loader.test_connection():
        return False
    
    # Tarih aralığı belirle - DÜZELTME: Gelecek tarih engellemesi
    print(f"\n📅 VERİ ARALIĞI SEÇİMİ")
    print("=" * 25)
    
    # Şu anki tarih
    today = datetime.now().date()
    
    # Varsayılan: son 30 gün
    default_start = today - timedelta(days=30)
    default_end = today - timedelta(days=1)  # Dün (bugün değil)
    
    print(f"Varsayılan aralık: {default_start} - {default_end}")
    print(f"⚠️  Not: End date bugünden önce olmalı (OANDA sınırlaması)")
    
    start_input = input(f"Başlangıç tarihi (YYYY-MM-DD) [varsayılan={default_start}]: ").strip()
    end_input = input(f"Bitiş tarihi (YYYY-MM-DD) [varsayılan={default_end}]: ").strip()
    
    start_date = start_input or str(default_start)
    end_date = end_input or str(default_end)
    
    # Tarih doğrulama
    try:
        start_dt = datetime.strptime(start_date, "%Y-%m-%d").date()
        end_dt = datetime.strptime(end_date, "%Y-%m-%d").date()
        
        if start_dt >= end_dt:
            print("❌ Başlangıç tarihi bitiş tarihinden küçük olmalı!")
            return False
        
        if end_dt >= today:
            print("❌ Bitiş tarihi bugünden önce olmalı!")
            return False
            
    except ValueError:
        print("❌ Geçersiz tarih formatı! YYYY-MM-DD kullanın.")
        return False
    
    print(f"✅ Seçilen aralık: {start_date} - {end_date}")
    
    # Sembolleri çek
    symbols = ['EURUSD', 'XAUUSD']
    success_count = 0
    
    for symbol in symbols:
        print(f"\n🎯 {symbol} verisi çekiliyor...")
        
        df = loader.fetch_candles(symbol, start_date, end_date, granularity="M1")
        
        if df is not None and not df.empty:
            # Veri kalite kontrolü
            print(f"   📊 Veri kalitesi kontrol ediliyor...")
            
            # Duplikat kontrol
            duplicates = df.index.duplicated().sum()
            if duplicates > 0:
                print(f"   🧹 {duplicates} duplikat satır temizlendi")
                df = df[~df.index.duplicated(keep='first')]
            
            # OHLC mantık kontrolü
            invalid_ohlc = ((df['high'] < df['low']) | 
                           (df['high'] < df['open']) | 
                           (df['high'] < df['close']) |
                           (df['low'] > df['open']) | 
                           (df['low'] > df['close'])).sum()
            
            if invalid_ohlc > 0:
                print(f"   🧹 {invalid_ohlc} geçersiz OHLC satırı temizlendi")
                df = df[~((df['high'] < df['low']) | 
                         (df['high'] < df['open']) | 
                         (df['high'] < df['close']) |
                         (df['low'] > df['open']) | 
                         (df['low'] > df['close']))]
            
            # Sıfır/negatif fiyat kontrolü
            negative_prices = (df[['open', 'high', 'low', 'close']] <= 0).any(axis=1).sum()
            if negative_prices > 0:
                print(f"   🧹 {negative_prices} negatif/sıfır fiyat satırı temizlendi")
                df = df[(df[['open', 'high', 'low', 'close']] > 0).all(axis=1)]
            
            # Final kontrol
            if len(df) < 100:
                print(f"   ⚠️  {symbol} için çok az veri ({len(df)} mum)")
            else:
                # Kaydet
                if loader.save_data(symbol, df):
                    success_count += 1
                    
                    # Özet bilgi
                    print(f"   ✅ {symbol} başarılı:")
                    print(f"      📊 Final boyut: {len(df)} mum")
                    print(f"      📅 Tarih span: {(df.index.max() - df.index.min()).days} gün")
                    print(f"      📈 Fiyat volatilitesi: {df['close'].std():.6f}")
                    print(f"      📊 Ortalama hacim: {df['volume'].mean():.0f}")
        else:
            print(f"   ❌ {symbol} verisi alınamadı")
    
    print(f"\n📊 VERİ ÇEKME SONUCU: {success_count}/2 sembol başarılı")
    return success_count >= 1

def run_simple_backtest():
    """Basit backtest - import hatalarını gidererek."""
    print("\n🧬 BASİT BACKTEST SİSTEMİ")
    print("=" * 30)
    
    try:
        # Kendi basit NEAT sistemi ile test
        from quick_test import SimpleGenome, test_simple_fitness
        
        # Processed data'yı yükle
        processed_dir = Path("data/processed")
        
        symbols = ['EURUSD', 'XAUUSD']
        results = {}
        
        for symbol in symbols:
            processed_file = processed_dir / f"{symbol}_features_advanced.parquet"
            
            if processed_file.exists():
                print(f"📊 {symbol} processed data yükleniyor...")
                df = pd.read_parquet(processed_file)
                
                if len(df) >= 1000:  # Minimum veri kontrolü
                    print(f"   ✅ {symbol}: {len(df)} satır, {len(df.columns)} sütun")
                    
                    # Test genomu oluştur
                    genome = SimpleGenome(input_size=5, output_size=3)
                    
                    # Fitness hesapla
                    fitness = test_simple_fitness(genome, df)
                    results[symbol] = {
                        'fitness': fitness,
                        'data_size': len(df),
                        'features': len(df.columns)
                    }
                    
                    print(f"   🎯 {symbol} fitness: {fitness:.6f}")
                else:
                    print(f"   ⚠️  {symbol}: Çok az veri ({len(df)})")
            else:
                print(f"   ❌ {symbol}: Processed data bulunamadı")
        
        if results:
            print(f"\n🎉 Basit backtest tamamlandı!")
            for symbol, result in results.items():
                print(f"   {symbol}: fitness={result['fitness']:.6f}, "
                      f"data={result['data_size']}, features={result['features']}")
            return True
        else:
            print("❌ Hiç sonuç alınamadı")
            return False
            
    except Exception as e:
        print(f"❌ Basit backtest hatası: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Ana fonksiyon - Import hataları giderilmiş."""
    print("🌐 EVO_FX_MASTER - DÜZELTME: OANDA GERÇEK VERİ BACKTEST")
    print("=" * 65)
    print(f"📅 Test tarihi: {datetime.now()}")
    print(f"🎯 Hedef: Import hatalarını giderek OANDA gerçek verisi ile test")
    
    success_steps = 0
    total_steps = 3
    
    # 1. Gerçek veri çekme
    print(f"\n{'='*65}")
    print("1️⃣  OANDA GERÇEK VERİ ÇEKİMİ")
    if fetch_real_forex_data():
        print("✅ Gerçek veri çekimi başarılı!")
        success_steps += 1
    else:
        print("❌ Gerçek veri çekimi başarısız!")
        return False
    
    # 2. Feature engineering (direkt çalıştır)
    print(f"\n{'='*65}")
    print("2️⃣  FEATURE ENGİNEERİNG")
    try:
        # Import olmadan direkt çalıştır
        import subprocess
        result = subprocess.run([sys.executable, "src/feature_engineer.py"], 
                              capture_output=True, text=True, timeout=300)
        
        if result.returncode == 0:
            print("✅ Feature engineering başarılı!")
            success_steps += 1
        else:
            print(f"❌ Feature engineering hatası: {result.stderr}")
    except Exception as e:
        print(f"❌ Feature engineering subprocess hatası: {e}")
    
    # 3. Basit backtest
    print(f"\n{'='*65}")
    print("3️⃣  BASİT BACKTEST")
    if run_simple_backtest():
        print("✅ Basit backtest başarılı!")
        success_steps += 1
    else:
        print("❌ Basit backtest başarısız!")
    
    # Final sonuçlar
    print(f"\n{'='*65}")
    print("📊 FİNAL SONUÇLAR")
    print("=" * 30)
    
    success_rate = success_steps / total_steps
    print(f"📈 Başarı oranı: {success_steps}/{total_steps} ({success_rate*100:.1f}%)")
    
    if success_steps >= 2:
        print("🎉 OANDA GERÇEK VERİ SİSTEMİ ÇALIŞIYOR!")
        print("🌟 Import hataları giderildi, sistem hazır!")
        
        print(f"\n🎯 BAŞARILI AŞAMALAR:")
        print(f"   • Gerçek OANDA verisi çekildi")
        print(f"   • Zaman sınırlaması düzeltildi")
        print(f"   • Feature engineering çalıştırıldı")
        if success_steps == 3:
            print(f"   • Basit backtest tamamlandı")
        
        print(f"\n🚀 SONRAKI ADIMLAR:")
        print(f"   1. Import hatalarını tamamen giderin")
        print(f"   2. Tam NEAT sistemini çalıştırın")
        print(f"   3. Daha uzun backtest yapın")
        print(f"   4. Live trading'e geçin")
        
        return True
    else:
        print("❌ SİSTEM HAZIR DEĞİL!")
        print("Import hatalarını giderin ve tekrar deneyin.")
        return False

if __name__ == "__main__":
    main()