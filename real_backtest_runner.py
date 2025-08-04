#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
EVO_FX_MASTER - Gerçek Sistem Backtest
=====================================

GitHub'daki tam NEAT sistemini kullanarak kapsamlı backtest.
"""

import sys
import os
import numpy as np
import pandas as pd
import logging
import warnings
from pathlib import Path
from datetime import datetime, timedelta
import time

# Proje kökünü path'e ekle
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root / "src"))

warnings.filterwarnings('ignore')
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_all_modules():
    """Tüm modülleri test et."""
    print("🧪 TÜM MODÜLLER TEST EDİLİYOR")
    print("=" * 50)
    
    modules_to_test = [
        ('genome', 'Genom sistemi'),
        ('feature_engineer', 'Feature engineering'),
        ('neat_engine', 'NEAT algoritması'),
        ('fitness', 'Fitness evaluation'),
        ('backtest', 'Backtest engine'),
        ('risk', 'Risk management'),
        ('viz', 'Görselleştirme')
    ]
    
    results = {}
    
    for module_name, description in modules_to_test:
        print(f"\n🔍 {description} test ediliyor...")
        try:
            # Import test
            module = __import__(module_name)
            print(f"   ✅ Import başarılı: {module_name}")
            
            # Main function test
            if hasattr(module, 'main'):
                print(f"   🏃 {module_name}.main() çalıştırılıyor...")
                module.main()
                print(f"   ✅ {description} test başarılı!")
                results[module_name] = True
            else:
                print(f"   ⚠️  {module_name} main() fonksiyonu yok")
                results[module_name] = True
                
        except Exception as e:
            print(f"   ❌ {description} test hatası: {e}")
            results[module_name] = False
    
    # Sonuçları özetle
    print(f"\n📊 MODÜL TEST SONUÇLARI")
    print("=" * 30)
    
    passed = sum(results.values())
    total = len(results)
    
    for module, success in results.items():
        status = "✅ BAŞARILI" if success else "❌ BAŞARISIZ"
        print(f"   {module.upper()}: {status}")
    
    print(f"\n📈 GENEL SONUÇ: {passed}/{total} modül başarılı")
    
    return passed == total

def run_comprehensive_test():
    """Kapsamlı modül testi."""
    print("\n🔬 KAPSAMLI MODÜL TESTİ")
    print("=" * 40)
    
    try:
        # Test modules script çalıştır
        print("🧪 test_modules.py çalıştırılıyor...")
        exec(open('test_modules.py').read())
        return True
    except Exception as e:
        print(f"❌ Kapsamlı test hatası: {e}")
        return False

def create_synthetic_data_environment():
    """Test için sentetik veri ortamı oluştur."""
    print("\n📊 SENTETİK VERİ ORTAMI OLUŞTURULUYOR")
    print("=" * 45)
    
    # Dizinleri oluştur
    data_dir = Path("data")
    raw_dir = data_dir / "raw"
    processed_dir = data_dir / "processed"
    
    for directory in [data_dir, raw_dir, processed_dir]:
        directory.mkdir(exist_ok=True)
        print(f"✅ Dizin oluşturuldu: {directory}")
    
    # Her sembol için sentetik veri oluştur
    symbols = ['EURUSD', 'XAUUSD']
    
    for symbol in symbols:
        print(f"\n🎯 {symbol} sentetik verisi oluşturuluyor...")
        
        # 60 günlük 1-dakikalık veri
        start_date = datetime.now() - timedelta(days=60)
        end_date = datetime.now()
        dates = pd.date_range(start_date, end_date, freq='1min', tz='UTC')
        
        # Market saatleri filtreleme (forex market 5 gün açık)
        dates = dates[dates.dayofweek < 5]  # Hafta içi
        
        n_samples = len(dates)
        print(f"   📅 Tarih aralığı: {dates[0]} - {dates[-1]}")
        print(f"   📊 Toplam mum sayısı: {n_samples}")
        
        # Sembol spesifik parametreler
        if symbol == 'EURUSD':
            base_price = 1.1000
            daily_vol = 0.008
            noise_level = 0.0001
        else:  # XAUUSD
            base_price = 2000.0
            daily_vol = 0.015
            noise_level = 0.0005
        
        # Price simulation
        np.random.seed(42 if symbol == 'EURUSD' else 43)
        
        # Trend + noise
        trend = np.cumsum(np.random.normal(0, noise_level/5, n_samples))
        noise = np.random.normal(0, noise_level, n_samples)
        
        # Session-based volatility
        volatility = np.ones(n_samples) * daily_vol / np.sqrt(1440)
        for i, date in enumerate(dates):
            hour = date.hour
            if 8 <= hour <= 16:      # London session
                volatility[i] *= 1.3
            elif 13 <= hour <= 21:   # NY session
                volatility[i] *= 1.2
            elif 0 <= hour <= 8:     # Asian session
                volatility[i] *= 0.8
        
        # Final returns
        returns = trend + noise + np.random.normal(0, 1, n_samples) * volatility
        prices = base_price * np.exp(np.cumsum(returns))
        
        # OHLCV construction
        df = pd.DataFrame(index=dates)
        df['close'] = prices
        df['open'] = df['close'].shift(1).fillna(base_price)
        
        # Realistic high/low
        true_range = np.abs(returns) * np.random.uniform(1.2, 2.5, n_samples)
        df['high'] = np.maximum(df['open'], df['close']) + true_range * prices * 0.5
        df['low'] = np.minimum(df['open'], df['close']) - true_range * prices * 0.5
        
        # Volume
        base_volume = 500 if symbol == 'EURUSD' else 200
        df['volume'] = (base_volume * np.random.lognormal(0, 0.3, n_samples)).astype(int)
        
        # Data quality
        df = df.dropna()
        df = df[df['high'] >= df['low']]
        df = df[df['volume'] > 0]
        
        # Parquet olarak kaydet
        raw_file = raw_dir / f"{symbol}_raw.parquet"
        df.to_parquet(raw_file, compression='snappy')
        
        print(f"   ✅ {symbol} verisi kaydedildi: {raw_file}")
        print(f"   📈 Final boyut: {len(df)} satır")
        print(f"   💰 Fiyat aralığı: {df['close'].min():.5f} - {df['close'].max():.5f}")
    
    return True

def run_feature_engineering():
    """Feature engineering çalıştır."""
    print("\n🔧 FEATURE ENGİNEERİNG ÇALIŞTIRILIYOR")
    print("=" * 45)
    
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
            for symbol, df in results.items():
                print(f"✅ {symbol}: {len(df)} satır, {len(df.columns)} özellik")
            
            print(f"🎉 Feature engineering başarılı: {len(results)}/2 sembol")
            return True
        else:
            print("❌ Feature engineering başarısız")
            return False
            
    except Exception as e:
        print(f"❌ Feature engineering hatası: {e}")
        return False

def run_neat_backtest():
    """NEAT backtest çalıştır."""
    print("\n🧬 NEAT BACKTEST ÇALIŞTIRILIYOR")
    print("=" * 40)
    
    try:
        from backtest import BacktestEngine, BacktestConfig
        
        print("⚙️  Backtest config oluşturuluyor...")
        
        # Test için optimize edilmiş config
        config = BacktestConfig(
            # Veri parametreleri
            start_date="2024-01-01",
            end_date="2024-12-31",
            symbols=['EURUSD', 'XAUUSD'],
            train_ratio=0.8,
            
            # Grid search (test için küçük)
            population_sizes=[16, 32],
            stagnation_generations=[10, 15],
            speciation_thresholds=[2.0, 2.5],
            n_runs_per_config=2,        # Az run
            max_generations=20,         # Kısa evrim
            num_workers=2,              # Paralel
            
            # Sonuç ayarları
            min_fitness_threshold=-0.5,
            top_n_results=5
        )
        
        print(f"📊 Backtest parametreleri:")
        print(f"   Popülasyon boyutları: {config.population_sizes}")
        print(f"   Nesil sayısı: {config.max_generations}")
        print(f"   Toplam konfigürasyon: {len(config.population_sizes) * len(config.stagnation_generations) * len(config.speciation_thresholds)}")
        print(f"   Toplam run: {len(config.population_sizes) * len(config.stagnation_generations) * len(config.speciation_thresholds) * config.n_runs_per_config}")
        
        # BacktestEngine oluştur
        engine = BacktestEngine(config)
        
        print("\n🚀 Tam backtest başlatılıyor...")
        start_time = time.time()
        
        analysis = engine.run_full_experiment()
        
        total_time = time.time() - start_time
        
        if analysis:
            print(f"\n🎉 NEAT BACKTEST TAMAMLANDI!")
            print(f"⏱️  Toplam süre: {total_time/60:.1f} dakika")
            print(f"📊 Toplam experiment: {analysis['total_experiments']}")
            print(f"✅ Başarılı experiment: {analysis['successful_experiments']}")
            print(f"🏆 En iyi fitness: {analysis['fitness_statistics']['best']:.6f}")
            print(f"📈 Ortalama fitness: {analysis['fitness_statistics']['mean']:.6f}")
            print(f"🔬 Final tür sayısı: {len(analysis.get('species_count_history', [0]))}")
            
            if 'notebook_path' in analysis:
                print(f"📓 Jupyter rapor: {analysis['notebook_path']}")
            
            # En iyi genomu göster
            if 'best_result' in analysis:
                best = analysis['best_result']
                print(f"\n🧬 EN İYİ GENOM:")
                print(f"   Config ID: {best['config_id']}")
                print(f"   Fitness: {best['fitness']:.6f}")
                print(f"   Validation: {best['validation_fitness']:.6f}")
                print(f"   Tamamlanan nesil: {best['generations']}")
                
                neat_config = best['neat_config']
                print(f"   Popülasyon: {neat_config['population_size']}")
                print(f"   Stagnasyon: {neat_config['stagnation_generations']}")
                print(f"   Spesiation: {neat_config['speciation_threshold']}")
            
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
    """Ana backtest runner."""
    print("🚀 EVO_FX_MASTER - GERÇEK SİSTEM BACKTEST")
    print("=" * 60)
    print(f"📅 Test tarihi: {datetime.now()}")
    print(f"📁 Çalışma dizini: {os.getcwd()}")
    
    success_count = 0
    total_tests = 5
    
    # 1. Modül testleri
    print(f"\n{'='*60}")
    print("1️⃣  MODÜL TESTLERİ")
    if test_all_modules():
        print("✅ Modül testleri başarılı!")
        success_count += 1
    else:
        print("❌ Modül testleri başarısız!")
    
    # 2. Kapsamlı test
    print(f"\n{'='*60}")
    print("2️⃣  KAPSAMLI TEST")
    if run_comprehensive_test():
        print("✅ Kapsamlı test başarılı!")
        success_count += 1
    else:
        print("❌ Kapsamlı test başarısız!")
    
    # 3. Veri ortamı
    print(f"\n{'='*60}")
    print("3️⃣  VERİ ORTAMI KURULUMU")
    if create_synthetic_data_environment():
        print("✅ Veri ortamı başarılı!")
        success_count += 1
    else:
        print("❌ Veri ortamı başarısız!")
    
    # 4. Feature engineering
    print(f"\n{'='*60}")
    print("4️⃣  FEATURE ENGİNEERİNG")
    if run_feature_engineering():
        print("✅ Feature engineering başarılı!")
        success_count += 1
    else:
        print("❌ Feature engineering başarısız!")
    
    # 5. NEAT backtest
    print(f"\n{'='*60}")
    print("5️⃣  NEAT BACKTEST")
    if run_neat_backtest():
        print("✅ NEAT backtest başarılı!")
        success_count += 1
    else:
        print("❌ NEAT backtest başarısız!")
    
    # Final sonuçlar
    print(f"\n{'='*60}")
    print("📊 FİNAL SONUÇLAR")
    print("=" * 30)
    
    print(f"📈 Başarı oranı: {success_count}/{total_tests} ({success_count/total_tests*100:.1f}%)")
    
    if success_count == total_tests:
        print("🎉 TÜM TESTLER BAŞARILI!")
        print("🚀 EVO_FX_MASTER sistemi tamamen hazır!")
        
        print(f"\n📁 Oluşturulan dosyalar:")
        print(f"   • data/raw/*.parquet - Ham veriler")
        print(f"   • data/processed/*.parquet - İşlenmiş veriler") 
        print(f"   • results/*.json - Backtest sonuçları")
        print(f"   • notebooks/*.ipynb - Jupyter raporları")
        print(f"   • logs/*.log - Detaylı loglar")
        
        print(f"\n🎯 Sonraki adımlar:")
        print(f"   1. OANDA API token ekleyerek gerçek veri")
        print(f"   2. MT5 terminal bağlayarak live trading")
        print(f"   3. Daha uzun backtest (100+ nesil)")
        print(f"   4. Risk parametreleri fine-tuning")
        
    elif success_count >= 3:
        print("⚠️  KISMEN BAŞARILI!")
        print("Temel sistem çalışıyor, bazı iyileştirmeler gerekli.")
    else:
        print("❌ ÇOK SAYIDA BAŞARISIZLIK!")
        print("Sistem kurulumunu kontrol edin.")
    
    print(f"\n💡 Yardım:")
    print(f"   • Hata logları: backtest_run.log")
    print(f"   • Modül testleri: python test_modules.py")
    print(f"   • Tek modül test: python src/[module_name].py")
    
    return success_count == total_tests

if __name__ == "__main__":
    success = main()
    
    if success:
        print(f"\n🏆 EVO_FX_MASTER hazır! Live trading başlatabilirsiniz:")
        print(f"python src/live_trader.py")
    else:
        print(f"\n🔧 Sorun giderme gerekli. Logları kontrol edin.")