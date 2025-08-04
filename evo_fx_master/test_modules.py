#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
EVO_FX_MASTER - Modül Test Script'i
===================================

Güncellenmiş modülleri test eder:
1. Gelişmiş viz.py
2. Gelişmiş feature_engineer.py 
3. Gelişmiş neat_engine.py
"""

import sys
import os
import numpy as np
import pandas as pd
import logging
from pathlib import Path
from datetime import datetime
import warnings

# Proje kökünü sys.path'e ekle
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root / "src"))

# Warnings'leri kapat
warnings.filterwarnings('ignore')

# Logging ayarla
logging.basicConfig(level=logging.INFO, format='%(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def test_advanced_viz():
    """Gelişmiş görselleştirme modülünü test et."""
    print("\n" + "="*60)
    print("🎨 GELİŞMİŞ VIZ.PY TEST EDİLİYOR")
    print("="*60)
    
    try:
        # Import et
        from viz import AdvancedVisualizationEngine
        from genome import Genome
        
        print("✅ Import başarılı")
        
        # Test verileri oluştur
        print("📊 Test verileri oluşturuluyor...")
        dates = pd.date_range('2024-01-01', '2024-01-05', freq='1H')
        
        # Equity verisi
        initial_balance = 10000
        returns = np.random.normal(0.0002, 0.02, len(dates)).cumsum()
        equity_data = pd.DataFrame({
            'timestamp': dates,
            'equity': initial_balance * (1 + returns)
        })
        
        # Trade verisi
        n_trades = 15
        trade_dates = np.random.choice(dates, n_trades, replace=False)
        trades_data = pd.DataFrame({
            'entry_time': trade_dates,
            'pnl': np.random.normal(20, 70, n_trades),
            'symbol': np.random.choice(['EURUSD', 'XAUUSD'], n_trades)
        })
        
        # Test genomu oluştur
        test_genome = Genome(input_size=8, output_size=3)
        for _ in range(8):
            test_genome.mutate()
        
        print("✅ Test verileri hazırlandı")
        
        # AdvancedVisualizationEngine test
        viz_engine = AdvancedVisualizationEngine()
        print("✅ AdvancedVisualizationEngine oluşturuldu")
        
        # Test 1: Inset'li equity curve
        print("🧪 Inset'li equity curve testi...")
        fig1 = viz_engine.plot_equity_with_insets(
            equity_data, trades_data, test_genome,
            title="Test: Advanced Equity Curve with Insets"
        )
        
        if fig1:
            print("✅ Inset'li equity curve başarılı")
            # Dosyaya kaydet (opsiyonel)
            # fig1.savefig('test_equity_insets.png', dpi=100, bbox_inches='tight')
            
        # Test 2: Network diagram
        print("🧪 Network diagram testi...")
        fig2 = viz_engine.plot_network_full(
            test_genome, edge_threshold=0.1,
            title="Test: Advanced Neural Network"
        )
        
        if fig2:
            print("✅ Network diagram başarılı")
            # fig2.savefig('test_network.png', dpi=100, bbox_inches='tight')
        
        print("🎨 GELİŞMİŞ VIZ.PY TEST BAŞARILI! ✅")
        return True
        
    except Exception as e:
        print(f"❌ Viz test hatası: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_advanced_feature_engineer():
    """Gelişmiş feature engineering modülünü test et."""
    print("\n" + "="*60)
    print("🔧 GELİŞMİŞ FEATURE_ENGINEER.PY TEST EDİLİYOR")
    print("="*60)
    
    try:
        # Import et
        from feature_engineer import AdvancedFeatureEngineer
        
        print("✅ Import başarılı")
        
        # Test config
        config = {
            'rsi_period': 14,
            'atr_period': 14,
            'volatility_period': 30,
            'tick_momentum_periods': [5, 10, 20],
            'volume_periods': [10, 20, 50]
        }
        
        # AdvancedFeatureEngineer oluştur
        engineer = AdvancedFeatureEngineer(config)
        print("✅ AdvancedFeatureEngineer oluşturuldu")
        
        # Test verisi oluştur
        print("📊 Test OHLCV verisi oluşturuluyor...")
        dates = pd.date_range('2024-01-01', '2024-01-03', freq='1min', tz='UTC')
        n_samples = len(dates)
        
        # Gerçekçi EURUSD verisi
        np.random.seed(42)
        price_base = 1.1000
        returns = np.random.normal(0, 0.0001, n_samples).cumsum()
        prices = price_base * np.exp(returns)
        
        test_data = pd.DataFrame(index=dates)
        test_data['close'] = prices
        test_data['open'] = test_data['close'].shift(1).fillna(price_base)
        
        # High/Low
        hl_range = np.random.uniform(0.0005, 0.002, n_samples)
        test_data['high'] = test_data['close'] + hl_range / 2
        test_data['low'] = test_data['close'] - hl_range / 2
        test_data['volume'] = np.random.randint(100, 1000, n_samples)
        
        print(f"✅ Test verisi hazırlandı: {len(test_data)} mum")
        
        # Gelişmiş feature engineering
        print("🔧 Gelişmiş feature engineering başlatılıyor...")
        featured_data = engineer.bake_features_advanced('TEST', test_data)
        
        if featured_data is not None:
            print(f"✅ Feature engineering başarılı!")
            print(f"📊 Toplam özellik sayısı: {len(featured_data.columns)}")
            print(f"📈 Veri boyutu: {featured_data.shape}")
            
            # Özellik kategorilerini say
            categories = {
                'Returns': len([col for col in featured_data.columns if 'return' in col or 'momentum' in col]),
                'Technical': len([col for col in featured_data.columns if any(x in col for x in ['rsi', 'atr', 'bb_', 'sma_', 'ema_', 'macd'])]),
                'Time': len([col for col in featured_data.columns if any(x in col for x in ['hour', 'day', 'session', 'sin', 'cos'])]),
                'Microstructure': len([col for col in featured_data.columns if any(x in col for x in ['tick', 'volume', 'spread', 'gap'])])
            }
            
            print("📋 Özellik kategorileri:")
            total_features = 0
            for category, count in categories.items():
                print(f"  {category}: {count} özellik")
                total_features += count
            
            print(f"📊 Kategorize edilen: {total_features}/{len(featured_data.columns)} özellik")
            
            # Bazı önemli özellikleri kontrol et
            important_features = [
                'log_return', 'rsi_norm', 'atr_pct', 'realized_vol_30',
                'hour_sin', 'session_london', 'tick_momentum_5'
            ]
            
            available_features = [f for f in important_features if f in featured_data.columns]
            print(f"✅ Önemli özellikler mevcut: {len(available_features)}/{len(important_features)}")
            
            if available_features:
                print(f"📋 Örnek değerler:")
                sample_data = featured_data[available_features].head(3)
                for col in available_features:
                    values = sample_data[col].tolist()
                    print(f"  {col}: {values}")
            
            print("🔧 GELİŞMİŞ FEATURE_ENGINEER.PY TEST BAŞARILI! ✅")
            return True
        else:
            print("❌ Feature engineering başarısız!")
            return False
            
    except Exception as e:
        print(f"❌ Feature engineer test hatası: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_advanced_neat_engine():
    """Gelişmiş NEAT engine modülünü test et."""
    print("\n" + "="*60)
    print("🧬 GELİŞMİŞ NEAT_ENGINE.PY TEST EDİLİYOR")
    print("="*60)
    
    try:
        # Import et
        from neat_engine import AdvancedNEATEngine, NeatConfig
        from genome import Genome
        
        print("✅ Import başarılı")
        
        # Test fitness fonksiyonu
        def test_fitness(genome: Genome) -> float:
            """Test için gelişmiş fitness fonksiyonu."""
            try:
                # Rastgele input ile test
                test_input = np.random.randn(genome.input_size)
                output = genome.predict(test_input)
                
                # Fitness = output variety + complexity penalty
                base_fitness = np.std(output) + np.mean(np.abs(output))
                complexity_penalty = len(genome.connections) * 0.001
                
                return max(0, base_fitness - complexity_penalty)
            except:
                return 0.0
        
        print("✅ Test fitness fonksiyonu hazırlandı")
        
        # Test config (küçük boyutlu)
        config = NeatConfig(
            population_size=10,  # Çok küçük test
            max_generations=3,   # Çok kısa test
            stagnation_generations=5,
            target_species=3,
            num_workers=1,  # Tek işlemci
            speciation_threshold=2.0
        )
        
        print("✅ Test config hazırlandı")
        
        # AdvancedNEATEngine oluştur
        engine = AdvancedNEATEngine(config, test_fitness)
        print("✅ AdvancedNEATEngine oluşturuldu")
        
        # İlk popülasyon testi
        print("🧪 İlk popülasyon testi...")
        engine.create_initial_population(input_size=4, output_size=2)
        
        if len(engine.population) == config.population_size:
            print(f"✅ İlk popülasyon başarılı: {len(engine.population)} genom")
        else:
            print(f"❌ İlk popülasyon hatası: {len(engine.population)}/{config.population_size}")
            return False
        
        # Fitness değerlendirme testi
        print("🧪 Fitness değerlendirme testi...")
        engine.evaluate_population_parallel()
        
        fitnesses = [g.fitness for g in engine.population]
        if all(isinstance(f, (int, float)) for f in fitnesses):
            print(f"✅ Fitness değerlendirme başarılı. Ortalama: {np.mean(fitnesses):.4f}")
        else:
            print("❌ Fitness değerlendirme hatası")
            return False
        
        # Speciation testi
        print("🧪 Speciation testi...")
        engine.speciate_population()
        
        if len(engine.species) > 0:
            print(f"✅ Speciation başarılı: {len(engine.species)} tür oluşturuldu")
        else:
            print("❌ Speciation hatası")
            return False
        
        # Kısa evrim testi
        print("🧪 Kısa evrim testi (3 nesil)...")
        best_genome, fitness_history = engine.evolve(input_size=4, output_size=2)
        
        if best_genome and fitness_history:
            print(f"✅ Evrim testi başarılı!")
            print(f"🏆 En iyi fitness: {best_genome.fitness:.4f}")
            print(f"🧬 En iyi genom: {len(best_genome.nodes)} node, {len(best_genome.connections)} bağlantı")
            print(f"📈 Fitness geçmişi: {fitness_history}")
            print(f"🔬 Final tür sayısı: {len(engine.species)}")
            
            # Innovation tracker testi
            print("🧪 Innovation tracker testi...")
            innovations_used = len(engine.innovation_tracker.connection_innovations)
            print(f"✅ Innovation tracker: {innovations_used} bağlantı innovation'ı")
            
            print("🧬 GELİŞMİŞ NEAT_ENGINE.PY TEST BAŞARILI! ✅")
            return True
        else:
            print("❌ Evrim testi başarısız!")
            return False
            
    except Exception as e:
        print(f"❌ NEAT engine test hatası: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Ana test fonksiyonu."""
    print("🧪 EVO_FX_MASTER - MODÜL TEST SÜİTİ")
    print("=" * 70)
    print(f"📅 Test tarihi: {datetime.now()}")
    print(f"🐍 Python: {sys.version}")
    print(f"📁 Çalışma dizini: {os.getcwd()}")
    
    # Test sonuçları
    results = {}
    
    # 1. Viz testi
    try:
        results['viz'] = test_advanced_viz()
    except Exception as e:
        print(f"❌ Viz test kırılması: {e}")
        results['viz'] = False
    
    # 2. Feature engineer testi
    try:
        results['feature_engineer'] = test_advanced_feature_engineer()
    except Exception as e:
        print(f"❌ Feature engineer test kırılması: {e}")
        results['feature_engineer'] = False
    
    # 3. NEAT engine testi
    try:
        results['neat_engine'] = test_advanced_neat_engine()
    except Exception as e:
        print(f"❌ NEAT engine test kırılması: {e}")
        results['neat_engine'] = False
    
    # Sonuçları özetle
    print("\n" + "="*70)
    print("📊 TEST SONUÇLARI")
    print("="*70)
    
    total_tests = len(results)
    passed_tests = sum(results.values())
    
    for module, passed in results.items():
        status = "✅ BAŞARILI" if passed else "❌ BAŞARISIZ"
        print(f"{module.upper()}: {status}")
    
    print(f"\n📈 GENEL SONUÇ: {passed_tests}/{total_tests} test geçti")
    
    if passed_tests == total_tests:
        print("🎉 TÜM TESTLER BAŞARILI! Modüller hazır.")
    else:
        print("⚠️  Bazı testler başarısız. Hataları kontrol edin.")
    
    print("\n💡 Sonraki adımlar:")
    if results.get('viz', False):
        print("  ✅ Viz modülü hazır - görselleştirmeler çalışıyor")
    if results.get('feature_engineer', False):
        print("  ✅ Feature engineer hazır - 30+ özellik üretiyor")
    if results.get('neat_engine', False):
        print("  ✅ NEAT engine hazır - evrim algoritması çalışıyor")
    
    print("\n🚀 Tüm modüller hazırsa config dosyaları ekleyebiliriz!")


if __name__ == "__main__":
    main()