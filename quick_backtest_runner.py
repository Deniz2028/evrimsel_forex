#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
EVO_FX_MASTER - Quick Backtest Runner
====================================

Mevcut GitHub kodlarını kullanarak hızlı backtest çalıştırır.
"""

import sys
import os
from pathlib import Path
import logging

# Proje yolunu ayarla
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root / "src"))

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def main():
    """Ana backtest runner."""
    print("🚀 EVO_FX_MASTER Quick Backtest")
    print("=" * 50)
    
    # 1. Test modüllerini çalıştır
    print("\n1️⃣ Modül testleri çalıştırılıyor...")
    try:
        exec(open('test_modules.py').read())
        print("✅ Modül testleri başarılı!")
    except Exception as e:
        print(f"❌ Modül test hatası: {e}")
        return
    
    # 2. Data loader test
    print("\n2️⃣ Veri yükleme testi...")
    try:
        from src.data_loader import main as data_main
        data_main()
        print("✅ Data loader testi tamamlandı!")
    except Exception as e:
        print(f"⚠️ Data loader uyarısı: {e}")
    
    # 3. Feature engineering test
    print("\n3️⃣ Feature engineering testi...")
    try:
        from src.feature_engineer import main as feature_main
        feature_main()
        print("✅ Feature engineering testi tamamlandı!")
    except Exception as e:
        print(f"❌ Feature engineering hatası: {e}")
    
    # 4. NEAT engine test
    print("\n4️⃣ NEAT algoritması testi...")
    try:
        from src.neat_engine import main as neat_main
        neat_main()
        print("✅ NEAT testi tamamlandı!")
    except Exception as e:
        print(f"❌ NEAT hatası: {e}")
    
    # 5. Backtest engine test
    print("\n5️⃣ Backtest engine testi...")
    try:
        from src.backtest import main as backtest_main
        backtest_main()
        print("✅ Backtest testi tamamlandı!")
    except Exception as e:
        print(f"❌ Backtest hatası: {e}")
    
    # 6. Fitness evaluation test
    print("\n6️⃣ Fitness evaluation testi...")
    try:
        from src.fitness import main as fitness_main
        fitness_main()
        print("✅ Fitness testi tamamlandı!")
    except Exception as e:
        print(f"❌ Fitness hatası: {e}")
    
    # 7. Risk management test
    print("\n7️⃣ Risk yönetimi testi...")
    try:
        from src.risk import main as risk_main
        risk_main()
        print("✅ Risk testi tamamlandı!")
    except Exception as e:
        print(f"❌ Risk hatası: {e}")
    
    # 8. Visualization test
    print("\n8️⃣ Görselleştirme testi...")
    try:
        from src.viz import main as viz_main
        viz_main()
        print("✅ Görselleştirme testi tamamlandı!")
    except Exception as e:
        print(f"❌ Görselleştirme hatası: {e}")
    
    print("\n🎉 TÜM TESTLER TAMAMLANDI!")
    print("\n📊 Sonuçlar:")
    print("  • results/ klasöründe JSON raporlar")
    print("  • notebooks/ klasöründe Jupyter raporları")
    print("  • logs/ klasöründe detaylı loglar")
    
    # 9. Tam backtest çalıştır (opsiyonel)
    answer = input("\n🤔 Tam backtest çalıştırmak ister misiniz? (y/n): ")
    if answer.lower() == 'y':
        print("\n🚀 Tam backtest başlatılıyor...")
        run_full_backtest()

def run_full_backtest():
    """Tam backtest çalıştırır."""
    try:
        from src.backtest import BacktestEngine, BacktestConfig
        
        # Orta seviye config
        config = BacktestConfig(
            population_sizes=[32, 64],
            stagnation_generations=[15, 25], 
            speciation_thresholds=[2.0, 2.5],
            n_runs_per_config=3,
            max_generations=50,
            num_workers=2
        )
        
        engine = BacktestEngine(config)
        print(f"📈 Grid search: {len(config.population_sizes) * len(config.stagnation_generations) * len(config.speciation_thresholds)} konfigürasyon")
        
        # Çalıştır
        analysis = engine.run_full_experiment()
        
        if analysis:
            print(f"🏆 En iyi fitness: {analysis['fitness_statistics']['best']:.6f}")
            print(f"📊 Başarılı experiment: {analysis['successful_experiments']}/{analysis['total_experiments']}")
            
            if 'notebook_path' in analysis:
                print(f"📓 Jupyter rapor oluşturuldu: {analysis['notebook_path']}")
        
    except Exception as e:
        print(f"❌ Tam backtest hatası: {e}")

if __name__ == "__main__":
    main()