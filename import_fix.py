#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
EVO_FX_MASTER - Import Hatalarını Düzelten Patch
==============================================

Bu script tüm import hatalarını düzeltir.
"""

import os
import re
from pathlib import Path

def fix_imports():
    """Tüm import hatalarını düzelt."""
    
    # 1. fitness.py'yi düzelt
    fitness_path = Path("src/fitness.py")
    if fitness_path.exists():
        with open(fitness_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Import'ları düzelt
        content = content.replace(
            "from .feature_engineer import FeatureEngineer",
            "from .feature_engineer import AdvancedFeatureEngineer as FeatureEngineer"
        )
        content = content.replace(
            "from feature_engineer import FeatureEngineer",
            "from feature_engineer import AdvancedFeatureEngineer as FeatureEngineer"
        )
        
        with open(fitness_path, 'w', encoding='utf-8') as f:
            f.write(content)
        print("✅ fitness.py düzeltildi")
    
    # 2. backtest.py'yi düzelt
    backtest_path = Path("src/backtest.py")
    if backtest_path.exists():
        with open(backtest_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Import'ları düzelt
        content = content.replace(
            "from .neat_engine import NEATEngine, NeatConfig",
            "from .neat_engine import AdvancedNEATEngine as NEATEngine, NeatConfig"
        )
        content = content.replace(
            "from neat_engine import NEATEngine, NeatConfig",
            "from neat_engine import AdvancedNEATEngine as NEATEngine, NeatConfig"
        )
        
        with open(backtest_path, 'w', encoding='utf-8') as f:
            f.write(content)
        print("✅ backtest.py düzeltildi")
    
    # 3. live_trader.py'yi düzelt
    live_trader_path = Path("src/live_trader.py")
    if live_trader_path.exists():
        with open(live_trader_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Import'ları düzelt
        content = content.replace(
            "from .feature_engineer import FeatureEngineer",
            "from .feature_engineer import AdvancedFeatureEngineer as FeatureEngineer"
        )
        content = content.replace(
            "from feature_engineer import FeatureEngineer",
            "from feature_engineer import AdvancedFeatureEngineer as FeatureEngineer"
        )
        
        with open(live_trader_path, 'w', encoding='utf-8') as f:
            f.write(content)
        print("✅ live_trader.py düzeltildi")
    
    # 4. __init__.py'yi düzelt
    init_path = Path("src/__init__.py")
    if init_path.exists():
        with open(init_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Import'ları düzelt
        content = content.replace(
            "from .feature_engineer import FeatureEngineer",
            "from .feature_engineer import AdvancedFeatureEngineer as FeatureEngineer"
        )
        content = content.replace(
            "from .data_loader import DataLoader",
            "from .data_loader import DataLoader"
        )
        
        # __all__ güncelle
        new_all = """__all__ = [
    'Genome', 'NodeGene', 'ConnectionGene', 'NodeType',
    'DataLoader', 'FeatureEngineer', 'AdvancedNEATEngine',
    'FitnessEvaluator', 'RiskManager', 'BacktestEngine',
    'LiveTrader', 'AdvancedVisualizationEngine'
]"""
        
        # Eski __all__'ı bul ve değiştir
        content = re.sub(r'__all__ = \[.*?\]', new_all, content, flags=re.DOTALL)
        
        with open(init_path, 'w', encoding='utf-8') as f:
            f.write(content)
        print("✅ __init__.py düzeltildi")

def fix_encoding_issues():
    """Encoding sorunlarını düzelt."""
    
    # test_modules.py'yi düzelt
    test_modules_path = Path("test_modules.py")
    if test_modules_path.exists():
        # UTF-8 encoding ile yeniden oku
        try:
            with open(test_modules_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # Başlığa encoding ekle
            if "# -*- coding: utf-8 -*-" not in content:
                lines = content.split('\n')
                if lines[0].startswith('#!/usr/bin/env python3'):
                    lines.insert(1, "# -*- coding: utf-8 -*-")
                else:
                    lines.insert(0, "# -*- coding: utf-8 -*-")
                content = '\n'.join(lines)
            
            with open(test_modules_path, 'w', encoding='utf-8') as f:
                f.write(content)
            print("✅ test_modules.py encoding düzeltildi")
            
        except UnicodeDecodeError:
            print("⚠️  test_modules.py encoding sorunu - manuel düzeltme gerekli")

def create_compatibility_imports():
    """Compatibility için alias import'lar oluştur."""
    
    # neat_engine.py'ye alias ekle
    neat_path = Path("src/neat_engine.py")
    if neat_path.exists():
        with open(neat_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Dosya sonuna alias ekle
        if "NEATEngine = AdvancedNEATEngine" not in content:
            content += "\n\n# Backward compatibility alias\nNEATEngine = AdvancedNEATEngine\n"
        
        with open(neat_path, 'w', encoding='utf-8') as f:
            f.write(content)
        print("✅ neat_engine.py alias eklendi")
    
    # feature_engineer.py'ye alias ekle
    feature_path = Path("src/feature_engineer.py")
    if feature_path.exists():
        with open(feature_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Dosya sonuna alias ekle
        if "FeatureEngineer = AdvancedFeatureEngineer" not in content:
            content += "\n\n# Backward compatibility alias\nFeatureEngineer = AdvancedFeatureEngineer\n"
        
        with open(feature_path, 'w', encoding='utf-8') as f:
            f.write(content)
        print("✅ feature_engineer.py alias eklendi")

def main():
    """Ana düzeltme fonksiyonu."""
    print("🔧 EVO_FX_MASTER Import Hatalarını Düzeltme")
    print("=" * 50)
    
    # Çalışma dizinini kontrol et
    if not Path("src").exists():
        print("❌ src/ dizini bulunamadı!")
        print("Bu scripti proje kök dizininde çalıştırın.")
        return
    
    # Düzeltmeleri uygula
    print("1️⃣  Import hatalarını düzeltiliyor...")
    fix_imports()
    
    print("\n2️⃣  Encoding sorunları düzeltiliyor...")
    fix_encoding_issues()
    
    print("\n3️⃣  Compatibility alias'ları ekleniyor...")
    create_compatibility_imports()
    
    print("\n✅ Tüm düzeltmeler tamamlandı!")
    print("\n🔄 Şimdi tekrar test edin:")
    print("   python real_backtest_runner.py")

if __name__ == "__main__":
    main()