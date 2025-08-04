#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
EVO_FX_MASTER - GPU Integration Wrapper (Fixed)
=============================================

Ana sistem ile GPU modüllerini entegre eder
"""

import logging
import sys
from pathlib import Path

logger = logging.getLogger(__name__)

# Path düzeltmesi
current_dir = Path(__file__).parent
sys.path.insert(0, str(current_dir))

# GPU availability check
try:
    from gpu.engines.gpu_neat_engine import GPUNEATEngine, GPU_AVAILABLE, check_gpu_setup, create_gpu_config
    from gpu.utils.memory_manager import gpu_memory
    from gpu.benchmarks.performance_test import benchmark_matrix_operations
    
    GPU_MODULES_AVAILABLE = True
    logger.info("🎮 GPU modülleri yüklendi")
    
except ImportError as e:
    print(f"⚠️  GPU modülleri yüklenemedi: {e}")
    
    # Fallback imports
    try:
        from neat_engine import AdvancedNEATEngine as GPUNEATEngine
        from neat_engine import NeatConfig
        
        def create_gpu_config():
            return NeatConfig(population_size=64, max_generations=30, num_workers=2)
        
        GPU_MODULES_AVAILABLE = False
        GPU_AVAILABLE = False
        gpu_memory = None
        
        def check_gpu_setup():
            print("❌ GPU modülleri yok")
            return False
        
        def benchmark_matrix_operations():
            print("❌ GPU modülleri yok, benchmark yapılamıyor")
        
    except ImportError as e2:
        print(f"❌ Fallback import de başarısız: {e2}")
        GPU_MODULES_AVAILABLE = False
        GPU_AVAILABLE = False
        gpu_memory = None
        
        def check_gpu_setup():
            return False
        
        def benchmark_matrix_operations():
            print("Modüller yok")
        
        def create_gpu_config():
            return None
        
        class GPUNEATEngine:
            def __init__(self, *args, **kwargs):
                raise ImportError("NEAT modülleri yok!")

def get_best_neat_engine(config, fitness_function):
    """En iyi NEAT engine'i döndür (GPU varsa GPU, yoksa CPU)."""
    
    if GPU_MODULES_AVAILABLE and GPU_AVAILABLE:
        logger.info("🎮 GPU NEAT Engine kullanılıyor")
        return GPUNEATEngine(config, fitness_function)
    else:
        logger.info("🔄 CPU NEAT Engine kullanılıyor")
        try:
            from neat_engine import AdvancedNEATEngine
            return AdvancedNEATEngine(config, fitness_function)
        except ImportError:
            raise ImportError("Ne GPU ne de CPU NEAT engine kullanılabilir!")

def get_optimal_config(base_config=None):
    """GPU/CPU'ya göre optimal config döndür."""
    
    try:
        from neat_engine import NeatConfig
        
        if GPU_MODULES_AVAILABLE and GPU_AVAILABLE:
            # GPU config
            if callable(create_gpu_config):
                return create_gpu_config()
            else:
                return NeatConfig(population_size=128, max_generations=50, num_workers=1)
        else:
            # CPU config
            return base_config or NeatConfig(population_size=64, max_generations=30, num_workers=2)
            
    except ImportError:
        print("❌ NeatConfig import edilemedi!")
        return None

def print_system_info():
    """Sistem bilgilerini yazdır."""
    print("🔍 SİSTEM BİLGİLERİ")
    print("=" * 30)
    print(f"GPU modülleri: {'✅ Yüklü' if GPU_MODULES_AVAILABLE else '❌ Yok'}")
    print(f"GPU hardware: {'✅ Aktif' if GPU_AVAILABLE else '❌ Yok'}")
    
    # NEAT modül kontrolü
    try:
        import neat_engine
        print(f"NEAT engine: ✅ Yüklü")
    except ImportError:
        print(f"NEAT engine: ❌ Yok")
    
    try:
        import genome
        print(f"Genome modülü: ✅ Yüklü")
    except ImportError:
        print(f"Genome modülü: ❌ Yok")
    
    if GPU_MODULES_AVAILABLE and gpu_memory:
        memory_info = gpu_memory.get_memory_info()
        if memory_info:
            print(f"GPU memory: {memory_info['used_mb']:.1f}/{memory_info['total_mb']:.1f} MB")

# Export public API
__all__ = [
    'get_best_neat_engine',
    'get_optimal_config', 
    'check_gpu_setup',
    'benchmark_matrix_operations',
    'print_system_info',
    'GPU_AVAILABLE',
    'GPU_MODULES_AVAILABLE'
]
