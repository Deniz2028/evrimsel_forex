#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
GPU Performance Benchmark (Fixed)
===============================

CPU vs GPU performans karşılaştırması
"""

import time
import numpy as np
import sys
from pathlib import Path

# CuPy import
try:
    import cupy as cp
    GPU_AVAILABLE = True
    print("🎮 CuPy algılandı!")
except ImportError:
    GPU_AVAILABLE = False
    print("⚠️  CuPy yok, sadece CPU benchmark")

def benchmark_matrix_operations():
    """Matrix işlemlerini benchmark et."""
    sizes = [500, 1000, 2000]
    
    print("🏁 CPU vs GPU Matrix Benchmark")
    print("=" * 50)
    
    for size in sizes:
        print(f"\n📊 Matrix size: {size}x{size}")
        
        # CPU benchmark
        a_cpu = np.random.randn(size, size).astype(np.float32)
        b_cpu = np.random.randn(size, size).astype(np.float32)
        
        start_time = time.time()
        c_cpu = np.matmul(a_cpu, b_cpu)
        cpu_time = time.time() - start_time
        
        print(f"   CPU: {cpu_time:.3f}s")
        
        # GPU benchmark
        if GPU_AVAILABLE:
            try:
                a_gpu = cp.asarray(a_cpu)
                b_gpu = cp.asarray(b_cpu)
                
                # Warm up
                cp.matmul(a_gpu, b_gpu)
                cp.cuda.Stream.null.synchronize()
                
                start_time = time.time()
                c_gpu = cp.matmul(a_gpu, b_gpu)
                cp.cuda.Stream.null.synchronize()
                gpu_time = time.time() - start_time
                
                speedup = cpu_time / gpu_time
                print(f"   GPU: {gpu_time:.3f}s ({speedup:.1f}x speedup)")
                
                # Memory info
                mempool = cp.get_default_memory_pool()
                print(f"   VRAM: {mempool.used_bytes() / 1024**2:.1f} MB")
                
            except Exception as e:
                print(f"   GPU: Hata - {e}")
        else:
            print(f"   GPU: Kullanılamıyor")
    
    # Özet
    print(f"\n📋 ÖZET:")
    if GPU_AVAILABLE:
        print(f"✅ GPU benchmark tamamlandı")
        print(f"🚀 Tipik hızlanma: 5-50x (matrix boyutuna göre)")
        print(f"💡 NEAT için beklenen: 10-50x")
    else:
        print(f"❌ GPU yok - CuPy kurulumu gerekli:")
        print(f"   pip install cupy-cuda11x")
        print(f"   pip install cupy-cuda12x")

def benchmark_simple_operations():
    """Basit GPU işlemleri test et."""
    print(f"\n🧪 Basit GPU işlemleri test ediliyor...")
    
    if not GPU_AVAILABLE:
        print("❌ GPU yok, test atlandı")
        return
    
    try:
        # Array oluşturma
        size = 1000000
        
        start_time = time.time()
        cpu_array = np.random.randn(size)
        cpu_sum = np.sum(cpu_array)
        cpu_time = time.time() - start_time
        
        start_time = time.time()
        gpu_array = cp.random.randn(size)
        gpu_sum = cp.sum(gpu_array)
        cp.cuda.Stream.null.synchronize()
        gpu_time = time.time() - start_time
        
        speedup = cpu_time / gpu_time
        print(f"✅ Array işlemleri:")
        print(f"   CPU: {cpu_time:.4f}s")
        print(f"   GPU: {gpu_time:.4f}s ({speedup:.1f}x)")
        
    except Exception as e:
        print(f"❌ GPU test hatası: {e}")

if __name__ == "__main__":
    print("🚀 EVO_FX_MASTER GPU Benchmark")
    print("=" * 40)
    
    # Ana benchmark
    benchmark_matrix_operations()
    
    # Basit işlemler
    benchmark_simple_operations()
    
    print(f"\n🏁 Benchmark tamamlandı!")
