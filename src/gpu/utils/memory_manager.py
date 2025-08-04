#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
GPU Memory Manager
================
"""

try:
    import cupy as cp
    GPU_AVAILABLE = True
except ImportError:
    GPU_AVAILABLE = False

class GPUMemoryManager:
    """GPU memory yöneticisi."""
    
    def __init__(self):
        self.peak_usage = 0
        
    def get_memory_info(self):
        """GPU memory bilgilerini al."""
        if not GPU_AVAILABLE:
            return None
            
        try:
            mempool = cp.get_default_memory_pool()
            return {
                'used_bytes': mempool.used_bytes(),
                'total_bytes': mempool.total_bytes(),
                'used_mb': mempool.used_bytes() / 1024**2,
                'total_mb': mempool.total_bytes() / 1024**2
            }
        except:
            return None
    
    def clear_cache(self):
        """GPU cache'i temizle."""
        if GPU_AVAILABLE:
            try:
                mempool = cp.get_default_memory_pool()
                mempool.free_all_blocks()
                return True
            except:
                return False
        return False

# Global instance
gpu_memory = GPUMemoryManager()
