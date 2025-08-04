# EVO_FX_MASTER - GPU Modülleri

🎮 NVIDIA GPU ile 10-50x hızlanma sağlayan modüller

## 📁 Dizin Yapısı

```
src/gpu/
├── engines/
│   └── gpu_neat_engine.py     # GPU hızlandırmalı NEAT
├── utils/
│   └── memory_manager.py      # GPU memory yönetimi
├── benchmarks/
│   └── performance_test.py    # CPU vs GPU benchmark
└── kernels/                   # Custom CUDA kernels (gelecek)
```

## 🚀 Kurulum

### GPU Gereksinimleri
- NVIDIA GPU (GTX 1060+ veya RTX serisi)
- CUDA 11.x veya 12.x
- Minimum 4GB VRAM

### CuPy Kurulumu
```bash
# CUDA 11.x için
pip install cupy-cuda11x

# CUDA 12.x için  
pip install cupy-cuda12x

# Kurulumu test et
python -c "import cupy; print('GPU OK!')"
```

## 🎯 Kullanım

### Otomatik GPU/CPU Detection
```python
from gpu_integration import get_best_neat_engine, print_system_info

# Sistem bilgilerini göster
print_system_info()

# En iyi engine'i al (GPU varsa GPU, yoksa CPU)
engine = get_best_neat_engine(config, fitness_function)
```

## 📊 Performans

| Metrik | CPU | GPU | Hızlanma |
|--------|-----|-----|----------|
| Nesil süresi | 90-120s | 2-10s | **10-50x** |
| Popülasyon | 32-64 | 128-512 | **4-8x** |
| Batch eval | Sequential | Parallel | **20-100x** |

## 🔧 Troubleshooting

### GPU Algılanmıyor
```bash
# NVIDIA sürücü kontrol
nvidia-smi

# CUDA versiyon kontrol
nvcc --version

# CuPy test
python -c "import cupy as cp; print(cp.cuda.runtime.getDeviceCount())"
```

## ⚠️ Notlar

- GPU modülleri opsiyoneldir
- GPU yoksa otomatik CPU fallback
- Büyük popülasyonlar için daha fazla VRAM gerekir
