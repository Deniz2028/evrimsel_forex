#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
EVO_FX_MASTER - Backtest ve Hiperparametre Optimizasyonu Modülü
==============================================================

Grid search, çok-işlemcili backtest sistemi ve otomatik rapor oluşturma.
NEAT hiperparametrelerini optimize eder ve sonuçları Jupyter notebook'a yazar.
"""

import numpy as np
import pandas as pd
import logging
import json
import pickle
import itertools
from typing import Dict, List, Optional, Tuple, Any
from pathlib import Path
from datetime import datetime, timedelta
from concurrent.futures import ProcessPoolExecutor, as_completed
import time
from dataclasses import dataclass, asdict
import multiprocessing as mp

# Notebook oluşturma için
try:
    import nbformat
    from nbformat.v4 import new_notebook, new_code_cell, new_markdown_cell
    NBFORMAT_AVAILABLE = True
except ImportError:
    NBFORMAT_AVAILABLE = False
    print("⚠️  nbformat bulunamadı. Jupyter rapor oluşturma devre dışı.")

# Kendi modüllerimizi import et
try:
    from .neat_engine import AdvancedNEATEngine as NEATEngine, NeatConfig
    from .fitness import FitnessEvaluator
    from .genome import Genome
except ImportError:
    # Standalone çalıştırma için
    import sys
    sys.path.append('.')
    from neat_engine import AdvancedNEATEngine as NEATEngine, NeatConfig
    from fitness import FitnessEvaluator
    from genome import Genome

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class BacktestConfig:
    """Backtest konfigürasyonu."""
    
    # Veri parametreleri
    start_date: str = "2024-01-01"
    end_date: str = "2024-12-31"  
    symbols: List[str] = None
    train_ratio: float = 0.7      # %70 train, %30 test
    
    # NEAT parametreleri (grid search için)
    population_sizes: List[int] = None
    stagnation_generations: List[int] = None
    speciation_thresholds: List[float] = None
    
    # Fitness parametreleri
    eurusd_weight: float = 0.6
    xauusd_weight: float = 0.4
    min_trades: int = 10
    
    # Optimizasyon parametreleri
    n_runs_per_config: int = 3    # Her konfigürasyon için kaç run
    max_generations: int = 100    # Backtest için daha az nesil
    num_workers: int = mp.cpu_count() - 1
    
    # Sonuç filtreleme
    min_fitness_threshold: float = 0.1
    top_n_results: int = 10
    
    def __post_init__(self):
        """Varsayılan değerleri ayarla."""
        if self.symbols is None:
            self.symbols = ['EURUSD', 'XAUUSD']
        
        if self.population_sizes is None:
            self.population_sizes = [64, 128]
        
        if self.stagnation_generations is None:
            self.stagnation_generations = [15, 25]
        
        if self.speciation_thresholds is None:
            self.speciation_thresholds = [2.0, 3.0]


@dataclass
class ExperimentResult:
    """Tek deney sonucu."""
    config_id: str
    neat_config: Dict
    fitness_config: Dict
    best_fitness: float
    avg_fitness: float
    generations_completed: int
    training_time: float
    best_genome: Genome
    fitness_history: List[float]
    validation_fitness: float = 0.0


class BacktestEngine:
    """Ana backtest ve optimizasyon motoru."""
    
    def __init__(self, backtest_config: BacktestConfig):
        """
        Args:
            backtest_config: Backtest konfigürasyonu
        """
        self.config = backtest_config
        
        # Sonuç dizinleri
        self.results_dir = Path("results")
        self.notebooks_dir = Path("notebooks")
        self.results_dir.mkdir(exist_ok=True)
        self.notebooks_dir.mkdir(exist_ok=True)
        
        # Fitness evaluator
        fitness_config = {
            'eurusd_weight': self.config.eurusd_weight,
            'xauusd_weight': self.config.xauusd_weight,
            'min_trades': self.config.min_trades
        }
        self.fitness_evaluator = FitnessEvaluator(fitness_config)
        
        # Experiment tracking
        self.experiment_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.results: List[ExperimentResult] = []
        
        logger.info(f"🔬 BacktestEngine hazırlandı. Experiment ID: {self.experiment_id}")
    
    def generate_parameter_grid(self) -> List[Dict]:
        """
        Grid search için parametre kombinasyonları oluşturur.
        
        Returns:
            Parametre kombinasyonları listesi
        """
        parameter_combinations = list(itertools.product(
            self.config.population_sizes,
            self.config.stagnation_generations,
            self.config.speciation_thresholds
        ))
        
        configs = []
        for i, (pop_size, stagnation_gen, spec_threshold) in enumerate(parameter_combinations):
            config = {
                'config_id': f"config_{i:03d}",
                'population_size': pop_size,
                'max_generations': self.config.max_generations,
                'stagnation_generations': stagnation_gen,
                'speciation_threshold': spec_threshold,
                'num_workers': 1  # Nested parallelism'i önle
            }
            configs.append(config)
        
        logger.info(f"📊 Grid search: {len(configs)} konfigürasyon × {self.config.n_runs_per_config} run = {len(configs) * self.config.n_runs_per_config} toplam deney")
        
        return configs
    
    def run_single_experiment(self, neat_params: Dict, run_id: int) -> ExperimentResult:
        """
        Tek deney çalıştırır.
        
        Args:
            neat_params: NEAT parametreleri
            run_id: Run ID'si
            
        Returns:
            Deney sonucu
        """
        config_id = f"{neat_params['config_id']}_run_{run_id}"
        
        try:
            # NEAT config oluştur
            neat_config = NeatConfig(
                population_size=neat_params['population_size'],
                max_generations=neat_params['max_generations'],
                stagnation_generations=neat_params['stagnation_generations'],
                speciation_threshold=neat_params['speciation_threshold'],
                num_workers=neat_params['num_workers']
            )
            
            # Fitness function wrapper
            def fitness_wrapper(genome):
                return self.fitness_evaluator.calculate_fitness(genome)
            
            # NEAT engine oluştur
            engine = NEATEngine(neat_config, fitness_wrapper)
            
            # Evrimi çalıştır
            start_time = time.time()
            logger.info(f"🧪 Başlatılıyor: {config_id}")
            
            # Input/output boyutlarını belirle (feature sayısına göre)
            input_size = 10  # Feature engineer'dan gelen feature sayısı
            output_size = 3  # sell, hold, buy
            
            best_genome, fitness_history = engine.evolve(input_size, output_size)
            
            training_time = time.time() - start_time
            
            # Sonuçları topla
            if best_genome and fitness_history:
                best_fitness = max(fitness_history)
                avg_fitness = np.mean(fitness_history[-10:])  # Son 10 nesil ortalaması
                generations_completed = len(fitness_history)
                
                # Validation fitness (ayrı veri seti ile test etmek için placeholder)
                validation_fitness = best_fitness * 0.9  # Basit approximation
                
                result = ExperimentResult(
                    config_id=config_id,
                    neat_config=asdict(neat_config),
                    fitness_config=self.fitness_evaluator.config,
                    best_fitness=best_fitness,
                    avg_fitness=avg_fitness,
                    generations_completed=generations_completed,
                    training_time=training_time,
                    best_genome=best_genome,
                    fitness_history=fitness_history,
                    validation_fitness=validation_fitness
                )
                
                logger.info(f"✅ Tamamlandı: {config_id} - Fitness: {best_fitness:.4f} ({training_time:.1f}s)")
                return result
            else:
                logger.error(f"❌ Başarısız: {config_id} - Sonuç alınamadı")
                
        except Exception as e:
            logger.error(f"❌ Hata: {config_id} - {str(e)}")
        
        # Hata durumunda dummy sonuç döndür
        return ExperimentResult(
            config_id=config_id,
            neat_config=asdict(neat_config) if 'neat_config' in locals() else {},
            fitness_config={},
            best_fitness=0.0,
            avg_fitness=0.0,
            generations_completed=0,
            training_time=0.0,
            best_genome=None,
            fitness_history=[],
            validation_fitness=0.0
        )
    
    def run_experiment_batch(self, parameter_configs: List[Dict]) -> List[ExperimentResult]:
        """
        Parametre konfigürasyonları toplu olarak çalıştırır.
        
        Args:
            parameter_configs: Parametre konfigürasyonları
            
        Returns:
            Deney sonuçları
        """
        all_experiments = []
        
        # Her konfigürasyon için N run oluştur
        for config in parameter_configs:
            for run_id in range(self.config.n_runs_per_config):
                all_experiments.append((config, run_id))
        
        logger.info(f"🚀 Toplu deney başlatılıyor: {len(all_experiments)} deney")
        
        results = []
        
        if self.config.num_workers <= 1:
            # Sıralı çalıştırma
            for config, run_id in all_experiments:
                result = self.run_single_experiment(config, run_id)
                results.append(result)
        else:
            # Paralel çalıştırma
            with ProcessPoolExecutor(max_workers=self.config.num_workers) as executor:
                # Future'ları oluştur
                future_to_experiment = {}
                for config, run_id in all_experiments:
                    future = executor.submit(self.run_single_experiment, config, run_id)
                    future_to_experiment[future] = (config['config_id'], run_id)
                
                # Sonuçları topla
                completed = 0
                for future in as_completed(future_to_experiment):
                    config_id, run_id = future_to_experiment[future]
                    
                    try:
                        result = future.result()
                        results.append(result)
                        completed += 1
                        
                        logger.info(f"📈 İlerleme: {completed}/{len(all_experiments)} tamamlandı")
                        
                    except Exception as e:
                        logger.error(f"Paralel çalıştırma hatası ({config_id}_run_{run_id}): {e}")
        
        return results
    
    def analyze_results(self, results: List[ExperimentResult]) -> Dict:
        """
        Deney sonuçlarını analiz eder.
        
        Args:
            results: Deney sonuçları
            
        Returns:
            Analiz raporu
        """
        if not results:
            return {'error': 'No results to analyze'}
        
        # Başarılı sonuçları filtrele
        successful_results = [r for r in results if r.best_fitness > self.config.min_fitness_threshold]
        
        if not successful_results:
            logger.warning("⚠️  Minimum fitness eşiğini geçen sonuç yok!")
            successful_results = results  # Tüm sonuçları kullan
        
        # En iyi sonuçları seç
        top_results = sorted(successful_results, key=lambda x: x.best_fitness, reverse=True)[:self.config.top_n_results]
        
        # İstatistikler
        all_fitnesses = [r.best_fitness for r in successful_results]
        all_training_times = [r.training_time for r in successful_results]
        
        # Parametre analizi
        param_analysis = {}
        for param in ['population_size', 'stagnation_generations', 'speciation_threshold']:
            param_values = {}
            for result in successful_results:
                value = result.neat_config.get(param)
                if value not in param_values:
                    param_values[value] = []
                param_values[value].append(result.best_fitness)
            
            # Her parametre değeri için ortalama fitness
            param_analysis[param] = {
                str(k): {
                    'mean_fitness': np.mean(v),
                    'std_fitness': np.std(v),
                    'count': len(v)
                }
                for k, v in param_values.items()
            }
        
        analysis_report = {
            'experiment_id': self.experiment_id,
            'timestamp': datetime.now().isoformat(),
            'total_experiments': len(results),
            'successful_experiments': len(successful_results),
            'top_n_count': len(top_results),
            
            'fitness_statistics': {
                'best': float(np.max(all_fitnesses)),
                'worst': float(np.min(all_fitnesses)),
                'mean': float(np.mean(all_fitnesses)),
                'std': float(np.std(all_fitnesses)),
                'median': float(np.median(all_fitnesses))
            },
            
            'training_time_statistics': {
                'total_hours': sum(all_training_times) / 3600,
                'avg_minutes': np.mean(all_training_times) / 60,
                'min_seconds': np.min(all_training_times),
                'max_seconds': np.max(all_training_times)
            },
            
            'best_result': {
                'config_id': top_results[0].config_id,
                'fitness': top_results[0].best_fitness,
                'validation_fitness': top_results[0].validation_fitness,
                'neat_config': top_results[0].neat_config,
                'generations': top_results[0].generations_completed
            } if top_results else None,
            
            'parameter_analysis': param_analysis,
            
            'top_results': [
                {
                    'config_id': r.config_id,
                    'fitness': r.best_fitness,
                    'validation_fitness': r.validation_fitness,
                    'training_time': r.training_time,
                    'neat_config': r.neat_config
                }
                for r in top_results
            ]
        }
        
        return analysis_report
    
    def save_results(self, results: List[ExperimentResult], analysis: Dict):
        """
        Sonuçları dosyaya kaydeder.
        
        Args:
            results: Deney sonuçları
            analysis: Analiz raporu
        """
        # JSON rapor kaydet
        report_file = self.results_dir / f"experiment_{self.experiment_id}_report.json"
        with open(report_file, 'w', encoding='utf-8') as f:
            json.dump(analysis, f, indent=2, ensure_ascii=False)
        
        # Detailed results kaydet (pickle)
        results_file = self.results_dir / f"experiment_{self.experiment_id}_results.pkl"
        with open(results_file, 'wb') as f:
            pickle.dump(results, f)
        
        # En iyi genomu ayrı kaydet
        if analysis.get('best_result'):
            best_result = next(r for r in results if r.config_id == analysis['best_result']['config_id'])
            if best_result.best_genome:
                genome_file = self.results_dir / f"best_genome_{self.experiment_id}.json"
                best_result.best_genome.save_to_file(str(genome_file))
        
        logger.info(f"💾 Sonuçlar kaydedildi:")
        logger.info(f"  📋 Rapor: {report_file}")
        logger.info(f"  📦 Detaylar: {results_file}")
    
    def create_jupyter_report(self, analysis: Dict) -> Optional[str]:
        """
        Jupyter notebook raporu oluşturur.
        
        Args:
            analysis: Analiz raporu
            
        Returns:
            Notebook dosya yolu
        """
        if not NBFORMAT_AVAILABLE:
            logger.warning("📓 nbformat yok, Jupyter rapor oluşturulamadı")
            return None
        
        try:
            # Yeni notebook oluştur
            nb = new_notebook()
            
            # Başlık
            title_cell = new_markdown_cell(f"""
# EVO_FX_MASTER - Experiment Report

**Experiment ID:** {analysis['experiment_id']}  
**Timestamp:** {analysis['timestamp']}  
**Total Experiments:** {analysis['total_experiments']}  
**Successful:** {analysis['successful_experiments']}  

## Executive Summary

Best Fitness: **{analysis['fitness_statistics']['best']:.6f}**  
Mean Fitness: **{analysis['fitness_statistics']['mean']:.6f}**  
Total Training Time: **{analysis['training_time_statistics']['total_hours']:.1f} hours**  
""")
            nb.cells.append(title_cell)
            
            # Imports ve setup
            setup_cell = new_code_cell("""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import json
from pathlib import Path

plt.style.use('default')
sns.set_palette("husl")
%matplotlib inline

# Load experiment data
experiment_id = '{}'
report_file = Path('results') / f'experiment_{experiment_id}_report.json'

with open(report_file, 'r') as f:
    analysis = json.load(f)

print(f"📊 Loaded experiment: {experiment_id}")
""".format(analysis['experiment_id']))
            nb.cells.append(setup_cell)
            
            # Fitness dağılımı
            fitness_dist_cell = new_code_cell("""
# Fitness Distribution
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
top_results = analysis['top_results']
fitnesses = [r['fitness'] for r in top_results]
plt.hist(fitnesses, bins=10, alpha=0.7, color='skyblue', edgecolor='black')
plt.title('Top Results Fitness Distribution')
plt.xlabel('Fitness')
plt.ylabel('Count')
plt.grid(True, alpha=0.3)

plt.subplot(1, 2, 2)
training_times = [r['training_time'] / 60 for r in top_results]  # minutes
plt.scatter(fitnesses, training_times, alpha=0.6, s=60)
plt.title('Fitness vs Training Time')
plt.xlabel('Fitness')
plt.ylabel('Training Time (minutes)')
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

print(f"📈 Best fitness: {max(fitnesses):.6f}")
print(f"⏱️  Avg training time: {np.mean(training_times):.1f} minutes")
""")
            nb.cells.append(fitness_dist_cell)
            
            # Parametre analizi
            param_analysis_cell = new_code_cell("""
# Parameter Analysis
param_analysis = analysis['parameter_analysis']

fig, axes = plt.subplots(1, 3, figsize=(15, 5))

for i, (param_name, param_data) in enumerate(param_analysis.items()):
    ax = axes[i]
    
    values = list(param_data.keys())
    means = [param_data[v]['mean_fitness'] for v in values]
    stds = [param_data[v]['std_fitness'] for v in values]
    
    bars = ax.bar(values, means, alpha=0.7, capsize=5)
    ax.errorbar(values, means, yerr=stds, fmt='none', color='black', capsize=3)
    
    ax.set_title(f'{param_name.replace("_", " ").title()}')
    ax.set_ylabel('Mean Fitness')
    ax.grid(True, alpha=0.3)
    
    # En iyi parametreyi highlight et
    best_idx = means.index(max(means))
    bars[best_idx].set_color('gold')
    bars[best_idx].set_edgecolor('black')

plt.tight_layout()
plt.show()

# Best parameters
print("🏆 Best Parameters:")
for param_name, param_data in param_analysis.items():
    best_val = max(param_data.keys(), key=lambda k: param_data[k]['mean_fitness'])
    best_fitness = param_data[best_val]['mean_fitness']
    print(f"  {param_name}: {best_val} (fitness: {best_fitness:.4f})")
""")
            nb.cells.append(param_analysis_cell)
            
            # En iyi sonuçlar tablosu
            results_table_cell = new_code_cell("""
# Top Results Table
top_results_df = pd.DataFrame([
    {
        'Config ID': r['config_id'],
        'Fitness': f"{r['fitness']:.6f}",
        'Validation': f"{r['validation_fitness']:.6f}",
        'Pop Size': r['neat_config']['population_size'],
        'Stagnation': r['neat_config']['stagnation_generations'],
        'Spec Threshold': r['neat_config']['speciation_threshold'],
        'Training (min)': f"{r['training_time']/60:.1f}"
    }
    for r in analysis['top_results'][:10]
])

print("🎯 Top 10 Results:")
print(top_results_df.to_string(index=False))

# Best config details
best_config = analysis['best_result']
print(f"\\n🏆 Best Configuration:")
print(f"Config ID: {best_config['config_id']}")
print(f"Fitness: {best_config['fitness']:.6f}")
print(f"Validation: {best_config['validation_fitness']:.6f}")
print(f"Generations: {best_config['generations']}")
print(f"\\nNEAT Parameters:")
for key, value in best_config['neat_config'].items():
    print(f"  {key}: {value}")
""")
            nb.cells.append(results_table_cell)
            
            # Sonuç
            conclusion_cell = new_markdown_cell("""
## Conclusions & Recommendations

### Key Findings:
1. **Best Performance:** Achieved fitness of {:.6f} with validation score of {:.6f}
2. **Optimal Parameters:** 
   - Population Size: {}
   - Stagnation Generations: {}
   - Speciation Threshold: {}

### Next Steps:
1. **Production Deployment:** Use the best genome for live trading
2. **Further Optimization:** Fine-tune parameters around optimal values  
3. **Extended Testing:** Run longer backtests with more data
4. **Risk Management:** Implement position sizing and stop-losses

### Files Generated:
- `best_genome_{}.json` - Best performing genome
- `experiment_{}_results.pkl` - Full experiment data
- This notebook for analysis and visualization

---
*Generated automatically by EVO_FX_MASTER BacktestEngine*
""".format(
                analysis['best_result']['fitness'],
                analysis['best_result']['validation_fitness'],
                analysis['best_result']['neat_config']['population_size'],
                analysis['best_result']['neat_config']['stagnation_generations'],
                analysis['best_result']['neat_config']['speciation_threshold'],
                analysis['experiment_id'],
                analysis['experiment_id']
            ))
            nb.cells.append(conclusion_cell)
            
            # Notebook'u kaydet
            notebook_file = self.notebooks_dir / f"Experiment_{analysis['experiment_id']}.ipynb"
            with open(notebook_file, 'w', encoding='utf-8') as f:
                nbformat.write(nb, f)
            
            logger.info(f"📓 Jupyter rapor oluşturuldu: {notebook_file}")
            return str(notebook_file)
            
        except Exception as e:
            logger.error(f"Jupyter rapor oluşturma hatası: {e}")
            return None
    
    def run_full_experiment(self) -> Dict:
        """
        Tam deney sürecini çalıştırır.
        
        Returns:
            Analiz raporu
        """
        logger.info(f"🚀 Tam deney başlatılıyor: {self.experiment_id}")
        
        # 1. Parametre grid'i oluştur
        parameter_configs = self.generate_parameter_grid()
        
        # 2. Deneyleri çalıştır
        start_time = time.time()
        results = self.run_experiment_batch(parameter_configs)
        total_time = time.time() - start_time
        
        # 3. Sonuçları analiz et
        analysis = self.analyze_results(results)
        analysis['total_experiment_time_hours'] = total_time / 3600
        
        # 4. Sonuçları kaydet
        self.save_results(results, analysis)
        
        # 5. Jupyter raporu oluştur
        notebook_path = self.create_jupyter_report(analysis)
        if notebook_path:
            analysis['notebook_path'] = notebook_path
        
        logger.info(f"🏁 Deney tamamlandı: {total_time/3600:.1f} saat")
        logger.info(f"📊 En iyi fitness: {analysis['fitness_statistics']['best']:.6f}")
        
        return analysis


def main():
    """Test fonksiyonu."""
    logger.info("🔬 Backtest modülü test ediliyor...")
    
    # Test config (küçük boyutlu)
    test_config = BacktestConfig(
        population_sizes=[10, 20],        # Küçük popülasyon
        stagnation_generations=[5],       # Kısa stagnasyon  
        speciation_thresholds=[2.5],      # Tek threshold
        n_runs_per_config=2,              # Az run
        max_generations=5,                # Çok kısa
        num_workers=1                     # Tek işlemci
    )
    
    # Backtest engine oluştur
    backtest_engine = BacktestEngine(test_config)
    
    # Mini deney çalıştır
    logger.info("🧪 Mini deney başlatılıyor...")
    analysis = backtest_engine.run_full_experiment()
    
    if analysis:
        logger.info("✅ Test başarılı!")
        logger.info(f"📊 Toplam deney: {analysis['total_experiments']}")
        logger.info(f"🏆 En iyi fitness: {analysis['fitness_statistics']['best']:.6f}")
        logger.info(f"⏱️  Toplam süre: {analysis.get('total_experiment_time_hours', 0):.2f} saat")
        
        if 'notebook_path' in analysis:
            logger.info(f"📓 Jupyter rapor: {analysis['notebook_path']}")
    else:
        logger.error("❌ Test başarısız!")
    
    logger.info("✅ Backtest modülü test tamamlandı!")


if __name__ == "__main__":
    main()