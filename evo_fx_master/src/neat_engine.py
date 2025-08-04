#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
EVO_FX_MASTER - NEAT Evrimsel Algoritma Motoru
=============================================

Neuroevolution of Augmenting Topologies (NEAT) benzeri algoritma.
Popülasyon yönetimi, mutasyon, çaprazlama ve tür ayrımı (speciation).
"""

import numpy as np
import pandas as pd
import logging
import random
import json
import multiprocessing as mp
from typing import Dict, List, Optional, Tuple, Callable
from pathlib import Path
from datetime import datetime
import time
from dataclasses import dataclass
from concurrent.futures import ProcessPoolExecutor, as_completed
import pickle
import copy

# Kendi modüllerimizi import et
try:
    from .genome import Genome, NodeGene, ConnectionGene, NodeType
except ImportError:
    # Standalone çalıştırma için
    import sys
    sys.path.append('.')
    from genome import Genome, NodeGene, ConnectionGene, NodeType

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class NeatConfig:
    """NEAT algoritması konfigürasyonu."""
    
    # Popülasyon parametreleri
    population_size: int = 128
    max_generations: int = 400
    stagnation_generations: int = 20
    elitism_ratio: float = 0.1  # En iyi %10'u korur
    
    # Tür ayrımı (speciation)
    speciation_threshold: float = 2.5
    target_species: int = 8
    survival_threshold: float = 0.3  # Türün %30'u hayatta kalır
    
    # Mutasyon oranları
    add_node_prob: float = 0.03
    add_connection_prob: float = 0.05
    weight_mutate_prob: float = 0.8
    weight_replace_prob: float = 0.1
    bias_mutate_prob: float = 0.7
    disable_connection_prob: float = 0.01
    
    # Mutasyon şiddetleri
    weight_mutate_power: float = 0.2
    bias_mutate_power: float = 0.1
    
    # Crossover parametreleri
    crossover_prob: float = 0.75
    interspecies_mating_prob: float = 0.001
    
    # Fitness parametreleri
    min_fitness_improvement: float = 0.001
    complexity_penalty: float = 0.0001  # Ağ karmaşıklığı cezası
    
    # Parallellik
    num_workers: int = mp.cpu_count() - 1


class NEATEngine:
    """Ana NEAT algoritması motoru."""
    
    def __init__(self, config: NeatConfig, fitness_function: Callable):
        """
        Args:
            config: NEAT konfigürasyonu
            fitness_function: Genom fitness'ı hesaplayan fonksiyon
        """
        self.config = config
        self.fitness_function = fitness_function
        
        # Algoritma durumu
        self.generation = 0
        self.population: List[Genome] = []
        self.species: List = []
        
        # İstatistikler
        self.best_genome: Optional[Genome] = None
        self.best_fitness_history: List[float] = []
        self.avg_fitness_history: List[float] = []
        
        # Stagnasyon takibi
        self.best_fitness_ever = float('-inf')
        self.stagnation_count = 0
        
        # Loglar
        self.logs_dir = Path("logs")
        self.logs_dir.mkdir(exist_ok=True)
        
        logger.info(f"🧬 NEAT Engine hazırlandı. Pop: {config.population_size}, Gen: {config.max_generations}")
    
    def create_initial_population(self, input_size: int, output_size: int):
        """
        İlk popülasyonu oluşturur.
        
        Args:
            input_size: Input node sayısı
            output_size: Output node sayısı
        """
        logger.info(f"🌱 İlk popülasyon oluşturuluyor: {self.config.population_size} genom")
        
        self.population = []
        for i in range(self.config.population_size):
            genome = Genome(input_size, output_size)
            
            # Hafif rastgele mutasyon
            for _ in range(random.randint(0, 3)):
                genome.mutate()
                
            self.population.append(genome)
        
        logger.info(f"✓ İlk popülasyon hazır: {len(self.population)} genom")
    
    def evaluate_population(self):
        """
        Popülasyonun fitness'ını hesaplar.
        """
        logger.info(f"🔄 Nesil {self.generation} değerlendiriliyor...")
        
        # Tek işlemci modu (basit)
        for genome in self.population:
            genome.fitness = self.fitness_function(genome)
    
    def update_statistics(self):
        """İstatistikleri güncelle."""
        if not self.population:
            return
            
        fitnesses = [g.fitness for g in self.population]
        avg_fitness = np.mean(fitnesses)
        best_fitness = max(fitnesses)
        best_genome = max(self.population, key=lambda x: x.fitness)
        
        # Geçmişe kaydet
        self.avg_fitness_history.append(avg_fitness)
        self.best_fitness_history.append(best_fitness)
        
        # En iyi genoma güncelleme
        if best_fitness > self.best_fitness_ever:
            self.best_fitness_ever = best_fitness
            self.best_genome = copy.deepcopy(best_genome)
            self.stagnation_count = 0
            
            logger.info(f"🏆 Yeni rekor! Fitness: {best_fitness:.6f}")
        else:
            self.stagnation_count += 1
        
        logger.info(f"📊 Gen {self.generation}: "
                   f"En iyi: {best_fitness:.6f}, "
                   f"Ortalama: {avg_fitness:.6f}, "
                   f"Stagnasyon: {self.stagnation_count}")
    
    def create_next_generation(self):
        """Sonraki nesli oluşturur (basit)."""
        logger.info(f"🔄 Nesil {self.generation + 1} oluşturuluyor...")
        
        # Fitness'a göre sırala
        self.population.sort(key=lambda x: x.fitness, reverse=True)
        
        # Elitizm - en iyileri koru
        elite_count = int(self.config.population_size * self.config.elitism_ratio)
        new_population = []
        
        for i in range(elite_count):
            new_population.append(copy.deepcopy(self.population[i]))
        
        # Kalan yerleri mutation ile doldur
        while len(new_population) < self.config.population_size:
            # En iyi %50'den rastgele seç
            parent_pool_size = self.config.population_size // 2
            parent = random.choice(self.population[:parent_pool_size])
            
            child = copy.deepcopy(parent)
            child.mutate()
            
            new_population.append(child)
        
        self.population = new_population
        logger.info(f"✓ Yeni nesil hazır: {len(self.population)} genom")
    
    def evolve(self, input_size: int, output_size: int) -> Tuple[Genome, List[float]]:
        """
        Ana evrim döngüsü.
        
        Args:
            input_size: Input boyutu
            output_size: Output boyutu
            
        Returns:
            (En iyi genom, fitness geçmişi)
        """
        logger.info(f"🚀 Evrim başlatılıyor: {self.config.max_generations} nesil")
        
        # İlk popülasyon
        self.create_initial_population(input_size, output_size)
        
        for generation in range(self.config.max_generations):
            self.generation = generation
            
            # Fitness değerlendirmesi
            start_time = time.time()
            self.evaluate_population()
            eval_time = time.time() - start_time
            
            # İstatistikleri güncelle
            self.update_statistics()
            
            # Stagnasyon kontrolü
            if self.stagnation_count >= self.config.stagnation_generations:
                logger.info(f"⏹️  Stagnasyon nedeniyle durduruluyor (nesil {generation})")
                break
            
            # Sonraki nesil
            if generation < self.config.max_generations - 1:
                self.create_next_generation()
            
            logger.info(f"⏱️  Nesil {generation} tamamlandı ({eval_time:.2f}s)")
        
        logger.info(f"🏁 Evrim tamamlandı! En iyi fitness: {self.best_fitness_ever:.6f}")
        
        return self.best_genome, self.best_fitness_history


def main():
    """Test fonksiyonu."""
    logger.info("🧬 NEAT Engine modülü test ediliyor...")
    
    # Dummy fitness fonksiyonu
    def dummy_fitness(genome: Genome) -> float:
        """Test için basit fitness fonksiyonu."""
        # Rastgele noise + ağ karmaşıklığı cezası
        base_fitness = random.uniform(0, 1)
        complexity_penalty = len(genome.connections) * 0.001
        return max(0, base_fitness - complexity_penalty)
    
    # Test config
    config = NeatConfig(
        population_size=20,  # Test için küçük
        max_generations=10,
        stagnation_generations=5,
        num_workers=1  # Test için tek işlemci
    )
    
    # NEAT engine oluştur
    engine = NEATEngine(config, dummy_fitness)
    
    # Kısa evrim testi
    logger.info("🧪 Kısa evrim testi başlatılıyor...")
    best_genome, fitness_history = engine.evolve(input_size=5, output_size=3)
    
    if best_genome:
        logger.info(f"✅ Test başarılı!")
        logger.info(f"🏆 En iyi fitness: {best_genome.fitness:.6f}")
        logger.info(f"🧬 En iyi genom: {best_genome}")
        logger.info(f"📈 Fitness geçmişi: {fitness_history}")
    else:
        logger.error("❌ Test başarısız!")
    
    logger.info("✅ NEAT Engine modülü test tamamlandı!")


if __name__ == "__main__":
    main()