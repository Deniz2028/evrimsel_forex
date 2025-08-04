#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
EVO_FX_MASTER - Gelişmiş NEAT Evrimsel Algoritma Motoru
=====================================================

MASTER PROMPT gereksinimlerine göre:
- Innovation ID sistemi eklendi
- Gelişmiş speciation algoritması 
- Gerçek multiprocessing implementasyonu
- Crossover fonksiyonu geliştirildi
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
from collections import defaultdict

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
    """Gelişmiş NEAT algoritması konfigürasyonu."""
    
    # Popülasyon parametreleri
    population_size: int = 128
    max_generations: int = 400
    stagnation_generations: int = 20
    elitism_ratio: float = 0.1  # En iyi %10'u korur
    
    # Speciation parametreleri
    speciation_threshold: float = 2.5
    target_species: int = 8
    survival_threshold: float = 0.3  # Türün %30'u hayatta kalır
    compatibility_weights: Tuple[float, float, float] = (1.0, 1.0, 0.4)  # c1, c2, c3
    
    # Mutasyon oranları
    add_node_prob: float = 0.03
    add_connection_prob: float = 0.05
    weight_mutate_prob: float = 0.8
    weight_replace_prob: float = 0.1
    bias_mutate_prob: float = 0.7
    disable_connection_prob: float = 0.01
    enable_connection_prob: float = 0.25
    
    # Mutasyon şiddetleri
    weight_mutate_power: float = 0.2
    bias_mutate_power: float = 0.1
    weight_init_range: Tuple[float, float] = (-1.0, 1.0)
    
    # Crossover parametreleri
    crossover_prob: float = 0.75
    interspecies_mating_prob: float = 0.001
    
    # Fitness parametreleri
    min_fitness_improvement: float = 0.001
    complexity_penalty: float = 0.0001  # Ağ karmaşıklığı cezası
    
    # Parallellik
    num_workers: int = mp.cpu_count() - 1


class InnovationTracker:
    """Innovation ID takip sistemi - MASTER PROMPT gereksinimi."""
    
    def __init__(self):
        self.innovation_counter = 0
        self.node_innovations = {}  # (from_node, to_node) -> innovation_id
        self.connection_innovations = {}  # (from_node, to_node) -> innovation_id
        self.node_counter = 0
    
    def get_connection_innovation(self, from_node: int, to_node: int) -> int:
        """Bağlantı için innovation ID al veya oluştur."""
        key = (from_node, to_node)
        if key not in self.connection_innovations:
            self.connection_innovations[key] = self.innovation_counter
            self.innovation_counter += 1
        return self.connection_innovations[key]
    
    def get_node_innovation(self, from_node: int, to_node: int) -> Tuple[int, int]:
        """Node ekleme için innovation ID al veya oluştur."""
        key = (from_node, to_node)
        if key not in self.node_innovations:
            new_node_id = self.node_counter
            self.node_counter += 1
            
            # İki yeni bağlantı innovation ID'si
            conn1_id = self.innovation_counter
            self.innovation_counter += 1
            conn2_id = self.innovation_counter
            self.innovation_counter += 1
            
            self.node_innovations[key] = (new_node_id, conn1_id, conn2_id)
        
        return self.node_innovations[key]
    
    def reset(self):
        """Yeni nesil için sıfırla."""
        self.node_innovations.clear()
        self.connection_innovations.clear()


class Species:
    """Tür (species) sınıfı - gelişmiş speciation."""
    
    def __init__(self, representative: Genome, species_id: int):
        self.representative = representative
        self.species_id = species_id
        self.members: List[Genome] = [representative]
        self.fitness_history: List[float] = []
        self.stagnation_count = 0
        self.best_fitness = representative.fitness
        self.adjusted_fitness_sum = 0.0
        self.offspring_count = 0
    
    def add_member(self, genome: Genome):
        """Türe yeni üye ekle."""
        self.members.append(genome)
    
    def update_fitness_stats(self):
        """Fitness istatistiklerini güncelle."""
        if not self.members:
            return
        
        # En iyi fitness
        current_best = max(self.members, key=lambda x: x.fitness).fitness
        
        if current_best > self.best_fitness:
            self.best_fitness = current_best
            self.stagnation_count = 0
        else:
            self.stagnation_count += 1
        
        self.fitness_history.append(current_best)
        
        # Adjusted fitness hesapla (fitness sharing)
        self.adjusted_fitness_sum = sum(g.fitness / len(self.members) for g in self.members)
    
    def calculate_offspring_count(self, total_adjusted_fitness: float, population_size: int):
        """Bu türün kaç offspring üretmesi gerektiğini hesapla."""
        if total_adjusted_fitness > 0:
            self.offspring_count = int((self.adjusted_fitness_sum / total_adjusted_fitness) * population_size)
        else:
            self.offspring_count = 0
    
    def select_parents(self, config: NeatConfig) -> List[Genome]:
        """Reproduction için ebeveynleri seç."""
        if not self.members:
            return []
        
        # Fitness'a göre sırala
        sorted_members = sorted(self.members, key=lambda x: x.fitness, reverse=True)
        
        # Survival threshold'a göre seç
        survival_count = max(1, int(len(sorted_members) * config.survival_threshold))
        survivors = sorted_members[:survival_count]
        
        return survivors


class AdvancedNEATEngine:
    """Gelişmiş NEAT algoritması motoru - MASTER PROMPT uyumlu."""
    
    def __init__(self, config: NeatConfig, fitness_function: Callable):
        """
        Args:
            config: NEAT konfigürasyonu
            fitness_function: Genom fitness'ı hesaplayan fonksiyon
        """
        self.config = config
        self.fitness_function = fitness_function
        
        # Innovation tracking
        self.innovation_tracker = InnovationTracker()
        
        # Algoritma durumu
        self.generation = 0
        self.population: List[Genome] = []
        self.species: List[Species] = []
        self.species_counter = 0
        
        # İstatistikler
        self.best_genome: Optional[Genome] = None
        self.best_fitness_history: List[float] = []
        self.avg_fitness_history: List[float] = []
        self.species_count_history: List[int] = []
        
        # Stagnasyon takibi
        self.best_fitness_ever = float('-inf')
        self.stagnation_count = 0
        
        # Loglar
        self.logs_dir = Path("logs")
        self.logs_dir.mkdir(exist_ok=True)
        
        logger.info(f"🧬 Advanced NEAT Engine hazırlandı. Pop: {config.population_size}, Gen: {config.max_generations}")
    
    def create_initial_population(self, input_size: int, output_size: int):
        """İlk popülasyonu oluşturur."""
        logger.info(f"🌱 İlk popülasyon oluşturuluyor: {self.config.population_size} genom")
        
        self.population = []
        
        # İlk genom (minimal topology)
        base_genome = Genome(input_size, output_size)
        self.population.append(base_genome)
        
        # Diğer genomlar (hafif mutasyonlarla)
        for i in range(1, self.config.population_size):
            genome = copy.deepcopy(base_genome)
            
            # Rastgele mutasyonlar uygula
            for _ in range(random.randint(0, 5)):
                self.mutate_genome(genome)
            
            self.population.append(genome)
        
        logger.info(f"✓ İlk popülasyon hazır: {len(self.population)} genom")
    
    def calculate_compatibility_distance(self, genome1: Genome, genome2: Genome) -> float:
        """İki genom arasındaki uyumluluk mesafesini hesaplar."""
        c1, c2, c3 = self.config.compatibility_weights
        
        # Connection genlerini innovation ID'ye göre sırala
        connections1 = {conn.innovation_id if hasattr(conn, 'innovation_id') else hash((conn.from_node, conn.to_node)): conn 
                       for conn in genome1.connections}
        connections2 = {conn.innovation_id if hasattr(conn, 'innovation_id') else hash((conn.from_node, conn.to_node)): conn 
                       for conn in genome2.connections}
        
        all_innovations = set(connections1.keys()) | set(connections2.keys())
        matching = set(connections1.keys()) & set(connections2.keys())
        
        # Disjoint ve excess hesapla
        max_innovation1 = max(connections1.keys()) if connections1 else 0
        max_innovation2 = max(connections2.keys()) if connections2 else 0
        max_innovation = max(max_innovation1, max_innovation2)
        
        disjoint = 0
        excess = 0
        
        for innovation in all_innovations:
            if innovation not in matching:
                if innovation < max_innovation:
                    disjoint += 1
                else:
                    excess += 1
        
        # Ortalama ağırlık farkı
        weight_diff = 0.0
        if matching:
            weight_diffs = []
            for innovation in matching:
                w1 = connections1[innovation].weight
                w2 = connections2[innovation].weight
                weight_diffs.append(abs(w1 - w2))
            weight_diff = np.mean(weight_diffs)
        
        # Normalizasyon faktörü
        N = max(len(connections1), len(connections2))
        if N < 20:  # Küçük genomlar için normalizasyon yapma
            N = 1
        
        # Compatibility distance
        distance = (c1 * excess / N) + (c2 * disjoint / N) + (c3 * weight_diff)
        
        return distance
    
    def speciate_population(self):
        """Popülasyonu türlere ayırır - gelişmiş speciation."""
        # Önceki türleri temizle
        for species in self.species:
            species.members = []
        
        # Her genomu uygun türe ata
        for genome in self.population:
            assigned = False
            
            # Mevcut türlerle uyumluluğu kontrol et
            for species in self.species:
                distance = self.calculate_compatibility_distance(genome, species.representative)
                
                if distance < self.config.speciation_threshold:
                    species.add_member(genome)
                    assigned = True
                    break
            
            # Hiçbir türe uymazsa yeni tür oluştur
            if not assigned:
                new_species = Species(genome, self.species_counter)
                self.species_counter += 1
                self.species.append(new_species)
        
        # Boş türleri kaldır
        self.species = [s for s in self.species if s.members]
        
        # Tür temsilcilerini güncelle (rastgele seç)
        for species in self.species:
            if species.members:
                species.representative = random.choice(species.members)
        
        # Fitness istatistiklerini güncelle
        for species in self.species:
            species.update_fitness_stats()
        
        logger.info(f"🔬 Speciation: {len(self.species)} tür oluşturuldu")
    
    def evaluate_population_parallel(self):
        """Popülasyonun fitness'ını paralel olarak hesaplar."""
        logger.info(f"🔄 Nesil {self.generation} paralel değerlendirme...")
        
        if self.config.num_workers <= 1:
            # Sıralı değerlendirme
            for genome in self.population:
                genome.fitness = self.fitness_function(genome)
        else:
            # Paralel değerlendirme
            with ProcessPoolExecutor(max_workers=self.config.num_workers) as executor:
                # Future mapping
                future_to_genome = {
                    executor.submit(self._evaluate_genome_wrapper, genome): genome
                    for genome in self.population
                }
                
                # Sonuçları topla
                for future in as_completed(future_to_genome):
                    genome = future_to_genome[future]
                    try:
                        fitness = future.result()
                        genome.fitness = fitness
                    except Exception as e:
                        logger.error(f"Paralel fitness hesaplama hatası: {e}")
                        genome.fitness = 0.0
    
    def _evaluate_genome_wrapper(self, genome: Genome) -> float:
        """Paralel execution için wrapper."""
        try:
            return self.fitness_function(genome)
        except Exception as e:
            logger.error(f"Fitness hesaplama hatası: {e}")
            return 0.0
    
    def mutate_genome(self, genome: Genome):
        """Gelişmiş genom mutasyonu."""
        # Add node mutation
        if random.random() < self.config.add_node_prob and genome.connections:
            self._mutate_add_node(genome)
        
        # Add connection mutation
        if random.random() < self.config.add_connection_prob:
            self._mutate_add_connection(genome)
        
        # Weight mutations
        for connection in genome.connections:
            if random.random() < self.config.weight_mutate_prob:
                if random.random() < self.config.weight_replace_prob:
                    # Replace weight
                    connection.weight = random.uniform(*self.config.weight_init_range)
                else:
                    # Perturb weight
                    connection.weight += random.gauss(0, self.config.weight_mutate_power)
                    connection.weight = np.clip(connection.weight, -5.0, 5.0)
        
        # Bias mutations
        for node in genome.nodes.values():
            if random.random() < self.config.bias_mutate_prob:
                node.bias += random.gauss(0, self.config.bias_mutate_power)
                node.bias = np.clip(node.bias, -5.0, 5.0)
        
        # Enable/disable connections
        for connection in genome.connections:
            if connection.enabled and random.random() < self.config.disable_connection_prob:
                connection.enabled = False
            elif not connection.enabled and random.random() < self.config.enable_connection_prob:
                connection.enabled = True
    
    def _mutate_add_node(self, genome: Genome):
        """Node ekle mutasyonu."""
        # Rastgele bir bağlantı seç
        enabled_connections = [c for c in genome.connections if c.enabled]
        if not enabled_connections:
            return
        
        connection = random.choice(enabled_connections)
        
        # Innovation ID al
        node_id, conn1_id, conn2_id = self.innovation_tracker.get_node_innovation(
            connection.from_node, connection.to_node
        )
        
        # Yeni node ekle
        new_node = NodeGene(node_id, NodeType.HIDDEN)
        genome.nodes[node_id] = new_node
        
        # Eski bağlantıyı devre dışı bırak
        connection.enabled = False
        
        # İki yeni bağlantı ekle
        conn1 = ConnectionGene(connection.from_node, node_id, 1.0, True)
        conn1.innovation_id = conn1_id
        
        conn2 = ConnectionGene(node_id, connection.to_node, connection.weight, True)
        conn2.innovation_id = conn2_id
        
        genome.connections.extend([conn1, conn2])
    
    def _mutate_add_connection(self, genome: Genome):
        """Bağlantı ekle mutasyonu."""
        node_ids = list(genome.nodes.keys())
        if len(node_ids) < 2:
            return
        
        # Rastgele iki node seç (recurrent bağlantılara izin ver)
        attempts = 0
        max_attempts = 20
        
        while attempts < max_attempts:
            from_node = random.choice(node_ids)
            to_node = random.choice(node_ids)
            
            # Zaten var olan bağlantıyı kontrol et
            existing = any(c.from_node == from_node and c.to_node == to_node 
                          for c in genome.connections)
            
            if not existing:
                # Innovation ID al
                innovation_id = self.innovation_tracker.get_connection_innovation(from_node, to_node)
                
                # Yeni bağlantı ekle
                weight = random.uniform(*self.config.weight_init_range)
                new_connection = ConnectionGene(from_node, to_node, weight, True)
                new_connection.innovation_id = innovation_id
                
                genome.connections.append(new_connection)
                break
            
            attempts += 1
    
    def crossover_genomes(self, parent1: Genome, parent2: Genome) -> Genome:
        """Gelişmiş genom crossover işlemi."""
        # Daha fit olan parent'ı belirle
        if parent1.fitness > parent2.fitness:
            fitter_parent, weaker_parent = parent1, parent2
        else:
            fitter_parent, weaker_parent = parent2, parent1
        
        # Yeni genom oluştur
        child = Genome(parent1.input_size, parent1.output_size)
        child.nodes = {}
        child.connections = []
        
        # Node'ları kopyala (her iki parent'tan)
        all_node_ids = set(parent1.nodes.keys()) | set(parent2.nodes.keys())
        for node_id in all_node_ids:
            if node_id in parent1.nodes and node_id in parent2.nodes:
                # Her ikisinde de var - rastgele seç
                source_node = random.choice([parent1.nodes[node_id], parent2.nodes[node_id]])
                child.nodes[node_id] = copy.deepcopy(source_node)
            elif node_id in fitter_parent.nodes:
                # Sadece daha fit parent'ta var
                child.nodes[node_id] = copy.deepcopy(fitter_parent.nodes[node_id])
        
        # Connection'ları crossover et
        p1_connections = {(c.from_node, c.to_node): c for c in parent1.connections}
        p2_connections = {(c.from_node, c.to_node): c for c in parent2.connections}
        
        all_connection_keys = set(p1_connections.keys()) | set(p2_connections.keys())
        
        for key in all_connection_keys:
            from_node, to_node = key
            
            # Her iki node'un child'da var olduğunu kontrol et
            if from_node not in child.nodes or to_node not in child.nodes:
                continue
            
            if key in p1_connections and key in p2_connections:
                # Matching genes - rastgele seç
                source_conn = random.choice([p1_connections[key], p2_connections[key]])
                child_conn = copy.deepcopy(source_conn)
                
                # Disabled gene handling
                if not p1_connections[key].enabled or not p2_connections[key].enabled:
                    # %75 şans ile disable et
                    if random.random() < 0.75:
                        child_conn.enabled = False
                
                child.connections.append(child_conn)
                
            elif key in p1_connections and parent1.fitness >= parent2.fitness:
                # Disjoint/excess gene - sadece daha fit parent'tan al
                child.connections.append(copy.deepcopy(p1_connections[key]))
                
            elif key in p2_connections and parent2.fitness >= parent1.fitness:
                # Disjoint/excess gene - sadece daha fit parent'tan al
                child.connections.append(copy.deepcopy(p2_connections[key]))
        
        return child
    
    def reproduce_species(self) -> List[Genome]:
        """Türlerin reproduction işlemi."""
        new_population = []
        
        # Her türün offspring sayısını hesapla
        total_adjusted_fitness = sum(s.adjusted_fitness_sum for s in self.species)
        
        for species in self.species:
            species.calculate_offspring_count(total_adjusted_fitness, self.config.population_size)
        
        # Elite preservation - en iyi genomları koru
        elite_count = max(1, int(self.config.population_size * self.config.elitism_ratio))
        all_genomes = []
        for species in self.species:
            all_genomes.extend(species.members)
        
        all_genomes.sort(key=lambda x: x.fitness, reverse=True)
        for i in range(min(elite_count, len(all_genomes))):
            new_population.append(copy.deepcopy(all_genomes[i]))
        
        # Her tür için offspring üret
        for species in self.species:
            parents = species.select_parents(self.config)
            
            if not parents:
                continue
            
            offspring_needed = species.offspring_count
            
            # Elit korumasından sonra kalan yerleri doldur
            if len(new_population) >= self.config.population_size:
                break
            
            offspring_needed = min(offspring_needed, 
                                 self.config.population_size - len(new_population))
            
            for _ in range(offspring_needed):
                if len(new_population) >= self.config.population_size:
                    break
                
                # Reproduction method seç
                if len(parents) == 1 or random.random() > self.config.crossover_prob:
                    # Asexual reproduction (mutation)
                    parent = random.choice(parents)
                    child = copy.deepcopy(parent)
                    self.mutate_genome(child)
                    
                else:
                    # Sexual reproduction (crossover + mutation)
                    parent1 = random.choice(parents)
                    parent2 = random.choice(parents)
                    
                    # Interspecies mating kontrolü
                    if random.random() < self.config.interspecies_mating_prob:
                        # Başka türden parent seç
                        other_species = [s for s in self.species if s != species and s.members]
                        if other_species:
                            other_species_choice = random.choice(other_species)
                            parent2 = random.choice(other_species_choice.members)
                    
                    child = self.crossover_genomes(parent1, parent2)
                    self.mutate_genome(child)
                
                new_population.append(child)
        
        # Popülasyon boyutunu tamamla (gerekirse)
        while len(new_population) < self.config.population_size:
            # En iyi genomları kopyala
            best_genome = max(all_genomes, key=lambda x: x.fitness)
            child = copy.deepcopy(best_genome)
            self.mutate_genome(child)
            new_population.append(child)
        
        return new_population[:self.config.population_size]
    
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
        self.species_count_history.append(len(self.species))
        
        # En iyi genoma güncelleme
        if best_fitness > self.best_fitness_ever:
            self.best_fitness_ever = best_fitness
            self.best_genome = copy.deepcopy(best_genome)
            self.stagnation_count = 0
            
            logger.info(f"🏆 Yeni rekor! Fitness: {best_fitness:.6f}, Tür sayısı: {len(self.species)}")
        else:
            self.stagnation_count += 1
        
        # Detaylı istatistikler
        complexity_stats = {
            'avg_nodes': np.mean([len(g.nodes) for g in self.population]),
            'avg_connections': np.mean([len(g.connections) for g in self.population]),
            'avg_enabled_connections': np.mean([sum(1 for c in g.connections if c.enabled) for g in self.population])
        }
        
        logger.info(f"📊 Gen {self.generation}: "
                   f"En iyi: {best_fitness:.6f}, "
                   f"Ortalama: {avg_fitness:.6f}, "
                   f"Türler: {len(self.species)}, "
                   f"Stagnasyon: {self.stagnation_count}")
        
        logger.info(f"🧬 Karmaşıklık: "
                   f"Ortalama node: {complexity_stats['avg_nodes']:.1f}, "
                   f"Ortalama bağlantı: {complexity_stats['avg_connections']:.1f}")
    
    def adjust_speciation_threshold(self):
        """Speciation threshold'u dinamik olarak ayarla."""
        current_species_count = len(self.species)
        target_species = self.config.target_species
        
        if current_species_count < target_species:
            # Çok az tür var - threshold'u düşür
            self.config.speciation_threshold *= 0.95
            logger.debug(f"Speciation threshold düşürüldü: {self.config.speciation_threshold:.3f}")
            
        elif current_species_count > target_species:
            # Çok fazla tür var - threshold'u yükselt
            self.config.speciation_threshold *= 1.05
            logger.debug(f"Speciation threshold yükseltildi: {self.config.speciation_threshold:.3f}")
        
        # Min/max sınırları
        self.config.speciation_threshold = np.clip(self.config.speciation_threshold, 0.5, 10.0)
    
    def remove_stagnant_species(self):
        """Durgun türleri kaldır."""
        # En iyi türü koru
        if not self.species:
            return
        
        best_species = max(self.species, key=lambda s: s.best_fitness)
        
        # Stagnant türleri filtrele
        active_species = []
        for species in self.species:
            if species == best_species or species.stagnation_count < 15:
                active_species.append(species)
            else:
                logger.info(f"🗑️  Stagnant tür kaldırıldı: ID {species.species_id}")
        
        self.species = active_species
    
    def evolve(self, input_size: int, output_size: int) -> Tuple[Genome, List[float]]:
        """
        Ana evrim döngüsü - Gelişmiş NEAT.
        
        Args:
            input_size: Input boyutu
            output_size: Output boyutu
            
        Returns:
            (En iyi genom, fitness geçmişi)
        """
        logger.info(f"🚀 Gelişmiş NEAT evrimi başlatılıyor: {self.config.max_generations} nesil")
        
        # İlk popülasyon
        self.create_initial_population(input_size, output_size)
        
        start_time = time.time()
        
        for generation in range(self.config.max_generations):
            self.generation = generation
            
            # Innovation tracker'ı sıfırla
            self.innovation_tracker.reset()
            
            # Fitness değerlendirmesi
            gen_start_time = time.time()
            self.evaluate_population_parallel()
            eval_time = time.time() - gen_start_time
            
            # Speciation
            self.speciate_population()
            
            # İstatistikleri güncelle
            self.update_statistics()
            
            # Stagnasyon kontrolü
            if self.stagnation_count >= self.config.stagnation_generations:
                logger.info(f"⏹️  Stagnasyon nedeniyle durduruluyor (nesil {generation})")
                break
            
            # Durgun türleri kaldır
            self.remove_stagnant_species()
            
            # Speciation threshold'u ayarla
            self.adjust_speciation_threshold()
            
            # Sonraki nesil
            if generation < self.config.max_generations - 1:
                self.population = self.reproduce_species()
            
            logger.info(f"⏱️  Nesil {generation} tamamlandı ({eval_time:.2f}s)")
        
        total_time = time.time() - start_time
        logger.info(f"🏁 Gelişmiş evrim tamamlandı! En iyi fitness: {self.best_fitness_ever:.6f} ({total_time/60:.1f} dk)")
        
        return self.best_genome, self.best_fitness_history
    
    def save_evolution_stats(self, filename: Optional[str] = None):
        """Evrim istatistiklerini kaydet."""
        if filename is None:
            filename = f"evolution_stats_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        
        stats = {
            'config': {
                'population_size': self.config.population_size,
                'max_generations': self.config.max_generations,
                'speciation_threshold': self.config.speciation_threshold,
                'crossover_prob': self.config.crossover_prob
            },
            'results': {
                'generations_completed': self.generation + 1,
                'best_fitness': self.best_fitness_ever,
                'final_species_count': len(self.species),
                'stagnation_count': self.stagnation_count
            },
            'history': {
                'best_fitness': self.best_fitness_history,
                'avg_fitness': self.avg_fitness_history,
                'species_count': self.species_count_history
            }
        }
        
        stats_file = self.logs_dir / filename
        with open(stats_file, 'w', encoding='utf-8') as f:
            json.dump(stats, f, indent=2)
        
        logger.info(f"📊 Evrim istatistikleri kaydedildi: {stats_file}")


def main():
    """Test fonksiyonu."""
    logger.info("🧬 Advanced NEAT Engine modülü test ediliyor...")
    
    # Dummy fitness fonksiyonu
    def advanced_dummy_fitness(genome: Genome) -> float:
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
    
    # Test config
    config = NeatConfig(
        population_size=20,  # Test için küçük
        max_generations=10,
        stagnation_generations=5,
        target_species=4,
        num_workers=1  # Test için tek işlemci
    )
    
    # Advanced NEAT engine oluştur
    engine = AdvancedNEATEngine(config, advanced_dummy_fitness)
    
    # Kısa evrim testi
    logger.info("🧪 Gelişmiş evrim testi başlatılıyor...")
    best_genome, fitness_history = engine.evolve(input_size=5, output_size=3)
    
    if best_genome:
        logger.info(f"✅ Test başarılı!")
        logger.info(f"🏆 En iyi fitness: {best_genome.fitness:.6f}")
        logger.info(f"🧬 En iyi genom: {len(best_genome.nodes)} node, {len(best_genome.connections)} bağlantı")
        logger.info(f"📈 Fitness geçmişi: {fitness_history[-5:]}")  # Son 5 nesil
        
        # Evolution stats kaydet
        engine.save_evolution_stats()
    else:
        logger.error("❌ Test başarısız!")
    
    logger.info("✅ Advanced NEAT Engine modülü test tamamlandı!")


if __name__ == "__main__":
    main()

# Backward compatibility alias
NEATEngine = AdvancedNEATEngine
