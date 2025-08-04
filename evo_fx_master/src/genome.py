#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
EVO_FX_MASTER - Genom ve Sinir Ağı Yapısı
=========================================

NEAT benzeri genom yapısı. Test için basitleştirilmiş versiyon.
"""

import numpy as np
import json
import logging
from typing import Dict, List, Optional
from dataclasses import dataclass
from enum import Enum
import random

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class NodeType(Enum):
    """Node türleri."""
    INPUT = "input"
    HIDDEN = "hidden" 
    OUTPUT = "output"


@dataclass
class NodeGene:
    """Sinir ağı düğümü geni."""
    node_id: int
    node_type: NodeType
    bias: float = 0.0


@dataclass 
class ConnectionGene:
    """Bağlantı geni."""
    from_node: int
    to_node: int
    weight: float
    enabled: bool = True


class Genome:
    """Ana genom sınıfı."""
    
    def __init__(self, input_size: int = 5, output_size: int = 3):
        self.input_size = input_size
        self.output_size = output_size
        self.nodes: Dict[int, NodeGene] = {}
        self.connections: List[ConnectionGene] = []
        self.fitness = 0.0
        
        # Temel yapı oluştur
        self._create_basic_structure()
    
    def _create_basic_structure(self):
        """Temel ağ yapısını oluştur."""
        node_id = 0
        
        # Input nodes
        for i in range(self.input_size):
            self.nodes[node_id] = NodeGene(node_id, NodeType.INPUT)
            node_id += 1
            
        # Output nodes  
        for i in range(self.output_size):
            self.nodes[node_id] = NodeGene(node_id, NodeType.OUTPUT)
            node_id += 1
            
        # Basit bağlantılar
        for out_id in range(self.input_size, self.input_size + self.output_size):
            for in_id in range(self.input_size):
                if random.random() < 0.5:  # %50 bağlantı
                    weight = random.uniform(-1, 1)
                    self.connections.append(
                        ConnectionGene(in_id, out_id, weight)
                    )
    
    def predict(self, inputs: np.ndarray) -> np.ndarray:
        """Basit forward pass."""
        if len(inputs) != self.input_size:
            raise ValueError(f"Input size mismatch: {len(inputs)} vs {self.input_size}")
        
        # Node değerleri
        node_values = {}
        
        # Input değerleri ata
        for i in range(self.input_size):
            node_values[i] = inputs[i]
            
        # Output hesapla
        outputs = []
        for out_id in range(self.input_size, self.input_size + self.output_size):
            total = self.nodes[out_id].bias
            
            # Gelen bağlantıları topla
            for conn in self.connections:
                if conn.to_node == out_id and conn.enabled:
                    if conn.from_node in node_values:
                        total += node_values[conn.from_node] * conn.weight
            
            # Sigmoid aktivasyon
            output = 1.0 / (1.0 + np.exp(-np.clip(total, -500, 500)))
            outputs.append(output)
            
        return np.array(outputs)
    
    def mutate(self):
        """Basit mutasyon."""
        # Ağırlık mutasyonu
        for conn in self.connections:
            if random.random() < 0.1:  # %10 mutasyon şansı
                conn.weight += random.uniform(-0.1, 0.1)
                conn.weight = np.clip(conn.weight, -5.0, 5.0)
    
    def save_to_file(self, filepath: str) -> bool:
        """JSON'a kaydet."""
        try:
            data = {
                'input_size': self.input_size,
                'output_size': self.output_size,
                'fitness': self.fitness,
                'nodes': {nid: {'type': n.node_type.value, 'bias': n.bias} 
                         for nid, n in self.nodes.items()},
                'connections': [{'from': c.from_node, 'to': c.to_node, 
                               'weight': c.weight, 'enabled': c.enabled}
                               for c in self.connections]
            }
            
            with open(filepath, 'w') as f:
                json.dump(data, f, indent=2)
            return True
        except Exception as e:
            logger.error(f"Save error: {e}")
            return False
    
    def __str__(self):
        return f"Genome(fitness={self.fitness:.4f}, nodes={len(self.nodes)}, connections={len(self.connections)})"


def main():
    """Test fonksiyonu."""
    logger.info("🧬 Genome modülü test ediliyor...")
    
    # Test genom oluştur
    genome = Genome(input_size=5, output_size=3)
    logger.info(f"Genom oluşturuldu: {genome}")
    
    # Test tahmini
    test_input = np.random.randn(5)
    output = genome.predict(test_input)
    logger.info(f"Test input: {test_input}")
    logger.info(f"Output: {output}")
    
    # Mutasyon testi
    old_fitness = genome.fitness
    genome.mutate()
    logger.info(f"Mutasyon sonrası fitness: {old_fitness} -> {genome.fitness}")
    
    # Kaydetme testi
    test_file = "test_genome.json"
    if genome.save_to_file(test_file):
        logger.info(f"✓ Genom kaydedildi: {test_file}")
        
        # Temizle
        try:
            os.remove(test_file)
        except:
            pass
    
    logger.info("✅ Genome modülü test tamamlandı!")


if __name__ == "__main__":
    main()
