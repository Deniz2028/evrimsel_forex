#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
EVO_FX_MASTER - Ana Paket
========================

NEAT tabanlı evrimsel forex trading sistemi.
"""

__version__ = "0.1.0"
__author__ = "EVO_FX_MASTER"

# Ana modülleri import et
try:
    from .genome import Genome, NodeGene, ConnectionGene, NodeType
    from .data_loader import DataLoader
    from .feature_engineer import AdvancedFeatureEngineer as FeatureEngineer
except ImportError as e:
    # Geliştirme aşamasında import hataları normal
    pass

__all__ = [
    'Genome', 'NodeGene', 'ConnectionGene', 'NodeType',
    'DataLoader', 'FeatureEngineer', 'AdvancedNEATEngine',
    'FitnessEvaluator', 'RiskManager', 'BacktestEngine',
    'LiveTrader', 'AdvancedVisualizationEngine'
]