#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
EVO_FX_MASTER - Düzeltilmiş Hızlı Test
=====================================

Scope hatası giderildi, tam evrim sistemi test ediyor.
"""

import sys
import os
import numpy as np
import pandas as pd
import logging
from pathlib import Path
from datetime import datetime, timedelta
import warnings

warnings.filterwarnings('ignore')
logging.basicConfig(level=logging.INFO, format='%(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class SimpleGenome:
    """Basit genom sınıfı - global scope."""
    def __init__(self, input_size=5, output_size=3):
        self.input_size = input_size
        self.output_size = output_size
        self.weights = np.random.randn(input_size, output_size) * 0.5
        self.bias = np.random.randn(output_size) * 0.1
        self.fitness = 0.0
    
    def predict(self, inputs):
        """Basit linear prediction."""
        if len(inputs) != self.input_size:
            return np.array([0.5, 0.0, 0.5])  # Default output
        
        outputs = np.dot(inputs, self.weights) + self.bias
        # Sigmoid activation
        outputs = 1 / (1 + np.exp(-np.clip(outputs, -500, 500)))
        return outputs
    
    def mutate(self):
        """Basit mutasyon."""
        self.weights += np.random.normal(0, 0.1, self.weights.shape)
        self.bias += np.random.normal(0, 0.05, self.bias.shape)
    
    def copy(self):
        """Genom kopyalama."""
        new_genome = SimpleGenome(self.input_size, self.output_size)
        new_genome.weights = self.weights.copy()
        new_genome.bias = self.bias.copy()
        new_genome.fitness = self.fitness
        return new_genome

def check_environment():
    """Ortam kontrolü."""
    print("🔍 Ortam Kontrolü")
    print("=" * 40)
    
    required_packages = {
        'pandas': 'Data manipulation',
        'numpy': 'Numerical operations', 
        'matplotlib': 'Plotting',
        'scipy': 'Scientific computing'
    }
    
    missing = []
    for package, desc in required_packages.items():
        try:
            __import__(package)
            print(f"✅ {package}: {desc}")
        except ImportError:
            print(f"❌ {package}: MISSING - {desc}")
            missing.append(package)
    
    if missing:
        print(f"\n⚠️  Eksik paketler: {missing}")
        print("Kurulum: pip install " + " ".join(missing))
        return False
    
    print("\n✅ Tüm temel paketler mevcut!")
    return True

def create_test_data():
    """Test verisi oluştur."""
    print("\n📊 Test verisi oluşturuluyor...")
    
    # 1 haftalık EURUSD verisi
    dates = pd.date_range('2024-01-01', '2024-01-07', freq='1H', tz='UTC')
    n = len(dates)
    
    # Basit price simulation
    np.random.seed(42)
    base_price = 1.1000
    returns = np.random.normal(0, 0.001, n).cumsum()
    prices = base_price * np.exp(returns)
    
    df = pd.DataFrame({
        'timestamp': dates,
        'open': np.roll(prices, 1),
        'high': prices * 1.001,
        'low': prices * 0.999, 
        'close': prices,
        'volume': np.random.randint(100, 1000, n)
    })
    
    df['open'].iloc[0] = base_price
    df.set_index('timestamp', inplace=True)
    
    print(f"✅ Test verisi hazır: {len(df)} satır")
    print(f"   Tarih aralığı: {df.index.min()} - {df.index.max()}")
    print(f"   Fiyat aralığı: {df['close'].min():.5f} - {df['close'].max():.5f}")
    
    return df

def test_basic_features(df):
    """Temel özellik hesaplamaları."""
    print("\n🔧 Temel özellikler hesaplanıyor...")
    
    # Returns
    df['log_return'] = np.log(df['close'] / df['close'].shift(1))
    
    # Simple moving averages
    df['sma_5'] = df['close'].rolling(5).mean()
    df['sma_20'] = df['close'].rolling(20).mean()
    
    # RSI (basit)
    delta = df['close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
    rs = gain / (loss + 1e-8)
    df['rsi'] = 100 - (100 / (1 + rs))
    
    # ATR (basit)
    high_low = df['high'] - df['low']
    high_close = np.abs(df['high'] - df['close'].shift())
    low_close = np.abs(df['low'] - df['close'].shift())
    tr = np.maximum(high_low, np.maximum(high_close, low_close))
    df['atr'] = tr.rolling(14).mean()
    
    # Time features
    df['hour'] = df.index.hour
    df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
    df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)
    
    # Temizlik
    df = df.dropna()
    
    feature_count = len([col for col in df.columns if col not in ['open', 'high', 'low', 'close', 'volume']])
    print(f"✅ {feature_count} özellik hesaplandı")
    print(f"   Final veri boyutu: {df.shape}")
    
    return df

def test_simple_fitness(genome, df):
    """Gelişmiş fitness hesaplaması."""
    # Feature seç
    feature_cols = ['log_return', 'rsi', 'atr', 'hour_sin', 'hour_cos']
    available_features = [col for col in feature_cols if col in df.columns]
    
    if len(available_features) < genome.input_size:
        genome.fitness = -1.0
        return -1.0
    
    features = df[available_features[:genome.input_size]].values
    
    # Signals generate et
    signals = []
    for i in range(len(features)):
        if not np.any(np.isnan(features[i])):
            output = genome.predict(features[i])
            # Threshold ile sinyal üret
            if output[0] > 0.6:  # Strong sell
                signals.append(-1)
            elif output[2] > 0.6:  # Strong buy
                signals.append(1)
            else:
                signals.append(0)  # Hold
        else:
            signals.append(0)
    
    # Gelişmiş backtest
    balance = 10000
    position = 0
    entry_price = 0
    trades = []
    equity_history = [balance]
    
    for i in range(1, len(df)):
        current_price = df['close'].iloc[i]
        signal = signals[i] if i < len(signals) else 0
        
        # Mevcut pozisyonu kapat (stop loss / take profit)
        if position != 0:
            # Simple stop/take profit
            price_change = (current_price - entry_price) / entry_price
            
            should_close = False
            if position > 0:  # Long position
                if price_change < -0.002 or price_change > 0.004:  # 0.2% SL, 0.4% TP
                    should_close = True
            else:  # Short position
                if price_change > 0.002 or price_change < -0.004:
                    should_close = True
            
            # Zıt sinyal ile de kapat
            if (position > 0 and signal == -1) or (position < 0 and signal == 1):
                should_close = True
            
            if should_close:
                if position > 0:
                    pnl = (current_price - entry_price) * 1000  # 1000 = lot size
                else:
                    pnl = (entry_price - current_price) * 1000
                
                balance += pnl
                trades.append(pnl)
                position = 0
        
        # Yeni pozisyon aç
        if signal != 0 and position == 0:
            position = signal  # 1 veya -1
            entry_price = current_price
        
        equity_history.append(balance)
    
    # Son pozisyonu kapat
    if position != 0:
        current_price = df['close'].iloc[-1]
        if position > 0:
            pnl = (current_price - entry_price) * 1000
        else:
            pnl = (entry_price - current_price) * 1000
        balance += pnl
        trades.append(pnl)
    
    # Gelişmiş fitness hesaplama
    if len(trades) > 5:  # Minimum trade sayısı
        avg_pnl = np.mean(trades)
        trade_count = len(trades)
        
        # Win rate
        winning_trades = [t for t in trades if t > 0]
        win_rate = len(winning_trades) / len(trades)
        
        # Max drawdown
        equity_series = pd.Series(equity_history)
        running_max = equity_series.expanding().max()
        drawdown = (running_max - equity_series) / running_max
        max_drawdown = drawdown.max()
        
        # Fitness = profit_per_trade × sqrt(trade_count) × win_rate - drawdown_penalty
        fitness = (avg_pnl * np.sqrt(trade_count) * win_rate / 1000) - (max_drawdown * 5)
        
        # Bonus for consistent positive trades
        if avg_pnl > 0 and win_rate > 0.5:
            fitness *= 1.2
        
    else:
        fitness = -0.5  # Çok az trade penalty
    
    genome.fitness = fitness
    return fitness

def test_advanced_evolution():
    """Gelişmiş evrim algoritması."""
    print("\n🧬 Gelişmiş evrim algoritması test ediliyor...")
    
    # Test verisi
    df = create_test_data()
    df = test_basic_features(df)
    
    # Evrim parametreleri
    population_size = 20
    generations = 10
    mutation_rate = 0.3
    crossover_rate = 0.7
    
    # İlk popülasyon
    population = []
    for _ in range(population_size):
        genome = SimpleGenome(5, 3)
        population.append(genome)
    
    print(f"📊 Popülasyon: {population_size}, Nesil: {generations}")
    print(f"🔧 Mutasyon oranı: {mutation_rate}, Crossover oranı: {crossover_rate}")
    
    fitness_history = []
    best_fitness_history = []
    
    for gen in range(generations):
        print(f"\n📈 Nesil {gen+1}/{generations}")
        
        # Fitness hesapla
        fitnesses = []
        for i, genome in enumerate(population):
            fitness = test_simple_fitness(genome, df)
            fitnesses.append(fitness)
            if i % 5 == 0:
                print(f"   Genom {i+1}: fitness={fitness:.6f}")
        
        # İstatistikler
        best_fitness = max(fitnesses)
        avg_fitness = np.mean(fitnesses)
        worst_fitness = min(fitnesses)
        
        fitness_history.append(avg_fitness)
        best_fitness_history.append(best_fitness)
        
        print(f"   📊 En iyi: {best_fitness:.6f}, Ortalama: {avg_fitness:.6f}, En kötü: {worst_fitness:.6f}")
        
        # Selection ve reproduction
        if gen < generations - 1:
            # Tournament selection
            new_population = []
            
            # Elite preservation (en iyi %20)
            population.sort(key=lambda x: x.fitness, reverse=True)
            elite_count = max(1, population_size // 5)
            for i in range(elite_count):
                new_population.append(population[i].copy())
            
            # Tournament selection + crossover/mutation
            while len(new_population) < population_size:
                # Tournament selection (2 ebeveyn seç)
                parent1 = tournament_selection(population, 3)
                parent2 = tournament_selection(population, 3)
                
                if np.random.random() < crossover_rate:
                    # Crossover
                    child = crossover(parent1, parent2)
                else:
                    # Single parent reproduction
                    child = parent1.copy()
                
                if np.random.random() < mutation_rate:
                    child.mutate()
                
                new_population.append(child)
            
            population = new_population
    
    # En iyi genom
    best_genome = max(population, key=lambda x: x.fitness)
    
    print(f"\n🏆 Evrim tamamlandı!")
    print(f"   En iyi fitness: {best_genome.fitness:.6f}")
    print(f"   Ortalama fitness gelişimi: {fitness_history[0]:.6f} → {fitness_history[-1]:.6f}")
    print(f"   En iyi fitness gelişimi: {best_fitness_history[0]:.6f} → {best_fitness_history[-1]:.6f}")
    
    return best_genome, fitness_history, best_fitness_history

def tournament_selection(population, tournament_size):
    """Tournament selection."""
    tournament = np.random.choice(population, tournament_size, replace=False)
    return max(tournament, key=lambda x: x.fitness)

def crossover(parent1, parent2):
    """Basit crossover."""
    child = SimpleGenome(parent1.input_size, parent1.output_size)
    
    # Weight crossover (uniform)
    mask = np.random.random(parent1.weights.shape) < 0.5
    child.weights = np.where(mask, parent1.weights, parent2.weights)
    
    # Bias crossover
    mask = np.random.random(parent1.bias.shape) < 0.5
    child.bias = np.where(mask, parent1.bias, parent2.bias)
    
    return child

def test_visualization(fitness_history, best_fitness_history):
    """Basit görselleştirme."""
    print("\n📊 Sonuçları görselleştirme...")
    
    try:
        import matplotlib.pyplot as plt
        
        plt.figure(figsize=(12, 4))
        
        plt.subplot(1, 2, 1)
        plt.plot(fitness_history, 'b-', label='Ortalama Fitness', linewidth=2)
        plt.plot(best_fitness_history, 'g-', label='En İyi Fitness', linewidth=2)
        plt.xlabel('Nesil')
        plt.ylabel('Fitness')
        plt.title('Evrim İlerlemesi')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        plt.subplot(1, 2, 2)
        improvements = np.diff(best_fitness_history)
        plt.bar(range(len(improvements)), improvements, alpha=0.7)
        plt.xlabel('Nesil')
        plt.ylabel('Fitness İyileşmesi')
        plt.title('Nesil Bazında İyileşme')
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('evolution_results.png', dpi=100, bbox_inches='tight')
        plt.show()
        
        print("✅ Görselleştirme kaydedildi: evolution_results.png")
        
    except ImportError:
        print("⚠️  Matplotlib yok, görselleştirme atlandı")
    except Exception as e:
        print(f"⚠️  Görselleştirme hatası: {e}")

def main():
    """Ana test fonksiyonu."""
    print("🚀 EVO_FX_MASTER - Gelişmiş Test")
    print("=" * 50)
    
    # 1. Ortam kontrolü
    if not check_environment():
        return
    
    try:
        # 2. Gelişmiş evrim testi
        best_genome, avg_fitness_history, best_fitness_history = test_advanced_evolution()
        
        # 3. Görselleştirme
        test_visualization(avg_fitness_history, best_fitness_history)
        
        # 4. Sonuçlar
        print(f"\n📊 TEST SONUÇLARI")
        print("=" * 30)
        print(f"✅ Sistem tamamen çalışıyor!")
        print(f"✅ Evrimsel algoritma: OK")
        print(f"✅ Fitness evaluation: OK") 
        print(f"✅ Selection & reproduction: OK")
        print(f"✅ Mutation & crossover: OK")
        
        print(f"\n🎯 Performans Metrikleri:")
        print(f"   En iyi final fitness: {best_genome.fitness:.6f}")
        
        improvement = best_fitness_history[-1] - best_fitness_history[0]
        print(f"   Toplam iyileşme: {improvement:.6f}")
        
        if improvement > 0:
            print(f"   💡 Algoritma öğrenme gösteriyor!")
        else:
            print(f"   📝 Not: Daha uzun evrim gerekebilir")
        
        print(f"\n🎉 EVO_FX_MASTER çekirdek sistemi hazır!")
        print(f"📝 Sonraki adımlar:")
        print(f"   1. Gerçek forex verisi entegrasyonu")
        print(f"   2. NEAT algoritması ile değiştirme")
        print(f"   3. Risk yönetimi ekleme")
        print(f"   4. MT5 live trading bağlantısı")
        print(f"   5. Daha karmaşık feature engineering")
        
        return True
        
    except Exception as e:
        print(f"\n❌ Test hatası: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = main()
    if success:
        print(f"\n🚀 Sistem hazır! Gerçek backtesting için:")
        print(f"python src/backtest.py")