#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
EVO_FX_MASTER - Görselleştirme Modülü
====================================

Equity curves, trade dağılımları, sinir ağı diyagramları ve 
NEAT evrim istatistiklerinin görselleştirilmesi.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import logging
from typing import Dict, List, Optional, Tuple, Any
from pathlib import Path
import json
from datetime import datetime
import warnings

warnings.filterwarnings('ignore')
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Modern stil ayarları
plt.style.use('default')
sns.set_palette("husl")


class VisualizationEngine:
    """Ana görselleştirme motoru."""
    
    def __init__(self, config: Dict = None):
        """
        Args:
            config: Görselleştirme ayarları
        """
        self.config = config or {}
        
        # Renkler ve tema
        self.colors = {
            'profit': '#2E7D32',      # Koyu yeşil
            'loss': '#C62828',        # Koyu kırmızı
            'equity': '#1976D2',      # Mavi
            'drawdown': '#FF9800',    # Turuncu
            'signal_buy': '#4CAF50',  # Yeşil
            'signal_sell': '#F44336', # Kırmızı
            'hold': '#9E9E9E',        # Gri
            'background': '#FFFFFF',  # Beyaz
            'grid': '#E0E0E0'         # Açık gri
        }
        
        # Figür ayarları
        self.figure_size = self.config.get('figure_size', (12, 8))
        self.dpi = self.config.get('dpi', 100)
        self.font_size = self.config.get('font_size', 10)
        
        # Dosya kaydetme
        self.output_dir = Path(self.config.get('output_dir', 'results/plots'))
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Matplotlib ayarları
        plt.rcParams.update({
            'font.size': self.font_size,
            'axes.grid': True,
            'grid.alpha': 0.3,
            'figure.facecolor': self.colors['background'],
            'axes.facecolor': self.colors['background']
        })
        
        logger.info(f"📊 VisualizationEngine hazırlandı. Çıktı dizini: {self.output_dir}")
    
    def plot_equity_curve(self, equity_df: pd.DataFrame, trades_df: pd.DataFrame = None,
                         title: str = "Equity Curve", save_path: str = None) -> plt.Figure:
        """
        Basit equity curve çizer.
        
        Args:
            equity_df: Equity DataFrame (timestamp, equity)
            trades_df: Trade DataFrame (opsiyonel)
            title: Grafik başlığı
            save_path: Kaydetme yolu
            
        Returns:
            Matplotlib figure
        """
        fig, ax = plt.subplots(figsize=self.figure_size)
        
        # Equity çizgisi
        if 'timestamp' in equity_df.columns:
            ax.plot(equity_df['timestamp'], equity_df['equity'], 
                   color=self.colors['equity'], linewidth=2, label='Equity')
        else:
            ax.plot(equity_df.index, equity_df['equity'], 
                   color=self.colors['equity'], linewidth=2, label='Equity')
        
        # Trade noktaları
        if trades_df is not None and not trades_df.empty and 'pnl' in trades_df.columns:
            profitable_trades = trades_df[trades_df['pnl'] > 0]
            losing_trades = trades_df[trades_df['pnl'] <= 0]
            
            if not profitable_trades.empty and 'entry_time' in profitable_trades.columns:
                ax.scatter(profitable_trades['entry_time'], 
                          profitable_trades['pnl'].cumsum(),
                          color=self.colors['profit'], marker='^', s=30, alpha=0.7, label='Wins')
            
            if not losing_trades.empty and 'entry_time' in losing_trades.columns:
                ax.scatter(losing_trades['entry_time'], 
                          losing_trades['pnl'].cumsum(),
                          color=self.colors['loss'], marker='v', s=30, alpha=0.7, label='Losses')
        
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.set_xlabel('Time')
        ax.set_ylabel('Equity ($)')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # Kaydet
        if save_path:
            plt.savefig(save_path, dpi=self.dpi, bbox_inches='tight')
            logger.info(f"💾 Equity curve kaydedildi: {save_path}")
        
        return fig
    
    def plot_trade_distribution(self, trades_df: pd.DataFrame, title: str = "Trade Analysis",
                               save_path: str = None) -> plt.Figure:
        """
        Trade dağılımı çizer.
        
        Args:
            trades_df: Trade DataFrame
            title: Grafik başlığı
            save_path: Kaydetme yolu
            
        Returns:
            Matplotlib figure
        """
        if trades_df.empty or 'pnl' not in trades_df.columns:
            logger.warning("Boş trade DataFrame veya PnL yok!")
            return None
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle(title, fontsize=16, fontweight='bold')
        
        # 1. P&L Dağılımı
        ax = axes[0, 0]
        profitable_trades = trades_df[trades_df['pnl'] > 0]
        losing_trades = trades_df[trades_df['pnl'] <= 0]
        
        if not profitable_trades.empty:
            ax.hist(profitable_trades['pnl'], bins=15, alpha=0.7, 
                   color=self.colors['profit'], label=f'Profitable ({len(profitable_trades)})')
        
        if not losing_trades.empty:
            ax.hist(losing_trades['pnl'], bins=15, alpha=0.7, 
                   color=self.colors['loss'], label=f'Losing ({len(losing_trades)})')
        
        ax.axvline(0, color='black', linestyle='--', alpha=0.5)
        ax.set_title('P&L Distribution')
        ax.set_xlabel('P&L ($)')
        ax.set_ylabel('Count')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # 2. Kümülatif P&L
        ax = axes[0, 1]
        cumulative_pnl = trades_df['pnl'].cumsum()
        ax.plot(range(len(cumulative_pnl)), cumulative_pnl, 
               color=self.colors['equity'], linewidth=2)
        ax.set_title('Cumulative P&L')
        ax.set_xlabel('Trade Number')
        ax.set_ylabel('Cumulative P&L ($)')
        ax.grid(True, alpha=0.3)
        
        # 3. Win Rate
        ax = axes[1, 0]
        win_count = len(profitable_trades)
        loss_count = len(losing_trades)
        
        labels = ['Wins', 'Losses']
        sizes = [win_count, loss_count]
        colors = [self.colors['profit'], self.colors['loss']]
        
        if sum(sizes) > 0:
            ax.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%', startangle=90)
            ax.set_title('Win/Loss Ratio')
        else:
            ax.text(0.5, 0.5, 'No Data', ha='center', va='center', transform=ax.transAxes)
            ax.set_title('Win/Loss Ratio')
        
        # 4. Stats
        ax = axes[1, 1]
        ax.axis('off')
        
        total_trades = len(trades_df)
        win_rate = (win_count / total_trades * 100) if total_trades > 0 else 0
        total_pnl = trades_df['pnl'].sum()
        avg_pnl = trades_df['pnl'].mean()
        
        stats_text = f"""Trade Statistics:
        
Total Trades: {total_trades}
Winning Trades: {win_count}
Losing Trades: {loss_count}
Win Rate: {win_rate:.1f}%

Total P&L: ${total_pnl:.2f}
Average P&L: ${avg_pnl:.2f}
        """
        
        if not profitable_trades.empty:
            avg_win = profitable_trades['pnl'].mean()
            stats_text += f"Average Win: ${avg_win:.2f}\n"
        
        if not losing_trades.empty:
            avg_loss = losing_trades['pnl'].mean()
            stats_text += f"Average Loss: ${avg_loss:.2f}\n"
        
        ax.text(0.1, 0.9, stats_text, transform=ax.transAxes, fontsize=11,
               va='top', ha='left', family='monospace',
               bbox=dict(boxstyle="round,pad=0.5", facecolor='lightgray', alpha=0.5))
        
        plt.tight_layout()
        
        # Kaydet
        if save_path:
            plt.savefig(save_path, dpi=self.dpi, bbox_inches='tight')
            logger.info(f"💾 Trade dağılımı kaydedildi: {save_path}")
        
        return fig
    
    def plot_network(self, genome, ax: plt.Axes = None, title: str = "Neural Network") -> plt.Figure:
        """
        Basit sinir ağı diyagramı çizer.
        
        Args:
            genome: Genom
            ax: Matplotlib axis (opsiyonel)
            title: Grafik başlığı
            
        Returns:
            Matplotlib figure
        """
        if ax is None:
            fig, ax = plt.subplots(figsize=(10, 6))
        else:
            fig = ax.figure
        
        try:
            # Basit ağ görselleştirmesi
            input_nodes = []
            hidden_nodes = []
            output_nodes = []
            
            # Node'ları türlerine göre grupla
            for node_id, node in genome.nodes.items():
                node_type_str = str(node.node_type).lower()
                if 'input' in node_type_str:
                    input_nodes.append(node_id)
                elif 'output' in node_type_str:
                    output_nodes.append(node_id)
                else:
                    hidden_nodes.append(node_id)
            
            # Pozisyonları hesapla
            positions = {}
            
            # Input nodes (sol)
            for i, node_id in enumerate(input_nodes):
                positions[node_id] = (0, i - len(input_nodes)/2)
            
            # Hidden nodes (orta)
            for i, node_id in enumerate(hidden_nodes):
                positions[node_id] = (1, i - len(hidden_nodes)/2)
            
            # Output nodes (sağ)
            for i, node_id in enumerate(output_nodes):
                positions[node_id] = (2, i - len(output_nodes)/2)
            
            # Bağlantıları çiz
            for conn in genome.connections:
                if conn.enabled and conn.from_node in positions and conn.to_node in positions:
                    x1, y1 = positions[conn.from_node]
                    x2, y2 = positions[conn.to_node]
                    
                    color = 'blue' if conn.weight > 0 else 'red'
                    alpha = min(abs(conn.weight), 1.0)
                    linewidth = max(abs(conn.weight) * 3, 0.5)
                    
                    ax.plot([x1, x2], [y1, y2], color=color, alpha=alpha, linewidth=linewidth)
            
            # Node'ları çiz
            for node_id, (x, y) in positions.items():
                if node_id in input_nodes:
                    color = 'lightgreen'
                elif node_id in output_nodes:
                    color = 'lightcoral'
                else:
                    color = 'lightblue'
                
                ax.scatter(x, y, s=300, c=color, edgecolors='black', zorder=5)
                ax.text(x, y, str(node_id), ha='center', va='center', fontweight='bold', zorder=6)
            
            ax.set_title(title, fontweight='bold')
            ax.set_xlim(-0.5, 2.5)
            ax.set_ylim(-max(len(input_nodes), len(hidden_nodes), len(output_nodes))/2 - 0.5,
                       max(len(input_nodes), len(hidden_nodes), len(output_nodes))/2 + 0.5)
            ax.axis('off')
            
            # Legend
            ax.text(0, -max(len(input_nodes), len(hidden_nodes), len(output_nodes))/2 - 1,
                   'Green: Input, Blue: Hidden, Red: Output\nBlue lines: +weight, Red lines: -weight',
                   ha='left', va='top', fontsize=9,
                   bbox=dict(boxstyle="round,pad=0.3", facecolor='white', alpha=0.8))
            
        except Exception as e:
            logger.error(f"Network çizim hatası: {e}")
            ax.text(0.5, 0.5, f'Network plot error: {str(e)[:30]}...', 
                   ha='center', va='center', transform=ax.transAxes)
            ax.set_title(title)
        
        return fig
    
    def plot_fitness_evolution(self, fitness_history: List[float], title: str = "Fitness Evolution",
                              save_path: str = None) -> plt.Figure:
        """
        Fitness evrimini çizer.
        
        Args:
            fitness_history: Nesil başına en iyi fitness
            title: Grafik başlığı
            save_path: Kaydetme yolu
            
        Returns:
            Matplotlib figure
        """
        fig, ax = plt.subplots(figsize=self.figure_size)
        
        generations = range(len(fitness_history))
        
        ax.plot(generations, fitness_history, color=self.colors['equity'], 
               linewidth=2, marker='o', markersize=4)
        
        # En iyi fitness'ı vurgula
        if len(fitness_history) > 0:
            best_gen = np.argmax(fitness_history)
            best_fitness = fitness_history[best_gen]
            ax.scatter([best_gen], [best_fitness], color='red', s=100, zorder=5, 
                      label=f'Best: {best_fitness:.4f}')
            ax.legend()
        
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.set_xlabel('Generation')
        ax.set_ylabel('Fitness')
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # Kaydet
        if save_path:
            plt.savefig(save_path, dpi=self.dpi, bbox_inches='tight')
            logger.info(f"💾 Fitness evolution kaydedildi: {save_path}")
        
        return fig


def main():
    """Test fonksiyonu."""
    logger.info("📊 Visualization modülü test ediliyor...")
    
    # Test verileri oluştur
    logger.info("🧪 Test verileri oluşturuluyor...")
    
    # Equity data
    dates = pd.date_range('2024-01-01', '2024-01-10', freq='1H')
    initial_balance = 10000
    returns = np.random.normal(0.0001, 0.01, len(dates)).cumsum()
    equity_data = pd.DataFrame({
        'timestamp': dates,
        'equity': initial_balance * (1 + returns)
    })
    
    # Trade data
    n_trades = 20
    trade_dates = np.random.choice(dates, n_trades, replace=False)
    trade_dates.sort()
    
    trades_data = pd.DataFrame({
        'entry_time': trade_dates,
        'pnl': np.random.normal(10, 50, n_trades),  # Ortalama +$10, std $50
        'symbol': np.random.choice(['EURUSD', 'XAUUSD'], n_trades)
    })
    
    # Fitness history
    fitness_history = np.random.random(15).cumsum() + np.random.normal(0, 0.1, 15)
    
    # Visualization engine
    viz_engine = VisualizationEngine()
    
    # Test 1: Equity Curve
    logger.info("📈 Equity curve testi...")
    fig1 = viz_engine.plot_equity_curve(
        equity_data, trades_data, title="Test Equity Curve"
    )
    if fig1:
        plt.show()
        plt.close(fig1)
        logger.info("✅ Equity curve başarılı")
    
    # Test 2: Trade Distribution  
    logger.info("📊 Trade distribution testi...")
    fig2 = viz_engine.plot_trade_distribution(
        trades_data, title="Test Trade Analysis"
    )
    if fig2:
        plt.show()
        plt.close(fig2)
        logger.info("✅ Trade distribution başarılı")
    
    # Test 3: Fitness Evolution
    logger.info("🧬 Fitness evolution testi...")
    fig3 = viz_engine.plot_fitness_evolution(
        fitness_history, title="Test Fitness Evolution"
    )
    if fig3:
        plt.show()
        plt.close(fig3)
        logger.info("✅ Fitness evolution başarılı")
    
    # Test 4: Network (basit test)
    logger.info("🧠 Network test...")
    try:
        # Basit test genomu
        class TestGenome:
            def __init__(self):
                self.nodes = {}
                self.connections = []
                
                # Test nodes
                for i in range(3):
                    node = type('Node', (), {'node_type': 'input'})()
                    self.nodes[i] = node
                
                for i in range(3, 5):
                    node = type('Node', (), {'node_type': 'output'})()
                    self.nodes[i] = node
                
                # Test connections
                for i in range(3):
                    for j in range(3, 5):
                        conn = type('Conn', (), {
                            'from_node': i, 'to_node': j, 
                            'weight': np.random.uniform(-1, 1),
                            'enabled': True
                        })()
                        self.connections.append(conn)
        
        test_genome = TestGenome()
        fig4 = viz_engine.plot_network(test_genome, title="Test Network")
        if fig4:
            plt.show()
            plt.close(fig4)
            logger.info("✅ Network test başarılı")
            
    except Exception as e:
        logger.warning(f"Network test atlandı: {e}")
    
    logger.info("✅ Visualization modülü test tamamlandı!")


if __name__ == "__main__":
    main()