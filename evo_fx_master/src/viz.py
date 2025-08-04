#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
EVO_FX_MASTER - Düzeltilmiş Görselleştirme Modülü
===============================================

MASTER PROMPT gereksinimlerine göre düzeltilmiş:
- Datetime comparison hatası düzeltildi
- Inset axes ile equity curve + trade histogram
- Sinir ağı diyagramı küçük resim olarak
- Edge threshold ile ağırlık filtreleme
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import FancyBboxPatch
import seaborn as sns
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
import logging
from typing import Dict, List, Optional, Tuple, Any
from pathlib import Path
import warnings

# Kendi modüllerimizi import et
try:
    from .genome import Genome
except ImportError:
    import sys
    sys.path.append('.')
    from genome import Genome

warnings.filterwarnings('ignore')
logger = logging.getLogger(__name__)


class AdvancedVisualizationEngine:
    """Düzeltilmiş gelişmiş görselleştirme motoru."""
    
    def __init__(self, config: Dict = None):
        self.config = config or {}
        
        # Modern renkler
        self.colors = {
            'profit': '#00C851',      # Parlak yeşil
            'loss': '#FF4444',        # Parlak kırmızı  
            'equity': '#2196F3',      # Material Blue
            'drawdown': '#FF9800',    # Turuncu
            'network_pos': '#4CAF50', # Pozitif ağırlık
            'network_neg': '#F44336', # Negatif ağırlık
            'node_input': '#E8F5E8',  # Açık yeşil
            'node_hidden': '#E3F2FD', # Açık mavi
            'node_output': '#FFEBEE'  # Açık kırmızı
        }
        
        # Stil ayarları
        plt.style.use('default')
        sns.set_palette("husl")
        
        logger.info("🎨 AdvancedVisualizationEngine hazırlandı")
    
    def plot_equity_with_insets(self, equity_df: pd.DataFrame, trades_df: pd.DataFrame = None,
                               genome: 'Genome' = None, title: str = "Advanced Equity Curve",
                               figsize: Tuple[int, int] = (15, 10)) -> plt.Figure:
        """
        ⚡️ MASTER PROMPT: Equity curve + inset trade histogram + inset network diagram
        
        Args:
            equity_df: Equity DataFrame
            trades_df: Trade DataFrame  
            genome: Sinir ağı genomu
            title: Grafik başlığı
            figsize: Figür boyutu
            
        Returns:
            Matplotlib figure with insets
        """
        fig, main_ax = plt.subplots(figsize=figsize, facecolor='white')
        
        # Ana equity curve
        if 'timestamp' in equity_df.columns:
            x_data = equity_df['timestamp']
        else:
            x_data = equity_df.index
            
        main_ax.plot(x_data, equity_df['equity'], 
                    color=self.colors['equity'], linewidth=2.5, label='Equity', alpha=0.9)
        
        # Drawdown gölgelendirme
        if len(equity_df) > 1:
            running_max = equity_df['equity'].expanding().max()
            drawdown = (running_max - equity_df['equity']) / running_max
            
            main_ax.fill_between(x_data, equity_df['equity'], running_max,
                                where=(drawdown > 0), alpha=0.3, 
                                color=self.colors['drawdown'], label='Drawdown')
        
        # Trade işaretleri - HATA DÜZELTİLDİ
        if trades_df is not None and not trades_df.empty and 'pnl' in trades_df.columns:
            profitable = trades_df[trades_df['pnl'] > 0]
            losing = trades_df[trades_df['pnl'] <= 0]
            
            if not profitable.empty and 'entry_time' in profitable.columns:
                # Equity curve üzerindeki trade noktalarını bul - DÜZELTİLMİŞ
                trade_equities = []
                trade_times = []
                
                for trade_time in profitable['entry_time']:
                    try:
                        # Timestamp dönüşümü
                        if hasattr(trade_time, 'to_pydatetime'):
                            trade_time_converted = trade_time.to_pydatetime()
                        else:
                            trade_time_converted = pd.to_datetime(trade_time)
                        
                        # Equity DataFrame'de en yakın zamanı bul
                        if 'timestamp' in equity_df.columns:
                            # Timestamp sütunu varsa
                            time_diffs = np.abs((equity_df['timestamp'] - trade_time_converted).dt.total_seconds())
                            closest_idx = time_diffs.idxmin()
                            closest_equity = equity_df.loc[closest_idx, 'equity']
                        else:
                            # Index datetime ise
                            equity_times = pd.to_datetime(equity_df.index)
                            time_diffs = np.abs((equity_times - trade_time_converted).total_seconds())
                            closest_idx = time_diffs.idxmin()
                            closest_equity = equity_df.iloc[closest_idx]['equity']
                        
                        trade_equities.append(closest_equity)
                        trade_times.append(trade_time)
                        
                    except Exception as e:
                        # Hata durumunda son equity değerini kullan
                        trade_equities.append(equity_df['equity'].iloc[-1])
                        trade_times.append(trade_time)
                
                if trade_times and trade_equities:
                    main_ax.scatter(trade_times, trade_equities,
                                  color=self.colors['profit'], marker='^', s=60, 
                                  alpha=0.8, label='Winning Trades', zorder=5)
            
            if not losing.empty and 'entry_time' in losing.columns:
                # Aynı logic losing trades için
                trade_equities = []
                trade_times = []
                
                for trade_time in losing['entry_time']:
                    try:
                        if hasattr(trade_time, 'to_pydatetime'):
                            trade_time_converted = trade_time.to_pydatetime()
                        else:
                            trade_time_converted = pd.to_datetime(trade_time)
                        
                        if 'timestamp' in equity_df.columns:
                            time_diffs = np.abs((equity_df['timestamp'] - trade_time_converted).dt.total_seconds())
                            closest_idx = time_diffs.idxmin()
                            closest_equity = equity_df.loc[closest_idx, 'equity']
                        else:
                            equity_times = pd.to_datetime(equity_df.index)
                            time_diffs = np.abs((equity_times - trade_time_converted).total_seconds())
                            closest_idx = time_diffs.idxmin()
                            closest_equity = equity_df.iloc[closest_idx]['equity']
                        
                        trade_equities.append(closest_equity)
                        trade_times.append(trade_time)
                        
                    except Exception as e:
                        trade_equities.append(equity_df['equity'].iloc[-1])
                        trade_times.append(trade_time)
                
                if trade_times and trade_equities:
                    main_ax.scatter(trade_times, trade_equities,
                                  color=self.colors['loss'], marker='v', s=60,
                                  alpha=0.8, label='Losing Trades', zorder=5)
        
        # Ana eksene stil
        main_ax.set_title(title, fontsize=16, fontweight='bold', pad=20)
        main_ax.set_xlabel('Time', fontsize=12)
        main_ax.set_ylabel('Equity ($)', fontsize=12)
        main_ax.legend(loc='upper left', frameon=True, fancybox=True, shadow=True)
        main_ax.grid(True, alpha=0.3)
        
        # ⚡️ İNSET 1: Trade P&L Histogram (Sol üst)
        if trades_df is not None and not trades_df.empty and 'pnl' in trades_df.columns:
            try:
                hist_ax = inset_axes(main_ax, width="25%", height="25%", 
                                   loc='upper left', bbox_to_anchor=(0.05, 0.95, 1, 1),
                                   bbox_transform=main_ax.transAxes)
                
                # Histogram
                profitable_pnl = trades_df[trades_df['pnl'] > 0]['pnl']
                losing_pnl = trades_df[trades_df['pnl'] <= 0]['pnl']
                
                if len(profitable_pnl) > 0:
                    hist_ax.hist(profitable_pnl, bins=10, alpha=0.7, 
                               color=self.colors['profit'], label=f'Wins ({len(profitable_pnl)})')
                
                if len(losing_pnl) > 0:
                    hist_ax.hist(losing_pnl, bins=10, alpha=0.7,
                               color=self.colors['loss'], label=f'Losses ({len(losing_pnl)})')
                
                hist_ax.axvline(0, color='black', linestyle='--', alpha=0.5, linewidth=1)
                hist_ax.set_title('P&L Distribution', fontsize=10, fontweight='bold')
                hist_ax.set_xlabel('P&L ($)', fontsize=8)
                hist_ax.set_ylabel('Count', fontsize=8)
                hist_ax.tick_params(labelsize=7)
                hist_ax.legend(fontsize=7)
                hist_ax.grid(True, alpha=0.2)
            except Exception as e:
                logger.warning(f"Histogram inset hatası: {e}")
        
        # ⚡️ İNSET 2: Neural Network Diagram (Sağ alt)
        if genome is not None:
            try:
                net_ax = inset_axes(main_ax, width="25%", height="25%",
                                  loc='lower right', bbox_to_anchor=(0.95, 0.05, 1, 1),
                                  bbox_transform=main_ax.transAxes)
                
                self.plot_network_compact(genome, net_ax, edge_threshold=0.15)
            except Exception as e:
                logger.warning(f"Network inset hatası: {e}")
        
        plt.tight_layout()
        return fig
    
    def plot_network_compact(self, genome: 'Genome', ax: plt.Axes, 
                           edge_threshold: float = 0.15, title: str = "NN Topology"):
        """
        ⚡️ Kompakt sinir ağı diyagramı - inset axes için optimize edilmiş
        
        Args:
            genome: Genom nesnesi
            ax: Matplotlib axis
            edge_threshold: Minimum ağırlık eşiği (küçük ağırlıklar gizlenir)
            title: Başlık
        """
        try:
            # Node'ları kategorize et
            input_nodes = []
            hidden_nodes = []
            output_nodes = []
            
            for node_id, node in genome.nodes.items():
                node_type_str = str(node.node_type).lower()
                if 'input' in node_type_str:
                    input_nodes.append(node_id)
                elif 'output' in node_type_str:
                    output_nodes.append(node_id)
                else:
                    hidden_nodes.append(node_id)
            
            # Pozisyonları hesapla (3 katman: input-hidden-output)
            positions = {}
            
            # Input nodes (sol, dikey sıralı)
            if input_nodes:
                for i, node_id in enumerate(input_nodes):
                    y_pos = (i - len(input_nodes)/2 + 0.5) * 0.8
                    positions[node_id] = (0, y_pos)
            
            # Hidden nodes (orta, dikey sıralı)  
            if hidden_nodes:
                for i, node_id in enumerate(hidden_nodes):
                    y_pos = (i - len(hidden_nodes)/2 + 0.5) * 0.6
                    positions[node_id] = (1, y_pos)
            
            # Output nodes (sağ, dikey sıralı)
            if output_nodes:
                for i, node_id in enumerate(output_nodes):
                    y_pos = (i - len(output_nodes)/2 + 0.5) * 0.8
                    positions[node_id] = (2, y_pos)
            
            # ⚡️ Edge threshold ile bağlantıları filtrele
            filtered_connections = []
            for conn in genome.connections:
                if (hasattr(conn, 'enabled') and conn.enabled and 
                    abs(conn.weight) >= edge_threshold and
                    conn.from_node in positions and conn.to_node in positions):
                    filtered_connections.append(conn)
            
            # Bağlantıları çiz
            for conn in filtered_connections:
                x1, y1 = positions[conn.from_node]
                x2, y2 = positions[conn.to_node]
                
                # Ağırlığa göre renk ve kalınlık
                color = self.colors['network_pos'] if conn.weight > 0 else self.colors['network_neg']
                alpha = min(abs(conn.weight), 1.0) * 0.8
                linewidth = max(abs(conn.weight) * 2, 0.5)
                
                ax.plot([x1, x2], [y1, y2], color=color, alpha=alpha, 
                       linewidth=linewidth, solid_capstyle='round')
            
            # Node'ları çiz
            for node_id, (x, y) in positions.items():
                if node_id in input_nodes:
                    color = self.colors['node_input']
                    shape = 's'  # kare
                elif node_id in output_nodes:
                    color = self.colors['node_output'] 
                    shape = 's'  # kare
                else:
                    color = self.colors['node_hidden']
                    shape = 'o'  # daire
                
                ax.scatter(x, y, s=120, c=color, marker=shape, 
                         edgecolors='black', linewidths=1, zorder=10)
                
                # Node ID (küçük font)
                ax.text(x, y, str(node_id), ha='center', va='center', 
                       fontsize=6, fontweight='bold', zorder=11)
            
            # Axis ayarları
            ax.set_title(title, fontsize=10, fontweight='bold')
            ax.set_xlim(-0.5, 2.5)
            
            if positions:
                y_values = [pos[1] for pos in positions.values()]
                if len(y_values) > 1:
                    y_range = max(y_values) - min(y_values)
                    y_center = sum(y_values) / len(y_values)
                    ax.set_ylim(y_center - y_range/2 - 0.5, y_center + y_range/2 + 0.5)
                else:
                    ax.set_ylim(-1, 1)
            
            ax.axis('off')
            
            # Mini legend
            legend_text = f"Nodes: {len(positions)}\nEdges: {len(filtered_connections)}\nThreshold: {edge_threshold}"
            ax.text(0.02, 0.02, legend_text, transform=ax.transAxes, fontsize=6,
                   bbox=dict(boxstyle="round,pad=0.3", facecolor='white', alpha=0.8))
            
        except Exception as e:
            logger.error(f"Network plot hatası: {e}")
            ax.text(0.5, 0.5, f'Network Error:\n{str(e)[:20]}...', 
                   ha='center', va='center', transform=ax.transAxes, fontsize=8)
            ax.set_title(title, fontsize=10)
    
    def plot_network_full(self, genome: 'Genome', edge_threshold: float = 0.15,
                         title: str = "Neural Network Topology", 
                         figsize: Tuple[int, int] = (12, 8)) -> plt.Figure:
        """
        ⚡️ Tam boyut sinir ağı diyagramı
        
        Args:
            genome: Genom nesnesi
            edge_threshold: Minimum ağırlık eşiği
            title: Başlık
            figsize: Figür boyutu
            
        Returns:
            Matplotlib figure
        """
        fig, ax = plt.subplots(figsize=figsize, facecolor='white')
        self.plot_network_compact(genome, ax, edge_threshold, title)
        
        # Büyük versiyonda daha detaylı bilgi
        try:
            if hasattr(genome, 'fitness'):
                info_text = f"Fitness: {genome.fitness:.4f}\n"
            else:
                info_text = ""
            
            info_text += f"Input Size: {genome.input_size}\n"
            info_text += f"Output Size: {genome.output_size}\n"
            info_text += f"Total Connections: {len(genome.connections)}\n"
            
            # Enabled connections sayısı
            enabled_count = sum(1 for c in genome.connections if hasattr(c, 'enabled') and c.enabled)
            info_text += f"Enabled Connections: {enabled_count}\n"
            info_text += f"Edge Threshold: {edge_threshold}"
            
            ax.text(0.02, 0.98, info_text, transform=ax.transAxes, fontsize=10,
                   verticalalignment='top',
                   bbox=dict(boxstyle="round,pad=0.5", facecolor='lightgray', alpha=0.8))
        except Exception as e:
            logger.warning(f"Info text hatası: {e}")
        
        plt.tight_layout()
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
            plt.savefig(save_path, dpi=100, bbox_inches='tight')
            logger.info(f"💾 Trade dağılımı kaydedildi: {save_path}")
        
        return fig


def main():
    """Test fonksiyonu."""
    logger.info("🎨 Düzeltilmiş AdvancedVisualizationEngine test ediliyor...")
    
    # Test verileri oluştur
    dates = pd.date_range('2024-01-01', '2024-01-05', freq='1H')
    initial_balance = 10000
    returns = np.random.normal(0.0002, 0.02, len(dates)).cumsum()
    equity_data = pd.DataFrame({
        'timestamp': dates,
        'equity': initial_balance * (1 + returns)
    })
    
    # Trade data
    n_trades = 15
    trade_dates = np.random.choice(dates, n_trades, replace=False)
    trades_data = pd.DataFrame({
        'entry_time': trade_dates,
        'pnl': np.random.normal(20, 70, n_trades),
        'symbol': np.random.choice(['EURUSD', 'XAUUSD'], n_trades)
    })
    
    # Test genomu
    from genome import Genome
    test_genome = Genome(input_size=6, output_size=3)
    for _ in range(8):
        test_genome.mutate()
    
    # Düzeltilmiş visualizer
    viz = AdvancedVisualizationEngine()
    
    # Test 1: Inset'li equity curve (DÜZELTME)
    logger.info("🧪 Düzeltilmiş inset'li equity curve testi...")
    try:
        fig1 = viz.plot_equity_with_insets(
            equity_data, trades_data, test_genome,
            title="FIXED: Advanced Equity Curve with Insets"
        )
        logger.info("✅ Düzeltilmiş equity curve başarılı!")
        plt.close(fig1)
    except Exception as e:
        logger.error(f"❌ Hala hata var: {e}")
    
    # Test 2: Tam boyut network
    logger.info("🧪 Full network diagram testi...")
    try:
        fig2 = viz.plot_network_full(
            test_genome, edge_threshold=0.1,
            title="FIXED: Detailed Neural Network"
        )
        logger.info("✅ Network diagram başarılı!")
        plt.close(fig2)
    except Exception as e:
        logger.error(f"❌ Network hatası: {e}")
    
    logger.info("✅ Düzeltilmiş AdvancedVisualizationEngine test tamamlandı!")


if __name__ == "__main__":
    main()