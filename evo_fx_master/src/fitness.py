#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
EVO_FX_MASTER - Fitness Değerlendirme Modülü
===========================================

EURUSD ve XAUUSD için backtest tabanlı fitness hesabı.
Metrik: profit_per_trade × trade_count / (1 + max_drawdown_bars)
"""

import numpy as np
import pandas as pd
import logging
from typing import Dict, List, Optional, Tuple, NamedTuple
from pathlib import Path
from datetime import datetime, timedelta
import warnings

# Kendi modüllerimizi import et
try:
    from .genome import Genome
    from .feature_engineer import AdvancedFeatureEngineer as FeatureEngineer
except ImportError:
    # Standalone çalıştırma için
    import sys
    sys.path.append('.')
    from genome import Genome
    from feature_engineer import AdvancedFeatureEngineer as FeatureEngineer

warnings.filterwarnings('ignore')
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class TradeResult(NamedTuple):
    """Tek trade sonucu."""
    entry_time: pd.Timestamp
    exit_time: pd.Timestamp
    entry_price: float
    exit_price: float
    direction: int  # 1: long, -1: short, 0: no trade
    pnl: float
    pnl_pct: float
    commission: float
    slippage: float


class BacktestResult(NamedTuple):
    """Backtest sonuç özeti."""
    total_trades: int
    winning_trades: int
    losing_trades: int
    win_rate: float
    total_pnl: float
    avg_profit_per_trade: float
    avg_win: float
    avg_loss: float
    max_drawdown: float
    max_drawdown_bars: int
    sharpe_ratio: float
    calmar_ratio: float
    profit_factor: float
    equity_curve: pd.Series
    trades: List[TradeResult]


class FitnessEvaluator:
    """Ana fitness değerlendirici sınıfı."""
    
    def __init__(self, config: Dict = None):
        """
        Args:
            config: Fitness parametreleri
        """
        self.config = config or {}
        
        # Trading parametreleri
        self.initial_balance = self.config.get('initial_balance', 10000.0)
        self.risk_per_trade = self.config.get('risk_per_trade', 0.01)  # %1 risk
        
        # Komisyon ve slippage (broker maliyetleri)
        self.commission = {
            'EURUSD': self.config.get('commission_eurusd', 0.0002),    # 2 pip
            'XAUUSD': self.config.get('commission_xauusd', 0.00035)    # 3.5 pip
        }
        
        self.slippage = {
            'EURUSD': self.config.get('slippage_eurusd', 0.0001),     # 1 pip
            'XAUUSD': self.config.get('slippage_xauusd', 0.0001)      # 1 pip  
        }
        
        # Sinyal eşikleri
        self.signal_threshold = self.config.get('signal_threshold', 0.1)
        self.strong_signal_threshold = self.config.get('strong_signal_threshold', 0.7)
        
        # Multi-symbol ağırlıkları
        self.symbol_weights = {
            'EURUSD': self.config.get('eurusd_weight', 0.6),
            'XAUUSD': self.config.get('xauusd_weight', 0.4)
        }
        
        # Minimum trade sayısı (çok az trade yapanları cezalandır)
        self.min_trades = self.config.get('min_trades', 10)
        
        # Veri dizinleri
        self.processed_dir = Path("data/processed")
        
        # Cache için veri yükleme
        self._cached_data = {}
        self._load_processed_data()
        
        logger.info(f"💪 FitnessEvaluator hazırlandı. Semboller: {list(self.symbol_weights.keys())}")
    
    def _load_processed_data(self):
        """İşlenmiş feature verilerini yükle (cache için)."""
        for symbol in ['EURUSD', 'XAUUSD']:
            data_file = self.processed_dir / f"{symbol}_features.parquet"
            
            if data_file.exists():
                try:
                    df = pd.read_parquet(data_file)
                    
                    # NaN kontrolü
                    df = df.dropna()
                    
                    if len(df) > 0:
                        self._cached_data[symbol] = df
                        logger.info(f"📊 {symbol} verisi yüklendi: {len(df)} satır, {len(df.columns)} özellik")
                    else:
                        logger.warning(f"⚠️  {symbol} verisi boş!")
                        
                except Exception as e:
                    logger.error(f"Veri yükleme hatası ({symbol}): {e}")
            else:
                logger.warning(f"📁 {symbol} feature dosyası bulunamadı: {data_file}")
    
    def get_trading_signals(self, genome: Genome, data: pd.DataFrame) -> pd.Series:
        """
        Genome'dan trading sinyalleri üretir.
        
        Args:
            genome: Sinyal üreten genom
            data: Feature DataFrame'i
            
        Returns:
            Sinyal serisi (-1: sell, 0: hold, 1: buy)
        """
        if data.empty:
            return pd.Series(dtype=int)
        
        # Feature sütunlarını belirle (OHLCV hariç)
        feature_cols = [col for col in data.columns 
                       if col not in ['open', 'high', 'low', 'close', 'volume']]
        
        if len(feature_cols) < genome.input_size:
            logger.warning(f"Yetersiz feature: {len(feature_cols)} < {genome.input_size}")
            return pd.Series(0, index=data.index)
        
        # İlk N feature'ı kullan
        selected_features = feature_cols[:genome.input_size]
        feature_data = data[selected_features].values
        
        signals = []
        
        for i in range(len(feature_data)):
            try:
                # Genomdan tahmin al
                inputs = feature_data[i]
                
                # NaN kontrolü
                if np.any(np.isnan(inputs)):
                    signals.append(0)
                    continue
                
                # Normalizasyon (basit)
                inputs = np.tanh(inputs / 2.0)  # [-1, 1] aralığına sıkıştır
                
                # Genom prediction
                outputs = genome.predict(inputs)
                
                # 3 output: [sell_prob, hold_prob, buy_prob]
                if len(outputs) >= 3:
                    sell_prob = outputs[0]
                    hold_prob = outputs[1] 
                    buy_prob = outputs[2]
                    
                    # En yüksek probabiliteyi seç
                    max_idx = np.argmax([sell_prob, hold_prob, buy_prob])
                    
                    if max_idx == 0 and sell_prob > self.signal_threshold:
                        signals.append(-1)  # Sell
                    elif max_idx == 2 and buy_prob > self.signal_threshold:
                        signals.append(1)   # Buy
                    else:
                        signals.append(0)   # Hold
                else:
                    # Fallback: tek output
                    output = outputs[0] if len(outputs) > 0 else 0.5
                    
                    if output < (0.5 - self.signal_threshold):
                        signals.append(-1)
                    elif output > (0.5 + self.signal_threshold):
                        signals.append(1)
                    else:
                        signals.append(0)
                        
            except Exception as e:
                logger.debug(f"Sinyal hesaplama hatası (satır {i}): {e}")
                signals.append(0)
        
        return pd.Series(signals, index=data.index)
    
    def calculate_position_size(self, signal_strength: float, balance: float, 
                              atr: float, price: float) -> float:
        """
        ATR tabanlı pozisyon boyutu hesaplar.
        
        Args:
            signal_strength: Sinyal kuvveti (0-1)
            balance: Mevcut bakiye
            atr: Average True Range
            price: Mevcut fiyat
            
        Returns:
            Pozisyon boyutu (lot/kontrat sayısı)
        """
        # Risk miktarı (sabit %1)
        risk_amount = balance * self.risk_per_trade
        
        # Stop loss mesafesi (ATR tabanlı)
        stop_distance = atr * 1.1  # ATR'ın 1.1 katı
        
        if stop_distance <= 0:
            return 0.0
        
        # Temel pozisyon boyutu
        base_position_size = risk_amount / stop_distance
        
        # Sinyal kuvvetine göre ölçekle
        if signal_strength < 0.3:
            multiplier = 0.5    # Zayıf sinyal
        elif signal_strength < 0.6:
            multiplier = 1.0    # Orta sinyal
        else:
            multiplier = 1.5    # Güçlü sinyal
        
        position_size = base_position_size * multiplier
        
        # Maksimum pozisyon limiti (%5 balance)
        max_position = balance * 0.05 / price
        position_size = min(position_size, max_position)
        
        return max(0.0, position_size)
    
    def backtest_symbol(self, genome: Genome, symbol: str, 
                       start_date: str = None, end_date: str = None) -> BacktestResult:
        """
        Tek sembol için backtest yapar.
        
        Args:
            genome: Test edilecek genom
            symbol: Sembol adı (EURUSD, XAUUSD)
            start_date: Başlangıç tarihi (opsiyonel)
            end_date: Bitiş tarihi (opsiyonel)
            
        Returns:
            Backtest sonuçları
        """
        # Veriyi al
        if symbol not in self._cached_data:
            logger.warning(f"Veri bulunamadı: {symbol}")
            return self._empty_backtest_result()
        
        data = self._cached_data[symbol].copy()
        
        # Tarih filtreleme
        if start_date:
            data = data[data.index >= start_date]
        if end_date:
            data = data[data.index <= end_date]
        
        if len(data) < 100:  # Minimum veri kontrolü
            logger.warning(f"Yetersiz veri: {symbol} ({len(data)} satır)")
            return self._empty_backtest_result()
        
        # Trading sinyalleri üret
        signals = self.get_trading_signals(genome, data)
        
        if signals.empty or signals.sum() == 0:
            logger.debug(f"Sinyal bulunamadı: {symbol}")
            return self._empty_backtest_result()
        
        # Backtest simülasyonu
        balance = self.initial_balance
        position = 0.0  # Mevcut pozisyon (+ long, - short)
        entry_price = 0.0
        entry_time = None
        trades = []
        equity_curve = [balance]
        max_balance = balance
        max_drawdown = 0.0
        max_drawdown_bars = 0
        current_drawdown_bars = 0
        
        for i in range(1, len(data)):
            current_time = data.index[i]
            current_price = data['close'].iloc[i]
            current_signal = signals.iloc[i]
            atr = data.get('atr', pd.Series(0.001, index=data.index)).iloc[i]
            
            # Mevcut pozisyon var mı?
            if position != 0:
                # Stop Loss / Take Profit kontrolü
                if position > 0:  # Long pozisyon
                    # Stop Loss: ATR * 1.1 altında
                    stop_loss = entry_price - (atr * 1.1)
                    # Take Profit: ATR * 2.2 üstünde  
                    take_profit = entry_price + (atr * 2.2)
                    
                    if current_price <= stop_loss or current_price >= take_profit:
                        # Pozisyonu kapat
                        exit_price = current_price
                        pnl = (exit_price - entry_price) * position
                        
                        # Komisyon ve slippage
                        commission_cost = abs(position) * exit_price * self.commission[symbol]
                        slippage_cost = abs(position) * self.slippage[symbol]
                        total_cost = commission_cost + slippage_cost
                        
                        net_pnl = pnl - total_cost
                        balance += net_pnl
                        
                        # Trade kaydı
                        trade = TradeResult(
                            entry_time=entry_time,
                            exit_time=current_time,
                            entry_price=entry_price,
                            exit_price=exit_price,
                            direction=1,
                            pnl=net_pnl,
                            pnl_pct=net_pnl / entry_price,
                            commission=commission_cost,
                            slippage=slippage_cost
                        )
                        trades.append(trade)
                        
                        position = 0.0
                        
                elif position < 0:  # Short pozisyon
                    stop_loss = entry_price + (atr * 1.1)
                    take_profit = entry_price - (atr * 2.2)
                    
                    if current_price >= stop_loss or current_price <= take_profit:
                        # Pozisyonu kapat
                        exit_price = current_price
                        pnl = (entry_price - exit_price) * abs(position)
                        
                        commission_cost = abs(position) * exit_price * self.commission[symbol]
                        slippage_cost = abs(position) * self.slippage[symbol]
                        total_cost = commission_cost + slippage_cost
                        
                        net_pnl = pnl - total_cost
                        balance += net_pnl
                        
                        trade = TradeResult(
                            entry_time=entry_time,
                            exit_time=current_time,
                            entry_price=entry_price,
                            exit_price=exit_price,
                            direction=-1,
                            pnl=net_pnl,
                            pnl_pct=net_pnl / entry_price,
                            commission=commission_cost,
                            slippage=slippage_cost
                        )
                        trades.append(trade)
                        
                        position = 0.0
            
            # Yeni pozisyon açma
            if position == 0 and current_signal != 0:
                # Sinyal kuvveti hesapla (basit)
                signal_strength = abs(current_signal) * 0.7  # Varsayılan kuvvet
                
                # Pozisyon boyutu hesapla
                position_size = self.calculate_position_size(
                    signal_strength, balance, atr, current_price
                )
                
                if position_size > 0:
                    if current_signal > 0:  # Buy
                        position = position_size
                    else:  # Sell
                        position = -position_size
                    
                    entry_price = current_price
                    entry_time = current_time
            
            # Equity curve güncelle
            if position != 0:
                # Unrealized P&L hesapla
                if position > 0:
                    unrealized_pnl = (current_price - entry_price) * position
                else:
                    unrealized_pnl = (entry_price - current_price) * abs(position)
                
                current_equity = balance + unrealized_pnl
            else:
                current_equity = balance
            
            equity_curve.append(current_equity)
            
            # Drawdown hesapla
            if current_equity > max_balance:
                max_balance = current_equity
                current_drawdown_bars = 0
            else:
                current_drawdown_bars += 1
                drawdown = (max_balance - current_equity) / max_balance
                if drawdown > max_drawdown:
                    max_drawdown = drawdown
                
                if current_drawdown_bars > max_drawdown_bars:
                    max_drawdown_bars = current_drawdown_bars
        
        # Son pozisyonu kapat
        if position != 0:
            exit_price = data['close'].iloc[-1]
            if position > 0:
                pnl = (exit_price - entry_price) * position
            else:
                pnl = (entry_price - exit_price) * abs(position)
            
            commission_cost = abs(position) * exit_price * self.commission[symbol]
            slippage_cost = abs(position) * self.slippage[symbol]
            net_pnl = pnl - commission_cost - slippage_cost
            
            balance += net_pnl
            
            trade = TradeResult(
                entry_time=entry_time,
                exit_time=data.index[-1],
                entry_price=entry_price,
                exit_price=exit_price,
                direction=1 if position > 0 else -1,
                pnl=net_pnl,
                pnl_pct=net_pnl / entry_price,
                commission=commission_cost,
                slippage=slippage_cost
            )
            trades.append(trade)
        
        # Sonuçları hesapla
        return self._calculate_backtest_metrics(trades, equity_curve)
    
    def _empty_backtest_result(self) -> BacktestResult:
        """Boş backtest sonucu döndürür."""
        return BacktestResult(
            total_trades=0,
            winning_trades=0,
            losing_trades=0,
            win_rate=0.0,
            total_pnl=0.0,
            avg_profit_per_trade=0.0,
            avg_win=0.0,
            avg_loss=0.0,
            max_drawdown=0.0,
            max_drawdown_bars=0,
            sharpe_ratio=0.0,
            calmar_ratio=0.0,
            profit_factor=0.0,
            equity_curve=pd.Series([self.initial_balance]),
            trades=[]
        )
    
    def _calculate_backtest_metrics(self, trades: List[TradeResult], 
                                  equity_curve: List[float]) -> BacktestResult:
        """
        Backtest metriklerini hesaplar.
        
        Args:
            trades: Trade listesi
            equity_curve: Equity curve
            
        Returns:
            Backtest sonuçları
        """
        if not trades:
            return self._empty_backtest_result()
        
        # Temel metrikler
        total_trades = len(trades)
        winning_trades = sum(1 for t in trades if t.pnl > 0)
        losing_trades = total_trades - winning_trades
        win_rate = winning_trades / total_trades if total_trades > 0 else 0
        
        total_pnl = sum(t.pnl for t in trades)
        avg_profit_per_trade = total_pnl / total_trades
        
        winning_pnls = [t.pnl for t in trades if t.pnl > 0]
        losing_pnls = [t.pnl for t in trades if t.pnl < 0]
        
        avg_win = np.mean(winning_pnls) if winning_pnls else 0
        avg_loss = np.mean(losing_pnls) if losing_pnls else 0
        
        # Profit factor
        gross_profit = sum(winning_pnls)
        gross_loss = abs(sum(losing_pnls))
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else float('inf')
        
        # Drawdown
        equity_series = pd.Series(equity_curve)
        running_max = equity_series.expanding().max()
        drawdown_series = (running_max - equity_series) / running_max
        max_drawdown = drawdown_series.max()
        
        # Drawdown bars
        max_drawdown_bars = 0
        current_drawdown_bars = 0
        peak = equity_curve[0]
        
        for equity in equity_curve[1:]:
            if equity > peak:
                peak = equity
                current_drawdown_bars = 0
            else:
                current_drawdown_bars += 1
                max_drawdown_bars = max(max_drawdown_bars, current_drawdown_bars)
        
        # Sharpe ratio (basitleştirilmiş)
        if len(equity_curve) > 1:
            returns = np.diff(equity_curve) / equity_curve[:-1]
            sharpe_ratio = np.mean(returns) / np.std(returns) if np.std(returns) > 0 else 0
        else:
            sharpe_ratio = 0
        
        # Calmar ratio
        annual_return = total_pnl / self.initial_balance
        calmar_ratio = annual_return / max_drawdown if max_drawdown > 0 else 0
        
        return BacktestResult(
            total_trades=total_trades,
            winning_trades=winning_trades,
            losing_trades=losing_trades,
            win_rate=win_rate,
            total_pnl=total_pnl,
            avg_profit_per_trade=avg_profit_per_trade,
            avg_win=avg_win,
            avg_loss=avg_loss,
            max_drawdown=max_drawdown,
            max_drawdown_bars=max_drawdown_bars,
            sharpe_ratio=sharpe_ratio,
            calmar_ratio=calmar_ratio,
            profit_factor=profit_factor,
            equity_curve=equity_series,
            trades=trades
        )
    
    def calculate_fitness(self, genome: Genome) -> float:
        """
        Ana fitness hesaplama fonksiyonu.
        
        Args:
            genome: Değerlendirilecek genom
            
        Returns:
            Fitness skoru
        """
        try:
            # Multi-symbol backtest
            results = {}
            
            for symbol in self.symbol_weights.keys():
                result = self.backtest_symbol(genome, symbol)
                results[symbol] = result
            
            # Ağırlıklı fitness hesaplama
            total_fitness = 0.0
            total_weight = 0.0
            
            for symbol, weight in self.symbol_weights.items():
                if symbol in results:
                    result = results[symbol]
                    
                    # NEAT fitness formülü:
                    # profit_per_trade × trade_count / (1 + max_drawdown_bars)
                    
                    if result.total_trades >= self.min_trades:
                        # Yeterli trade var
                        trade_factor = result.total_trades
                        profit_factor = result.avg_profit_per_trade
                        drawdown_penalty = 1 + (result.max_drawdown_bars / 100.0)
                        
                        symbol_fitness = (profit_factor * trade_factor) / drawdown_penalty
                        
                        # Negatif fitness'ı cezalandır
                        if symbol_fitness < 0:
                            symbol_fitness *= 2  # Negatif fitness'ı daha da kötü yap
                            
                    else:
                        # Çok az trade - ceza
                        symbol_fitness = -0.1 * (self.min_trades - result.total_trades)
                    
                    total_fitness += symbol_fitness * weight
                    total_weight += weight
            
            # Normalize et
            if total_weight > 0:
                final_fitness = total_fitness / total_weight
            else:
                final_fitness = 0.0
            
            # Ağ karmaşıklığı cezası
            complexity_penalty = len(genome.connections) * self.config.get('complexity_penalty', 0.0001)
            final_fitness -= complexity_penalty
            
            # Minimum fitness
            final_fitness = max(final_fitness, -1.0)
            
            return float(final_fitness)
            
        except Exception as e:
            logger.error(f"Fitness hesaplama hatası: {e}")
            return 0.0


def main():
    """Test fonksiyonu."""
    logger.info("💪 Fitness modülü test ediliyor...")
    
    # Test config
    config = {
        'initial_balance': 10000,
        'risk_per_trade': 0.01,
        'signal_threshold': 0.1,
        'min_trades': 5,  # Test için düşük
        'complexity_penalty': 0.0001
    }
    
    # Fitness evaluator oluştur
    evaluator = FitnessEvaluator(config)
    
    # Test genomu oluştur
    from genome import Genome
    test_genome = Genome(input_size=10, output_size=3)
    
    # Hafif mutasyon
    for _ in range(5):
        test_genome.mutate()
    
    logger.info(f"🧬 Test genomu: {test_genome}")
    
    # Fitness hesapla
    logger.info("🧪 Fitness hesaplanıyor...")
    fitness_score = evaluator.calculate_fitness(test_genome)
    
    logger.info(f"🎯 Test fitness: {fitness_score:.6f}")
    
    # Detaylı test (eğer veri varsa)
    if evaluator._cached_data:
        logger.info("📊 Detaylı backtest testi...")
        
        for symbol in evaluator._cached_data.keys():
            result = evaluator.backtest_symbol(test_genome, symbol)
            
            logger.info(f"📈 {symbol} sonuçları:")
            logger.info(f"  Trade sayısı: {result.total_trades}")
            logger.info(f"  Kazanma oranı: {result.win_rate:.2%}")
            logger.info(f"  Toplam P&L: {result.total_pnl:.2f}")
            logger.info(f"  Ortalama P&L: {result.avg_profit_per_trade:.4f}")
            logger.info(f"  Max Drawdown: {result.max_drawdown:.2%}")
            logger.info(f"  Max DD Bars: {result.max_drawdown_bars}")
    else:
        logger.warning("⚠️  Test verisi yok - sadece temel test yapıldı")
    
    logger.info("✅ Fitness modülü test tamamlandı!")


if __name__ == "__main__":
    main()