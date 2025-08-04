#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
EVO_FX_MASTER - Fitness Modülü Veri Yolu Düzeltmesi
==================================================

🔧 PROBLEM: Fitness modülü processed data'yı bulamıyor
🎯 ÇÖZÜM: Dosya yollarını ve adlarını düzelt
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


class FitnessEvaluatorFixed:
    """🔧 DÜZELTME: Veri yolu düzeltilmiş fitness değerlendirici."""
    
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
        
        # 🔧 DÜZELTME: Veri dizinleri - hem raw hem processed dene
        self.processed_dir = Path("data/processed")
        self.raw_dir = Path("data/raw")
        
        # Cache için veri yükleme
        self._cached_data = {}
        self._load_processed_data_fixed()
        
        logger.info(f"💪 FitnessEvaluatorFixed hazırlandı. Semboller: {list(self.symbol_weights.keys())}")
    
    def _find_data_files(self, symbol: str) -> Optional[Path]:
        """🔧 DÜZELTME: Veri dosyalarını akıllıca bul."""
        
        # Olası dosya adları (öncelik sırasına göre)
        possible_names = [
            f"{symbol}_features_advanced.parquet",  # Feature engineer advanced
            f"{symbol}_features.parquet",           # Feature engineer normal
            f"{symbol}_processed.parquet",          # Generic processed
            f"{symbol}_raw.parquet"                 # Ham veri (son çare)
        ]
        
        # Önce processed dizininde ara
        for name in possible_names:
            file_path = self.processed_dir / name
            if file_path.exists():
                logger.info(f"📁 {symbol} verisi bulundu: {file_path}")
                return file_path
        
        # Sonra raw dizininde ara
        for name in possible_names:
            file_path = self.raw_dir / name
            if file_path.exists():
                logger.warning(f"⚠️  {symbol} sadece raw veri: {file_path}")
                return file_path
        
        # Tüm data klasörünü tara
        data_dir = Path("data")
        if data_dir.exists():
            for file_path in data_dir.rglob(f"*{symbol}*.parquet"):
                logger.info(f"🔍 {symbol} alternatif veri: {file_path}")
                return file_path
        
        logger.error(f"❌ {symbol} için hiç veri bulunamadı!")
        return None
    
    def _load_processed_data_fixed(self):
        """🔧 DÜZELTME: İşlenmiş feature verilerini akıllıca yükle."""
        
        print("🔍 Veri dosyaları taranıyor...")
        
        # Dizin varlığını kontrol et
        print(f"📁 Processed dir: {self.processed_dir.exists()} - {self.processed_dir}")
        print(f"📁 Raw dir: {self.raw_dir.exists()} - {self.raw_dir}")
        
        # Tüm parquet dosyalarını listele
        all_parquet_files = []
        data_dir = Path("data")
        if data_dir.exists():
            for file_path in data_dir.rglob("*.parquet"):
                all_parquet_files.append(file_path)
        
        print(f"📊 Bulunan tüm parquet dosyalar:")
        for file_path in all_parquet_files:
            file_size = file_path.stat().st_size / 1024 / 1024  # MB
            print(f"   {file_path} ({file_size:.1f} MB)")
        
        # Her sembol için veri yükle
        for symbol in ['EURUSD', 'XAUUSD']:
            logger.info(f"🎯 {symbol} verisi aranıyor...")
            
            data_file = self._find_data_files(symbol)
            
            if data_file and data_file.exists():
                try:
                    df = pd.read_parquet(data_file)
                    
                    # NaN kontrolü
                    initial_len = len(df)
                    df = df.dropna()
                    final_len = len(df)
                    
                    if final_len > 0:
                        self._cached_data[symbol] = df
                        
                        nan_removed = initial_len - final_len
                        if nan_removed > 0:
                            logger.warning(f"🧹 {symbol}: {nan_removed} NaN satır temizlendi")
                        
                        logger.info(f"✅ {symbol} verisi yüklendi: {final_len} satır, {len(df.columns)} özellik")
                        
                        # Özellik türlerini analiz et
                        feature_cols = [col for col in df.columns 
                                       if col not in ['open', 'high', 'low', 'close', 'volume']]
                        
                        print(f"   📊 OHLCV sütunlar: {len(df.columns) - len(feature_cols)}")
                        print(f"   🔧 Feature sütunlar: {len(feature_cols)}")
                        
                        if len(feature_cols) >= 5:
                            print(f"   📋 İlk 5 feature: {feature_cols[:5]}")
                        
                        # Veri kalitesi kontrol
                        if 'close' in df.columns:
                            price_range = df['close'].max() - df['close'].min()
                            print(f"   💰 Fiyat aralığı: {df['close'].min():.5f} - {df['close'].max():.5f}")
                            print(f"   📈 Volatilite: {price_range:.5f}")
                        
                        print(f"   📅 Zaman aralığı: {df.index.min()} - {df.index.max()}")
                        
                    else:
                        logger.warning(f"⚠️  {symbol} verisi NaN temizleme sonrası boş!")
                        
                except Exception as e:
                    logger.error(f"❌ {symbol} veri yükleme hatası: {e}")
                    import traceback
                    traceback.print_exc()
            else:
                logger.error(f"❌ {symbol} veri dosyası bulunamadı!")
        
        # Sonuç özeti
        loaded_symbols = list(self._cached_data.keys())
        print(f"\n📊 VERİ YÜKLEME SONUCU:")
        print(f"   ✅ Yüklenen semboller: {loaded_symbols} ({len(loaded_symbols)}/2)")
        
        if not loaded_symbols:
            print(f"   ❌ HİÇ VERİ YÜKLENEMEDİ!")
            print(f"   🔧 Çözüm önerileri:")
            print(f"      1. Feature engineering'i tekrar çalıştırın")
            print(f"      2. Veri dosyalarının varlığını kontrol edin")
            print(f"      3. Dosya izinlerini kontrol edin")
        else:
            for symbol in loaded_symbols:
                df = self._cached_data[symbol]
                print(f"   📈 {symbol}: {len(df)} satır, {len(df.columns)} sütun")
    
    def get_trading_signals(self, genome: Genome, data: pd.DataFrame) -> pd.Series:
        """
        Genome'dan trading sinyalleri üretir - veri kontrolü eklenmiş.
        
        Args:
            genome: Sinyal üreten genom
            data: Feature DataFrame'i
            
        Returns:
            Sinyal serisi (-1: sell, 0: hold, 1: buy)
        """
        if data.empty:
            logger.warning("Boş DataFrame - sinyal üretilemiyor")
            return pd.Series(dtype=int)
        
        # Feature sütunlarını belirle (OHLCV hariç)
        feature_cols = [col for col in data.columns 
                       if col not in ['open', 'high', 'low', 'close', 'volume']]
        
        if len(feature_cols) < genome.input_size:
            logger.warning(f"Yetersiz feature: {len(feature_cols)} < {genome.input_size}")
            # Mevcut feature'ları tekrarla
            if feature_cols:
                while len(feature_cols) < genome.input_size:
                    feature_cols.extend(feature_cols[:min(len(feature_cols), genome.input_size - len(feature_cols))])
                feature_cols = feature_cols[:genome.input_size]
            else:
                return pd.Series(0, index=data.index)
        
        # İlk N feature'ı kullan
        selected_features = feature_cols[:genome.input_size]
        
        # Veri kalitesi kontrol
        feature_data = data[selected_features]
        
        # NaN satırları atla
        valid_mask = ~feature_data.isnull().any(axis=1)
        valid_indices = data.index[valid_mask]
        valid_data = feature_data.loc[valid_mask].values
        
        if len(valid_data) == 0:
            logger.warning("Geçerli feature verisi yok")
            return pd.Series(0, index=data.index)
        
        logger.debug(f"Sinyal üretimi: {len(valid_data)}/{len(data)} geçerli satır")
        
        signals = pd.Series(0, index=data.index)
        
        for i, idx in enumerate(valid_indices):
            try:
                # Genomdan tahmin al
                inputs = valid_data[i]
                
                # Normalizasyon (robust)
                inputs = np.tanh(inputs / (2.0 + 1e-8))  # [-1, 1] aralığına sıkıştır
                
                # Genom prediction
                outputs = genome.predict(inputs)
                
                # 3 output: [sell_prob, hold_prob, buy_prob]
                if len(outputs) >= 3:
                    sell_prob = outputs[0]
                    hold_prob = outputs[1] 
                    buy_prob = outputs[2]
                    
                    # En yüksek probabiliteyi seç
                    max_idx = np.argmax([sell_prob, hold_prob, buy_prob])
                    max_prob = max([sell_prob, hold_prob, buy_prob])
                    
                    if max_idx == 0 and sell_prob > self.signal_threshold:
                        signals.iloc[data.index.get_loc(idx)] = -1  # Sell
                    elif max_idx == 2 and buy_prob > self.signal_threshold:
                        signals.iloc[data.index.get_loc(idx)] = 1   # Buy
                    else:
                        signals.iloc[data.index.get_loc(idx)] = 0   # Hold
                else:
                    # Fallback: tek output
                    output = outputs[0] if len(outputs) > 0 else 0.5
                    
                    if output < (0.5 - self.signal_threshold):
                        signals.iloc[data.index.get_loc(idx)] = -1
                    elif output > (0.5 + self.signal_threshold):
                        signals.iloc[data.index.get_loc(idx)] = 1
                    else:
                        signals.iloc[data.index.get_loc(idx)] = 0
                        
            except Exception as e:
                logger.debug(f"Sinyal hesaplama hatası (satır {i}): {e}")
                continue
        
        signal_counts = signals.value_counts()
        logger.debug(f"Sinyal dağılımı: {signal_counts.to_dict()}")
        
        return signals
    
    def calculate_fitness(self, genome: Genome) -> float:
        """
        🔧 DÜZELTME: Ana fitness hesaplama fonksiyonu - veri kontrolü eklenmiş.
        
        Args:
            genome: Değerlendirilecek genom
            
        Returns:
            Fitness skoru
        """
        try:
            # Cache durumunu kontrol et
            if not self._cached_data:
                logger.error("Hiç cached veri yok - fitness hesaplanamıyor!")
                return -1.0
            
            # Multi-symbol backtest
            results = {}
            
            for symbol in self.symbol_weights.keys():
                if symbol in self._cached_data:
                    result = self.backtest_symbol(genome, symbol)
                    results[symbol] = result
                else:
                    logger.warning(f"⚠️  {symbol} verisi cache'de yok")
            
            if not results:
                logger.error("Hiç backtest sonucu yok!")
                return -1.0
            
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
                        trade_factor = min(result.total_trades, 100)  # Cap at 100
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
                    
                    logger.debug(f"{symbol} fitness: {symbol_fitness:.6f} (trades: {result.total_trades})")
            
            # Normalize et
            if total_weight > 0:
                final_fitness = total_fitness / total_weight
            else:
                final_fitness = -1.0
            
            # Ağ karmaşıklığı cezası
            complexity_penalty = len(genome.connections) * self.config.get('complexity_penalty', 0.0001)
            final_fitness -= complexity_penalty
            
            # Minimum fitness
            final_fitness = max(final_fitness, -1.0)
            
            return float(final_fitness)
            
        except Exception as e:
            logger.error(f"Fitness hesaplama hatası: {e}")
            import traceback
            traceback.print_exc()
            return -1.0
    
    def backtest_symbol(self, genome: Genome, symbol: str, 
                       start_date: str = None, end_date: str = None) -> BacktestResult:
        """
        🔧 DÜZELTME: Tek sembol için backtest - veri kontrolü eklenmiş.
        
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
            logger.warning(f"⚠️  Veri bulunamadı: {symbol}")
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
        
        if signals.empty or abs(signals.sum()) < 5:  # Minimum sinyal kontrolü
            logger.debug(f"Yetersiz sinyal: {symbol} (toplam: {signals.sum()})")
            return self._empty_backtest_result()
        
        # Basit backtest simülasyonu (hızlı versiyon)
        try:
            balance = self.initial_balance
            trades = []
            
            # Sinyal değişiklik noktalarını bul
            signal_changes = signals.diff().fillna(0)
            trade_points = signal_changes[signal_changes != 0].index
            
            if len(trade_points) < 2:
                return self._empty_backtest_result()
            
            # Basit trade simülasyonu
            for i in range(0, len(trade_points)-1, 2):  # Çift çift al (giriş-çıkış)
                try:
                    entry_time = trade_points[i]
                    exit_time = trade_points[i+1] if i+1 < len(trade_points) else data.index[-1]
                    
                    entry_price = data.loc[entry_time, 'close']
                    exit_price = data.loc[exit_time, 'close']
                    
                    direction = signals.loc[entry_time]
                    
                    if direction != 0:
                        # P&L hesapla
                        if direction > 0:  # Long
                            pnl = (exit_price - entry_price) * 1000  # 1000 = lot size
                        else:  # Short
                            pnl = (entry_price - exit_price) * 1000
                        
                        # Komisyon ve slippage
                        commission_cost = abs(1000) * exit_price * self.commission[symbol]
                        slippage_cost = abs(1000) * self.slippage[symbol]
                        
                        net_pnl = pnl - commission_cost - slippage_cost
                        balance += net_pnl
                        
                        # Trade kaydı
                        trade = TradeResult(
                            entry_time=entry_time,
                            exit_time=exit_time,
                            entry_price=entry_price,
                            exit_price=exit_price,
                            direction=direction,
                            pnl=net_pnl,
                            pnl_pct=net_pnl / entry_price,
                            commission=commission_cost,
                            slippage=slippage_cost
                        )
                        trades.append(trade)
                        
                except (KeyError, IndexError):
                    continue
            
            # Sonuçları hesapla
            return self._calculate_backtest_metrics(trades, [self.initial_balance, balance])
            
        except Exception as e:
            logger.error(f"Backtest simülasyon hatası ({symbol}): {e}")
            return self._empty_backtest_result()
    
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
        Backtest metriklerini hesaplar - hata kontrolü eklenmiş.
        
        Args:
            trades: Trade listesi
            equity_curve: Equity curve
            
        Returns:
            Backtest sonuçları
        """
        if not trades:
            return self._empty_backtest_result()
        
        try:
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
            
            # Basit drawdown (equity curve ile)
            if len(equity_curve) > 1:
                equity_series = pd.Series(equity_curve)
                running_max = equity_series.expanding().max()
                drawdown_series = (running_max - equity_series) / running_max
                max_drawdown = drawdown_series.max()
                
                # Drawdown bars (basit)
                max_drawdown_bars = len([d for d in drawdown_series if d > 0.01])
            else:
                max_drawdown = 0.0
                max_drawdown_bars = 0
            
            # Basit Sharpe ratio
            if len(trades) > 1:
                returns = [t.pnl for t in trades]
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
                equity_curve=pd.Series(equity_curve),
                trades=trades
            )
            
        except Exception as e:
            logger.error(f"Metrik hesaplama hatası: {e}")
            return self._empty_backtest_result()


def main():
    """Test fonksiyonu - düzeltilmiş fitness."""
    logger.info("💪 Düzeltilmiş Fitness modülü test ediliyor...")
    
    # Test config
    config = {
        'initial_balance': 10000,
        'risk_per_trade': 0.01,
        'signal_threshold': 0.1,
        'min_trades': 5,  # Test için düşük
        'complexity_penalty': 0.0001
    }
    
    # Düzeltilmiş fitness evaluator oluştur
    evaluator = FitnessEvaluatorFixed(config)
    
    # Test genomu oluştur
    from genome import Genome
    test_genome = Genome(input_size=10, output_size=3)
    
    # Hafif mutasyon
    for _ in range(5):
        test_genome.mutate()
    
    logger.info(f"🧬 Test genomu: {test_genome}")
    
    # Fitness hesapla
    logger.info("🧪 Düzeltilmiş fitness hesaplanıyor...")
    fitness_score = evaluator.calculate_fitness(test_genome)
    
    logger.info(f"🎯 Test fitness: {fitness_score:.6f}")
    
    # Cache durumunu göster
    cache_info = {}
    for symbol, df in evaluator._cached_data.items():
        cache_info[symbol] = {
            'rows': len(df),
            'columns': len(df.columns),
            'features': len([col for col in df.columns if col not in ['open', 'high', 'low', 'close', 'volume']])
        }
    
    print(f"\n📊 Cached Data Durumu:")
    for symbol, info in cache_info.items():
        print(f"   {symbol}: {info['rows']} satır, {info['features']} feature")
    
    logger.info("✅ Düzeltilmiş Fitness modülü test tamamlandı!")


if __name__ == "__main__":
    main()