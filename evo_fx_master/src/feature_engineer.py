#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
EVO_FX_MASTER - Gelişmiş Özellik Mühendisliği Modülü
==================================================

MASTER PROMPT gereksinimlerine göre:
- Cross-asset korelasyon optimize edildi
- TA-Lib entegrasyonu geliştirildi  
- Mikroyapısal özellikler genişletildi
- 30+ teknik indikatör ve zaman özellikleri
"""

import pandas as pd
import numpy as np
import logging
from typing import Dict, List, Optional, Tuple
from pathlib import Path
import warnings
from datetime import datetime
from scipy import stats
from sklearn.preprocessing import StandardScaler

# Teknik analiz kütüphanesi
try:
    import talib
    TALIB_AVAILABLE = True
except ImportError:
    TALIB_AVAILABLE = False
    print("⚠️  TA-Lib bulunamadı. Manuel hesaplamalar kullanılacak.")

warnings.filterwarnings('ignore')
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class AdvancedFeatureEngineer:
    """Gelişmiş özellik mühendisliği sınıfı - MASTER PROMPT uyumlu."""
    
    def __init__(self, config: Dict = None):
        """
        Args:
            config: Özellik parametreleri içeren sözlük
        """
        self.config = config or {}
        
        # Teknik indikatör parametreleri
        self.rsi_period = self.config.get('rsi_period', 14)
        self.atr_period = self.config.get('atr_period', 14) 
        self.volatility_period = self.config.get('volatility_period', 30)
        self.correlation_period = self.config.get('correlation_period', 50)
        self.bollinger_period = self.config.get('bollinger_period', 20)
        self.bollinger_std = self.config.get('bollinger_std', 2)
        
        # Moving average periods
        self.ma_short = self.config.get('ma_short', 12)
        self.ma_long = self.config.get('ma_long', 26)
        self.ma_signal = self.config.get('ma_signal', 9)
        
        # Mikroyapısal parametreler
        self.tick_momentum_periods = self.config.get('tick_momentum_periods', [5, 10, 20])
        self.volume_periods = self.config.get('volume_periods', [10, 20, 50])
        
        # Veri dizinleri
        self.raw_dir = Path("data/raw")
        self.processed_dir = Path("data/processed")
        self.processed_dir.mkdir(parents=True, exist_ok=True)
        
        # Feature scaling
        self.scaler = StandardScaler()
        
        logger.info(f"🔧 AdvancedFeatureEngineer hazırlandı. TA-Lib: {TALIB_AVAILABLE}")
    
    def calculate_advanced_returns(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Gelişmiş return hesaplamaları.
        
        Args:
            df: OHLCV DataFrame'i
            
        Returns:
            Return sütunları eklenmiş DataFrame
        """
        result = df.copy()
        
        # Temel log returns
        result['log_return'] = np.log(result['close'] / result['close'].shift(1))
        result['log_return_5m'] = np.log(result['close'] / result['close'].shift(5))
        result['log_return_15m'] = np.log(result['close'] / result['close'].shift(15))
        result['log_return_1h'] = np.log(result['close'] / result['close'].shift(60))
        result['log_return_4h'] = np.log(result['close'] / result['close'].shift(240))
        result['log_return_1d'] = np.log(result['close'] / result['close'].shift(1440))
        
        # Realized volatility (different periods)
        for period in [10, 30, 60]:
            result[f'realized_vol_{period}'] = result['log_return'].rolling(period).std() * np.sqrt(period)
        
        # Return skewness and kurtosis
        for period in [20, 50]:
            result[f'return_skew_{period}'] = result['log_return'].rolling(period).skew()
            result[f'return_kurt_{period}'] = result['log_return'].rolling(period).kurt()
        
        # Price momentum indicators
        for period in [5, 10, 20, 50]:
            result[f'momentum_{period}'] = result['close'] / result['close'].shift(period) - 1
        
        # OHLC relationships
        result['hl_ratio'] = (result['high'] - result['low']) / result['close']
        result['oc_ratio'] = (result['close'] - result['open']) / result['open']
        result['ho_ratio'] = (result['high'] - result['open']) / result['open']
        result['lo_ratio'] = (result['low'] - result['open']) / result['open']
        
        # Intrabar pressure
        result['buying_pressure'] = (result['close'] - result['low']) / (result['high'] - result['low'])
        result['selling_pressure'] = (result['high'] - result['close']) / (result['high'] - result['low'])
        
        return result
    
    def calculate_advanced_technical_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Gelişmiş teknik indikatörler - 30+ indikatör.
        
        Args:
            df: OHLCV DataFrame'i
            
        Returns:
            Teknik indikatör sütunları eklenmiş DataFrame
        """
        result = df.copy()
        
        # RSI variations
        if TALIB_AVAILABLE:
            result['rsi'] = talib.RSI(result['close'].values, timeperiod=self.rsi_period)
            result['rsi_5'] = talib.RSI(result['close'].values, timeperiod=5)
            result['rsi_21'] = talib.RSI(result['close'].values, timeperiod=21)
        else:
            # Manuel RSI hesaplama
            for period in [5, 14, 21]:
                delta = result['close'].diff()
                gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
                loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
                rs = gain / (loss + 1e-8)
                result[f'rsi' if period == 14 else f'rsi_{period}'] = 100 - (100 / (1 + rs))
        
        # Normalize RSI values
        for col in ['rsi', 'rsi_5', 'rsi_21']:
            if col in result.columns:
                result[f'{col}_norm'] = result[col] / 100.0
                result[f'{col}_overbought'] = (result[col] > 70).astype(int)
                result[f'{col}_oversold'] = (result[col] < 30).astype(int)
        
        # ATR variations
        high_low = result['high'] - result['low']
        high_close = np.abs(result['high'] - result['close'].shift())
        low_close = np.abs(result['low'] - result['close'].shift())
        true_range = np.maximum(high_low, np.maximum(high_close, low_close))
        
        for period in [7, 14, 21]:
            atr_name = f'atr' if period == 14 else f'atr_{period}'
            result[atr_name] = true_range.rolling(period).mean()
            result[f'{atr_name}_pct'] = result[atr_name] / result['close']
            result[f'{atr_name}_zscore'] = (result[atr_name] - result[atr_name].rolling(100).mean()) / result[atr_name].rolling(100).std()
        
        # Bollinger Bands variations
        for period, std_dev in [(20, 2), (20, 1.5), (50, 2)]:
            bb_name = f'bb_{period}' if std_dev == 2 else f'bb_{period}_{int(std_dev*10)}'
            bb_middle = result['close'].rolling(period).mean()
            bb_std = result['close'].rolling(period).std()
            
            result[f'{bb_name}_upper'] = bb_middle + (bb_std * std_dev)
            result[f'{bb_name}_lower'] = bb_middle - (bb_std * std_dev)
            result[f'{bb_name}_position'] = (result['close'] - result[f'{bb_name}_lower']) / (result[f'{bb_name}_upper'] - result[f'{bb_name}_lower'])
            result[f'{bb_name}_width'] = (result[f'{bb_name}_upper'] - result[f'{bb_name}_lower']) / bb_middle
        
        # Moving averages and crossovers
        ma_periods = [5, 10, 20, 50, 100, 200]
        for period in ma_periods:
            result[f'sma_{period}'] = result['close'].rolling(period).mean()
            result[f'ema_{period}'] = result['close'].ewm(span=period).mean()
            result[f'price_vs_sma_{period}'] = (result['close'] - result[f'sma_{period}']) / result[f'sma_{period}']
            result[f'price_vs_ema_{period}'] = (result['close'] - result[f'ema_{period}']) / result[f'ema_{period}']
        
        # MA crossover signals
        result['sma_cross_5_20'] = np.where(result['sma_5'] > result['sma_20'], 1, 0)
        result['sma_cross_20_50'] = np.where(result['sma_20'] > result['sma_50'], 1, 0)
        result['ema_cross_12_26'] = np.where(result['ema_12'] > result['ema_26'], 1, 0)
        
        # MACD family
        ema_12 = result['close'].ewm(span=12).mean()
        ema_26 = result['close'].ewm(span=26).mean()
        result['macd'] = ema_12 - ema_26
        result['macd_signal'] = result['macd'].ewm(span=9).mean()
        result['macd_histogram'] = result['macd'] - result['macd_signal']
        result['macd_cross'] = np.where(result['macd'] > result['macd_signal'], 1, 0)
        
        # Stochastic oscillator
        lowest_low = result['low'].rolling(14).min()
        highest_high = result['high'].rolling(14).max()
        result['stoch_k'] = 100 * (result['close'] - lowest_low) / (highest_high - lowest_low)
        result['stoch_d'] = result['stoch_k'].rolling(3).mean()
        result['stoch_cross'] = np.where(result['stoch_k'] > result['stoch_d'], 1, 0)
        
        # Williams %R
        result['williams_r'] = -100 * (highest_high - result['close']) / (highest_high - lowest_low)
        
        # Commodity Channel Index (CCI)
        typical_price = (result['high'] + result['low'] + result['close']) / 3
        cci_period = 20
        sma_tp = typical_price.rolling(cci_period).mean()
        mad = typical_price.rolling(cci_period).apply(lambda x: np.mean(np.abs(x - x.mean())))
        result['cci'] = (typical_price - sma_tp) / (0.015 * mad)
        
        # Rate of Change (ROC)
        for period in [5, 10, 20]:
            result[f'roc_{period}'] = (result['close'] - result['close'].shift(period)) / result['close'].shift(period) * 100
        
        return result
    
    def calculate_microstructure_features_advanced(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Gelişmiş mikro-yapısal piyasa özellikler.
        
        Args:
            df: OHLCV DataFrame'i
            
        Returns:
            Mikro-yapısal özellikli DataFrame
        """
        result = df.copy()
        
        # Tick direction and momentum
        price_change = result['close'].diff()
        result['tick_direction'] = np.sign(price_change)
        
        for period in self.tick_momentum_periods:
            result[f'tick_momentum_{period}'] = result['tick_direction'].rolling(period).sum()
            result[f'tick_momentum_{period}_norm'] = result[f'tick_momentum_{period}'] / period
        
        # Price acceleration
        result['price_acceleration'] = result['close'].diff().diff()
        result['price_acceleration_ma'] = result['price_acceleration'].rolling(10).mean()
        
        # Volume analysis (if available)
        if 'volume' in result.columns and result['volume'].sum() > 0:
            for period in self.volume_periods:
                result[f'volume_ma_{period}'] = result['volume'].rolling(period).mean()
                result[f'volume_ratio_{period}'] = result['volume'] / (result[f'volume_ma_{period}'] + 1e-8)
                result[f'volume_zscore_{period}'] = (result['volume'] - result[f'volume_ma_{period}']) / (result['volume'].rolling(period).std() + 1e-8)
            
            # Volume-price trend (VPT)
            result['vpt'] = (result['volume'] * result['log_return']).cumsum()
            result['vpt_ma'] = result['vpt'].rolling(20).mean()
            result['vpt_signal'] = np.where(result['vpt'] > result['vpt_ma'], 1, -1)
            
            # On-Balance Volume (OBV)
            result['obv'] = (result['volume'] * result['tick_direction']).cumsum()
            result['obv_ma'] = result['obv'].rolling(20).mean()
            result['obv_signal'] = np.where(result['obv'] > result['obv_ma'], 1, -1)
            
            # Volume Weighted Average Price (VWAP)
            typical_price = (result['high'] + result['low'] + result['close']) / 3
            result['vwap'] = (typical_price * result['volume']).cumsum() / result['volume'].cumsum()
            result['price_vs_vwap'] = (result['close'] - result['vwap']) / result['vwap']
        
        # Advanced candle patterns
        result['body_size'] = np.abs(result['close'] - result['open']) / result['close']
        result['upper_shadow'] = (result['high'] - np.maximum(result['open'], result['close'])) / result['close']
        result['lower_shadow'] = (np.minimum(result['open'], result['close']) - result['low']) / result['close']
        result['shadow_ratio'] = result['upper_shadow'] / (result['lower_shadow'] + 1e-8)
        
        # Pattern recognition
        result['doji'] = ((result['body_size'] < 0.001) & (result['upper_shadow'] > 0.002) & (result['lower_shadow'] > 0.002)).astype(int)
        result['hammer'] = ((result['lower_shadow'] > 2 * result['body_size']) & (result['upper_shadow'] < result['body_size'])).astype(int)
        result['shooting_star'] = ((result['upper_shadow'] > 2 * result['body_size']) & (result['lower_shadow'] < result['body_size'])).astype(int)
        result['spinning_top'] = ((result['body_size'] < 0.005) & (result['upper_shadow'] > result['body_size']) & (result['lower_shadow'] > result['body_size'])).astype(int)
        
        # Gap analysis
        result['gap'] = result['open'] - result['close'].shift(1)
        result['gap_pct'] = result['gap'] / result['close'].shift(1)
        result['gap_up'] = (result['gap'] > 0).astype(int)
        result['gap_down'] = (result['gap'] < 0).astype(int)
        
        # Gap filling
        result['gap_filled'] = ((result['gap'] > 0) & (result['low'] <= result['close'].shift(1))) | ((result['gap'] < 0) & (result['high'] >= result['close'].shift(1)))
        result['gap_filled'] = result['gap_filled'].astype(int)
        
        # Spread proxy and liquidity measures
        result['spread_proxy'] = (result['high'] - result['low']) / result['close']
        result['spread_ma'] = result['spread_proxy'].rolling(20).mean()
        result['spread_ratio'] = result['spread_proxy'] / (result['spread_ma'] + 1e-8)
        result['spread_zscore'] = (result['spread_proxy'] - result['spread_ma']) / (result['spread_proxy'].rolling(20).std() + 1e-8)
        
        return result
    
    def calculate_time_features_advanced(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Gelişmiş zaman bazlı özellikler.
        
        Args:
            df: Timestamp index'li DataFrame
            
        Returns:
            Zaman özellikli DataFrame
        """
        result = df.copy()
        
        # Temel zaman özellikleri
        result['hour'] = result.index.hour
        result['minute'] = result.index.minute
        result['day_of_week'] = result.index.dayofweek
        result['day_of_month'] = result.index.day
        result['month'] = result.index.month
        result['quarter'] = result.index.quarter
        result['week_of_year'] = result.index.isocalendar().week
        
        # Trading session detection (UTC times)
        # Sydney: 22:00-06:00 UTC, Tokyo: 00:00-08:00 UTC
        # London: 08:00-16:00 UTC, New York: 13:00-21:00 UTC
        result['session_sydney'] = ((result['hour'] >= 22) | (result['hour'] < 6)).astype(int)
        result['session_tokyo'] = ((result['hour'] >= 0) & (result['hour'] < 8)).astype(int)
        result['session_london'] = ((result['hour'] >= 8) & (result['hour'] < 16)).astype(int)
        result['session_newyork'] = ((result['hour'] >= 13) & (result['hour'] < 21)).astype(int)
        
        # Session overlaps (high liquidity periods)
        result['session_overlap_london_ny'] = ((result['hour'] >= 13) & (result['hour'] < 16)).astype(int)
        result['session_overlap_sydney_tokyo'] = ((result['hour'] >= 0) & (result['hour'] < 6)).astype(int)
        result['session_overlap_tokyo_london'] = ((result['hour'] >= 8) & (result['hour'] < 8)).astype(int)  # No overlap
        
        # Market activity levels
        result['high_activity'] = (result['session_london'] | result['session_newyork']).astype(int)
        result['low_activity'] = (result['session_sydney'] & ~result['session_tokyo']).astype(int)
        
        # Trigonometric encoding for cyclical features
        result['hour_sin'] = np.sin(2 * np.pi * result['hour'] / 24)
        result['hour_cos'] = np.cos(2 * np.pi * result['hour'] / 24)
        result['minute_sin'] = np.sin(2 * np.pi * result['minute'] / 60)
        result['minute_cos'] = np.cos(2 * np.pi * result['minute'] / 60)
        result['dow_sin'] = np.sin(2 * np.pi * result['day_of_week'] / 7)
        result['dow_cos'] = np.cos(2 * np.pi * result['day_of_week'] / 7)
        result['month_sin'] = np.sin(2 * np.pi * result['month'] / 12)
        result['month_cos'] = np.cos(2 * np.pi * result['month'] / 12)
        
        # Week patterns
        result['is_monday'] = (result['day_of_week'] == 0).astype(int)
        result['is_friday'] = (result['day_of_week'] == 4).astype(int)
        result['is_weekend'] = (result['day_of_week'] >= 5).astype(int)
        result['is_month_end'] = (result['day_of_month'] >= 28).astype(int)
        result['is_month_start'] = (result['day_of_month'] <= 3).astype(int)
        
        # Economic calendar approximation
        result['potential_nfp'] = ((result['day_of_week'] == 4) & (result.index.day <= 7)).astype(int)
        result['potential_fomc'] = ((result['day_of_week'] == 2) & (result.index.day >= 14) & (result.index.day <= 21)).astype(int)
        
        return result
    
    def calculate_cross_asset_features_optimized(self, eur_data: pd.DataFrame, 
                                               gold_data: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Optimize edilmiş cross-asset korelasyon özellikleri.
        
        Args:
            eur_data: EURUSD DataFrame'i
            gold_data: XAUUSD DataFrame'i
            
        Returns:
            Korelasyon özellikli (EUR_DF, GOLD_DF) tuple'ı
        """
        # Ortak timestamp'leri bul
        common_index = eur_data.index.intersection(gold_data.index)
        
        if len(common_index) < self.correlation_period:
            logger.warning("Yeterli ortak veri yok, korelasyon özelliği atlanıyor")
            return eur_data, gold_data
        
        # Ortak indexe göre filtrele
        eur_common = eur_data.loc[common_index].copy()
        gold_common = gold_data.loc[common_index].copy()
        
        # Multiple correlation periods
        for period in [20, 50, 100]:
            # Return correlations
            correlation = eur_common['log_return'].rolling(period).corr(gold_common['log_return'])
            
            # Volatility correlations
            eur_vol = eur_common['log_return'].rolling(period).std()
            gold_vol = gold_common['log_return'].rolling(period).std()
            vol_correlation = eur_vol.rolling(period).corr(gold_vol)
            
            # Price level correlations
            price_correlation = eur_common['close'].rolling(period).corr(gold_common['close'])
            
            # EUR features
            eur_common[f'correlation_{period}'] = correlation
            eur_common[f'vol_correlation_{period}'] = vol_correlation
            eur_common[f'price_correlation_{period}'] = price_correlation
            eur_common[f'gold_momentum_{period}'] = gold_common['log_return'].rolling(period).mean()
            eur_common[f'gold_volatility_{period}'] = gold_vol
            
            # GOLD features
            gold_common[f'correlation_{period}'] = correlation
            gold_common[f'vol_correlation_{period}'] = vol_correlation
            gold_common[f'price_correlation_{period}'] = price_correlation
            gold_common[f'eur_momentum_{period}'] = eur_common['log_return'].rolling(period).mean()
            gold_common[f'eur_volatility_{period}'] = eur_vol
        
        # Spread and ratio analysis
        eur_common['eur_gold_return_spread'] = eur_common['log_return'] - gold_common['log_return']
        gold_common['gold_eur_return_spread'] = gold_common['log_return'] - eur_common['log_return']
        
        # Normalized price ratio
        eur_norm = eur_common['close'] / eur_common['close'].iloc[0]
        gold_norm = gold_common['close'] / gold_common['close'].iloc[0]
        
        ratio = eur_norm / gold_norm
        ratio_ma = ratio.rolling(50).mean()
        ratio_std = ratio.rolling(50).std()
        
        eur_common['eur_gold_ratio_zscore'] = (ratio - ratio_ma) / (ratio_std + 1e-8)
        gold_common['gold_eur_ratio_zscore'] = -eur_common['eur_gold_ratio_zscore']
        
        # Beta calculation (EUR vs GOLD)
        covariance = eur_common['log_return'].rolling(50).cov(gold_common['log_return'])
        gold_variance = gold_common['log_return'].rolling(50).var()
        eur_common['beta_vs_gold'] = covariance / (gold_variance + 1e-8)
        gold_common['beta_vs_eur'] = 1 / (eur_common['beta_vs_gold'] + 1e-8)
        
        # Orijinal DataFrame'lere geri map et
        eur_result = eur_data.copy()
        gold_result = gold_data.copy()
        
        # Cross-asset features'ları ekle
        cross_features = [col for col in eur_common.columns if col not in eur_data.columns]
        
        for feature in cross_features:
            if feature in eur_common.columns:
                eur_result[feature] = eur_common[feature].reindex(eur_result.index)
        
        gold_cross_features = [col for col in gold_common.columns if col not in gold_data.columns]
        
        for feature in gold_cross_features:
            if feature in gold_common.columns:
                gold_result[feature] = gold_common[feature].reindex(gold_result.index)
        
        logger.info("✓ Optimize edilmiş cross-asset özellikler hesaplandı")
        return eur_result, gold_result
    
    def bake_features_advanced(self, symbol: str, df: pd.DataFrame = None) -> Optional[pd.DataFrame]:
        """
        Tüm gelişmiş özellik mühendisliği işlemlerini uygular.
        
        Args:
            symbol: Sembol adı (EURUSD, XAUUSD)
            df: Opsiyonel DataFrame (None ise dosyadan yükler)
            
        Returns:
            Özellik mühendisliği yapılmış DataFrame
        """
        # Veri yükleme
        if df is None:
            raw_file = self.raw_dir / f"{symbol}_raw.parquet"
            if not raw_file.exists():
                logger.error(f"Ham veri bulunamadı: {raw_file}")
                return None
            
            try:
                df = pd.read_parquet(raw_file)
                logger.info(f"📁 Ham veri yüklendi: {symbol} ({len(df)} satır)")
            except Exception as e:
                logger.error(f"Veri yükleme hatası: {e}")
                return None
        
        if df.empty:
            logger.warning(f"Boş DataFrame: {symbol}")
            return None
        
        logger.info(f"🔧 Gelişmiş feature engineering başlatılıyor: {symbol}")
        
        # 1. Advanced returns
        df = self.calculate_advanced_returns(df)
        logger.info("✓ Gelişmiş returns hesaplandı")
        
        # 2. Advanced technical indicators (30+)
        df = self.calculate_advanced_technical_indicators(df)
        logger.info("✓ Gelişmiş teknik indikatörler hesaplandı")
        
        # 3. Advanced time features
        df = self.calculate_time_features_advanced(df)
        logger.info("✓ Gelişmiş zaman özellikleri hesaplandı")
        
        # 4. Advanced microstructure features
        df = self.calculate_microstructure_features_advanced(df)
        logger.info("✓ Gelişmiş mikro-yapısal özellikler hesaplandı")
        
        # 5. Feature cleaning and preprocessing
        df = self._clean_and_preprocess_features(df)
        logger.info("✓ Feature temizleme ve preprocessing tamamlandı")
        
        logger.info(f"✅ Gelişmiş feature engineering tamamlandı: {symbol} ({len(df)} satır, {len(df.columns)} özellik)")
        return df
    
    def _clean_and_preprocess_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Feature temizleme ve preprocessing."""
        # Sonsuz değer kontrolü
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        df[numeric_cols] = df[numeric_cols].replace([np.inf, -np.inf], np.nan)
        
        # Feature sayısını logla
        initial_len = len(df)
        
        # NaN temizleme (daha agresif)
        df = df.dropna()
        
        final_len = len(df)
        if final_len < initial_len:
            logger.info(f"🧹 NaN temizleme: {initial_len} -> {final_len} satır")
        
        # Outlier detection and clipping (optional)
        for col in numeric_cols:
            if col in df.columns and col not in ['open', 'high', 'low', 'close', 'volume']:
                # 99.5th percentile clipping
                upper_bound = df[col].quantile(0.995)
                lower_bound = df[col].quantile(0.005)
                df[col] = df[col].clip(lower_bound, upper_bound)
        
        return df
    
    def process_all_symbols_advanced(self, eur_data: pd.DataFrame = None, 
                                   gold_data: pd.DataFrame = None) -> Dict[str, pd.DataFrame]:
        """
        Tüm semboller için gelişmiş feature engineering yapar.
        
        Args:
            eur_data: Opsiyonel EURUSD DataFrame
            gold_data: Opsiyonel XAUUSD DataFrame
            
        Returns:
            İşlenmiş DataFrame'ler sözlüğü
        """
        results = {}
        
        # EURUSD işle
        logger.info("🎯 EURUSD gelişmiş feature engineering...")
        eur_features = self.bake_features_advanced('EURUSD', eur_data)
        
        # XAUUSD işle
        logger.info("🎯 XAUUSD gelişmiş feature engineering...")
        gold_features = self.bake_features_advanced('XAUUSD', gold_data)
        
        # Cross-asset features (optimize edilmiş)
        if eur_features is not None and gold_features is not None:
            logger.info("🔗 Optimize edilmiş cross-asset features hesaplanıyor...")
            eur_features, gold_features = self.calculate_cross_asset_features_optimized(
                eur_features, gold_features
            )
        
        # Kaydet
        if eur_features is not None:
            if self._save_processed_data_advanced('EURUSD', eur_features):
                results['EURUSD'] = eur_features
        
        if gold_features is not None:
            if self._save_processed_data_advanced('XAUUSD', gold_features):
                results['XAUUSD'] = gold_features
        
        logger.info(f"🏁 Gelişmiş feature engineering tamamlandı. Başarılı: {len(results)}/2")
        return results
    
    def _save_processed_data_advanced(self, symbol: str, df: pd.DataFrame) -> bool:
        """Gelişmiş işlenmiş veriyi kaydet."""
        if df is None or df.empty:
            return False
        
        try:
            output_path = self.processed_dir / f"{symbol}_features_advanced.parquet"
            df.to_parquet(output_path, compression='snappy')
            
            logger.info(f"💾 Gelişmiş işlenmiş veri kaydedildi: {output_path}")
            
            # Detaylı özet bilgi kaydet
            info_path = self.processed_dir / f"{symbol}_feature_info_advanced.txt"
            with open(info_path, 'w', encoding='utf-8') as f:
                f.write(f"EVO_FX_MASTER - Gelişmiş Feature Engineering Raporu\n")
                f.write("=" * 60 + "\n\n")
                f.write(f"Sembol: {symbol}\n")
                f.write(f"İşlem Tarihi: {datetime.now()}\n")
                f.write(f"Satır Sayısı: {len(df):,}\n")
                f.write(f"Toplam Özellik Sayısı: {len(df.columns)}\n")
                f.write(f"Tarih Aralığı: {df.index.min()} - {df.index.max()}\n\n")
                
                # Özellik kategorileri
                categories = {
                    'OHLCV': ['open', 'high', 'low', 'close', 'volume'],
                    'Returns': [col for col in df.columns if 'return' in col or 'momentum' in col],
                    'Technical': [col for col in df.columns if any(x in col for x in ['rsi', 'atr', 'bb_', 'sma_', 'ema_', 'macd', 'stoch'])],
                    'Time': [col for col in df.columns if any(x in col for x in ['hour', 'day', 'month', 'session', 'sin', 'cos'])],
                    'Microstructure': [col for col in df.columns if any(x in col for x in ['tick', 'volume', 'spread', 'gap', 'body', 'shadow'])],
                    'Cross-Asset': [col for col in df.columns if any(x in col for x in ['correlation', 'gold_', 'eur_', 'beta'])]
                }
                
                for category, features in categories.items():
                    actual_features = [f for f in features if f in df.columns]
                    if actual_features:
                        f.write(f"{category} Özellikleri ({len(actual_features)}):\n")
                        for i, feature in enumerate(actual_features, 1):
                            f.write(f"  {i:2d}. {feature}\n")
                        f.write("\n")
                
                # İstatistikler
                f.write("Özellik İstatistikleri:\n")
                f.write("-" * 30 + "\n")
                numeric_cols = df.select_dtypes(include=[np.number]).columns
                if len(numeric_cols) > 0:
                    stats_df = df[numeric_cols].describe()
                    f.write(stats_df.to_string())
                
            return True
            
        except Exception as e:
            logger.error(f"Kaydetme hatası: {e}")
            return False


def main():
    """Test fonksiyonu."""
    logger.info("🔧 AdvancedFeatureEngineer modülü test ediliyor...")
    
    # Test config
    config = {
        'rsi_period': 14,
        'atr_period': 14,
        'volatility_period': 30,
        'correlation_period': 50,
        'tick_momentum_periods': [5, 10, 20],
        'volume_periods': [10, 20, 50]
    }
    
    # AdvancedFeatureEngineer oluştur
    engineer = AdvancedFeatureEngineer(config)
    
    # Dummy test verisi oluştur
    logger.info("🧪 Gelişmiş test verisi oluşturuluyor...")
    
    dates = pd.date_range('2024-01-01', '2024-01-07', freq='1min', tz='UTC')
    n_samples = len(dates)
    
    # Gerçekçi EURUSD verisi simülasyonu
    np.random.seed(42)
    price_base = 1.1000
    returns = np.random.normal(0, 0.0001, n_samples).cumsum()
    prices = price_base * np.exp(returns)
    
    # OHLCV verisi oluştur
    test_data = pd.DataFrame(index=dates)
    test_data['close'] = prices
    test_data['open'] = test_data['close'].shift(1).fillna(price_base)
    
    # High/Low simulasyonu
    hl_range = np.random.uniform(0.0005, 0.002, n_samples)
    test_data['high'] = test_data['close'] + hl_range / 2
    test_data['low'] = test_data['close'] - hl_range / 2
    test_data['volume'] = np.random.randint(100, 1000, n_samples)
    
    logger.info(f"✓ Test verisi hazırlandı: {len(test_data)} mum")
    
    # Gelişmiş feature engineering test
    logger.info("🔧 Gelişmiş feature engineering testi...")
    featured_data = engineer.bake_features_advanced('TEST', test_data)
    
    if featured_data is not None:
        logger.info(f"✅ Test başarılı!")
        logger.info(f"📊 Toplam özellik sayısı: {len(featured_data.columns)}")
        logger.info(f"📈 Veri boyutu: {featured_data.shape}")
        
        # Özellik kategorilerini say
        categories = {
            'Returns': len([col for col in featured_data.columns if 'return' in col or 'momentum' in col]),
            'Technical': len([col for col in featured_data.columns if any(x in col for x in ['rsi', 'atr', 'bb_', 'sma_', 'ema_', 'macd'])]),
            'Time': len([col for col in featured_data.columns if any(x in col for x in ['hour', 'day', 'session', 'sin', 'cos'])]),
            'Microstructure': len([col for col in featured_data.columns if any(x in col for x in ['tick', 'volume', 'spread', 'gap'])])
        }
        
        logger.info("📋 Özellik kategorileri:")
        for category, count in categories.items():
            logger.info(f"  {category}: {count} özellik")
        
        # Bazı önemli features'ları göster
        important_features = [
            'log_return', 'rsi_norm', 'atr_pct', 'realized_vol_30',
            'hour_sin', 'session_london', 'tick_momentum_5', 'bb_20_position'
        ]
        
        available_features = [f for f in important_features if f in featured_data.columns]
        if available_features:
            logger.info(f"📋 Örnek özellikler:\n{featured_data[available_features].head()}")
    else:
        logger.error("❌ Gelişmiş feature engineering testi başarısız!")
    
    logger.info("✅ AdvancedFeatureEngineer modülü test tamamlandı!")


if __name__ == "__main__":
    main()