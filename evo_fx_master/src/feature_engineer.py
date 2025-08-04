#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
EVO_FX_MASTER - Özellik Mühendisliği Modülü
==========================================

OHLCV verilerinden teknik indikatörler ve mikro-yapısal özellikler çıkarır.
Forex ve altın için optimize edilmiş feature set.
"""

import pandas as pd
import numpy as np
import logging
from typing import Dict, List, Optional, Tuple
from pathlib import Path
import warnings
from datetime import datetime

# Teknik analiz kütüphanesi (ta-lib alternative)
try:
    import talib
    TALIB_AVAILABLE = True
except ImportError:
    TALIB_AVAILABLE = False
    print("⚠️  TA-Lib bulunamadı. Manuel hesaplamalar kullanılacak.")

warnings.filterwarnings('ignore')
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class FeatureEngineer:
    """Ana özellik mühendisliği sınıfı."""
    
    def __init__(self, config: Dict = None):
        """
        Args:
            config: Özellik parametreleri içeren sözlük
        """
        self.config = config or {}
        
        # Varsayılan parametreler
        self.rsi_period = self.config.get('rsi_period', 14)
        self.atr_period = self.config.get('atr_period', 14) 
        self.volatility_period = self.config.get('volatility_period', 30)
        self.correlation_period = self.config.get('correlation_period', 50)
        
        # Veri dizinleri
        self.raw_dir = Path("data/raw")
        self.processed_dir = Path("data/processed")
        self.processed_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"🔧 FeatureEngineer hazırlandı. TA-Lib: {TALIB_AVAILABLE}")
    
    def calculate_returns(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Log-return hesaplar.
        
        Args:
            df: OHLCV DataFrame'i
            
        Returns:
            Return sütunları eklenmiş DataFrame
        """
        result = df.copy()
        
        # Log returns (daha stabil)
        result['log_return'] = np.log(result['close'] / result['close'].shift(1))
        result['log_return_1h'] = np.log(result['close'] / result['close'].shift(60))  # 1 saatlik
        result['log_return_4h'] = np.log(result['close'] / result['close'].shift(240))  # 4 saatlik
        
        # Basit returns (yüzde)
        result['pct_return'] = result['close'].pct_change()
        
        # Volume-weighted return (eğer volume varsa)
        if 'volume' in result.columns:
            result['volume_return'] = result['log_return'] * np.log(result['volume'] + 1)
        
        # High-Low spread 
        result['hl_spread'] = (result['high'] - result['low']) / result['close']
        
        # Open-Close spread
        result['oc_spread'] = (result['close'] - result['open']) / result['open']
        
        return result
    
    def calculate_volatility_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Volatilite özelliklerini hesaplar.
        
        Args:
            df: Return sütunları içeren DataFrame
            
        Returns:
            Volatilite sütunları eklenmiş DataFrame
        """
        result = df.copy()
        
        # Rolling volatility (z-score normalized)
        volatility_raw = result['log_return'].rolling(self.volatility_period).std()
        volatility_mean = volatility_raw.rolling(100).mean()
        volatility_std = volatility_raw.rolling(100).std()
        
        result['volatility_zscore'] = (volatility_raw - volatility_mean) / (volatility_std + 1e-8)
        
        # Parkinson volatility (high-low based)
        if TALIB_AVAILABLE:
            # TA-Lib ile daha doğru hesaplama
            pass  # Şimdilik manuel hesaplama kullan
        
        # Manuel Parkinson volatility
        hl_ratio = np.log(result['high'] / result['low'])
        result['parkinson_vol'] = np.sqrt(hl_ratio.rolling(self.volatility_period).mean() / (4 * np.log(2)))
        
        # Garman-Klass volatility 
        gk_vol = (
            0.5 * np.log(result['high'] / result['low'])**2 -
            (2 * np.log(2) - 1) * np.log(result['close'] / result['open'])**2
        )
        result['gk_volatility'] = np.sqrt(gk_vol.rolling(self.volatility_period).mean())
        
        # Volatility regime (düşük/orta/yüksek)
        vol_q33 = result['volatility_zscore'].rolling(500).quantile(0.33)
        vol_q67 = result['volatility_zscore'].rolling(500).quantile(0.67)
        
        result['vol_regime'] = 1  # orta (varsayılan)
        result.loc[result['volatility_zscore'] < vol_q33, 'vol_regime'] = 0  # düşük
        result.loc[result['volatility_zscore'] > vol_q67, 'vol_regime'] = 2  # yüksek
        
        return result
    
    def calculate_technical_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Teknik indikatörleri hesaplar.
        
        Args:
            df: OHLCV DataFrame'i
            
        Returns:
            Teknik indikatör sütunları eklenmiş DataFrame
        """
        result = df.copy()
        
        # RSI (Relative Strength Index)
        if TALIB_AVAILABLE:
            result['rsi'] = talib.RSI(result['close'].values, timeperiod=self.rsi_period)
        else:
            # Manuel RSI hesaplama
            delta = result['close'].diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=self.rsi_period).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=self.rsi_period).mean()
            rs = gain / (loss + 1e-8)
            result['rsi'] = 100 - (100 / (1 + rs))
        
        # RSI normalizasyonu (0-1 arası)
        result['rsi_norm'] = result['rsi'] / 100.0
        
        # ATR (Average True Range)
        high_low = result['high'] - result['low']
        high_close = np.abs(result['high'] - result['close'].shift())
        low_close = np.abs(result['low'] - result['close'].shift())
        
        true_range = np.maximum(high_low, np.maximum(high_close, low_close))
        result['atr'] = true_range.rolling(self.atr_period).mean()
        
        # ATR yüzdesi (normalizasyon)
        result['atr_pct'] = result['atr'] / result['close']
        
        # Bollinger Bands
        bb_period = 20
        bb_std = 2
        bb_middle = result['close'].rolling(bb_period).mean()
        bb_std_dev = result['close'].rolling(bb_period).std()
        
        result['bb_upper'] = bb_middle + (bb_std_dev * bb_std)
        result['bb_lower'] = bb_middle - (bb_std_dev * bb_std)
        result['bb_position'] = (result['close'] - result['bb_lower']) / (result['bb_upper'] - result['bb_lower'])
        
        # Moving averages
        result['sma_20'] = result['close'].rolling(20).mean()
        result['sma_50'] = result['close'].rolling(50).mean()
        result['ema_12'] = result['close'].ewm(span=12).mean()
        result['ema_26'] = result['close'].ewm(span=26).mean()
        
        # MA crossover signals
        result['sma_cross'] = np.where(result['sma_20'] > result['sma_50'], 1, 0)
        result['price_vs_sma20'] = (result['close'] - result['sma_20']) / result['sma_20']
        
        # MACD
        macd_line = result['ema_12'] - result['ema_26']
        macd_signal = macd_line.ewm(span=9).mean()
        result['macd'] = macd_line
        result['macd_signal'] = macd_signal
        result['macd_histogram'] = macd_line - macd_signal
        
        # Momentum indicators
        result['momentum_5'] = result['close'] / result['close'].shift(5) - 1
        result['momentum_10'] = result['close'] / result['close'].shift(10) - 1
        result['momentum_20'] = result['close'] / result['close'].shift(20) - 1
        
        return result
    
    def calculate_time_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Zaman bazlı özellikler hesaplar.
        
        Args:
            df: Timestamp index'li DataFrame
            
        Returns:
            Zaman özellikli DataFrame
        """
        result = df.copy()
        
        # Saat bazlı özellikler (UTC)
        result['hour'] = result.index.hour
        result['minute'] = result.index.minute
        result['day_of_week'] = result.index.dayofweek
        result['day_of_month'] = result.index.day
        result['month'] = result.index.month
        
        # Session features (Forex seansları)
        # Sydney: 22:00-06:00 UTC, Tokyo: 00:00-08:00 UTC
        # London: 08:00-16:00 UTC, New York: 13:00-21:00 UTC
        
        result['session_sydney'] = ((result['hour'] >= 22) | (result['hour'] < 6)).astype(int)
        result['session_tokyo'] = ((result['hour'] >= 0) & (result['hour'] < 8)).astype(int)
        result['session_london'] = ((result['hour'] >= 8) & (result['hour'] < 16)).astype(int)
        result['session_newyork'] = ((result['hour'] >= 13) & (result['hour'] < 21)).astype(int)
        
        # Overlapping sessions (yüksek likidite)
        result['session_overlap_london_ny'] = (
            (result['hour'] >= 13) & (result['hour'] < 16)
        ).astype(int)
        result['session_overlap_sydney_tokyo'] = (
            (result['hour'] >= 0) & (result['hour'] < 6)
        ).astype(int)
        
        # Trigonometric encoding (döngüsel özellikler)
        result['hour_sin'] = np.sin(2 * np.pi * result['hour'] / 24)
        result['hour_cos'] = np.cos(2 * np.pi * result['hour'] / 24)
        result['dow_sin'] = np.sin(2 * np.pi * result['day_of_week'] / 7)
        result['dow_cos'] = np.cos(2 * np.pi * result['day_of_week'] / 7)
        result['month_sin'] = np.sin(2 * np.pi * result['month'] / 12)
        result['month_cos'] = np.cos(2 * np.pi * result['month'] / 12)
        
        # Weekend flag
        result['is_weekend'] = (result['day_of_week'] >= 5).astype(int)
        
        # Major economic calendar (simplified)
        # NFP: Her ayın ilk Cuma günü (yaklaşık)
        first_friday = (result['day_of_week'] == 4) & (result.index.day <= 7)
        result['potential_nfp'] = first_friday.astype(int)
        
        return result
    
    def calculate_cross_asset_features(self, eur_data: pd.DataFrame, 
                                     gold_data: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        EURUSD ve XAUUSD arasındaki korelasyon özelliklerini hesaplar.
        
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
        
        # Rolling korelasyon hesapla
        correlation = eur_common['log_return'].rolling(self.correlation_period).corr(
            gold_common['log_return']
        )
        
        # Cross-asset features
        # EUR için
        eur_common['eur_gold_correlation'] = correlation
        eur_common['gold_momentum'] = gold_common['log_return']
        eur_common['gold_volatility'] = gold_common['log_return'].rolling(20).std()
        
        # Spread features
        eur_common['eur_gold_spread'] = eur_common['log_return'] - gold_common['log_return']
        eur_common['eur_gold_ratio'] = eur_common['close'] / gold_common['close']
        eur_common['eur_gold_ratio_ma'] = eur_common['eur_gold_ratio'].rolling(50).mean()
        eur_common['eur_gold_ratio_zscore'] = (
            (eur_common['eur_gold_ratio'] - eur_common['eur_gold_ratio_ma']) / 
            eur_common['eur_gold_ratio'].rolling(50).std()
        )
        
        # GOLD için
        gold_common['eur_gold_correlation'] = correlation
        gold_common['eur_momentum'] = eur_common['log_return']
        gold_common['eur_volatility'] = eur_common['log_return'].rolling(20).std()
        gold_common['eur_gold_spread'] = gold_common['log_return'] - eur_common['log_return']
        gold_common['gold_eur_ratio'] = gold_common['close'] / eur_common['close']
        gold_common['gold_eur_ratio_ma'] = gold_common['gold_eur_ratio'].rolling(50).mean()
        gold_common['gold_eur_ratio_zscore'] = (
            (gold_common['gold_eur_ratio'] - gold_common['gold_eur_ratio_ma']) / 
            gold_common['gold_eur_ratio'].rolling(50).std()
        )
        
        # Orijinal DataFrame'lere geri map et
        eur_result = eur_data.copy()
        gold_result = gold_data.copy()
        
        # Cross-asset features'ları ekle
        cross_features = [
            'eur_gold_correlation', 'gold_momentum', 'gold_volatility', 
            'eur_gold_spread', 'eur_gold_ratio', 'eur_gold_ratio_ma', 'eur_gold_ratio_zscore'
        ]
        
        for feature in cross_features:
            if feature in eur_common.columns:
                eur_result[feature] = eur_common[feature].reindex(eur_result.index)
        
        gold_cross_features = [
            'eur_gold_correlation', 'eur_momentum', 'eur_volatility',
            'eur_gold_spread', 'gold_eur_ratio', 'gold_eur_ratio_ma', 'gold_eur_ratio_zscore'
        ]
        
        for feature in gold_cross_features:
            if feature in gold_common.columns:
                gold_result[feature] = gold_common[feature].reindex(gold_result.index)
        
        logger.info("✓ Cross-asset özellikler hesaplandı")
        return eur_result, gold_result
    
    def calculate_microstructure_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Mikro-yapısal piyasa özelliklerini hesaplar.
        
        Args:
            df: OHLCV DataFrame'i
            
        Returns:
            Mikro-yapısal özellikli DataFrame
        """
        result = df.copy()
        
        # Tick direction (yön momentum)
        result['tick_direction'] = np.sign(result['close'].diff())
        result['tick_momentum_5'] = result['tick_direction'].rolling(5).sum()
        result['tick_momentum_10'] = result['tick_direction'].rolling(10).sum()
        
        # Volume profile (eğer volume varsa)
        if 'volume' in result.columns and result['volume'].sum() > 0:
            result['volume_ma'] = result['volume'].rolling(20).mean()
            result['volume_ratio'] = result['volume'] / (result['volume_ma'] + 1e-8)
            result['volume_zscore'] = (
                (result['volume'] - result['volume_ma']) / 
                (result['volume'].rolling(20).std() + 1e-8)
            )
            
            # Volume-price trend
            result['vpt'] = (result['volume'] * result['pct_return']).cumsum()
            result['vpt_ma'] = result['vpt'].rolling(20).mean()
            result['vpt_signal'] = np.where(result['vpt'] > result['vpt_ma'], 1, -1)
        else:
            # Volume yoksa dummy değerler
            result['volume_ratio'] = 1.0
            result['volume_zscore'] = 0.0
            result['vpt_signal'] = 0
        
        # Bid-ask spread proxy (High-Low range)
        result['spread_proxy'] = (result['high'] - result['low']) / result['close']
        result['spread_ma'] = result['spread_proxy'].rolling(20).mean()
        result['spread_ratio'] = result['spread_proxy'] / (result['spread_ma'] + 1e-8)
        
        # Candle patterns (basit)
        result['body_size'] = np.abs(result['close'] - result['open']) / result['close']
        result['upper_shadow'] = (result['high'] - np.maximum(result['open'], result['close'])) / result['close']
        result['lower_shadow'] = (np.minimum(result['open'], result['close']) - result['low']) / result['close']
        
        # Doji pattern (small body)
        result['is_doji'] = (result['body_size'] < 0.001).astype(int)
        
        # Hammer/shooting star patterns
        result['is_hammer'] = (
            (result['lower_shadow'] > 2 * result['body_size']) & 
            (result['upper_shadow'] < result['body_size'])
        ).astype(int)
        
        result['is_shooting_star'] = (
            (result['upper_shadow'] > 2 * result['body_size']) & 
            (result['lower_shadow'] < result['body_size'])
        ).astype(int)
        
        # Gap analysis
        result['gap'] = result['open'] - result['close'].shift(1)
        result['gap_pct'] = result['gap'] / result['close'].shift(1)
        result['gap_filled'] = (
            (result['gap'] > 0) & (result['low'] <= result['close'].shift(1))
        ) | (
            (result['gap'] < 0) & (result['high'] >= result['close'].shift(1))
        )
        result['gap_filled'] = result['gap_filled'].astype(int)
        
        return result
    
    def bake_features(self, symbol: str, df: pd.DataFrame = None) -> Optional[pd.DataFrame]:
        """
        Tüm özellik mühendisliği işlemlerini uygular.
        
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
        
        logger.info(f"🔧 Feature engineering başlatılıyor: {symbol}")
        
        # 1. Returns
        df = self.calculate_returns(df)
        logger.info("✓ Returns hesaplandı")
        
        # 2. Volatility features
        df = self.calculate_volatility_features(df)
        logger.info("✓ Volatility features hesaplandı")
        
        # 3. Technical indicators
        df = self.calculate_technical_indicators(df)
        logger.info("✓ Teknik indikatörler hesaplandı")
        
        # 4. Time features
        df = self.calculate_time_features(df)
        logger.info("✓ Zaman özellikleri hesaplandı")
        
        # 5. Microstructure features
        df = self.calculate_microstructure_features(df)
        logger.info("✓ Mikro-yapısal özellikler hesaplandı")
        
        # NaN temizleme
        initial_len = len(df)
        df = df.dropna()
        final_len = len(df)
        
        if final_len < initial_len:
            logger.info(f"🧹 NaN temizleme: {initial_len} -> {final_len} satır")
        
        # Sonsuz değer kontrolü
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        df[numeric_cols] = df[numeric_cols].replace([np.inf, -np.inf], np.nan)
        df = df.dropna()
        
        if len(df) == 0:
            logger.error(f"Temizleme sonrası veri kalmadı: {symbol}")
            return None
        
        logger.info(f"✅ Feature engineering tamamlandı: {symbol} ({len(df)} satır, {len(df.columns)} özellik)")
        return df
    
    def save_processed_data(self, symbol: str, df: pd.DataFrame) -> bool:
        """
        İşlenmiş veriyi kaydet.
        
        Args:
            symbol: Sembol adı
            df: İşlenmiş DataFrame
            
        Returns:
            Başarı durumu
        """
        if df is None or df.empty:
            return False
        
        try:
            output_path = self.processed_dir / f"{symbol}_features.parquet"
            df.to_parquet(output_path, compression='snappy')
            
            logger.info(f"💾 İşlenmiş veri kaydedildi: {output_path}")
            
            # Özet bilgi kaydet
            info_path = self.processed_dir / f"{symbol}_feature_info.txt"
            with open(info_path, 'w', encoding='utf-8') as f:
                f.write(f"Sembol: {symbol}\n")
                f.write(f"İşlem Tarihi: {datetime.now()}\n")
                f.write(f"Satır Sayısı: {len(df)}\n")
                f.write(f"Özellik Sayısı: {len(df.columns)}\n")
                f.write(f"Tarih Aralığı: {df.index.min()} - {df.index.max()}\n\n")
                f.write("Özellik Listesi:\n")
                for i, col in enumerate(df.columns, 1):
                    f.write(f"{i:2d}. {col}\n")
            
            return True
            
        except Exception as e:
            logger.error(f"Kaydetme hatası: {e}")
            return False
    
    def process_all_symbols(self, eur_data: pd.DataFrame = None, 
                           gold_data: pd.DataFrame = None) -> Dict[str, pd.DataFrame]:
        """
        Tüm semboller için feature engineering yapar.
        
        Args:
            eur_data: Opsiyonel EURUSD DataFrame
            gold_data: Opsiyonel XAUUSD DataFrame
            
        Returns:
            İşlenmiş DataFrame'ler sözlüğü
        """
        results = {}
        
        # EURUSD işle
        logger.info("🎯 EURUSD feature engineering...")
        eur_features = self.bake_features('EURUSD', eur_data)
        
        # XAUUSD işle
        logger.info("🎯 XAUUSD feature engineering...")
        gold_features = self.bake_features('XAUUSD', gold_data)
        
        # Cross-asset features (eğer her ikisi de varsa)
        if eur_features is not None and gold_features is not None:
            logger.info("🔗 Cross-asset features hesaplanıyor...")
            eur_features, gold_features = self.calculate_cross_asset_features(
                eur_features, gold_features
            )
        
        # Kaydet
        if eur_features is not None:
            if self.save_processed_data('EURUSD', eur_features):
                results['EURUSD'] = eur_features
        
        if gold_features is not None:
            if self.save_processed_data('XAUUSD', gold_features):
                results['XAUUSD'] = gold_features
        
        logger.info(f"🏁 Feature engineering tamamlandı. Başarılı: {len(results)}/2")
        return results


def main():
    """Test fonksiyonu."""
    logger.info("🔧 FeatureEngineer modülü test ediliyor...")
    
    # Test config
    config = {
        'rsi_period': 14,
        'atr_period': 14,
        'volatility_period': 30,
        'correlation_period': 50
    }
    
    # FeatureEngineer oluştur
    engineer = FeatureEngineer(config)
    
    # Dummy test verisi oluştur
    logger.info("🧪 Test verisi oluşturuluyor...")
    
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
    
    # Feature engineering test
    logger.info("🔧 Feature engineering testi...")
    featured_data = engineer.bake_features('TEST', test_data)
    
    if featured_data is not None:
        logger.info(f"✅ Test başarılı!")
        logger.info(f"📊 Özellik sayısı: {len(featured_data.columns)}")
        logger.info(f"📈 Veri boyutu: {featured_data.shape}")
        
        # Bazı önemli features'ları göster
        important_features = [
            'log_return', 'rsi_norm', 'atr_pct', 'volatility_zscore',
            'hour_sin', 'session_london', 'tick_momentum_5'
        ]
        
        available_features = [f for f in important_features if f in featured_data.columns]
        if available_features:
            logger.info(f"📋 Örnek özellikler:\n{featured_data[available_features].head()}")
    else:
        logger.error("❌ Feature engineering testi başarısız!")
    
    logger.info("✅ FeatureEngineer modülü test tamamlandı!")


if __name__ == "__main__":
    main()