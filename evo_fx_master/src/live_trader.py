#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
EVO_FX_MASTER - Canlı Trading Modülü
===================================

MT5 Python API ile gerçek zamanlı sinyal üretimi ve emir yönetimi.
En iyi genomu kullanarak EURUSD ve XAUUSD'de otomatik trading yapar.
"""

import numpy as np
import pandas as pd
import logging
import json
import time
from typing import Dict, List, Optional, Tuple, Any
from pathlib import Path
from datetime import datetime, timedelta
import threading
import queue
from dataclasses import dataclass
from enum import Enum

# MT5 bağlantısı
try:
    import MetaTrader5 as mt5
    MT5_AVAILABLE = True
except ImportError:
    MT5_AVAILABLE = False
    print("⚠️  MT5 kütüphanesi bulunamadı. Live trading devre dışı.")

# Kendi modüllerimizi import et
try:
    from .genome import Genome
    from .feature_engineer import FeatureEngineer
    from .risk import RiskManager
except ImportError:
    # Standalone çalıştırma için
    import sys
    sys.path.append('.')
    from genome import Genome
    from feature_engineer import FeatureEngineer
    from risk import RiskManager

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class OrderType(Enum):
    """Emir türleri."""
    BUY = "buy"
    SELL = "sell"
    CLOSE = "close"


@dataclass
class LiveSignal:
    """Canlı sinyal verisi."""
    timestamp: datetime
    symbol: str
    signal: int  # -1: sell, 0: hold, 1: buy
    signal_strength: float
    features: Dict[str, float]
    price: float
    atr: float


@dataclass
class LiveTrade:
    """Canlı trade verisi."""
    ticket: int
    symbol: str
    order_type: OrderType
    lot_size: float
    open_price: float
    stop_loss: float
    take_profit: float
    open_time: datetime
    close_time: Optional[datetime] = None
    close_price: Optional[float] = None
    profit: Optional[float] = None
    comment: str = "EVO_FX_NEAT"


class LiveTrader:
    """Ana canlı trading sınıfı."""
    
    def __init__(self, config: Dict = None):
        """
        Args:
            config: Live trading konfigürasyonu
        """
        self.config = config or {}
        
        # MT5 bağlantı ayarları
        self.mt5_login = self.config.get('mt5_login')
        self.mt5_password = self.config.get('mt5_password') 
        self.mt5_server = self.config.get('mt5_server')
        self.mt5_path = self.config.get('mt5_path', '')
        
        # Trading ayarları
        self.symbols = self.config.get('symbols', ['EURUSD', 'XAUUSD'])
        self.timeframe = self.config.get('timeframe', mt5.TIMEFRAME_M1 if MT5_AVAILABLE else 1)
        self.max_positions = self.config.get('max_positions', 4)
        
        # Sinyal ayarları
        self.signal_threshold = self.config.get('signal_threshold', 0.1)
        self.signal_cooldown_minutes = self.config.get('signal_cooldown_minutes', 5)
        
        # Modüller
        self.feature_engineer = FeatureEngineer()
        self.risk_manager = RiskManager(config.get('risk_config', {}))
        
        # Trading durumu
        self.is_running = False
        self.genome: Optional[Genome] = None
        self.active_trades: Dict[int, LiveTrade] = {}
        self.signal_history: List[LiveSignal] = []
        self.last_signal_time: Dict[str, datetime] = {}
        
        # Threading
        self.signal_queue = queue.Queue()
        self.trade_queue = queue.Queue()
        
        # Log dosyası
        self.log_dir = Path("logs")
        self.log_dir.mkdir(exist_ok=True)
        self.trade_journal_file = self.log_dir / f"trade_journal_{datetime.now().strftime('%Y%m%d')}.csv"
        
        logger.info(f"🤖 LiveTrader hazırlandı. MT5: {MT5_AVAILABLE}")
    
    def initialize_mt5(self) -> bool:
        """
        MT5 bağlantısını başlatır.
        
        Returns:
            Başarı durumu
        """
        if not MT5_AVAILABLE:
            logger.error("MT5 kütüphanesi yok!")
            return False
        
        try:
            # MT5 başlat
            if self.mt5_path:
                if not mt5.initialize(path=self.mt5_path):
                    logger.error(f"MT5 başlatılamadı: {mt5.last_error()}")
                    return False
            else:
                if not mt5.initialize():
                    logger.error(f"MT5 başlatılamadı: {mt5.last_error()}")
                    return False
            
            # Login bilgileri varsa bağlan
            if self.mt5_login and self.mt5_password and self.mt5_server:
                if not mt5.login(self.mt5_login, self.mt5_password, self.mt5_server):
                    logger.error(f"MT5 login başarısız: {mt5.last_error()}")
                    return False
                    
                logger.info(f"✅ MT5 bağlantısı başarılı: {self.mt5_server}")
            else:
                logger.info("✅ MT5 başlatıldı (demo hesap)")
            
            # Hesap bilgileri
            account_info = mt5.account_info()
            if account_info:
                logger.info(f"💰 Hesap bakiyesi: ${account_info.balance:.2f}")
                logger.info(f"📊 Equity: ${account_info.equity:.2f}")
                logger.info(f"🏦 Broker: {account_info.company}")
            
            # Symbol bilgileri kontrol et
            for symbol in self.symbols:
                symbol_info = mt5.symbol_info(symbol)
                if symbol_info is None:
                    logger.warning(f"⚠️  Sembol bulunamadı: {symbol}")
                else:
                    if not symbol_info.visible:
                        if not mt5.symbol_select(symbol, True):
                            logger.warning(f"⚠️  Sembol aktifleştirilemedi: {symbol}")
                        else:
                            logger.info(f"✅ Sembol aktif: {symbol}")
                    else:
                        logger.info(f"✅ Sembol hazır: {symbol}")
            
            return True
            
        except Exception as e:
            logger.error(f"MT5 başlatma hatası: {e}")
            return False
    
    def load_best_genome(self, genome_path: str) -> bool:
        """
        En iyi genomu yükler.
        
        Args:
            genome_path: Genom dosya yolu
            
        Returns:
            Başarı durumu
        """
        try:
            if genome_path.endswith('.json'):
                # JSON'dan yükle (implement gerekli)
                logger.warning("JSON genom yükleme henüz desteklenmiyor")
                return False
            else:
                # Pickle'dan yükle
                import pickle
                with open(genome_path, 'rb') as f:
                    self.genome = pickle.load(f)
                
                logger.info(f"🧬 Genom yüklendi: {genome_path}")
                logger.info(f"📊 Genom bilgisi: {self.genome}")
                return True
                
        except Exception as e:
            logger.error(f"Genom yükleme hatası: {e}")
            return False
    
    def get_current_data(self, symbol: str, bars: int = 100) -> Optional[pd.DataFrame]:
        """
        Güncel market verisi çeker.
        
        Args:
            symbol: Sembol adı
            bars: Çekilecek bar sayısı
            
        Returns:
            OHLCV DataFrame'i
        """
        if not MT5_AVAILABLE:
            logger.warning("MT5 yok, test verisi oluşturuluyor")
            # Daha gerçekçi test verisi oluştur
            dates = pd.date_range(end=datetime.now(), periods=bars, freq='1min')
            base_price = 1.1000 if symbol == 'EURUSD' else 2000.0
            
            # Daha stabil price action
            price_changes = np.random.normal(0, 0.00001, bars)  # Çok küçük noise
            prices = [base_price]
            
            for change in price_changes[1:]:
                new_price = prices[-1] * (1 + change)
                prices.append(new_price)
            
            df = pd.DataFrame(index=dates)
            df['close'] = prices
            df['open'] = np.roll(prices, 1)
            df['open'].iloc[0] = base_price
            
            # High/Low mantıklı aralıklarda
            spread = 0.0001 if symbol == 'EURUSD' else 0.1
            df['high'] = df[['open', 'close']].max(axis=1) + np.random.uniform(0, spread, bars)
            df['low'] = df[['open', 'close']].min(axis=1) - np.random.uniform(0, spread, bars)
            
            df['volume'] = np.random.randint(500, 2000, bars)
            
            # Index'i UTC yap
            df.index = df.index.tz_localize('UTC')
            
            # Sonsuz değer kontrolü
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            df[numeric_cols] = df[numeric_cols].replace([np.inf, -np.inf], np.nan)
            df = df.fillna(method='ffill').fillna(method='bfill')
            
            return df
        
        try:
            # MT5'ten veri çek
            rates = mt5.copy_rates_from_pos(symbol, self.timeframe, 0, bars)
            
            if rates is None or len(rates) == 0:
                logger.warning(f"Veri alınamadı: {symbol}")
                return None
            
            # DataFrame'e dönüştür
            df = pd.DataFrame(rates)
            df['timestamp'] = pd.to_datetime(df['time'], unit='s')
            df.set_index('timestamp', inplace=True)
            
            # Gereksiz sütunları kaldır
            df = df[['open', 'high', 'low', 'close', 'tick_volume']].copy()
            df.rename(columns={'tick_volume': 'volume'}, inplace=True)
            
            return df
            
        except Exception as e:
            logger.error(f"Veri çekme hatası ({symbol}): {e}")
            return None
    
    def generate_signal(self, symbol: str) -> Optional[LiveSignal]:
        """
        Sembol için trading sinyali üretir.
        
        Args:
            symbol: Sembol adı
            
        Returns:
            Sinyal verisi
        """
        if not self.genome:
            return None
        
        # Güncel veri çek
        data = self.get_current_data(symbol, bars=200)
        if data is None or len(data) < 50:
            return None
        
        try:
            # Feature engineering
            featured_data = self.feature_engineer.bake_features(symbol, data)
            if featured_data is None or len(featured_data) == 0:
                return None
            
            # Son satırı al (en güncel)
            latest_data = featured_data.iloc[-1]
            
            # Feature sütunlarını belirle
            feature_cols = [col for col in featured_data.columns 
                           if col not in ['open', 'high', 'low', 'close', 'volume']]
            
            if len(feature_cols) < self.genome.input_size:
                logger.warning(f"Yetersiz feature: {len(feature_cols)} < {self.genome.input_size}")
                return None
            
            # İlk N feature'ı kullan
            selected_features = feature_cols[:self.genome.input_size]
            inputs = latest_data[selected_features].values
            
            # NaN kontrolü
            if np.any(np.isnan(inputs)):
                logger.debug(f"NaN feature bulundu: {symbol}")
                return None
            
            # Normalizasyon
            inputs = np.tanh(inputs / 2.0)  # [-1, 1] aralığına sıkıştır
            
            # Genom prediction
            outputs = self.genome.predict(inputs)
            
            # Sinyali çıkar
            if len(outputs) >= 3:
                sell_prob = outputs[0]
                hold_prob = outputs[1]
                buy_prob = outputs[2]
                
                # En yüksek probabiliteyi seç
                max_idx = np.argmax([sell_prob, hold_prob, buy_prob])
                max_prob = max([sell_prob, hold_prob, buy_prob])
                
                if max_idx == 0 and sell_prob > self.signal_threshold:
                    signal = -1  # Sell
                    signal_strength = sell_prob
                elif max_idx == 2 and buy_prob > self.signal_threshold:
                    signal = 1   # Buy
                    signal_strength = buy_prob
                else:
                    signal = 0   # Hold
                    signal_strength = hold_prob
            else:
                # Fallback: tek output
                output = outputs[0] if len(outputs) > 0 else 0.5
                
                if output < (0.5 - self.signal_threshold):
                    signal = -1
                    signal_strength = abs(output - 0.5)
                elif output > (0.5 + self.signal_threshold):
                    signal = 1
                    signal_strength = abs(output - 0.5)
                else:
                    signal = 0
                    signal_strength = abs(output - 0.5)
            
            # Sinyal oluştur
            live_signal = LiveSignal(
                timestamp=datetime.now(),
                symbol=symbol,
                signal=signal,
                signal_strength=signal_strength,
                features={col: float(latest_data[col]) for col in selected_features},
                price=float(latest_data['close']),
                atr=float(latest_data.get('atr', 0.001))
            )
            
            return live_signal
            
        except Exception as e:
            logger.error(f"Sinyal üretme hatası ({symbol}): {e}")
            return None
    
    def should_trade_signal(self, signal: LiveSignal) -> bool:
        """
        Sinyalin trade edilip edilmeyeceğini kontrol eder.
        
        Args:
            signal: Live sinyal
            
        Returns:
            Trade edilebilir mi?
        """
        # Sinyal yok
        if signal.signal == 0:
            return False
        
        # Cooldown kontrolü
        last_signal_time = self.last_signal_time.get(signal.symbol)
        if last_signal_time:
            time_diff = (signal.timestamp - last_signal_time).total_seconds() / 60
            if time_diff < self.signal_cooldown_minutes:
                logger.debug(f"Cooldown aktif: {signal.symbol} ({time_diff:.1f} dk)")
                return False
        
        # Risk limitleri
        account_info = None
        if MT5_AVAILABLE:
            account_info = mt5.account_info()
        
        current_balance = account_info.balance if account_info else 10000.0
        
        daily_limits = self.risk_manager.check_daily_limits(current_balance)
        if not daily_limits['can_trade']:
            logger.info(f"Günlük limit: {daily_limits['reason']}")
            return False
        
        # Maksimum pozisyon kontrolü
        if len(self.active_trades) >= self.max_positions:
            logger.debug(f"Max pozisyon limiti: {len(self.active_trades)}/{self.max_positions}")
            return False
        
        # Aynı sembollde zaten pozisyon var mı?
        for trade in self.active_trades.values():
            if trade.symbol == signal.symbol:
                logger.debug(f"Zaten pozisyon var: {signal.symbol}")
                return False
        
        return True
    
    def execute_trade(self, signal: LiveSignal) -> Optional[LiveTrade]:
        """
        Sinyal doğrultusunda trade açar.
        
        Args:
            signal: Live sinyal
            
        Returns:
            Açılan trade bilgisi
        """
        if not MT5_AVAILABLE:
            logger.info(f"🧪 DEMO: {signal.symbol} {signal.signal} sinyali (güçlü: {signal.signal_strength:.2f})")
            
            # Demo trade oluştur
            demo_trade = LiveTrade(
                ticket=int(time.time()),  # Fake ticket
                symbol=signal.symbol,
                order_type=OrderType.BUY if signal.signal == 1 else OrderType.SELL,
                lot_size=0.1,
                open_price=signal.price,
                stop_loss=signal.price - (signal.atr * 1.1) if signal.signal == 1 else signal.price + (signal.atr * 1.1),
                take_profit=signal.price + (signal.atr * 2.2) if signal.signal == 1 else signal.price - (signal.atr * 2.2),
                open_time=signal.timestamp,
                comment="EVO_FX_DEMO"
            )
            
            self.active_trades[demo_trade.ticket] = demo_trade
            return demo_trade
        
        try:
            # Hesap bilgileri
            account_info = mt5.account_info()
            if not account_info:
                logger.error("Hesap bilgisi alınamadı")
                return None
            
            # Pozisyon boyutu hesapla
            position_result = self.risk_manager.calculate_position_size(
                signal_strength=signal.signal_strength,
                balance=account_info.balance,
                atr=signal.atr,
                price=signal.price,
                symbol=signal.symbol
            )
            
            if position_result.lot_size <= 0:
                logger.warning(f"Pozisyon boyutu 0: {signal.symbol}")
                return None
            
            # Stop-loss ve Take-profit hesapla
            sl_tp_result = self.risk_manager.calculate_stop_loss_take_profit(
                entry_price=signal.price,
                direction=signal.signal,
                atr=signal.atr,
                symbol=signal.symbol
            )
            
            # MT5 emir parametreleri
            order_type = mt5.ORDER_TYPE_BUY if signal.signal == 1 else mt5.ORDER_TYPE_SELL
            
            request = {
                "action": mt5.TRADE_ACTION_DEAL,
                "symbol": signal.symbol,
                "volume": position_result.lot_size,
                "type": order_type,
                "price": signal.price,
                "sl": sl_tp_result.stop_price,
                "tp": sl_tp_result.take_profit_price,
                "deviation": 20,
                "magic": 12345,
                "comment": "EVO_FX_NEAT",
                "type_time": mt5.ORDER_TIME_GTC,
                "type_filling": mt5.ORDER_FILLING_IOC,
            }
            
            # Emri gönder
            result = mt5.order_send(request)
            
            if result.retcode != mt5.TRADE_RETCODE_DONE:
                logger.error(f"Emir hatası: {result.retcode} - {result.comment}")
                return None
            
            # Trade bilgisini oluştur
            live_trade = LiveTrade(
                ticket=result.order,
                symbol=signal.symbol,
                order_type=OrderType.BUY if signal.signal == 1 else OrderType.SELL,
                lot_size=position_result.lot_size,
                open_price=result.price,
                stop_loss=sl_tp_result.stop_price,
                take_profit=sl_tp_result.take_profit_price,
                open_time=signal.timestamp,
                comment="EVO_FX_NEAT"
            )
            
            self.active_trades[result.order] = live_trade
            
            logger.info(f"✅ Trade açıldı: {live_trade.symbol} {live_trade.order_type.value.upper()} {live_trade.lot_size} @ {live_trade.open_price}")
            
            # Trade journal'a kaydet
            self.log_trade_to_journal(live_trade, 'OPEN')
            
            return live_trade
            
        except Exception as e:
            logger.error(f"Trade açma hatası: {e}")
            return None
    
    def check_active_trades(self):
        """Aktif trade'leri kontrol eder ve gerekirse kapatır."""
        if not self.active_trades:
            return
        
        if not MT5_AVAILABLE:
            # Demo mode - rastgele kapatma
            for ticket, trade in list(self.active_trades.items()):
                if np.random.random() < 0.01:  # %1 şans ile kapat
                    trade.close_time = datetime.now()
                    trade.close_price = trade.open_price + np.random.uniform(-0.01, 0.01)
                    trade.profit = (trade.close_price - trade.open_price) * trade.lot_size * 1000
                    
                    logger.info(f"🔚 DEMO Trade kapatıldı: {trade.symbol} P&L: ${trade.profit:.2f}")
                    del self.active_trades[ticket]
            return
        
        try:
            # MT5'ten açık pozisyonları al
            positions = mt5.positions_get()
            if positions is None:
                return
            
            # MT5'teki ticket'ları al
            mt5_tickets = {pos.ticket for pos in positions}
            
            # Kapatılan trade'leri bul
            closed_tickets = []
            for ticket in self.active_trades.keys():
                if ticket not in mt5_tickets:
                    closed_tickets.append(ticket)
            
            # Kapatılan trade'leri işle
            for ticket in closed_tickets:
                trade = self.active_trades[ticket]
                
                # MT5'ten deal bilgisini al
                deals = mt5.history_deals_get(ticket=ticket)
                if deals and len(deals) >= 2:  # Açma ve kapama deal'i
                    close_deal = deals[-1]  # Son deal kapama olmalı
                    
                    trade.close_time = datetime.fromtimestamp(close_deal.time)
                    trade.close_price = close_deal.price
                    trade.profit = close_deal.profit
                    
                    logger.info(f"✅ Trade kapatıldı: {trade.symbol} P&L: ${trade.profit:.2f}")
                    
                    # Trade journal'a kaydet
                    self.log_trade_to_journal(trade, 'CLOSE')
                    
                    # Risk manager'ı güncelle
                    account_info = mt5.account_info()
                    if account_info:
                        self.risk_manager.update_daily_stats(trade.profit, account_info.balance)
                
                del self.active_trades[ticket]
                
        except Exception as e:
            logger.error(f"Trade kontrol hatası: {e}")
    
    def log_trade_to_journal(self, trade: LiveTrade, action: str):
        """
        Trade'i journal dosyasına kaydeder.
        
        Args:
            trade: Trade bilgisi
            action: OPEN veya CLOSE
        """
        try:
            # CSV header kontrolü
            write_header = not self.trade_journal_file.exists()
            
            with open(self.trade_journal_file, 'a', encoding='utf-8') as f:
                if write_header:
                    f.write("timestamp,action,ticket,symbol,type,lot_size,open_price,close_price,sl,tp,profit,comment\n")
                
                # Trade bilgilerini yaz
                f.write(f"{datetime.now().isoformat()},{action},{trade.ticket},{trade.symbol},"
                       f"{trade.order_type.value},{trade.lot_size},{trade.open_price},"
                       f"{trade.close_price or 0},{trade.stop_loss},{trade.take_profit},"
                       f"{trade.profit or 0},{trade.comment}\n")
                
        except Exception as e:
            logger.error(f"Journal kayıt hatası: {e}")
    
    def signal_thread(self):
        """Sinyal üretme thread'i."""
        logger.info("🔄 Sinyal thread'i başlatıldı")
        
        while self.is_running:
            try:
                for symbol in self.symbols:
                    if not self.is_running:
                        break
                    
                    # Sinyal üret
                    signal = self.generate_signal(symbol)
                    if signal:
                        self.signal_queue.put(signal)
                        self.signal_history.append(signal)
                        
                        # Geçmişi sınırla (son 1000 sinyal)
                        if len(self.signal_history) > 1000:
                            self.signal_history = self.signal_history[-1000:]
                
                # 1 dakika bekle
                time.sleep(60)
                
            except Exception as e:
                logger.error(f"Sinyal thread hatası: {e}")
                time.sleep(5)
        
        logger.info("🔄 Sinyal thread'i durduruldu")
    
    def trading_thread(self):
        """Trading thread'i."""
        logger.info("📈 Trading thread'i başlatıldı")
        
        while self.is_running:
            try:
                # Aktif trade'leri kontrol et
                self.check_active_trades()
                
                # Yeni sinyalleri işle
                while not self.signal_queue.empty():
                    signal = self.signal_queue.get()
                    
                    # Trade edilebilir mi?
                    if self.should_trade_signal(signal):
                        trade = self.execute_trade(signal)
                        if trade:
                            self.last_signal_time[signal.symbol] = signal.timestamp
                    
                    self.signal_queue.task_done()
                
                # 10 saniye bekle
                time.sleep(10)
                
            except Exception as e:
                logger.error(f"Trading thread hatası: {e}")
                time.sleep(5)
        
        logger.info("📈 Trading thread'i durduruldu")
    
    def start_live_trading(self, genome_path: str) -> bool:
        """
        Canlı trading'i başlatır.
        
        Args:
            genome_path: En iyi genom dosya yolu
            
        Returns:
            Başarı durumu
        """
        logger.info("🚀 Canlı trading başlatılıyor...")
        
        # MT5 bağlantısı
        if MT5_AVAILABLE:
            if not self.initialize_mt5():
                logger.error("MT5 bağlantısı başarısız!")
                return False
        else:
            logger.info("🧪 Demo mode aktif (MT5 yok)")
        
        # Genomu yükle
        if not self.load_best_genome(genome_path):
            logger.error("Genom yüklenemedi!")
            return False
        
        # Trading'i başlat
        self.is_running = True
        
        # Thread'leri başlat
        signal_thread = threading.Thread(target=self.signal_thread, daemon=True)
        trading_thread = threading.Thread(target=self.trading_thread, daemon=True)
        
        signal_thread.start()
        trading_thread.start()
        
        logger.info("✅ Canlı trading başlatıldı!")
        logger.info(f"📊 Semboller: {self.symbols}")
        logger.info(f"🧬 Genom: {self.genome}")
        logger.info(f"📝 Trade journal: {self.trade_journal_file}")
        
        try:
            # Ana döngü
            while self.is_running:
                # Status raporu (her 5 dakikada)
                time.sleep(300)
                
                if self.is_running:
                    logger.info(f"📊 Durum: {len(self.active_trades)} aktif trade, {len(self.signal_history)} sinyal geçmişi")
                    
                    # Hesap durumu
                    if MT5_AVAILABLE:
                        account_info = mt5.account_info()
                        if account_info:
                            logger.info(f"💰 Bakiye: ${account_info.balance:.2f}, Equity: ${account_info.equity:.2f}")
        
        except KeyboardInterrupt:
            logger.info("⌨️  Kullanıcı tarafından durduruldu")
        except Exception as e:
            logger.error(f"Ana döngü hatası: {e}")
        finally:
            self.stop_live_trading()
        
        return True
    
    def stop_live_trading(self):
        """Canlı trading'i durdurur."""
        logger.info("🛑 Canlı trading durduruluyor...")
        
        self.is_running = False
        
        # Thread'lerin bitmesini bekle
        time.sleep(2)
        
        # MT5 bağlantısını kapat
        if MT5_AVAILABLE:
            try:
                mt5.shutdown()
                logger.info("✅ MT5 bağlantısı kapatıldı")
            except:
                pass
        
        logger.info("🏁 Canlı trading durduruldu")
    
    def get_status(self) -> Dict[str, Any]:
        """
        Mevcut durumu döndürür.
        
        Returns:
            Durum bilgisi
        """
        status = {
            'is_running': self.is_running,
            'genome_loaded': self.genome is not None,
            'active_trades_count': len(self.active_trades),
            'signals_generated': len(self.signal_history),
            'last_signal_times': self.last_signal_time,
            'mt5_connected': MT5_AVAILABLE and mt5.terminal_info() is not None if MT5_AVAILABLE else False,
            'symbols': self.symbols,
            'active_trades': [
                {
                    'ticket': trade.ticket,
                    'symbol': trade.symbol,
                    'type': trade.order_type.value,
                    'lot_size': trade.lot_size,
                    'open_price': trade.open_price,
                    'current_profit': trade.profit
                }
                for trade in self.active_trades.values()
            ]
        }
        
        # MT5 hesap bilgisi ekle
        if MT5_AVAILABLE and self.is_running:
            try:
                account_info = mt5.account_info()
                if account_info:
                    status['account'] = {
                        'balance': account_info.balance,
                        'equity': account_info.equity,
                        'margin': account_info.margin,
                        'free_margin': account_info.margin_free,
                        'profit': account_info.profit
                    }
            except:
                pass
        
        return status


def main():
    """Test fonksiyonu."""
    logger.info("🤖 LiveTrader modülü test ediliyor...")
    
    # Test config
    test_config = {
        'symbols': ['EURUSD', 'XAUUSD'],
        'signal_threshold': 0.1,
        'max_positions': 2,
        'signal_cooldown_minutes': 1,  # Test için kısa
        'risk_config': {
            'base_risk_percent': 0.01,
            'max_trades_per_day': 5
        }
    }
    
    # LiveTrader oluştur
    trader = LiveTrader(test_config)
    
    # Status testi
    logger.info("📊 Status testi...")
    status = trader.get_status()
    logger.info(f"Durum: {json.dumps(status, indent=2, default=str)}")
    
    # Sinyal üretme testi
    logger.info("🎯 Sinyal üretme testi...")
    
    # Test genomu oluştur
    from genome import Genome
    test_genome = Genome(input_size=10, output_size=3)
    for _ in range(5):
        test_genome.mutate()
    
    trader.genome = test_genome
    
    # Her sembol için sinyal üret
    for symbol in trader.symbols:
        logger.info(f"🧪 {symbol} sinyali test ediliyor...")
        signal = trader.generate_signal(symbol)
        
        if signal:
            logger.info(f"✅ Sinyal üretildi:")
            logger.info(f"  Sembol: {signal.symbol}")
            logger.info(f"  Sinyal: {signal.signal} (güçlü: {signal.signal_strength:.3f})")
            logger.info(f"  Fiyat: {signal.price:.5f}")
            logger.info(f"  ATR: {signal.atr:.5f}")
            
            # Trade edilebilirlik testi
            can_trade = trader.should_trade_signal(signal)
            logger.info(f"  Trade edilebilir: {can_trade}")
            
            if can_trade and signal.signal != 0:
                logger.info("🧪 Demo trade testi...")
                demo_trade = trader.execute_trade(signal)
                if demo_trade:
                    logger.info(f"  ✅ Demo trade açıldı: {demo_trade.symbol} {demo_trade.order_type.value}")
        else:
            logger.warning(f"  ⚠️  Sinyal üretilemedi: {symbol}")
    
    # Aktif trade kontrolü
    if trader.active_trades:
        logger.info("🔍 Aktif trade kontrolü...")
        trader.check_active_trades()
        
        logger.info(f"📈 Aktif trade sayısı: {len(trader.active_trades)}")
    
    # MT5 bağlantı testi (eğer varsa)
    if MT5_AVAILABLE:
        logger.info("🔌 MT5 bağlantı testi...")
        connection_success = trader.initialize_mt5()
        if connection_success:
            logger.info("✅ MT5 bağlantısı başarılı")
            
            # Terminal bilgileri
            terminal_info = mt5.terminal_info()
            if terminal_info:
                logger.info(f"📊 Terminal: {terminal_info.name} {terminal_info.build}")
            
            trader.stop_live_trading()
        else:
            logger.warning("⚠️  MT5 bağlantısı başarısız (normal - ayarlar gerekli)")
    else:
        logger.info("⚠️  MT5 yok - sadece demo mode test edildi")
    
    logger.info("✅ LiveTrader modülü test tamamlandı!")


if __name__ == "__main__":
    main()