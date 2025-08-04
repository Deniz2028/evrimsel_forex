#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
EVO_FX_MASTER - Risk ve Para Yönetimi Modülü
===========================================

ATR tabanlı pozisyon boyutu hesaplama, stop-loss/take-profit yönetimi,
günlük drawdown kontrolü ve equity koruma sistemi.
"""

import numpy as np
import pandas as pd
import logging
from typing import Dict, List, Optional, Tuple, NamedTuple
from datetime import datetime, timedelta
from pathlib import Path
import json
from dataclasses import dataclass
from enum import Enum

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class RiskLevel(Enum):
    """Risk seviyeleri."""
    CONSERVATIVE = "conservative"  # %0.5 risk
    MODERATE = "moderate"         # %1.0 risk  
    AGGRESSIVE = "aggressive"     # %2.0 risk
    VERY_AGGRESSIVE = "very_aggressive"  # %3.0 risk


@dataclass
class PositionSizeResult:
    """Pozisyon boyutu hesaplama sonucu."""
    lot_size: float
    risk_amount: float
    stop_distance: float
    risk_percent: float
    max_loss_usd: float
    leverage_used: float


@dataclass
class StopLossResult:
    """Stop-loss hesaplama sonucu."""
    stop_price: float
    stop_distance_pips: float
    risk_reward_ratio: float
    take_profit_price: float
    tp_distance_pips: float


class RiskManager:
    """Ana risk yönetimi sınıfı."""
    
    def __init__(self, config: Dict = None):
        """
        Args:
            config: Risk yönetimi parametreleri
        """
        self.config = config or {}
        
        # Temel risk parametreleri
        self.base_risk_percent = self.config.get('base_risk_percent', 0.01)  # %1
        self.max_risk_percent = self.config.get('max_risk_percent', 0.03)    # %3
        self.min_risk_percent = self.config.get('min_risk_percent', 0.005)   # %0.5
        
        # Position sizing parametreleri  
        self.max_position_percent = self.config.get('max_position_percent', 0.2)  # %20 max
        self.min_lot_size = self.config.get('min_lot_size', 0.01)
        self.max_lot_size = self.config.get('max_lot_size', 10.0)
        
        # ATR tabanlı stop-loss
        self.atr_multiplier_sl = self.config.get('atr_multiplier_sl', 1.1)  # SL = ATR × 1.1
        self.atr_multiplier_tp = self.config.get('atr_multiplier_tp', 2.2)  # TP = ATR × 2.2
        self.min_rr_ratio = self.config.get('min_rr_ratio', 1.5)            # Min R:R 1:1.5
        
        # Günlük risk limitleri
        self.daily_loss_limit_percent = self.config.get('daily_loss_limit', 0.04)    # %4
        self.daily_profit_target_percent = self.config.get('daily_profit_target', 0.06)  # %6
        self.max_trades_per_day = self.config.get('max_trades_per_day', 20)
        
        # Equity koruma
        self.equity_stop_percent = self.config.get('equity_stop_percent', 0.15)  # %15
        self.trailing_stop_percent = self.config.get('trailing_stop_percent', 0.05)  # %5
        
        # Sembol spesifik ayarlar
        self.symbol_configs = {
            'EURUSD': {
                'pip_value': 0.0001,
                'pip_cost_per_lot': 1.0,  # $1 per pip per standard lot
                'min_spread': 0.1,        # 0.1 pip minimum spread
                'max_spread': 3.0,        # 3 pip maximum spread
                'session_multiplier': {   # Oturum bazlı risk çarpanları
                    'sydney': 0.8,
                    'tokyo': 0.9, 
                    'london': 1.2,
                    'newyork': 1.1,
                    'overlap_london_ny': 1.3
                }
            },
            'XAUUSD': {
                'pip_value': 0.01,
                'pip_cost_per_lot': 1.0,
                'min_spread': 0.2,
                'max_spread': 5.0,
                'session_multiplier': {
                    'sydney': 0.7,
                    'tokyo': 0.8,
                    'london': 1.0,
                    'newyork': 1.1,
                    'overlap_london_ny': 1.2
                }
            }
        }
        
        # Günlük takip
        self.daily_stats = {
            'date': datetime.now().date(),
            'trades_count': 0,
            'realized_pnl': 0.0,
            'max_drawdown': 0.0,
            'peak_balance': 0.0,
            'is_trading_stopped': False,
            'stop_reason': None
        }
        
        logger.info(f"🛡️  RiskManager hazırlandı. Temel risk: {self.base_risk_percent:.1%}")
    
    def calculate_position_size(self, signal_strength: float, balance: float,
                              atr: float, price: float, symbol: str = 'EURUSD',
                              session: str = 'london') -> PositionSizeResult:
        """
        ATR ve sinyal kuvvetine göre pozisyon boyutu hesaplar.
        
        Args:
            signal_strength: Sinyal kuvveti (0.0-1.0)
            balance: Mevcut hesap bakiyesi
            atr: Average True Range
            price: Mevcut fiyat
            symbol: Trading sembolü
            session: Trading seansı
            
        Returns:
            Pozisyon boyutu detayları
        """
        # Sembol konfigürasyonu
        symbol_config = self.symbol_configs.get(symbol, self.symbol_configs['EURUSD'])
        
        # Dinamik risk hesaplama
        base_risk = self.base_risk_percent
        
        # Sinyal gücüne göre risk ayarlama
        if signal_strength < 0.3:
            risk_percent = base_risk * 0.5      # Zayıf sinyal: %0.5
        elif signal_strength < 0.6:
            risk_percent = base_risk * 1.0      # Orta sinyal: %1.0  
        elif signal_strength < 0.8:
            risk_percent = base_risk * 1.5      # Güçlü sinyal: %1.5
        else:
            risk_percent = base_risk * 2.0      # Çok güçlü: %2.0
        
        # Oturum çarpanı
        session_multiplier = symbol_config['session_multiplier'].get(session, 1.0)
        risk_percent *= session_multiplier
        
        # Risk limitlerini kontrol et
        risk_percent = np.clip(risk_percent, self.min_risk_percent, self.max_risk_percent)
        
        # Risk miktarı (USD)
        risk_amount = balance * risk_percent
        
        # Stop-loss mesafesi (ATR tabanlı)
        stop_distance = atr * self.atr_multiplier_sl
        
        if stop_distance <= 0:
            return PositionSizeResult(0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
        
        # Pip değeri hesaplama
        pip_value = symbol_config['pip_value']
        stop_distance_pips = stop_distance / pip_value
        
        # Pozisyon boyutu hesaplama
        # Risk Amount = Position Size × Stop Distance × Pip Cost
        pip_cost_per_lot = symbol_config['pip_cost_per_lot']
        
        if symbol == 'EURUSD':
            # Forex: 1 standard lot = 100,000 units
            position_size_lots = risk_amount / (stop_distance_pips * pip_cost_per_lot)
        else:
            # Gold: 1 lot = 100 oz, pip cost farklı
            position_size_lots = risk_amount / (stop_distance_pips * pip_cost_per_lot)
        
        # Lot boyutu limitlerini uygula
        position_size_lots = np.clip(position_size_lots, self.min_lot_size, self.max_lot_size)
        
        # Maksimum pozisyon kontrolü (bakiyenin %20'si)
        max_position_value = balance * self.max_position_percent
        max_lots_by_value = max_position_value / price
        position_size_lots = min(position_size_lots, max_lots_by_value)
        
        # Leverage hesaplama
        position_value = position_size_lots * price
        leverage_used = position_value / balance if balance > 0 else 0
        
        # Maksimum kayıp hesaplama
        max_loss_usd = position_size_lots * stop_distance_pips * pip_cost_per_lot
        
        return PositionSizeResult(
            lot_size=round(position_size_lots, 2),
            risk_amount=risk_amount,
            stop_distance=stop_distance,
            risk_percent=risk_percent,
            max_loss_usd=max_loss_usd,
            leverage_used=leverage_used
        )
    
    def calculate_stop_loss_take_profit(self, entry_price: float, direction: int,
                                       atr: float, symbol: str = 'EURUSD') -> StopLossResult:
        """
        ATR tabanlı stop-loss ve take-profit seviyeleri hesaplar.
        
        Args:
            entry_price: Giriş fiyatı
            direction: Pozisyon yönü (1: long, -1: short)
            atr: Average True Range
            symbol: Trading sembolü
            
        Returns:
            Stop-loss ve take-profit detayları
        """
        symbol_config = self.symbol_configs.get(symbol, self.symbol_configs['EURUSD'])
        pip_value = symbol_config['pip_value']
        
        # Stop-loss hesaplama
        sl_distance = atr * self.atr_multiplier_sl
        
        if direction == 1:  # Long pozisyon
            stop_price = entry_price - sl_distance
            tp_distance = atr * self.atr_multiplier_tp
            take_profit_price = entry_price + tp_distance
        else:  # Short pozisyon  
            stop_price = entry_price + sl_distance
            tp_distance = atr * self.atr_multiplier_tp
            take_profit_price = entry_price - tp_distance
        
        # Pip cinsinden mesafeler
        sl_distance_pips = sl_distance / pip_value
        tp_distance_pips = tp_distance / pip_value
        
        # Risk/Reward oranı
        risk_reward_ratio = tp_distance_pips / sl_distance_pips if sl_distance_pips > 0 else 0
        
        # Minimum R:R kontrolü
        if risk_reward_ratio < self.min_rr_ratio:
            # TP'yi uzat
            required_tp_distance = sl_distance * self.min_rr_ratio
            if direction == 1:
                take_profit_price = entry_price + required_tp_distance
            else:
                take_profit_price = entry_price - required_tp_distance
            
            tp_distance_pips = required_tp_distance / pip_value
            risk_reward_ratio = self.min_rr_ratio
        
        return StopLossResult(
            stop_price=round(stop_price, 5),
            stop_distance_pips=round(sl_distance_pips, 1),
            risk_reward_ratio=round(risk_reward_ratio, 2),
            take_profit_price=round(take_profit_price, 5),
            tp_distance_pips=round(tp_distance_pips, 1)
        )
    
    def check_daily_limits(self, current_balance: float) -> Dict[str, bool]:
        """
        Günlük risk limitlerini kontrol eder.
        
        Args:
            current_balance: Mevcut bakiye
            
        Returns:
            Limit kontrol sonuçları
        """
        today = datetime.now().date()
        
        # Yeni gün kontrolü
        if self.daily_stats['date'] != today:
            self.reset_daily_stats(today, current_balance)
        
        # Günlük limit kontrolü - güvenlik kontrolü ile
        if self.daily_stats['peak_balance'] > 0:
            daily_pnl_percent = self.daily_stats['realized_pnl'] / self.daily_stats['peak_balance']
        else:
            daily_pnl_percent = 0.0
        
        results = {
            'can_trade': True,
            'reason': None,
            'daily_pnl_percent': daily_pnl_percent,
            'trades_remaining': self.max_trades_per_day - self.daily_stats['trades_count']
        }
        
        # Loss limit kontrolü
        if daily_pnl_percent <= -self.daily_loss_limit_percent:
            results['can_trade'] = False
            results['reason'] = f'Günlük kayıp limiti aşıldı: {daily_pnl_percent:.1%}'
            self.daily_stats['is_trading_stopped'] = True
            self.daily_stats['stop_reason'] = 'daily_loss_limit'
        
        # Profit target kontrolü
        elif daily_pnl_percent >= self.daily_profit_target_percent:
            results['can_trade'] = False
            results['reason'] = f'Günlük kar hedefi ulaşıldı: {daily_pnl_percent:.1%}'
            self.daily_stats['is_trading_stopped'] = True
            self.daily_stats['stop_reason'] = 'daily_profit_target'
        
        # Trade sayısı kontrolü
        elif self.daily_stats['trades_count'] >= self.max_trades_per_day:
            results['can_trade'] = False
            results['reason'] = f'Günlük maksimum trade sayısı: {self.max_trades_per_day}'
            self.daily_stats['is_trading_stopped'] = True
            self.daily_stats['stop_reason'] = 'max_trades'
        
        # Daha önce durdurulmuş mu?
        elif self.daily_stats['is_trading_stopped']:
            results['can_trade'] = False
            results['reason'] = f'Trading durduruldu: {self.daily_stats["stop_reason"]}'
        
        return results
    
    def check_equity_protection(self, current_balance: float, 
                               initial_balance: float) -> Dict[str, any]:
        """
        Equity koruma sistemini kontrol eder.
        
        Args:
            current_balance: Mevcut bakiye
            initial_balance: Başlangıç bakiyesi
            
        Returns:
            Equity koruma sonuçları
        """
        equity_drawdown = (initial_balance - current_balance) / initial_balance
        
        results = {
            'emergency_stop': False,
            'reduce_risk': False,
            'equity_drawdown_percent': equity_drawdown,
            'action': None
        }
        
        # Acil durum durdurma (%15 kayıp)
        if equity_drawdown >= self.equity_stop_percent:
            results['emergency_stop'] = True
            results['action'] = 'STOP_ALL_TRADING'
            logger.error(f"🚨 ACIL DURUM! Equity kaybı: {equity_drawdown:.1%}")
            
            # E-posta bildirimi (placeholder)
            self._send_emergency_alert(equity_drawdown)
        
        # Risk azaltma (%8 kayıp)
        elif equity_drawdown >= self.equity_stop_percent * 0.5:
            results['reduce_risk'] = True  
            results['action'] = 'REDUCE_RISK'
            logger.warning(f"⚠️  Risk azaltma aktif. Equity kaybı: {equity_drawdown:.1%}")
        
        return results
    
    def update_daily_stats(self, trade_pnl: float, current_balance: float):
        """
        Günlük istatistikleri günceller.
        
        Args:
            trade_pnl: Trade P&L
            current_balance: Güncel bakiye
        """
        self.daily_stats['trades_count'] += 1
        self.daily_stats['realized_pnl'] += trade_pnl
        
        # Peak balance takibi
        if current_balance > self.daily_stats['peak_balance']:
            self.daily_stats['peak_balance'] = current_balance
        
        # Drawdown hesaplama
        current_drawdown = (self.daily_stats['peak_balance'] - current_balance) / self.daily_stats['peak_balance']
        if current_drawdown > self.daily_stats['max_drawdown']:
            self.daily_stats['max_drawdown'] = current_drawdown
    
    def reset_daily_stats(self, new_date: datetime.date, starting_balance: float):
        """
        Günlük istatistikleri sıfırlar.
        
        Args:
            new_date: Yeni tarih
            starting_balance: Başlangıç bakiyesi
        """
        self.daily_stats = {
            'date': new_date,
            'trades_count': 0,
            'realized_pnl': 0.0,
            'max_drawdown': 0.0,
            'peak_balance': starting_balance,
            'is_trading_stopped': False,
            'stop_reason': None
        }
        
        logger.info(f"📅 Yeni gün başladı: {new_date}. Bakiye: ${starting_balance:,.2f}")
    
    def get_risk_level_recommendation(self, volatility_zscore: float, 
                                    session: str, market_conditions: Dict) -> RiskLevel:
        """
        Piyasa koşullarına göre risk seviyesi önerir.
        
        Args:
            volatility_zscore: Volatilite z-score
            session: Trading seansı
            market_conditions: Piyasa koşulları
            
        Returns:
            Önerilen risk seviyesi
        """
        risk_score = 0
        
        # Volatilite faktörü
        if abs(volatility_zscore) > 2.0:
            risk_score += 2  # Yüksek volatilite = daha konservatif
        elif abs(volatility_zscore) < 0.5:
            risk_score -= 1  # Düşük volatilite = daha agresif
        
        # Seans faktörü
        if session in ['london', 'overlap_london_ny']:
            risk_score -= 1  # Yüksek likidite = daha agresif
        elif session in ['sydney', 'tokyo']:
            risk_score += 1  # Düşük likidite = daha konservatif
        
        # Piyasa koşulları
        if market_conditions.get('trend_strength', 0) > 0.7:
            risk_score -= 1  # Güçlü trend = daha agresif
        
        if market_conditions.get('news_risk', False):
            risk_score += 2  # Haber riski = çok konservatif
        
        # Risk seviyesi belirle
        if risk_score >= 3:
            return RiskLevel.CONSERVATIVE
        elif risk_score >= 1:
            return RiskLevel.MODERATE
        elif risk_score >= -1:
            return RiskLevel.AGGRESSIVE
        else:
            return RiskLevel.VERY_AGGRESSIVE
    
    def _send_emergency_alert(self, drawdown_percent: float):
        """
        Acil durum bildirimi gönderir (placeholder).
        
        Args:
            drawdown_percent: Drawdown yüzdesi
        """
        # Telegram/E-posta bildirimi placeholder
        alert_message = f"""
        🚨 ACIL DURUM ALARMI! 🚨
        
        Equity Kaybı: {drawdown_percent:.1%}
        Zaman: {datetime.now()}
        Durum: TÜM TRADING DURDURULDU
        
        Lütfen hesabınızı kontrol edin!
        """
        
        logger.critical(alert_message)
        
        # Gerçek implementasyonda:
        # - Telegram bot bildirimi
        # - E-posta gönderimi  
        # - SMS uyarısı
        # - Discord webhook
    
    def export_daily_report(self) -> Dict:
        """
        Günlük risk raporu oluşturur.
        
        Returns:
            Günlük risk raporu
        """
        return {
            'date': self.daily_stats['date'].isoformat(),
            'trades_count': self.daily_stats['trades_count'],
            'realized_pnl': round(self.daily_stats['realized_pnl'], 2),
            'max_drawdown_percent': round(self.daily_stats['max_drawdown'] * 100, 2),
            'peak_balance': round(self.daily_stats['peak_balance'], 2),
            'is_trading_stopped': self.daily_stats['is_trading_stopped'],
            'stop_reason': self.daily_stats['stop_reason'],
            'risk_config': {
                'base_risk_percent': self.base_risk_percent,
                'daily_loss_limit': self.daily_loss_limit_percent,
                'daily_profit_target': self.daily_profit_target_percent,
                'max_trades_per_day': self.max_trades_per_day
            }
        }


def main():
    """Test fonksiyonu."""
    logger.info("🛡️  Risk modülü test ediliyor...")
    
    # Test config
    config = {
        'base_risk_percent': 0.01,
        'daily_loss_limit': 0.04,
        'atr_multiplier_sl': 1.1,
        'atr_multiplier_tp': 2.2
    }
    
    # RiskManager oluştur
    risk_manager = RiskManager(config)
    
    # Test parametreleri
    test_balance = 10000.0
    test_atr = 0.001  # EURUSD için tipik ATR
    test_price = 1.1000
    test_symbol = 'EURUSD'
    
    logger.info("🧪 Pozisyon boyutu testi...")
    
    # Farklı sinyal kuvvetleri test et
    signal_strengths = [0.2, 0.5, 0.8, 0.95]
    
    for signal_strength in signal_strengths:
        result = risk_manager.calculate_position_size(
            signal_strength=signal_strength,
            balance=test_balance,
            atr=test_atr,
            price=test_price,
            symbol=test_symbol,
            session='london'
        )
        
        logger.info(f"📊 Sinyal kuvveti {signal_strength:.1f}:")
        logger.info(f"  Lot boyutu: {result.lot_size}")
        logger.info(f"  Risk miktarı: ${result.risk_amount:.2f}")
        logger.info(f"  Risk yüzdesi: {result.risk_percent:.2%}")
        logger.info(f"  Leverage: {result.leverage_used:.1f}x")
    
    # Stop-loss/Take-profit testi
    logger.info("\n🎯 Stop-loss/Take-profit testi...")
    
    sl_tp_result = risk_manager.calculate_stop_loss_take_profit(
        entry_price=test_price,
        direction=1,  # Long
        atr=test_atr,
        symbol=test_symbol
    )
    
    logger.info(f"📈 Long pozisyon (Giriş: {test_price}):")
    logger.info(f"  Stop-loss: {sl_tp_result.stop_price}")
    logger.info(f"  Take-profit: {sl_tp_result.take_profit_price}")
    logger.info(f"  R:R oranı: 1:{sl_tp_result.risk_reward_ratio}")
    logger.info(f"  SL mesafesi: {sl_tp_result.stop_distance_pips} pip")
    logger.info(f"  TP mesafesi: {sl_tp_result.tp_distance_pips} pip")
    
    # Günlük limit testi
    logger.info("\n📅 Günlük limit testi...")
    
    daily_check = risk_manager.check_daily_limits(test_balance)
    logger.info(f"Trading yapılabilir: {daily_check['can_trade']}")
    logger.info(f"Kalan trade sayısı: {daily_check['trades_remaining']}")
    
    # Equity koruma testi
    logger.info("\n🛡️  Equity koruma testi...")
    
    # %10 kayıp simülasyonu
    test_balance_loss = test_balance * 0.9
    equity_check = risk_manager.check_equity_protection(test_balance_loss, test_balance)
    
    logger.info(f"Equity kaybı: {equity_check['equity_drawdown_percent']:.1%}")
    logger.info(f"Risk azaltma: {equity_check['reduce_risk']}")
    logger.info(f"Acil durum: {equity_check['emergency_stop']}")
    
    # Günlük rapor
    logger.info("\n📋 Günlük rapor testi...")
    daily_report = risk_manager.export_daily_report()
    logger.info(f"Rapor: {json.dumps(daily_report, indent=2)}")
    
    logger.info("✅ Risk modülü test tamamlandı!")


if __name__ == "__main__":
    main()