#!/usr/bin/env python
# -*- coding=utf-8 -*-
"""
trading_system_ft.py

核心交易系统模块 (Core Trading System Module)

架构概述:
1. 多线程模型: 采用主监控、卖出监控、盘后监控、期权监控多线程并行，通过 Threading.Event 协调。
2. 状态机驱动: Position 对象维护完整的生命周期状态 (Building -> Running -> Closed)。
3. 双重风控: 
   - 静态风控: 资金占用、最大持仓限制。
   - 动态风控: ATR波动率止损、持续新低动能止损、盘前/夜盘特殊风控。
4. 数据一致性: 关键状态 (Positions, Orders, PnL) 实时持久化到 JSON，支持故障恢复。

核心组件 (Core Components):
- TradingSystem: 整个系统的总调度和执行中心。
- Position: 单个股票的持仓对象，封装了其所有状态和交易历史。
- StrategyEvaluator: 策略评估器，负责从股票池中发现买入和卖出信号。
- RiskManager: 风险管理器，负责组合风险和单笔交易风险的检查。
- StockTechAdvisor: 基于LLM的二次复核模块，为交易信号提供额外决策支持。

核心方法
- _handle_pending_add_order:加仓订单处理
- run_strategy_loop: 主策略循环（信号扫描入口）
- _execute_confirmed_buy: [关键修改点] 确认买入执行器（负责本地执行 + 信号分发）
- process_buy_signal: 买入信号处理（风控、资金检查、下单路由）
- _handle_initial_buy: 首次建仓逻辑（计算仓位、R值、止损）
- _check_position_signals: 持仓监控主入口（止盈、止损、状态维护）
- _check_scale_in_conditions: 补仓检查（3-3-4模型/ATR补仓）
- _check_confirmation_add_opportunities: 确认加仓检查（右侧确认逻辑）
- _check_pending_position_transactions: 在途订单状态轮询（处理加仓/卖出回报）
- _is_entry_price_overextended: 价格过热/追高风控检查
- _check_trade_safety_gate: 可以的安全检查
- _check_and_execute_daily_profit_target
- _after_hours_monitor_loop
- _execute_extended_hours_pending_buy
- _pre_market_monitor_loop
- _check_continuous_low_stop_loss_pro
- _check_pending_option_orders【期权】
- _process_micro_building_continuation
- _verify_buy_signal_viability
- _option_monitor_loop
- _execute_infant_flash_trial
- _verify_trade_quality_gate
- _reconcile_positions_with_broker
- _reconcile_option_positions_with_broker
"""

import os, sys
project_path = os.path.dirname(os.path.abspath(__file__))
sys.path.append(project_path)
import logging
import threading
import time
from datetime import datetime,date,timezone,timedelta
from dataclasses import asdict
from typing import Dict, Optional, Tuple,List,Any
import json
from longport.openapi import (
    Config,
    TradeContext,
    QuoteContext,
    TimeInForceType,
    OrderStatus,
    OrderType,
    OrderSide,
    Period,
    AdjustType,
    OutsideRTH
    )
from utils.log_utils import setup_logging
from strategy_evaluator_ft import StrategyEvaluator
from configs.config_ft import get_longbridge_config, TradingConfig
from positions import (
    Position,
    PurchaseActionType,
    PositionOverallPhase,
    )
from utils.cfg_utils import load_yaml2cfg
from risk_manager import RiskManager,ExtendedHoursRiskEngine
from utils.common_utils import (
    get_stock_list,
    normalize_symbol,
    EnhancedJSONEncoder,
    parse_symbol,
    resolve_underlying_symbol,
    denormalize_hs_option_symbol,
    load_json_data,
    extract_candidates_pools,
    get_root_symbol,
    get_underlying_from_option_symbol
    )
from utils.notification_utils import send_email, send_weixin_notice
from utils.market_time_utils import (
    is_any_market_open,
    is_in_eod_buy_window,
    is_entering_weekend_risk_for_symbol,
    is_in_opening_window,
    is_any_market_in_grace_period,
    is_in_post_market_sell_window,
    is_in_trading_grace_period,
    MarketType,
    get_market_type,
    normalize_to_utc,
    is_opened_today,
    is_us_market_open,
    is_in_custom_trading_window,
    get_timezone_for_symbol,
    get_current_market_session,
    MARKET_TRADING_HOURS,
    is_market_in_trading_hours,
    get_last_business_close_time,
    TradingSession
    )
from utils.longport_api_utils import (
    get_stock_static_info,
    get_yesterday_close_price,
    get_raw_quote,
    get_historical_atr,
    get_dynamic_atr,
    is_pure_stock,
    check_tactical_entry_signal,
    check_extended_hours_tactical_entry_signal,
    check_tactical_exit_signal,
    check_extended_hours_tactical_exit_signal,
    get_klines_data,
    get_dynamic_atr_multiplier,
    is_making_new_high,
    get_rsi,
    get_adx,
    get_institutional_net_flow_ratio,
    get_capital_flow_vectors,
    get_intraday_poc,
    get_smart_quote,
    get_disable_trade,
    get_enable_bearish,
    get_custom_watchlist_group
)
from utils.market_regime import MarketRegimeEngine,MarketRegime,IntradayHealthType,AvalancheStatus
from sentiment_service import SentimentAnalysisPipeline
import pandas as pd
from api.data_provider import FutuDataProvider
from api.data_provider import HuaShengtDataProvider
from concurrent.futures import ThreadPoolExecutor
from engines.gex_data_engine import GEXDataEngine
from utils.pending_buy_cache import PendingBuySignalCache
from utils.trading_clock import get_trading_window_status, TradingWindowStatus,get_dynamic_k_minutes
from engines.anomaly_detection_engine import AnomalyDetectionEngine
from adaptive_stop_loss import AdaptiveStopLoss
from notification_manager import NotificationManager
from decimal import Decimal
import pytz
import random
import numpy as np
import pandas_ta as ta
import warnings
warnings.filterwarnings('ignore')

logger = logging.getLogger('trading_system')
trade_logger = logging.getLogger('trade_signals')
sell_logger = logging.getLogger('sell_signals')
buy_logger = logging.getLogger('buy_signals')

class TradingSystem:
    """
    交易系统主类 (Main Trading System Class)

    - 实现了基于风险的仓位管理。
    - 支持非阻塞的订单处理和状态恢复。
    - 核心策略：两阶段加仓（建仓期分批建仓 + 盈利期金字塔加仓）。
    - 融合了多种动态风险管理技术（保本止损、追踪止损）。
    - 采用多线程架构，将高耗时的信号扫描与低延迟的风险监控分离。
    """
    # ==============================================================================
    # I. 系统初始化与核心状态管理 (Initialization & Core State)
    # ==============================================================================
    def __init__(self, cfg_file, longbridge_config: Config):
        """
        初始化交易系统
        """
        logger.warning("交易系统初始化中...")
        
        # === 1. 基础配置与API上下文 ===
        self.cfg_file = cfg_file
        self.config: Optional[TradingConfig] = None
        self.longbridge_config = longbridge_config
        self.trade_ctx = TradeContext(longbridge_config)
        self.quote_ctx = QuoteContext(longbridge_config)
        self._cfg_mtime = 0 # 配置文件修改时间
        self.script_name = os.path.splitext(os.path.basename(__file__))[0]
        
        # === 2. 核心策略组件 (Lazy Load) ===
        self.risk_manager: Optional[RiskManager] = None
        self.strategy_evaluator: Optional[StrategyEvaluator] = None
        self.stock_tech_advisor = None
        self.adaptive_stop_loss = None
        self.extended_hours_risk_engine: Optional[ExtendedHoursRiskEngine] = None #初始化夜盘风控引擎
        
        # === 3. 数据引擎与服务 ===
        self.market_regime_engine = MarketRegimeEngine(self.quote_ctx)
        self.data_provider = FutuDataProvider(futu_config={})
        self.hs_data_provider = HuaShengtDataProvider()
        self.gex_engine = GEXDataEngine(db_path='', api_client=self.hs_data_provider)
        self.sentiment_analysis = SentimentAnalysisPipeline(self.quote_ctx)
        self.anomaly_engine = AnomalyDetectionEngine(
            db_path=os.path.join(project_path, 'data', 'option_anomaly_engine.db'),
            api_client=self.hs_data_provider,
            symbols_to_track=[],
            quote_ctx=self.quote_ctx
        )
        self.notification_manager = NotificationManager(self)
        
        # === 4. 状态持久化路径 ===
        self.position_file = os.path.join(project_path, 'data/positions_ft.json')
        self.trade_history_file = os.path.join(project_path, 'data/trade_history_ft.json')
        self.pending_orders_file = os.path.join(project_path, 'data/pending_orders_ft.json')
        self.sell_locks_file = os.path.join(project_path, 'data/sell_locks_ft.json')
        self.conservative_stock_pool_file = os.path.join(project_path, 'data/us_alpha_stock_by.json')
        self.pre_market_states_file = os.path.join(project_path, 'data/pre_market_states_ft.json')
        self.daily_pnl_file = os.path.join(project_path, 'data/daily_pnl_state_ft.json') # 每日盈亏文件
        self.intraday_blacklist: set = set()
        self.blacklist_file = os.path.join(getattr(self.config,'master_project_path','/home/nexus/project/trader'), 'data/intraday_blacklist.json') # 持久化文件

        # === 5. 运行时内存状态 (需持久化) ===
        self.positions: Dict[str, Position] = {}
        self.pending_orders: Dict[str, Dict] = {}
        self.sell_locks: Dict[Tuple[str, str], float] = {}
        self.pre_market_states: Dict[str, Dict] = {}
        # 日内交易记忆体：记录当日卖出的最高/最后价格，防止 T+0 反向亏损
        # 结构: { "TSLL.US": 23.085, "AAPL.US": 150.5 }
        self.intraday_trade_history: Dict[str, float] = {}
        self.clean_second_tier_stocks: List[str] = []
        self.clean_first_tier_stocks: List[str] = []
        self.risky_second_tier_stocks: List[str] = []
        
        # 每日盈亏状态 (内存+持久化)
        self.daily_realized_pnl_hk = 0.0
        self.daily_realized_pnl_us = 0.0
        self.current_trading_date = date.today()
        self.daily_hk_profit_target_hit = False
        self.daily_us_profit_target_hit = False
        # ▼▼▼ 30分钟逃跑计划状态变量 ▼▼▼
        # 记录倒计时开始的时间戳 (None 表示未触发)
        self.daily_profit_countdown_start_ts: Optional[float] = None
        # 记录今天是否已经执行过“半程逃跑” (防止反复触发卖出)
        self.half_target_escape_triggered: bool = False
        # 倒计时期间的最高盈利水位线 (用于回撤熔断)
        self.daily_pnl_high_water_mark: float = 0.0
        # 上次重置倒计时时的基准盈利 (用于动能续杯)
        self.daily_pnl_last_reset_base: float = 0.0
        self.daily_pnl_us_rth_last_reset_date: Optional[str] = None
        self.daily_equity_baseline_hkd: float = 0.0
        self.monthly_equity_baseline_hkd: float = 0.0
        self.monthly_equity_baseline_key: Optional[str] = None
        self.daily_loss_freeze_date: Optional[str] = None

        self.current_trading_date_hk = None
        self.current_trading_date_us = None

        # === 6. 缓存与信号处理 ===
        self.pending_buy_cache = PendingBuySignalCache(db_path=os.path.join(project_path, 'data/pending_buy_signals_ft.db'))
        self.signals_in_flight = set() # 正在处理中的信号锁
        
        # === 7. 线程与并发控制 ===
        self.task_executor = ThreadPoolExecutor(max_workers=10, thread_name_prefix='TaskExecutor')
        
        # 线程事件控制
        self.stop_main_monitor = threading.Event()
        self.stop_sell_monitor = threading.Event()
        self.stop_option_monitor = threading.Event()
        self.stop_pending_monitor = threading.Event()
        self.stop_pre_market_monitor = threading.Event()
        
        # 线程句柄
        self.main_monitor_thread: Optional[threading.Thread] = None
        self.sell_signal_monitor_thread: Optional[threading.Thread] = None
        self.ah_monitor_thread: Optional[threading.Thread] = None
        self.pending_signal_monitor_thread: Optional[threading.Thread] = None
        self.pre_market_monitor_thread: Optional[threading.Thread] = None

        # 线程锁
        self.position_lock = threading.RLock()
        self.pending_orders_lock = threading.RLock()
        self.notification_lock = threading.RLock()
        self.active_confirmations_lock = threading.Lock()
        self.signals_in_flight_lock = threading.Lock()
        self.pending_option_orders_lock = threading.RLock()

        # === 8. 辅助变量 ===
        self.account_info = None
        self.cash_info_hk = None
        self.cash_info_us = None
        self._account_cache = None
        self._account_cache_ts = 0
        self._account_cache_ttl = 5.0
        self.last_reconcile_timestamp = 0
        self.reconcile_cooldown_seconds = 45
        self.buy_signal_notifications: Dict[str, int] = {}
        self.notified_this_cycle = set()
        self.sell_lock_duration_seconds = 2*60
        self.extended_hours_order_timers = {} # 用于记录夜盘挂单时间：{symbol: timestamp}
        self.sentiment_cache_timers: Dict[str, float] = {} # 舆情风控节流计时器 (防止高频读取本地缓存)
        self.last_sync_task_time = 0.0          # 用于 run_strategy_loop 的主同步计时
        self.last_non_rth_sync_time = 0.0       # 用于 _main_monitor_loop 的非盘中同步计时
        self.enable_bearish = False
        self.disable_trade = False
        self.conservative_paused_month: Optional[str] = None
        # === 影子标签系统与雷达状态 (Shadow Tags & Radar) ===
        # 你的四大专业级控制面板
        self.shadow_tags = {
            'tactical_liquidation':[], # 对应分组名: tactical_liquidation (战术清仓)
            'profit_only_mode':[],     # 对应分组名: profit_only_mode (仅卖盈利)
            'strategic_hold':[],       # 对应分组名: strategic_hold (战略锁定/绝对不卖)
            'macro_tactical_radar':[]  # 对应分组名: macro_tactical_radar (大盘战术雷达)
        }
        # 防护机制：防止清仓指令和雷达告警无限狂刷
        self._last_liquidation_hash = ""
        self._radar_alert_cooldowns = {}
        self._last_clearance_hash = "" # 清空状态

        # === 9. 启动加载流程 ===
        self._check_and_perform_updates(force=True)
        self._load_all_states() # 加载所有状态，包括 Daily PnL
        self._initialize_account_info()
        logger.info("保守策略模式：启动阶段跳过券商持仓自动同步。")
        self._recover_intraday_history_from_logs() # 系统重启后的记忆恢复 (必须在启动监控线程之前执行)
        
        # 启动后台线程
        self._start_all_monitors()
        
        logger.warning("交易系统初始化完成。")
    
    def _check_and_perform_updates(self, force: bool = False):
        """
        检查是否需要更新配置和Watchlist，如果需要则执行。
        一次性完成所有热更新检查和执行，避免冗余IO。
        """
        # now = time.time()
        # # 确保 self.config 已初始化后再访问其属性
        # if not force and self.config and (now - self.last_update_time < self.config.updater_interval_seconds):
        #     return
        logger.info("触发热更新检查...")
        try:
            self.enable_bearish = get_enable_bearish(self.quote_ctx)
            self.disable_trade = get_disable_trade(self.quote_ctx)

            # 只有文件修改时间变了才重新加载
            current_mtime = os.path.getmtime(self.cfg_file)
            if not force and current_mtime == self._cfg_mtime:
                return
            
            self._cfg_mtime = current_mtime

            cfg = load_yaml2cfg(self.cfg_file)
            
            if self.config is None:
                self.config = TradingConfig()
                logger.info("首次创建TradingConfig对象。")

            updated_fields = []
            new_config_data = {
                # === 通用参数 (common section) ===
                # 注意：您的yaml中test_mode在common下，但旧代码在trading下，这里根据yaml结构调整
                'test_mode': cfg.common.get('test_mode', False),
                'max_signal_notifications': cfg.common.get('max_signal_notifications', 1),
                'updater_interval_seconds': cfg.common.get('updater_interval_seconds', 600),
                'check_interval': cfg.common.get('check_interval', 20),
                'email_receivers': cfg.common.get('email_receivers', '')
            }
            config_updated = False
            for key, value in new_config_data.items():
                if not hasattr(self.config, key) or getattr(self.config, key) != value:
                    setattr(self.config, key, value)
                    updated_fields.append(f"{key}={value}")
                    config_updated = True
            
            '''
            merge_stock_list = []
            try:
                hk_file_path = os.path.join(project_path, 'data/hk_stock.json')
                us_file_path = os.path.join(project_path, 'data/us_stock.json')
                hk_stock_list = get_symbol_codes_from_json(hk_file_path)
                us_stock_list = get_symbol_codes_from_json(us_file_path)
                merge_stock_list = hk_stock_list + us_stock_list
            except (FileNotFoundError, ValueError) as e:
                logger.error(f'get_symbol_codes_from_json error: {e}', exc_info=True)
            
            print(merge_stock_list)
            
            print(self.config.vip_symbols)
            if merge_stock_list and  len(merge_stock_list) > 0:
                self.config.vip_symbols = set(self.config.vip_symbols) | set(merge_stock_list)
                print("self.config.vip_symbols")
                print(self.config.vip_symbols)
            '''
            
            if updated_fields:
                logger.info(f"交易配置已更新: {', '.join(updated_fields)}")
            
            if self.strategy_evaluator is None:
                self.strategy_evaluator = StrategyEvaluator(self.quote_ctx, self.config)
                # 在 StrategyEvaluator 创建后，立即将 positions 字典的引用赋给它。
                # 这样 StrategyEvaluator 就能“看到”主系统的实时持仓，而无需文件I/O。
                self.strategy_evaluator.positions_ref = self.positions
                logger.info("StrategyEvaluator已初始化并成功关联持仓引用。")
            elif config_updated:
                self.strategy_evaluator.config = self.config
            
            if self.risk_manager is None:
                self.risk_manager = RiskManager(self.config, self.quote_ctx)
                logger.info("RiskManager已初始化。")
            elif config_updated:
                self.risk_manager.config = self.config
                logger.info("RiskManager配置已同步更新。")
            
            if self.adaptive_stop_loss is None:
                self.adaptive_stop_loss = AdaptiveStopLoss(self.quote_ctx, self.config)
                logger.info("AdaptiveStopLoss已初始化。")
            elif config_updated:
                self.adaptive_stop_loss.config = self.config
                logger.info("AdaptiveStopLoss配置已同步更新。")
            
            self.stock_tech_advisor = None
            self.extended_hours_risk_engine = None
            logger.info("保守策略模式：LLM复核与夜盘风控模块不初始化。")

        except Exception as e:
            logger.error(f"热更新交易配置失败: {e}", exc_info=True)

    def _load_all_states(self):
        """统一加载所有持久化的状态。"""
        logger.info("正在从文件加载所有系统状态...")
        self._load_positions()
        self._load_pending_orders()
        self._load_sell_locks()
        self.pre_market_states = {}
        self._load_daily_pnl_state() # 启动时立即尝试加载状态
        self._load_blacklist()
        self.clean_second_tier_stocks = []
        self.clean_first_tier_stocks = []
        self.risky_second_tier_stocks = []

    def _start_all_monitors(self):
        """统一启动所有后台监控线程。"""
        self._start_main_monitor()
        logger.info("保守策略模式：只启动持仓/订单监控线程，停用卖出信号、待买入DB和盘前盘后线程。")
    
    def _reconcile_positions_with_broker(self, force: bool = False):
        """
        与券商的实际持仓进行核对与同步 (Reconciliation - The "Unbreakable" V4.0)
        
        Review 修正记录:
        1. 修复【空仓死锁】Bug：正确区分 API 故障与真实空仓。
        2. 捍卫【本地为魂】：存量持仓绝不覆盖本地成本价。
        3. 修复【新股致盲】：新增持仓时，若取不到现价，强制使用成本价兜底。
        4. 优化【时段限制】：允许盘前盘后进行强制或初始化同步。
        """
        # logger.info("保守策略模式：券商持仓自动同步已禁用。")
        # return

        now_ts = time.time()
        # 节流检查：如果不是强制执行(force=True)，且未过冷却期，直接跳过
        if not force and (now_ts - self.last_reconcile_timestamp < self.reconcile_cooldown_seconds):
            # logger.debug("持仓核对处于冷却期，跳过...")
            return
        
        logger.info(f"启动与券商的持仓核对 (Force={force})...")
        discrepancy_messages = []
        
        try:
            # --- 步骤 1: 获取外部数据 (Source of Truth) ---
            # 如果是强制同步，或者美股开盘，或者刚启动(这里简单用force代理)，都应该跑
            if not is_us_market_open() and not force:
                # logger.debug("非交易时段，跳过自动同步。")
                return

            # 调用 DataProvider
            broker_positions = self.data_provider.get_stock_positions(MarketType.US)

            # 🛑 1：空仓死锁防御
            # 如果 broker_positions 是 None，说明 API 崩了 -> 也就是真正的 P1 告警
            if broker_positions is None:
                error_msg = "P1级告警：DataProvider 返回 None，疑似接口通信失败，拒绝执行同步！"
                logger.critical(error_msg)
                # self._send_critical_alert(error_msg)
                return
            
            # 如果 broker_positions 是 [] (空列表)，说明真的空仓了，允许继续执行（以便删除本地持仓）
            # 但为了防止瞬间暴毙，我们可以加个日志
            with self.position_lock:
                local_count = len(self.positions)
            
            if len(broker_positions) == 0 and local_count > 0:
                logger.warning(f"⚠️ 注意：券商返回空持仓，而本地有 {local_count} 个持仓。系统将执行拒绝清仓同步。")
                return

            # 构建映射表
            actual_holdings = {}
            for actual_pos in broker_positions:
                if actual_pos.get('quantity', 0) > 0:
                    normalized_symbol = normalize_symbol(actual_pos['symbol'])
                    actual_holdings[normalized_symbol] = actual_pos

            # --- 步骤 2: 无菌室手术 ---
            with self.position_lock:
                local_symbols = set(self.positions.keys())
                actual_symbols = set(actual_holdings.keys())
                
                # 剔除正在路上的订单 (Pending Orders)
                with self.pending_orders_lock:
                    pending_symbols = set(self.pending_orders.keys())
                
                # 集合运算
                symbols_to_remove = local_symbols - actual_symbols
                symbols_to_add = actual_symbols - local_symbols - pending_symbols
                symbols_to_update = local_symbols.intersection(actual_symbols)

                # --- A. 执行删除 ---
                for symbol in symbols_to_remove:
                    try:
                        pos = self.positions.get(symbol)
                        if not pos: continue
                        if pos:
                            # 🛡️ 保护期检查
                            is_protected = False
                            holding_duration_minutes = pos.get_minutes_since_first_buy()
                            if holding_duration_minutes < 5:
                                is_protected = True
                            if is_protected:
                                logger.warning(f"🛡️ [保护盾生效] 本地持仓 {symbol} 在券商端消失，但因建立时间不足15分钟，强制保留！")
                                continue

                            msg = f"场景A: 本地持仓 {symbol} 在券商端消失，执行归档删除。"
                            logger.warning(msg)
                            discrepancy_messages.append(msg)
                            
                            # 归档
                            self._archive_completed_trade(
                                position=pos,
                                exit_price=pos.avg_cost, # 外部平仓无成交价，用成本价兜底
                                exit_quantity=pos.total_quantity,
                                exit_reason="BrokerSync_Missing"
                            )
                            # 删除
                            del self.positions[symbol]

                        # 状态机重置
                        if symbol in self.pre_market_states:
                            state = self.pre_market_states[symbol]
                            if state.get('status') == 'BOUGHT':
                                logger.info(f"[{symbol}] 状态机重置: BOUGHT -> WATCHING")
                                state['status'] = 'WATCHING'
                                state['last_update_ts'] = time.time()
                                self._save_pre_market_states()

                    except Exception as e:
                        logger.error(f"移除仓位 {symbol} 失败: {e}", exc_info=True)

                # --- B. 执行新增 (Handle "Alien" Positions) ---
                for symbol in symbols_to_add:
                    # Double Check Pending (极致防御)
                    with self.pending_orders_lock:
                        if symbol in self.pending_orders: continue
                    
                    actual_pos = actual_holdings[symbol]
                    actual_qty = int(actual_pos['quantity'])
                    actual_cost = float(actual_pos['cost_price'])
                    
                    msg = f"场景B: 发现外部新持仓 {symbol} (Qty:{actual_qty}, Cost:{actual_cost})"
                    logger.warning(msg)
                    discrepancy_messages.append(msg)
                    
                    try:
                        market = get_market_type(symbol)
                        current_price = self.get_current_price(symbol)

                        # 如果拿不到实时价格，暂时用成本价占位，但必须记录警告
                        if current_price and current_price > 0:
                            current_price_for_init = current_price
                        else:
                            logger.warning(f"⚠️ 同步 {symbol} 时无法获取实时行情，暂时使用成本价 {actual_cost} 初始化风控参数（存在风险）！")
                            current_price_for_init = actual_cost# 如果拿不到实时价格，暂时用成本价占位，但必须记录警告
                        
                        stop_loss_price = self.adaptive_stop_loss.calculate_stop_loss(symbol, current_price_for_init, 'long')
                            
                        # 对于“空降”的仓位，我们完全信任券商的成本和数量作为初始状态
                        new_pos = Position(
                            symbol=symbol,
                            market=market,
                            initial_price=actual_cost,
                            total_quantity=actual_qty,
                            avg_cost=actual_cost,
                            initial_scout_price=actual_cost,
                            overall_phase=PositionOverallPhase.RUNNING,
                            triggering_strategy="BrokerSync",
                            strategy_class_name="Manual/External"
                        )
                        new_pos.initial_risk_per_share = actual_cost * self.config.stop_loss_ratio
                        new_pos.initial_stop_loss_price = stop_loss_price
                        new_pos.add_purchase_record(PurchaseActionType.INITIAL_SCOUT, actual_cost, actual_qty, actual_cost * actual_qty)
                        
                        self.positions[symbol] = new_pos
                        
                    except Exception as e:
                        logger.error(f"新增仓位 {symbol} 失败: {e}", exc_info=True)

                # --- C. 执行更新---
                for symbol in symbols_to_update:
                    local_pos = self.positions[symbol]
                    actual_pos = actual_holdings[symbol]
                    actual_qty = int(actual_pos['quantity'])
                    actual_cost = float(actual_pos['cost_price'])
                    
                    if local_pos.total_quantity != actual_qty:
                        msg = f"场景C: 同步 {symbol} 数量: 本地({local_pos.total_quantity}) -> 券商({actual_qty})"
                        logger.warning(msg)
                        discrepancy_messages.append(msg)
                        local_pos.total_quantity = actual_qty
                    
                    avg_cost_diff = abs(local_pos.avg_cost - actual_cost)
                    if  avg_cost_diff > 0.1:

                        # ==============================================================================
                        # 🔥 数据驱动判断 (覆盖 main_force_add/混合交易)
                        # ==============================================================================
                        
                        # --- 1. 提取关键数据真相 ---
                        # 获取首仓数量，用于判断是否发生过加仓
                        initial_scout_records = local_pos.phase_records.get('initial_scout', [])
                        initial_scout_qty = sum(r.get('quantity', 0) for r in initial_scout_records) if initial_scout_records else 0
                        
                        total_buy_value, total_buy_qty = 0.0, 0
                        # last_buy_ts = None
                        for records in local_pos.phase_records.values():
                            if isinstance(records, list):
                                for r in records:
                                    # ts = r.get('timestamp', '')
                                    # if ts > last_buy_ts: last_buy_ts = ts
                                    # 累加用于本地重算
                                    p, q = r.get('price', 0), r.get('quantity', 0)
                                    if p > 0 and q > 0:
                                        total_buy_value += p * q
                                        total_buy_qty += q
                        
                        # 逻辑：如果 总买入量 > 首仓量，说明必然发生过加仓 (无论 key 是 main_force_add 还是 dip_add)
                        has_add_action = (total_buy_qty > initial_scout_qty) or (local_pos.dip_adds_done > 0)
                        
                        # 卖出判断：本地记录 OR 交易日志
                        has_sell_record = len(local_pos.sell_records) > 0  # 严格：有实际卖出记录
                        has_intraday_activity = symbol in self.intraday_trade_history  # 宽松：今日有交易痕迹
                        has_sell_signal = has_sell_record or has_intraday_activity     # 合并信号
                        
                        # 混合交易判断：必须严格依赖 sell_records 的时间戳 (防止 intraday_history 误判方向)
                        last_sell_ts = local_pos.sell_records[-1].get('timestamp', '') if has_sell_record else ''
                        is_mixed_trade = False
                        if has_sell_record and last_sell_ts:
                            # 检查是否有买入发生在卖出之后
                            for records in local_pos.phase_records.values():
                                if isinstance(records, list):
                                    for r in records:
                                        if r.get('timestamp', '') > last_sell_ts:
                                            is_mixed_trade = True
                                            break
                                if is_mixed_trade: break

                        # --- 2. 场景化同步策略 ---
                        is_clean_first = (not has_sell_record and not has_add_action and len(local_pos.phase_records) <= 1)
                        is_add_only = (not has_sell_record and has_add_action)

                        # 本地重算成本 (用于校验券商数据合理性)
                        local_calculated_cost = round(total_buy_value / total_buy_qty, 3) if total_buy_qty > 0 else 0.0
                        
                        if is_clean_first:
                            # ✅ 干净首仓：无条件信任券商
                            local_pos.avg_cost = actual_cost
                            local_pos.initial_price = actual_cost
                            local_pos.initial_scout_price = actual_cost
                            logger.info(f"[{symbol}] 首仓同步券商成本 {actual_cost:.3f}")

                        elif is_mixed_trade:
                            # 🔄 混合交易 (先卖后买)：高风险，校验后同步
                            if actual_cost > 0 and local_calculated_cost > 0:
                                pct_diff = abs(actual_cost - local_calculated_cost) / local_calculated_cost
                                if pct_diff < 0.15:  # 允许 15% 偏差 (混合交易计算复杂)
                                    local_pos.avg_cost = actual_cost
                                    logger.warning(f"[{symbol}] 混合交易同步券商成本 {actual_cost:.3f} (本地重算{local_calculated_cost:.3f})")
                                else:
                                    logger.error(f"[{symbol}] 混合交易成本分歧过大！券商{actual_cost:.3f} vs 本地{local_calculated_cost:.3f}，保持本地")
                            else:
                                local_pos.avg_cost = actual_cost

                        elif is_add_only:
                            # ➕ 纯加仓场景 (覆盖 main_force_add)：本地重算 vs 券商，取最优
                            if abs(actual_cost - local_calculated_cost) <= 0.5:
                                local_pos.avg_cost = actual_cost  # 差异小，信任券商真实扣款
                                logger.debug(f"[{symbol}] 加仓成本校准：本地重算{local_calculated_cost:.3f} ≈ 券商{actual_cost:.3f}，采用券商")
                            else:
                                local_pos.avg_cost = local_calculated_cost  # 差异大，信任本地逻辑自洽
                                logger.warning(f"[{symbol}] 加仓成本分歧：本地重算{local_calculated_cost:.3f} ≠ 券商{actual_cost:.3f}，采用本地重算")

                        elif has_sell_signal:
                            # 💰 卖出信号场景 (含 intraday_history 兜底)
                            # 注意：如果是纯 intraday_history 触发 (无 sell_records)，则不执行移动成本同步，防止误判
                            if has_sell_record and actual_cost > 0:
                                local_pos.avg_cost = actual_cost
                                logger.info(f"[{symbol}] 卖出后剩余持仓，同步券商移动成本 {actual_cost:.3f}")
                            elif has_intraday_activity and not has_sell_record:
                                # 仅有交易日志但无卖出记录，可能是纯买入，保持谨慎
                                logger.debug(f"[{symbol}] 仅有交易日志无卖出记录，保持本地成本 {local_pos.avg_cost:.3f}")
                            else:
                                logger.error(f"[{symbol}] 卖出场景券商成本为 0，异常！保持本地 {local_pos.avg_cost:.3f}")

                        else:
                            # 🛡️ 兜底保护
                            if avg_cost_diff <= 0.5:
                                logger.info(f"[{symbol}] 非首仓小差异，坚守本地 {local_pos.avg_cost:.3f}")
                            elif has_intraday_activity:
                                if random.random() > 0.90:
                                    logger.warning(f"[{symbol}] 【灵魂保护】大差异 + 当日交易，拒绝券商 {actual_cost:.3f}")
            
            # --- 步骤 3: 手术完成，持久化状态 ---
            self.last_reconcile_timestamp = now_ts
            # 只有在锁内的所有操作都成功后，才保存状态
            self._save_positions()
            
            if discrepancy_messages:
                summary_message = "P2级告警: 本轮仓位校对发现并处理了以下差异:\n" + "\n".join(discrepancy_messages)
                logger.warning(summary_message)
                # self.notification_manager.send_critical_alert(summary_message) # 可选：发送通知

            logger.info(" 持仓核对与同步完成。")

        except Exception as e:
            error_msg = f"P1级告警: 与券商同步持仓时发生致命错误，系统状态可能已不一致! 错误: {e}"
            logger.critical(error_msg, exc_info=True)
            self.notification_manager.send_critical_alert(error_msg)

    def shutdown(self):
        """安全地关闭所有后台线程和系统资源"""
        logger.info("正在关闭交易系统...")
        # 第一步：停止所有监控线程
        self.stop_main_monitor.set()
        self.stop_sell_monitor.set()

        # 第二步：优雅关闭线程池
        if hasattr(self, 'task_executor'):
            self.task_executor.shutdown(wait=True)
            logger.info("任务执行器已关闭")

        # 第三步：关闭策略评估器
        if self.strategy_evaluator:
            self.strategy_evaluator.shutdown()

        # 第四步：等待线程终止（带超时）
        threads_to_join = [
        (self.main_monitor_thread, "主监控线程"),
        (self.sell_signal_monitor_thread, "卖出监控线程"), 
        (self.pending_signal_monitor_thread, "待买入监控线程")
        ]
        
        for thread, name in threads_to_join:
            if thread and thread.is_alive():
                thread.join(timeout=10)
                if thread.is_alive():
                    logger.warning(f"{name} 未能及时终止")
                else:
                    logger.info(f"{name} 已终止")
        
        # if self.main_monitor_thread and self.main_monitor_thread.is_alive(): self.main_monitor_thread.join(timeout=10)
        # if self.sell_signal_monitor_thread and self.sell_signal_monitor_thread.is_alive(): self.sell_signal_monitor_thread.join(timeout=10)
        # if self.option_monitor_thread and self.option_monitor_thread.is_alive(): self.option_monitor_thread.join(timeout=10)

        
        # 第五步：保存最终状态
        self._save_all_states()
        logger.info("交易系统已成功关闭。")

    # ==============================================================================
    # II. 主循环与信号生成 (Main Loop & Signal Generation)
    # ==============================================================================
    def run_strategy_loop(self):
        """
        保守策略主循环。
        直接读取 data/us_stock.json 作为股票池，用日线数据验证买入条件。
        """
        logger.info("已启动保守策略主扫描循环...")

        while True:
            try:
                loop_config = self.config
                if not loop_config:
                    logger.warning("配置尚未加载，主循环等待...")
                    time.sleep(5)
                    continue

                with self.notification_lock:
                    self.notified_this_cycle.clear()

                self._check_and_perform_updates()

                if self._is_account_daily_loss_limit_hit():
                    logger.critical("账户当日总亏损达到 3%，触发一键清仓。")
                    self._liquidate_all_positions("账户当日总亏损达到3%")
                    time.sleep(loop_config.check_interval)
                    continue

                if self._is_account_monthly_loss_limit_hit():
                    logger.critical("当月累计账户总亏损达到 6%，触发一键清仓并暂停当月交易。")
                    self._liquidate_all_positions("当月累计账户总亏损达到6%")
                    time.sleep(loop_config.check_interval)
                    continue

                if self._is_monthly_loss_pause_active():
                    logger.warning("当月累计亏损风控已触发，本月停止新开仓。")
                    time.sleep(loop_config.check_interval)
                    continue
                
                if not is_us_market_open(): continue

                symbols = self._load_conservative_stock_pool()
                if not symbols:
                    logger.warning("保守策略股票池为空，等待下一轮。")
                    time.sleep(loop_config.check_interval)
                    continue

                for symbol in symbols:
                    print(f'symbol:{symbol}')
                    if self.stop_main_monitor.is_set():
                        break

                    with self.position_lock, self.pending_orders_lock:
                        if symbol in self.positions or symbol in self.pending_orders:
                            continue

                    passed, reason, candidate = self._verify_buy_signal_viability(symbol)
                    if not passed:
                        logger.debug(f"[{symbol}] 买入条件未通过: {reason}")
                        print(f"[{symbol}] 买入条件未通过: {reason}")
                        continue

                    logger.warning(f"[{symbol}] 保守策略买入信号通过: {reason}")
                    self.process_buy_signal(candidate)
                    time.sleep(0.2)

                time.sleep(loop_config.check_interval)
                
            except Exception as e:
                error_msg = f"P2级告警: 策略扫描主循环遇到严重错误，已自动恢复: {e}"
                logger.error(error_msg, exc_info=True)
                self.notification_manager.send_critical_alert(error_msg)
                time.sleep(30)  # 发生异常后，长一点时间休眠，防止快速连续失败

    def _load_conservative_stock_pool(self) -> List[str]:
        """读取 data/us_stock.json，返回保守策略可扫描的美股代码。"""
        try:
            with open(self.conservative_stock_pool_file, 'r', encoding='utf-8') as f:
                raw_data = json.load(f)
        except Exception as e:
            logger.error(f"读取保守策略股票池失败: {e}", exc_info=True)
            return []

        if isinstance(raw_data, dict):
            raw_items = raw_data.items()
        elif isinstance(raw_data, list):
            raw_items = []
            for item in raw_data:
                if isinstance(item, dict):
                    raw_items.append((item.get('symbol'), item.get('name') or item.get('name_cn') or item.get('name_en') or ''))
                else:
                    raw_items.append((item, ''))
        else:
            logger.error(f"股票池格式不支持: {type(raw_data)}")
            return []

        leverage_tokens = ('2倍', '3倍', '三倍', '4倍', '四倍', '2x', '3x', '4x', 'leveraged', 'inverse', '做空')
        symbols: List[str] = []
        skipped_leverage = 0

        for raw_symbol, raw_name in raw_items:
            if not raw_symbol:
                continue
            symbol = normalize_symbol(str(raw_symbol).strip())
            if '.' not in symbol:
                symbol = f"{symbol}.US"
            if not symbol.endswith('.US'):
                continue

            desc = f"{raw_symbol} {raw_name}".lower().replace(' ', '').replace('\t', '')
            if any(token in desc for token in leverage_tokens):
                skipped_leverage += 1
                continue

            if symbol not in symbols:
                symbols.append(symbol)

        if skipped_leverage:
            logger.info(f"保守策略跳过 {skipped_leverage} 个杠杆/反向类标的。")
        return symbols

    def _get_total_stock_exposure_hkd(self, include_pending: bool = True) -> float:
        """按成本估算当前股票持仓占用资金，统一折算为 HKD。"""
        exchange_rate = getattr(self.config, 'exchange_rate_usd_to_hkd', 7.8)
        exposure_hkd = 0.0

        with self.position_lock:
            positions_snapshot = list(self.positions.values())

        for position in positions_snapshot:
            try:
                if not self._is_pure_stock(position.symbol) or position.total_quantity <= 0:
                    continue
                mark_price = self.get_current_price(position.symbol)
                if mark_price is None or mark_price <= 0:
                    mark_price = position.get_avg_cost(self.config) if self.config else position.avg_cost
                value = max(0.0, float(mark_price) * float(position.total_quantity))
                if position.market == MarketType.US:
                    value *= exchange_rate
                exposure_hkd += value
            except Exception as e:
                logger.warning(f"[{getattr(position, 'symbol', '?')}] 估算持仓资金占用失败: {e}")

        if include_pending:
            with self.pending_orders_lock:
                pending_snapshot = list(self.pending_orders.values())
            for pending in pending_snapshot:
                try:
                    plan_info = pending.get("plan_info", {}) if isinstance(pending, dict) else {}
                    exposure_hkd += float(plan_info.get("estimated_trade_value_hkd", 0.0) or 0.0)
                except Exception:
                    continue

        return exposure_hkd

    def _ensure_loss_baselines(self) -> float:
        """确保日内和月度账户净值基准存在，返回当前净值(HKD)。"""
        us_tz = pytz.timezone(MARKET_TRADING_HOURS["US"]["timezone"])
        today_str = str(datetime.now(us_tz).date())
        month_key = today_str[:7]
        current_equity_hkd = self.get_net_equity_value(strict=True)

        if current_equity_hkd <= 0:
            return 0.0

        changed = False
        if self.current_trading_date_us != today_str or self.daily_equity_baseline_hkd <= 0:
            self.current_trading_date_us = today_str
            self.daily_equity_baseline_hkd = current_equity_hkd
            self.daily_loss_freeze_date = None
            changed = True

        if self.monthly_equity_baseline_key != month_key or self.monthly_equity_baseline_hkd <= 0:
            self.monthly_equity_baseline_key = month_key
            self.monthly_equity_baseline_hkd = current_equity_hkd
            if self.conservative_paused_month != month_key:
                self.conservative_paused_month = None
            changed = True

        if changed:
            self._save_daily_pnl_state()

        return current_equity_hkd

    def _is_account_daily_loss_limit_hit(self) -> bool:
        current_equity_hkd = self._ensure_loss_baselines()
        if current_equity_hkd <= 0 or self.daily_equity_baseline_hkd <= 0:
            return False

        loss_ratio = (self.daily_equity_baseline_hkd - current_equity_hkd) / self.daily_equity_baseline_hkd
        if loss_ratio >= 0.03:
            us_tz = pytz.timezone(MARKET_TRADING_HOURS["US"]["timezone"])
            self.daily_loss_freeze_date = str(datetime.now(us_tz).date())
            self._save_daily_pnl_state()
            logger.critical(
                f"账户日内回撤 {loss_ratio:.2%} >= 3.00%，"
                f"基准={self.daily_equity_baseline_hkd:.2f} HKD, 当前={current_equity_hkd:.2f} HKD"
            )
            return True
        return False

    def _is_account_monthly_loss_limit_hit(self) -> bool:
        current_equity_hkd = self._ensure_loss_baselines()
        if current_equity_hkd <= 0 or self.monthly_equity_baseline_hkd <= 0:
            return False

        loss_ratio = (self.monthly_equity_baseline_hkd - current_equity_hkd) / self.monthly_equity_baseline_hkd
        if loss_ratio >= 0.06:
            us_tz = pytz.timezone(MARKET_TRADING_HOURS["US"]["timezone"])
            self.conservative_paused_month = datetime.now(us_tz).strftime('%Y-%m')
            self._save_daily_pnl_state()
            logger.critical(
                f"账户月度回撤 {loss_ratio:.2%} >= 6.00%，"
                f"基准={self.monthly_equity_baseline_hkd:.2f} HKD, 当前={current_equity_hkd:.2f} HKD"
            )
            return True
        return False

    def _is_monthly_loss_pause_active(self) -> bool:
        if not self.conservative_paused_month:
            return False
        us_tz = pytz.timezone(MARKET_TRADING_HOURS["US"]["timezone"])
        current_month = datetime.now(us_tz).strftime('%Y-%m')
        if self.conservative_paused_month == current_month:
            return True
        self.conservative_paused_month = None
        self._save_daily_pnl_state()
        return False

    def _liquidate_all_positions(self, reason: str):
        """账户级风控触发时，全仓提交卖出。"""
        with self.position_lock:
            symbols = [symbol for symbol, pos in self.positions.items() if pos.total_quantity > 0]

        if not symbols:
            logger.info(f"账户级清仓触发但当前无持仓: {reason}")
            return

        for symbol in symbols:
            self._execute_full_sell(symbol, reason)

    def _verify_buy_signal_viability(self, symbol: str) -> Tuple[bool, str, dict]:
        """
        保守策略买入准入：
        1. 收盘价从下向上突破 20 日均线。
        2. 最新成交量至少达到前 20 日均量的 2 倍 or 最新成交量至少达到昨天量的 1.5 倍。
        3. RSI(14) 位于 50-70。
        """
        try:
            df = get_klines_data(self.quote_ctx, symbol, count=60, period=Period.Day, adjust_type=AdjustType.NoAdjust)
            if df is None or len(df) < 35:
                return False, "日线数据不足，无法计算 MA20/RSI14", {}

            if 'timestamp' in df.columns:
                df = df.sort_values('timestamp')
            df = df.copy()
            df['close'] = pd.to_numeric(df['close'], errors='coerce')
            df['volume'] = pd.to_numeric(df['volume'], errors='coerce')
            df = df.dropna(subset=['close', 'volume'])

            if len(df) < 35:
                return False, "有效日线数据不足", {}

            close = df['close']
            volume = df['volume']
            ma20 = close.rolling(window=20).mean()
            delta = close.diff()
            gain = delta.clip(lower=0).rolling(window=14).mean()
            loss = (-delta.clip(upper=0)).rolling(window=14).mean()
            rs = gain / loss.replace(0, np.nan)
            rsi = 100 - (100 / (1 + rs))

            latest_close = float(close.iloc[-1])
            previous_close = float(close.iloc[-2])
            latest_ma20 = float(ma20.iloc[-1])
            previous_ma20 = float(ma20.iloc[-2])
            latest_volume = float(volume.iloc[-1])
            avg_volume_20 = float(volume.iloc[-21:-1].mean())
            yesterday_volume = float(volume.iloc[-2])
            latest_rsi = float(rsi.iloc[-1])

            if any(pd.isna(x) for x in [latest_close, previous_close, latest_ma20, previous_ma20, avg_volume_20, latest_rsi]):
                return False, "指标存在空值", {}

            crossed_up = previous_close <= previous_ma20 and latest_close > latest_ma20
            if not crossed_up:
                return False, f"未形成收盘价突破 MA20: 前收={previous_close:.3f}/前MA20={previous_ma20:.3f}, 收={latest_close:.3f}/MA20={latest_ma20:.3f}", {}

            volume_vs_avg20 = latest_volume / avg_volume_20 if avg_volume_20 > 0 else 0.0
            volume_vs_yesterday = latest_volume / yesterday_volume if yesterday_volume > 0 else 0.0
            yesterday_multiplier = 1.5 
            volume_ok_yesterday = volume_vs_yesterday >= yesterday_multiplier
            avg20_multiplier = 2.0
            volume_ok_avg20 = volume_vs_avg20 >= avg20_multiplier

            # if volume_ratio < 2.0:
            #     return False, f"成交量未达到放量1倍以上: {volume_ratio:.2f}x < 2.00x", {}

            if not (volume_ok_yesterday or volume_ok_avg20):
                return False,f"未满足放量条件: 今日/昨日={volume_vs_yesterday:.2f}x<{yesterday_multiplier:.2f}x,今日/20日均量={volume_vs_avg20:.2f}x<{avg20_multiplier:.2f}x "

            if not (50 <= latest_rsi <= 70):
                return False, f"RSI14 不在 50-70 区间: {latest_rsi:.2f}", {}

            metrics = {
                "latest_close": round(latest_close, 4),
                "previous_close": round(previous_close, 4),
                "latest_ma20": round(latest_ma20, 4),
                "previous_ma20": round(previous_ma20, 4),
                "latest_volume": latest_volume,
                "avg_volume_20": avg_volume_20,
                "volume_vs_yesterday": round(volume_vs_yesterday, 4),
                "volume_vs_avg20": round(volume_vs_avg20, 4),
                "rsi14": round(latest_rsi, 4),
            }
            reason = f"收盘突破MA20，成交量{volume_vs_avg20:.2f}x，RSI14={latest_rsi:.2f}"
            candidate = {
                "symbol": symbol,
                "trigger_price": latest_close,
                "strategy_name": "ConservativeMA20Breakout",
                "strategy_class_name": "ConservativeMA20Breakout",
                "strategy_params": {
                    "entry_signal": metrics,
                    "conservative_exit_state": {
                        "highest_price": latest_close,
                        "stage_10_taken": False,
                        "stage_15_taken": False,
                    }
                },
                "buy_percentage": 0.085,
                "final_confirmation": True,
                "reason": reason,
            }
            return True, reason, candidate
        except Exception as e:
            logger.error(f"[{symbol}] 保守策略买入准入检查失败: {e}", exc_info=True)
            return False, f"检查异常: {e}", {}

    def _sell_signal_monitor_loop(self):
        """
        独立的策略性卖出信号监控循环。
        此循环以低延迟持续检查持仓，发现并处理基于策略的卖出信号。
        """
        logger.info("保守策略模式：策略性卖出信号监控已禁用。")
        return

        logger.info("策略性卖出信号监控循环已开始...")
        while not self.stop_sell_monitor.is_set():
            try:
                loop_config = self.config
                if not loop_config or not is_any_market_open():
                    logger.debug("当前为非交易时间或配置未加载，卖出信号监控器休眠中...")
                    time.sleep(60)
                    continue

                with self.position_lock:
                    self.strategy_evaluator.positions_ref = self.positions
                    # 严格筛选：只取当前处于交易中(Market Open)的股票
                    self.strategy_evaluator.positions_ref = self.positions 
                    active_symbols = [
                        s for s in self.positions.keys() 
                        if is_any_market_open(s) and self._is_pure_stock(s)
                    ]
                
                # 如果工作范围内没有任何股票，则本轮循环无事可做，直接休眠。
                if not active_symbols:
                    time.sleep(loop_config.check_interval)
                    continue
                
                sell_candidates = self.strategy_evaluator.find_sell_signals(active_symbols)
                for candidate in sell_candidates:
                    symbol_to_sell = candidate['symbol']
                    if not is_any_market_open(symbol_to_sell): continue

                    # 在执行卖出前，获取当前的时间窗口状态
                    # current_status = get_trading_window_status(symbol_to_sell)
                    
                    # 定义一个“有利卖出窗口”的集合
                    # favorable_sell_windows = self.config.favorable_sell_windows
                    
                    strategy_name = candidate.get('strategy_name','')
                    # 如果当前不处于有利的卖出窗口，则跳过本次执行，等待下一轮检查
                    # if current_status not in favorable_sell_windows:
                    #     if (strategy_name not in ['双核大脑·自适应利润保护','自适盈利润保护(日内交易)','IntradayDynamicGuardian','自适盈利润保护(无死区版)','日内收盘前(10min)清仓','其他盈利润保护(日内交易)','次日智能卖出','大盘状态防御策略']):
                    #         logger.warning(f"[{symbol_to_sell}] 策略卖出信号已触发，但当前时间窗口 ({current_status.name}) 不利于卖出，延迟执行。")
                    #         # send_email(subject=f"[{symbol_to_sell}] 策略卖出信号已触发,但被延迟执行。",content=f"[{symbol_to_sell}] 策略卖出信号已触发，但当前时间窗口 ({current_status.name}) 不利于卖出，延迟执行。")
                    #         continue

                    # 如果LLM同意卖出
                    llm_reason = 'LLM检查被全局禁用'
                    # if (strategy_name not in ['双核大脑·自适应利润保护','自适盈利润保护(日内交易)','日内收盘前(10min)清仓','其他盈利润保护(日内交易)']):
                    #     llm_approved, llm_reason = self._get_llm_decision(candidate, 'sell')

                    #     if not llm_approved:
                    #         logger.warning(f"LLM复核建议不卖出 {symbol_to_sell}，忽略策略卖出信号。原因: {llm_reason}")
                    #         continue

                    reason = candidate['reason']

                    if not reason or not reason.strip():
                        strategy_name = candidate.get('strategy_name', 'UnknownStrategy')
                        reason = f"策略'{strategy_name}'触发，但未提供具体原因"
                        logger.warning(f"[{symbol_to_sell}] 发现上游策略 '{strategy_name}' 未提供有效卖出原因，已生成默认描述。")
                    
                    symbol_info = self.get_cached_stock_static_info(symbol_to_sell)
                    symbol_name = symbol_info.get('name_cn') or symbol_info.get('name_en')
                    percentage_to_sell = candidate.get('percentage', 0.5)
                    post_market_price = self.get_current_price(symbol_to_sell)

                    log_msg = f"-- 🚨 发现卖出信号 --\n股票代码: {symbol_to_sell}\n股票名称: {symbol_name}\n股票价格: {post_market_price}\n触发策略: {reason}\nLLM意见: 同意卖出 - {llm_reason}\n触发时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
                    logger.warning(log_msg)

                    # 使用独立的sell_logger记录结构化卖出数据
                    sell_logger.info(f"symbol:{symbol_to_sell},name:{symbol_name},price:{post_market_price},strategy_reason:{reason},llm_reason:{llm_reason}")
                    
                    if not loop_config.test_mode:
                        with self.position_lock:
                            position = self.positions.get(symbol_to_sell)
                            if position:
                                # if position.total_quantity<=self.config.min_residual_quantity: #持有10股以下，直接全部卖出了
                                #     percentage_to_sell = 1.0

                                # 网格豁免权】
                                # 如果处于防御模式，说明我们留着这些碎股是为了监控回补机会（作为Seed），绝对不能清仓。
                                # 除非触发了 ATR 2.0 死亡线（那是 check_stop_loss 管的事），或者是财报风控。
                                is_immune = position.is_defense_mode_active
                                is_dust = position.total_quantity <= self.config.min_residual_quantity
                                
                                if is_dust and not is_immune:
                                    percentage_to_sell = 1.0
                                    logger.info(f"[{symbol_to_sell}] 碎股清理触发 (非防御模式)。")
                                elif is_dust and is_immune:
                                    logger.info(f"[{symbol_to_sell}] 碎股清理被豁免 (防御模式激活中，保留火种)。")
                                
                                if hasattr(self.strategy_evaluator, 'earnings_manager') and self.strategy_evaluator.earnings_manager:
                                    # 变量重命名为 earnings_reason，避免污染主 reason
                                    is_danger, earnings_reason = self.strategy_evaluator.earnings_manager.is_in_danger_zone(symbol_to_sell)
                                    if is_danger:
                                        # 仅在真正触发财报风控时，才覆盖卖出原因和比例
                                        logger.critical(f"⚠️ 财报风控触发强制卖出: {symbol_to_sell} - {earnings_reason}")
                                        percentage_to_sell = 1.0
                                        # 明确将原因更新为财报相关
                                        reason = f"财报风控: {earnings_reason}"
                        
                        if self.process_sell_signal(symbol_to_sell, percentage=percentage_to_sell, reason=reason):
                                self.notification_manager.send_trade_execution(
                                    action='LIQUIDATE' if percentage_to_sell >= 1.0 else 'PARTIAL SELL',
                                    symbol=symbol_to_sell,
                                    quantity=int(position.total_quantity * percentage_to_sell),
                                    price=post_market_price,
                                    reason=f"{reason} (LLM确认)"
                                )
                    
                time.sleep(loop_config.check_interval)
            except Exception as e:
                logger.error(f"策略性卖出信号监控循环出错: {e}", exc_info=True)
                # 【告警集成】 8. 核心线程崩溃告警
                error_msg = f"P1级告警: 策略卖出监控线程崩溃! 系统部分功能可能已失效。错误: {e}"
                logger.error(error_msg, exc_info=True)
                self.notification_manager.send_critical_alert(error_msg)
                time.sleep(30)
        logger.info("策略性卖出信号监控循环已停止...")
    
    def _after_hours_monitor_loop(self):
        """
        【盘后智能清算系统 (AH-ICS) - 完美架构版】
        
        设计哲学：
        1. 全局联动：监控窗口参数化，杜绝硬编码 Bug。
        2. 0-60分钟 (全天候)：动态阶梯止盈，让利润奔跑。
        3. 5分钟 (抢收窗口)：开盘情绪释放，战术减仓 40%。
        4. 55-60分钟 (终极清场)：流动性枯竭前，战术减仓 60%。
        """
        logger.info(">>> 盘后智能清算线程已启动 (等待收盘信号)...")

        # --- [Configuration] 核心参数配置区 ---
        # 将常量提取到循环外，方便维护
        MAX_AH_WINDOW = 1110  # 盘后最大监控时长 (分钟),18*60+30
        BUFFER_MINUTES = 30 # 外层检查的冗余缓冲时间 (分钟)

        # [State] 线程局部的高点记录器 {symbol: max_price}
        ah_high_water_marks = {}
        # 全局决策缓存
        current_global_action = "HOLD_DIP" # 默认防守
        last_reset_date = None

        while not self.stop_sell_monitor.is_set():
            try:
                # --- 1. 环境检查与休眠控制 ---
                # 获取美股当前日期字符串 (解决时区问题)
                us_tz = pytz.timezone(MARKET_TRADING_HOURS["US"]["timezone"])
                current_date_str = datetime.now(us_tz).strftime('%Y%m%d')
                
                # 获取当前 Session 状态
                us_session = get_current_market_session(MarketType.US)

                # ==============================================================================
                # ▼▼▼ 盘中自动重置 (Re-arm Trigger) ▼▼▼
                # 逻辑：只要进入盘中，就把“上次复盘日期”抹掉。这意味着今天收盘后，是一个全新的机会。
                # 这完美解决了“午夜0点更新导致下午16点不执行”的死锁 Bug。
                # ==============================================================================
                if us_session == TradingSession.REGULAR_TRADING:
                    if last_reset_date is not None:
                        logger.info("🔔 [系统状态] 进入美股盘中，重置盘后决策状态，等待收盘触发...")
                        last_reset_date = None

                # 每天重置一次水位线记录，防止跨日污染
                if current_date_str != last_reset_date and us_session != TradingSession.REGULAR_TRADING:
                    ah_high_water_marks.clear()
                    self.pre_market_states.clear()

                    try:
                        logger.info(f"⚡ [盘后塔台] 日期变更或系统重启 ({last_reset_date} -> {current_date_str})，执行决策分析...")
                        decision = self.market_regime_engine.analyze_post_market_decision(MarketType.US)
                        current_global_action = decision.get('action', 'HOLD_DIP')
                        
                        if current_global_action != "HOLD_DIP":
                            logger.warning(f"⚡ [盘后塔台] 全局指令更新: {current_global_action} | 原因: {decision.get('reason')}")
                    except Exception as e:
                        logger.error(f"获取盘后决策失败: {e}")
                        # 失败时不更新 last_reset_date，保证下次循环重试
                        time.sleep(5)
                        continue


                    last_reset_date = current_date_str
                    logger.warning(f"盘后缓存已自动重置 (锚定日期: {last_reset_date})")

                # 动态计算宽限期
                # 只要市场处于 (收盘 + MAX_AH_WINDOW + 缓冲) 的时间内，线程就保持清醒
                # 例如：60 + 10 = 70分钟。这确保了 55-60 分钟的【终极清场】绝对能被执行到。
                if not is_any_market_in_grace_period(grace_minutes=MAX_AH_WINDOW + BUFFER_MINUTES) and us_session == TradingSession.MARKET_CLOSED:
                    # 如果不在宽限期，说明是深夜或盘中，长休眠节省资源
                    time.sleep(60)
                    continue

                # --- 2. 获取持仓快照 ---
                # 必须加锁复制，避免遍历时字典大小改变报错
                with self.position_lock:
                    # 过滤掉非股票资产（如现金、期货等，根据你的 _is_pure_stock 逻辑）
                    positions_snapshot = [
                        (s, p) for s, p in self.positions.items() if self._is_pure_stock(s)
                    ]
                
                if not positions_snapshot:
                    time.sleep(10)
                    continue
                # --- 3. 逐个股票执行微操 ---
                for symbol, position in positions_snapshot:
                    try:
                        # =======================================================
                        # [Step 0] 基础状态与数据准备
                        # =======================================================
                        # 0.1 物理锁：有挂单则跳过
                        if position.pending_sell_order_id:
                            continue

                        # 0.2 市场与时间计算
                        market = get_market_type(symbol)
                        if market==MarketType.HK: continue
                        market_info = MARKET_TRADING_HOURS.get(market.name)
                        if not market_info: continue

                        market_tz = pytz.timezone(market_info["timezone"])
                        now_local = datetime.now(market_tz)
                        close_dt = get_last_business_close_time(market_tz, market_info["sessions"], now_local)

                        # 计算距离收盘的分钟数 (T+N)
                        delta_seconds = (now_local - close_dt).total_seconds()
                        delta_minutes = delta_seconds / 60.0
                        
                        # 0.3 跨日/交易日过滤
                        # 逻辑：如果已经过了很久(例如6小时)，且市场又开盘了，说明是第二天了，跳过。
                        if delta_minutes > 360 and is_market_in_trading_hours(market):
                            continue

                        ##持仓时间
                        holding_duration_minutes = position.get_minutes_since_first_buy()
                        if holding_duration_minutes <12:
                            continue
                        
                        # 0.4 获取行情 (智能路由)
                        quote_data = get_smart_quote(self.quote_ctx,symbol)
                        if not quote_data:
                             try: quote_data = self.hs_data_provider.get_smart_quote(symbol)
                             except: pass
                        
                        if not quote_data:
                            logger.warning(f"[{symbol}] 无法获取有效行情，跳过。")
                            continue
                        
                        current_price = quote_data.get('last_price',0.0)
                        static_close_price = quote_data.get('prev_close_price',0.0)
                        
                        # 防御性检查：防止数据源返回 0 或 None 导致计算炸裂
                        if not current_price or current_price <= 0.01 or static_close_price <= 0.01:
                            continue

                        # --- 4. 策略执行核心 (The Smart Brain) ---
                        
                        # =======================================================
                        # [Step 1] 核心决策变量计算
                        # =======================================================
                        # 计算基础指标
                        cost_price = position.get_avg_cost(self.config)
                        # ROI: 相对成本的盈亏率
                        roi_pct = (current_price - cost_price) / cost_price if cost_price > 0 else 0
                        current_status = get_trading_window_status(symbol)
                        # 初始化决策标志
                        should_sell = False
                        sell_reason = ""
                        target_sell_ratio = 0.0 # 默认为0，安全第一

                        # --- 时间窗口判定 (Time Flags) ---
                        # 抢跑/躁动窗口
                        is_5min_rush = 4.0 <= delta_minutes <= 10.5
                        is_profit_run = 55 <= delta_minutes <= 60
                        
                        # 枚举状态判定
                        is_euro_rush = current_status == TradingWindowStatus.NIGHT_EURO_OPEN_RUSH
                        is_asia_open = current_status == TradingWindowStatus.NIGHT_ASIA_OPEN_HYSTERIA
                        is_midnight_pump = current_status == TradingWindowStatus.NIGHT_MIDNIGHT_PUMP
                        is_pre_open_impulse = current_status == TradingWindowStatus.PRE_MARKET_OPEN_IMPULSE
                        is_pre_market_marco_sneak = current_status == TradingWindowStatus.PRE_MARKET_MACRO_SNEAK
                        is_pre_market_marco_data = current_status == TradingWindowStatus.PRE_MARKET_MACRO_DATA
                        
                        # 风险窗口
                        is_pre_trap = current_status == TradingWindowStatus.PRE_MARKET_FINAL_TRAP
                        is_deadline = (MAX_AH_WINDOW - 5) <= delta_minutes <= MAX_AH_WINDOW

                        # =======================================================
                        # [Step 2] 盈亏双轨制决策 (Dual-Track Logic)
                        # =======================================================

                        # ==================================================================
                        # ▼▼▼ 盘后六宫格路由 (After-Hours Regime Router)▼▼▼
                        # 不再无脑卖出，按 "日内盈亏 × 盘后走势" 六宫格路由：
                        #   1) 盈 + 盘后涨  -> STRONG_HOLD   （完全持有 + 拉盘后追踪止损）
                        #   2) 盈 + 盘后平  -> FLAT_LOCK     （锁 40% 留 60% 过夜）
                        #   3) 盈 + 盘后跌  -> WEAK_DUMP     （卖 85% 留 15%）
                        #   4) 亏 + 盘后回收盘 -> LOSS_HOLD_ORB （持有博 ORB 翻身）
                        #   5) 亏 + 盘后继续跌 -> LOSS_DUMP   （沿用原周历逻辑）
                        #   6) 盘中被 Guardian 触发过 -> 剩余仓位只追踪不追卖
                        # ==================================================================
                        ah_cfg = getattr(self.config, 'after_hours_strong_hold_config', {}) or {}
                        ah_regime = 'NEUTRAL'
                        is_strong_hold = False
                        if ah_cfg.get('enabled', True) and static_close_price > 0:
                            rise_th = ah_cfg.get('rise_threshold_pct', 0.003)
                            fall_th = ah_cfg.get('fall_threshold_pct', 0.003)
                            ah_delta = (current_price - static_close_price) / static_close_price

                            # ---- 盘后 ROI 分层 map：决定"盈利单"是否有资格进入被动持有 ----
                            profit_tier = ah_cfg.get('profit_tier_map', {}) or {}
                            sh_min_roi = profit_tier.get('strong_hold_min_roi', 0.005)
                            fl_min_roi = profit_tier.get('flat_lock_min_roi',   0.003)

                            if roi_pct >= sh_min_roi:
                                # 真盈利：六宫格完整路由
                                if ah_delta >= rise_th:
                                    ah_regime = 'STRONG_HOLD'
                                elif ah_delta <= -fall_th:
                                    ah_regime = 'WEAK_DUMP'
                                else:
                                    ah_regime = 'FLAT_LOCK'
                            elif roi_pct >= fl_min_roi:
                                # 微利（0.3%~0.5%）：禁止 STRONG_HOLD，最多 FLAT_LOCK
                                if ah_delta <= -fall_th:
                                    ah_regime = 'WEAK_DUMP'
                                else:
                                    ah_regime = 'FLAT_LOCK'
                            elif roi_pct > 0:
                                # 鸡肋利润（<0.3%）：等同于"准亏损"，按弱势处理
                                ah_regime = 'WEAK_DUMP'
                            else:
                                # 亏损单：看是否反弹回收盘附近
                                proximity = ah_cfg.get('loss_rebound_proximity_pct', 0.008)
                                if abs(ah_delta) <= proximity:
                                    ah_regime = 'LOSS_HOLD_ORB'
                                else:
                                    ah_regime = 'LOSS_DUMP'

                            # 强势持有安全兜底：跌破成本 -1% 立刻降级
                            fail_safe = ah_cfg.get('strong_hold_max_loss_pct', -0.01)
                            if ah_regime == 'STRONG_HOLD' and roi_pct <= fail_safe:
                                ah_regime = 'FLAT_LOCK'
                            is_strong_hold = (ah_regime == 'STRONG_HOLD')

                            # 场景 6：盘中已被 Guardian 部分减仓的剩余仓位，只追踪不追卖
                            guardian_fired_today = False
                            try:
                                guardian_fired_today = (getattr(position, 'partial_sell_price', 0) or 0) > 0
                            except Exception:
                                pass

                            if ah_cfg.get('respect_guardian_fired_flag', True) and guardian_fired_today \
                                    and ah_regime in ('STRONG_HOLD', 'FLAT_LOCK'):
                                is_strong_hold = True  # 强制进入"只追踪不卖出"分支

                            # ---- STRONG_HOLD 分支：拉盘后追踪止损而非立即卖出 ----
                            if is_strong_hold:
                                # 更新盘后高水位
                                prev_high = ah_high_water_marks.get(symbol, current_price)
                                ah_high_water_marks[symbol] = max(prev_high, current_price)
                                hwm = ah_high_water_marks[symbol]

                                # 计算 盘后高点 − N × ATR 的追踪线
                                try:
                                    ah_atr = get_historical_atr(self.quote_ctx, symbol) or 0.0
                                except Exception:
                                    ah_atr = 0.0
                                atr_mult = ah_cfg.get('strong_hold_trailing_atr_multiplier', 2.0)
                                if ah_atr > 0:
                                    trailing_line = hwm - ah_atr * atr_mult
                                else:
                                    # ATR 不可用时用 1.5% 兜底
                                    trailing_line = hwm * (1 - 0.015)

                                # 追踪线不能低于成本（至少保本）
                                trailing_line = max(trailing_line, cost_price)

                                if current_price <= trailing_line:
                                    # 盘后追踪止损被击穿 → 大比例清
                                    target_sell_ratio = ah_cfg.get('weak_sell_ratio', 0.85)
                                    sell_reason = (f"盘后·强势持有追踪止损触发 | ROI:{roi_pct:.2%} | "
                                                   f"高点:{hwm:.3f}→现价:{current_price:.3f} | "
                                                   f"线:{trailing_line:.3f}")
                                    should_sell = True
                                else:
                                    # 仍在趋势内 → 本轮直接跳过所有卖出逻辑
                                    if now_local.second % 30 == 0:
                                        logger.info(f"🟢 [{symbol}] 盘后强势持有中 | ROI:{roi_pct:.2%} | "
                                                    f"盘后涨:{ah_delta:.2%} | 追踪线:{trailing_line:.3f}")
                                    # 直接跳到 Step 3 后的 执行块（should_sell=False）
                                    continue

                            # ---- FLAT_LOCK 分支：锁部分利润 + 低优先级触发常规逻辑 ----
                            elif ah_regime == 'FLAT_LOCK' and roi_pct > profit_tier.get('flat_lock_sell_min_roi', 0.005) \
                                    and not position.has_executed_action_today("AH_FLAT_LOCK"):
                                # 只在特定窗口才落袋（避免与机会窗口重复触发）
                                if is_5min_rush or is_profit_run or is_euro_rush or is_asia_open:
                                    target_sell_ratio = ah_cfg.get('flat_partial_sell_ratio', 0.4)
                                    sell_reason = (f"盘后·走平锁利[AH_FLAT_LOCK] | ROI:{roi_pct:.2%} | "
                                                   f"盘后Δ:{ah_delta:+.2%} | 减仓{target_sell_ratio:.0%}")
                                    should_sell = True

                            # ---- LOSS_HOLD_ORB 分支：亏损但反弹回收盘 → 持有博 ORB ----
                            elif ah_regime == 'LOSS_HOLD_ORB' and not is_deadline and not is_pre_trap:
                                # 非死线窗口直接跳过卖出逻辑
                                if now_local.second % 60 == 0:
                                    logger.info(f"🟡 [{symbol}] 盘后·亏损反弹持有 | ROI:{roi_pct:.2%} | "
                                                f"盘后Δ:{ah_delta:+.2%} (≤{proximity:.1%}) | 博次日 ORB")
                                continue
                            # 其他 regime (WEAK_DUMP / LOSS_DUMP) 继续走原有逻辑
                        # ==================================================================

                        # ----------------------------------------------------------------------
                        # [逻辑 A] 场景: 盘后开盘冲高 / 欧盘抢跑 / 盘前陷阱(盈利单)
                        # 特征：只在盈利时触发，且通常只触发一次
                        # 【优化】把"盈利门票"从硬编码 0 提升为可配阈值，消除微利噪声触发；
                        #        真正的卖出条件仍由下游 is_high_confirmed 技术 K 线确认把关。
                        # ----------------------------------------------------------------------
                        ah_min_profit_to_rush = float(
                            (getattr(self.config, 'after_hours_strong_hold_config', {}) or {})
                            .get('min_profit_to_rush', 0.005)
                        )
                        if roi_pct > ah_min_profit_to_rush:

                            # 1. 处理“只做一次”的机会性卖出 (抢收、欧盘、亚洲开盘躁动、夜真空陷阱)
                            # === 分支 1: 盘外机会 (只做盈利单) ===
                            if is_5min_rush or is_euro_rush or is_profit_run or is_asia_open or \
                                is_midnight_pump or is_pre_open_impulse or is_pre_market_marco_data or is_pre_market_marco_sneak:
                                action_keyword = "特殊窗口抢跑"
                                ratio_candidate = 0.8 # 默认减半
                                dedup_tag = "AH_RUSH" # 默认特征码
                                # 细化比例
                                if is_profit_run:
                                    action_keyword = "盈利抢跑" # 55-60分
                                    # ratio_candidate = 1.0      # 抢跑通常是为了落袋为安，全走
                                    dedup_tag = "AH_CLOSING_RUN"
                                    # 如果全局策略是激进做多，且利润丰厚(>2%)，留 20% 仓位过夜
                                    if current_global_action == "AGGRESSIVE_LONG" and roi_pct > 0.02:
                                        ratio_candidate = 0.8
                                    else:
                                        ratio_candidate = 1.0
                                elif is_euro_rush:
                                    action_keyword = "欧盘抢跑"  # 欧盘抢跑
                                    ratio_candidate = 0.8      # 留一半博美股
                                    dedup_tag = "EURO_OPEN_RUSH"
                                elif is_asia_open:
                                    action_keyword = "亚洲开盘躁动" # 亚洲开盘躁动
                                    ratio_candidate = 0.8      # 留一半博美股
                                    dedup_tag = "ASIA_OPEN_RUSH"
                                elif is_midnight_pump:
                                    action_keyword = "夜真空陷阱"  # 夜真空陷阱
                                    # 如果收益极高 (>5%)，说明是极其虚假的拉升，多卖点
                                    ratio_candidate = 1.0 if roi_pct > 0.05 else 0.8
                                    dedup_tag = "MIDNIGHT_PUMP"
                                elif is_pre_open_impulse:
                                    action_keyword = "盘前刚开脉冲" # 盘前刚开脉冲
                                    ratio_candidate = 0.8 if roi_pct > 0.05 else 0.7
                                    dedup_tag = "PRE_OPEN_IMPULSE"
                                else:
                                    action_keyword = "抢收"     # 收盘5分钟
                                    ratio_candidate = 0.8      # 战术减仓
                                    dedup_tag = "AH_5MIN_RUSH"

                                # === [核心查重拦截 (Signature Lock) ===
                                if not position.has_executed_action_today(dedup_tag):
                                    # 4. 盈利检查 (共同条件)
                                    # 抢跑逻辑额外要求：大于昨收 (防止低开后的假反弹)
                                    is_price_valid = current_price > cost_price
                                    # is_price_valid = is_price_valid and (current_price > static_close_price)
                                    k_mins_check = self.config.tactical_k_mins_map.get('SCALP_EXIT', 3)
                                    # 死线临头，回撤阈值放宽(0.2%)；诱多陷阱，要求严格(0.5%)
                                    rebound_pct_threshold = 0.002 if is_deadline else 0.005
                                    
                                    is_high_confirmed = check_extended_hours_tactical_exit_signal(
                                        self.hs_data_provider, symbol, k_mins_check, rebound_pct_threshold
                                    )
                                    
                                    if (is_price_valid) and is_high_confirmed: #or roi_pct>0.010
                                        sell_reason = (
                                            f"盘外机会[{action_keyword}][{dedup_tag}] | "
                                            f"T+{delta_minutes:.1f}m | "
                                            f"盈率:{roi_pct:.2%} | "
                                            f"执行:锁定利润"
                                        )
                                        logger.warning(sell_reason)
                                        target_sell_ratio = ratio_candidate
                                        should_sell = True
                                    
                                    # 场景 X: 崩盘保命 (FLEE_CRASH)
                                    elif current_global_action == "FLEE_CRASH" and is_high_confirmed:
                                        # 盈利单也要跑，防止补跌。卖出比例提高。
                                        target_sell_ratio = 1.0 if roi_pct < 0.02 else 0.8
                                        sell_reason = f"全局熔断(FLEE_CRASH)[{dedup_tag}]: 保护利润 (ROI {roi_pct:.2%})"
                                        should_sell = True

                                    # 场景 Y: 积极止盈 (LOCK_PROFIT)
                                    elif current_global_action == "LOCK_PROFIT" and is_high_confirmed:
                                        # 只要有微利(>0.5%)，不需要等待特定窗口，直接激活卖出
                                        if roi_pct > 0.005:
                                            # 提高卖出比例 (0.5 -> 0.8)
                                            target_sell_ratio = 0.8
                                            sell_reason = f"全局积极止盈(LOCK_PROFIT)[{dedup_tag}]: 顺势落袋 (ROI {roi_pct:.2%})"
                                            should_sell = True

                            # ----------------------------------------------------------------------
                            # [逻辑 B] 场景: 动态阶梯止盈 (Dynamic Trailing Stop) -> 卖 100%
                            # ----------------------------------------------------------------------
                            # 只有在还没触发 A 逻辑的情况下才执行 B
                            # if not should_sell:
                            #     # 获取或初始化历史最高价
                            #     previous_high = ah_high_water_marks.get(symbol, max(current_price, static_close_price))
                                
                            #     if current_price > previous_high:
                            #         # 创新高：更新水位线
                            #         ah_high_water_marks[symbol] = current_price
                            #     else:
                            #         # 计算回撤
                            #         drawdown_pct = (previous_high - current_price) / previous_high
                                    
                            #         # 动态阈值计算
                            #         stop_threshold = 0.02 # 默认 2%
                            #         if roi_pct >= 0.05: stop_threshold = 0.005    # 赚>5% -> 0.5%
                            #         elif roi_pct >= 0.03: stop_threshold = 0.008  # 赚>3% -> 0.8%
                            #         elif roi_pct >= 0.01: stop_threshold = 0.012  # 赚>1% -> 1.2%
                            #         else: stop_threshold = 0.015                  # 赚<1% -> 1.5%

                            #         if drawdown_pct > stop_threshold:
                            #             sell_reason = (f"盘后动态止盈 (ROI:{roi_pct*100:.1f}% | "
                            #                         f"最高:{previous_high:.2f}->现价:{current_price:.2f}, "
                            #                         f"回撤:{drawdown_pct*100:.2f}% > 阈值:{stop_threshold*100:.1f}%)")
                            #             target_sell_ratio = 1.0 # <--- 趋势坏了，全跑
                            #             should_sell = True
                        # ==============================================================================
                        #  轨道 B: 亏损单处理 (Loss Mitigation Logic)
                        #  重构架构：优先级管道 (Priority Pipeline)
                        #  ① 日历断头台(高) → ② risk_engine(常规) → 统一流到 Step 3 执行
                        #  所有层只设 should_sell=True，不 continue，最终统一在 Step 3 执行
                        # ==============================================================================
                        else:
                            is_immune = position.triggering_strategy in self.config.strategies_immune_to_exit
                            is_bearish = self._is_bearish_symbol(symbol)

                            # ── 免疫策略：完全跳过亏损轨道 ──
                            if is_immune or symbol in self.shadow_tags.get('strategic_hold', []):
                                continue

                            # ══════════════════════════════════════════════════════
                            # ① 日历断头台 (Calendar Guillotine) — bearish 不参与
                            # ══════════════════════════════════════════════════════
                            if not is_bearish:
                                weekday = now_local.weekday()  # 0=周一 ... 4=周五

                                # ──【盘后反弹豁免】(Rebound Immunity) ──
                                # 若盘后正在反弹 (ah_delta ≥ 阈值) 且亏损未到极限，
                                # 暂缓砍仓；只要反弹持续，每个 tick 都会持续豁免，
                                # 一旦反弹熄火 (ah_delta 跌回阈值之下) 自动解除。
                                rebound_imm_cfg = ah_cfg.get('guillotine_rebound_immunity', {}) or {}
                                rebound_immunity_active = False
                                if rebound_imm_cfg.get('enabled', True):
                                    imm_min_delta = rebound_imm_cfg.get('min_ah_delta_pct', 0.003)
                                    imm_max_loss  = rebound_imm_cfg.get('max_loss_for_immunity', -0.025)
                                    imm_tag       = rebound_imm_cfg.get('log_tag', 'GUILLOTINE_REBOUND_PASS')
                                    if ah_delta >= imm_min_delta and roi_pct >= imm_max_loss:
                                        rebound_immunity_active = True

                                # 场景 1: 周五大清洗 — 拒绝亏损过周末 (容忍度 -0.3%)
                                if weekday == 4 and us_session == TradingSession.AFTER_MARKET_EXTENDED:
                                    if roi_pct < -0.003:
                                        if rebound_immunity_active:
                                            logger.warning(
                                                f"🛡️ [{symbol}] 日历断头台·周五[{imm_tag}] 豁免: "
                                                f"盘后反弹 {ah_delta:+.2%} ≥ {imm_min_delta:.2%}, "
                                                f"ROI {roi_pct:.2%} 暂缓砍仓"
                                            )
                                        else:
                                            should_sell = True
                                            target_sell_ratio = 0.8
                                            sell_reason = f"💀 [日历断头台] 周五死线: 拒绝亏损过周末 (ROI {roi_pct:.2%})"

                                # 场景 2: 周四高危夜 — 亏损超 -1.0% 则杀
                                elif (weekday == 3 or weekday == 4) and is_pre_market_marco_data:
                                    if roi_pct < -0.010:
                                        if rebound_immunity_active:
                                            logger.warning(
                                                f"🛡️ [{symbol}] 日历断头台·周四[{imm_tag}] 豁免: "
                                                f"盘后反弹 {ah_delta:+.2%} ≥ {imm_min_delta:.2%}"
                                            )
                                        else:
                                            should_sell = True
                                            target_sell_ratio = 1.0
                                            sell_reason = f"🔪 [日历断头台] 周四警戒: 亏损超标 (ROI {roi_pct:.2%} < -1.0%)"

                                # 场景 3: 周三驼峰清洗 — 亏损超 -1.0% 的劣质资产
                                elif weekday == 2 and us_session == TradingSession.AFTER_MARKET_EXTENDED:
                                    if roi_pct < -0.010:
                                        if rebound_immunity_active:
                                            logger.warning(
                                                f"🛡️ [{symbol}] 日历断头台·周三[{imm_tag}] 豁免: "
                                                f"盘后反弹 {ah_delta:+.2%} ≥ {imm_min_delta:.2%}"
                                            )
                                        else:
                                            should_sell = True
                                            target_sell_ratio = 0.5
                                            sell_reason = f"🐪 [日历断头台] 周三清洗: 剔除劣质持仓 (ROI {roi_pct:.2%} < -1.0%)"

                            # ══════════════════════════════════════════════════════
                            # ② 常规风控引擎 — 日历断头台未触发时才进入
                            #    (bearish 也走这条路，不被跳过)
                            # ══════════════════════════════════════════════════════
                            if not should_sell and not is_5min_rush:
                                is_risk, ratio, risk_msg = self.extended_hours_risk_engine.check_risk_action(
                                    symbol, position, current_price, quote_data,
                                    global_action=current_global_action
                                )
                                if is_risk:
                                    # 防抖："回本逃生" 减仓操作 10 分钟内不重复
                                    if "回本逃生" in risk_msg and ratio < 1.0:
                                        if self._is_action_recently_taken(position, "回本逃生", lookback_minutes=10):
                                            pass  # 跳过本次 risk_engine 结果，但不 continue
                                        else:
                                            should_sell = True
                                            target_sell_ratio = ratio
                                            sell_reason = risk_msg
                                    else:
                                        should_sell = True
                                        target_sell_ratio = ratio
                                        sell_reason = risk_msg
                                    # 亏损单不需要高水位记录，清理以防污染
                                    if should_sell and symbol in ah_high_water_marks:
                                        del ah_high_water_marks[symbol]

                        # =======================================================
                        # [Step 3] 公共兜底逻辑 (Deadline Safety Net)
                        # =======================================================
                        # 无论刚才盈亏逻辑如何，如果到了死线或陷阱区，还没卖出，强制检查
                        if not should_sell and (is_pre_trap or is_deadline):
                            # 逻辑：
                            # 1. 如果是盈利的 (roi > 0) -> 使用技术指标确认是否要在死线卖出 (你原来的逻辑)。
                            # 2. 如果是亏损的 (roi < 0) -> 死线到了，通常需要无脑跑，或者基于更宽松的指标跑。
                            
                            # 3.1 盈利单的最后离场 (Technical Check)
                            if roi_pct > ah_min_profit_to_rush: # 
                                k_mins_check = self.config.tactical_k_mins_map.get('SCALP_EXIT', 3)
                                # 死线临头，回撤阈值放宽(0.2%)；诱多陷阱，要求严格(0.5%)
                                rebound_pct_threshold = 0.002 if is_deadline else 0.005
                                
                                is_high_confirmed = check_extended_hours_tactical_exit_signal(
                                    self.hs_data_provider, symbol, k_mins_check, rebound_pct_threshold
                                )
                                
                                if is_high_confirmed:
                                    tag = "终极清场" if is_deadline else "盘前诱多"
                                    sell_reason = f"兜底-盈利离场[{tag}] | ROI:{roi_pct:.2%}"
                                    target_sell_ratio = 0.5 if is_pre_trap else 1.0
                                    should_sell = True
                            # 3.2 亏损单的强制熔断 (Hard Exit) - 你的救命稻草
                            # 只有在死线时刻(09:25)，且确实亏损，才强制跑
                            # elif roi_pct < 0 and is_deadline:
                            #     # 只有【没加过仓】的才触发死线强制清仓
                            #     if position.dip_adds_done == 0:
                            #         # 只有亏损超过一定幅度才强制跑，微亏(0.1%)可能还可以抗一下?
                            #         # 不，夜盘死线由于马上接 MOO (Market On Open)，为了防止不可控波动，建议清仓。
                            #         sell_reason = f"兜底-亏损熔断[盘前死线] | ROI:{roi_pct:.2%} | 规避开盘波动"
                            #         target_sell_ratio = 1.0
                            #         should_sell = True
                            #         logger.critical(f"⏰ [{symbol}] 盘前死线触发！亏损单强制离场。")
                            #     else:
                            #         # 如果加过仓，即使到了死线也不卖，留着过夜/赌开盘
                            #         logger.warning(f"🎰 [{symbol}] 盘前死线豁免: 已补仓{position.dip_adds_done}次，执行[赌一把]协议，持仓过夜。")
                            #         should_sell = False # 再次确认


                        # ==============================================================================
                        # ▼▼▼【黄昏审判庭】(The Dusk Tribunal)  ▼▼▼
                        # 逻辑：盘后/盘前流动性差，且面临隔夜/周末风险。
                        # 对于亏损单，算法通常倾向于止损，但此处引入 LLM 进行“死刑复核”。
                        # ==============================================================================
                        # if should_sell and roi_pct < 0 and self.config.enable_llm_check:
                        #     try:
                        #         # 1. 识别特殊风险场景
                        #         is_risk = is_entering_weekend_risk_for_symbol(symbol, wrp_activation_days=[2]) # 周五/周三
                        #         if is_risk:
                        #             should_sell = True
                        #         else:
                        #             # 2. 构造审判请求 (Candidate Context)
                        #             # 明确告诉 LLM 这是盘后亏损单，需要它做最终裁决
                        #             tribunal_candidate = {
                        #                 'symbol': symbol,
                        #                 'avg_cost': cost_price,
                        #                 'trigger_price': current_price, # 兼容接口
                        #                 'strategy_name': position.triggering_strategy or 'AfterHours',
                        #                 'reason': f"{sell_reason} (盘后亏损单复核)", # 传递原始卖出理由
                        #                 # --- 注入元数据供 Prompt 优化 ---
                        #                 'is_after_hours': True,
                        #                 'is_weekend_risk': is_risk,
                        #                 'roi_pct': roi_pct
                        #             }

                        #             # 3. 记录日志：审判开始
                        #             logger.info(f"⚖️ [黄昏审判] 启动复核: {symbol} (ROI {roi_pct:.2%}) | 原始判决: 卖出 | 原因: {sell_reason}")

                        #             # 4. 调用 LLM (re_check_sell)
                        #             is_confirmed_sell, llm_verdict = self._get_llm_decision(tribunal_candidate, 'sell')

                        #             # 5. 执行裁决
                        #             if not is_confirmed_sell:
                        #                 # === 结果 A: 刀下留人 (VETO) ===
                        #                 # LLM 认为不该卖 (可能发现了主力吸筹或超跌背离)
                        #                 should_sell = False
                        #                 if random.random() > 0.90:
                        #                     logger.warning(f"🛡️ [黄昏审判] LLM 否决卖出指令: {symbol} | LLM理由: {llm_verdict}")
                                        
                        #                 # [可选] 记录被否决的事件，防止下一轮循环立刻又触发
                        #                 # position.add_log(f"LLM否决盘后止损: {llm_verdict}")
                        #             else:
                        #                 # === 结果 B: 维持原判 (CONFIRM) ===
                        #                 # LLM 同意卖出，将 LLM 的理由追加到 sell_reason 中，增强日志可读性
                        #                 # 提取 LLM 返回的精简理由 (通常 formatted_reason 比较长，这里取前段)
                        #                 clean_llm_msg = llm_verdict.split('\n')[0].replace('【决策: ', '').replace('】', '')
                        #                 sell_reason = f"{sell_reason} | 🤖LLM核准: {clean_llm_msg}"
                        #                 logger.info(f"💀 [黄昏审判] LLM 维持死刑判决: {symbol}")

                        #     except Exception as e:
                        #         # 审判庭崩溃不能影响执法，默认维持原判 (卖出)
                        #         logger.error(f"[{symbol}] 黄昏审判执行异常，维持原判: {e}")

                        # --- 5. 交易执行 ---
                        if should_sell:
                            # 周三风控检查 (强制全仓)
                            # if is_entering_weekend_risk_for_symbol(symbol, wrp_activation_days=[2]):
                            #     sell_reason += " [周三风控生效]"
                            #     target_sell_ratio = 1.0 # ⚠️ 强行覆盖：如果是周三风控，不管上面算出来多少，全部清仓过夜！
                            # if random.random() > 0.90:
                            #     logger.critical(f"🚀 [盘后交易触发] {symbol} | {sell_reason} | 比例: {target_sell_ratio}")
                            
                            if not self.config.test_mode and position.triggering_strategy != "NightHunter":
                                self._execute_extended_hours_sell(symbol, sell_reason, sell_ratio=target_sell_ratio)
                                
                                # 如果是全仓卖出，才清除水位线；减仓的话保留水位线继续监控剩余仓位
                                if target_sell_ratio >= 1.0 and symbol in ah_high_water_marks:
                                    del ah_high_water_marks[symbol]
                                    # 必须同步清除风控引擎中的僵尸单记忆，否则再次买入会误判为持有过久
                                    self.extended_hours_risk_engine.clear_zombie_memory(symbol)
                                    
                    except Exception as e_inner:
                        logger.error(f"处理股票 {symbol} 盘后逻辑时出错: {e_inner}", exc_info=True)
                        continue

                # 盘后轮询间隔：建议 3-5 秒，既能捕捉高点，又不至于被API限流
                time.sleep(self.config.loop_sleep_interval)

            except Exception as e_outer:
                logger.error("盘后监控线程主循环异常，正在自动恢复...", exc_info=True)
                time.sleep(30) # 发生严重错误时冷却一下
   
    # 检查已持仓的MACD策略头寸是否有加仓机会
    def _check_confirmation_add_opportunities(self):
        """
        遍历持仓，寻找策略确认加仓机会 (包含二次探底与策略确认)。
        """
        logger.info("保守策略模式：确认加仓检查已禁用。")
        return

        with self.position_lock:
            positions_to_check = list(self.positions.values())

        for pos in positions_to_check:
            symbol = pos.symbol
            
            # --- 前置过滤 (保持静默) ---
            current_status = get_trading_window_status(symbol)
            favorable_buy_windows = self.config.favorable_buy_windows
            
            # 必须满足基础条件才进入昂贵的计算逻辑
            if not (pos.triggering_strategy and
                    not pos.confirmation_add_done and
                    pos.strategy_class_name and
                    current_status in favorable_buy_windows):
                continue

            try:
                # --- 数据准备 ---
                current_price = self.get_current_price(symbol)
                real_cost = pos.get_avg_cost(self.config)
                if not current_price or real_cost <= 0: continue 
                roi = (current_price - real_cost) / real_cost
                # market = get_market_type(symbol)
                
                # 动态参数计算
                # k_mins_check = get_dynamic_k_minutes(current_status, market)
                # threshold = self.config.rebound_pct_threshold_map.get(current_status, self.config.rebound_pct_threshold_map['default'])
                # is_entry_signal = check_tactical_entry_signal(self.quote_ctx, symbol, k_mins_check, threshold)
                
                is_confirmed = False
                trigger_type = "" # 用于日志区分触发类型
                trigger_msg = ""

                # ==============================================================================
                # 逻辑分支 A: 用户自定义 [二次探底亏损加仓] (Secondary Dip Rescue)
                # ==============================================================================
                # if (current_status in [TradingWindowStatus.MORNING_DIP_BUY, TradingWindowStatus.AFTERNOON_GOLDEN_PIT] and
                #     roi <= -0.013 and
                #     is_entry_signal and
                #     is_opened_today(pos, symbol) and
                #     pos.dip_adds_done == 1):
                    
                #     is_confirmed = True
                #     trigger_type = "二次探底救援 (Dip Rescue)"
                #     trigger_msg = f"满足苛刻条件: ROI({roi:.2%})<=-1.3% & 已补仓1次 & 窗口({current_status.name})"

                # ==============================================================================
                # 逻辑分支 B: 策略原生 [右侧确认加仓] (Strategy Final Confirmation)
                # ==============================================================================
                if -0.025 <= roi <= -0.0023 and (current_status in [TradingWindowStatus.MORNING_DIP_BUY, TradingWindowStatus.AFTERNOON_GOLDEN_PIT]):
                    # 动态复活策略实例
                    strategy_info = {
                        "strategy_class_name": pos.strategy_class_name,
                        "strategy_params": pos.strategy_params
                    }
                    strategy = self.strategy_evaluator.get_strategy_instance_for_recheck(strategy_info)
                    
                    if not strategy:
                        logger.error(f"[{symbol}] 无法重建策略实例 {pos.strategy_class_name}，跳过复核。")
                        continue

                    # 调用策略内部确认
                    strat_confirmed, strat_msg = strategy._final_confirmation(symbol)
                    if strat_confirmed:
                        is_confirmed = True
                        trigger_type = "策略右侧确认 (Strategy Confirm)"
                        trigger_msg = f"策略逻辑通过: {strat_msg}"
                
                # ==============================================================================
                # 统一执行与日志 (Execution & Standardized Logging)
                # ==============================================================================
                if is_confirmed:
                    # 双重互斥检查 (防止计算期间状态突变)
                    if pos.dip_pending_state is not None: continue
                    
                    log_payload = (
                        f"🚀 [确认加仓触发] {symbol} | "
                        f"类型: {trigger_type} | "
                        f"现价: {current_price} (成本: {pos.avg_cost:.3f}) | "
                        f"ROI: {roi:.2%} | "
                        f"详情: {trigger_msg}"
                    )
                    logger.warning(log_payload)
                    
                    # 传入完整的 trigger_msg 作为原因
                    self._handle_confirmation_add(pos, f"{trigger_type}|{trigger_msg}")
                    
            except Exception as e:
                logger.error(f"[{symbol}] 确认加仓逻辑发生异常: {e}", exc_info=True)
        
            # ==============================================================================
            # 逻辑分支 C: [夜猎者] 专属加仓 (保持独立)
            # ==============================================================================
            if (pos.triggering_strategy == "NightHunter" and
                not pos.confirmation_add_done and
                pos.strategy_class_name and
                current_status in [TradingWindowStatus.PRE_MARKET_LONDON_FIX,
                                   TradingWindowStatus.NIGHT_LUNCH_DIP,
                                   TradingWindowStatus.NIGHT_ASIA_CORRELATION]):
                
                real_cost = pos.get_avg_cost(self.config)
                if not current_price or real_cost <= 0: continue 
                roi = (current_price - real_cost) / real_cost
                
                if roi <= -0.010:
                    is_night_confirmed = False
                    night_msg = ""
                    
                    # 轨道 A: K=5
                    thresh_a = self.config.rebound_pct_threshold_map.get('default', 0.008)
                    if check_extended_hours_tactical_entry_signal(self.hs_data_provider, symbol, 5, thresh_a):
                        is_night_confirmed = True
                        night_msg = f"夜盘短线反弹(K=5|T={thresh_a:.1%})"

                    # 轨道 B: K=25
                    if not is_night_confirmed:
                        thresh_b = 0.006
                        if check_extended_hours_tactical_entry_signal(self.hs_data_provider, symbol, 25, thresh_b):
                            is_night_confirmed = True
                            night_msg = f"夜盘深跌企稳(K=25|T={thresh_b:.1%})"

                    if is_night_confirmed:
                        full_reason = f"夜猎者确认(NightHunter)|{night_msg}|ROI:{roi:.2%}"
                        
                        # 统一夜盘日志格式
                        logger.warning(f"🚀 [夜盘加仓触发] {symbol} | 现价: {current_price} | ROI: {roi:.2%} | 原因: {night_msg}")
                        
                        self._handle_confirmation_add(pos, full_reason, triggering_strategy=self.config.night_config.get('strategy_name', 'NightHunter'))

    def _handle_confirmation_add(self, position: Position,reason:str,triggering_strategy:str = None):
        """处理确认加仓的具体下单逻辑"""
        with self.position_lock:
            # 再次检查，防止并发问题
            if position.confirmation_add_done or position.pending_pyramid_order_id:
                logger.warning(f"{position.symbol} 已在处理加仓或已完成确认加仓，本次忽略。")
                return

            # 加仓数量：这里我们定义为当前持仓数量的100%，即“再买一倍”
            add_ratio = getattr(self.config, 'confirmation_add_ratio', 0.5)
            if '符合二次探底亏损加仓' == reason: add_ratio = 0.3
            quantity_to_add = position.total_quantity * add_ratio
            
            symbol_info = self.get_cached_stock_static_info(position.symbol)
            quantity_to_add = self._adjust_quantity(quantity_to_add, position.market, lot_size=symbol_info.get('lot_size', 100))

            if quantity_to_add <= 0:
                logger.error(f"确认加仓计算数量为0，无法为 {position.symbol} 加仓。")
                # 标记为完成，以防止因持续计算为0而无限重试
                position.confirmation_add_done = True
                self._save_positions()
                return # <--- 直接返回，中止操作

            if triggering_strategy==self.config.night_config.get('strategy_name', 'NightHunter'):#夜盘加仓策略
                if self._execute_extended_hours_add_position(position.symbol, quantity_to_add, position, reason):
                    # 如果下单成功，则立即标记，防止重复下单
                    position.confirmation_add_done = True
                    self._save_positions()
                    logger.warning(f"已为 {position.symbol} 提交确认加仓订单，数量: {quantity_to_add}。")
            
            else:
                # 使用通用的加仓执行函数
                if self._execute_add_position(position.symbol, quantity_to_add, position, reason):
                    # 如果下单成功，则立即标记，防止重复下单
                    position.confirmation_add_done = True
                    self._save_positions()
                    logger.warning(f"已为 {position.symbol} 提交确认加仓订单，数量: {quantity_to_add}。")
                    current_price = self.get_current_price(position.symbol)
                    self.notification_manager.send_trade_execution("ADD", position.symbol,quantity_to_add, current_price, reason)
    
    # ==============================================================================
    # III. 开仓生命周期 (Entry Lifecycle)
    # ==============================================================================
    def process_buy_signal(self, candidate: dict):
        symbol = candidate['symbol']
        try:
            market = get_market_type(symbol)

            if self._is_monthly_loss_pause_active() or self._is_account_daily_loss_limit_hit() or self._is_account_monthly_loss_limit_hit():
                logger.warning(f"[{symbol}] 账户级亏损风控生效，拒绝新开仓。")
                return

            target_root = get_root_symbol(symbol)
            with self.position_lock:
                for pos_symbol in self.positions.keys():
                    if get_root_symbol(pos_symbol) == target_root:
                        logger.warning(f"[{symbol}] 已持有同根标的 {pos_symbol}，拒绝重复暴露。")
                        return

            with self.pending_orders_lock:
                for pending_symbol in self.pending_orders.keys():
                    if get_root_symbol(pending_symbol) == target_root:
                        logger.warning(f"[{symbol}] 同根标的 {pending_symbol} 已有在途买单，拒绝重复开仓。")
                        return

            # if self.get_current_positions_count(market) >= self.get_max_positions(market):
            #     logger.error(f"{market.value}市场达到最大持仓数({self.get_max_positions(market)})，无法买入 {symbol}")
            #     return

            current_price = self.get_current_price(symbol) or candidate.get('trigger_price')
            if current_price is None or current_price <= 0:
                logger.error(f"[{symbol}] 无法获取有效买入价格，放弃开仓。")
                return

            self._handle_initial_buy(symbol, market, float(current_price), candidate)
            return

            # =========================================================
            # 0. 相关性风控拦截 (Correlation Risk Gate) - Priority 0+
            # =========================================================
            # 逻辑：禁止“左手 NVDA，右手 NVDX”。
            # 如果即将买入的标的，其【根资产】在当前持仓或在途订单中已存在，
            # 视为重复暴露风险，直接熔断。
            # =========================================================
            
            # 1. 计算当前信号的 Root ID
            target_root = get_root_symbol(symbol)
            
            conflict_found = False
            conflict_holder = ""
            conflict_source = ""

            # 2. 检查【已持仓】 (Holding Positions)
            # 必须加锁，虽然只是读操作，但为了严谨性 (The Geek Way)
            with self.position_lock:
                for pos_symbol in self.positions.keys():
                    # 计算持仓的 Root ID
                    if get_root_symbol(pos_symbol) == target_root:
                        conflict_found = True
                        conflict_holder = pos_symbol
                        conflict_source = "已持仓"
                        break
            
            # 3. 检查【在途订单】 (Pending Orders)
            # 如果持仓没冲突，再看看是不是已经在路上了
            if not conflict_found:
                with self.pending_orders_lock:
                    for pending_symbol in self.pending_orders.keys():
                        if get_root_symbol(pending_symbol) == target_root:
                            conflict_found = True
                            conflict_holder = pending_symbol
                            conflict_source = "在途订单"
                            break
            
            # 4. 执行熔断
            if conflict_found:
                # 只有当 目标代码 != 冲突代码 时才认为是“关联冲突” (防止同代码的去重逻辑被这里误报)
                # 不过，即使是同代码，这里拦截也是对的（防止加仓）。
                # 但为了日志准确性，我们区分一下文案。
                
                log_msg = f"🚫 [风控拦截] {symbol} 关联标的冲突：{conflict_source} {conflict_holder} (Root: {target_root})"
                
                # 如果是自己冲突自己（比如 pending 里有 NVDA，又来个 NVDA 信号）
                if symbol == conflict_holder:
                    # 这种情况通常由 pending_orders_lock 下面的逻辑处理，但这里拦截更早，效率更高
                    logger.warning(f"🚫 [重复拦截] {symbol} {conflict_source}已存在，拒绝重复开仓。")
                else:
                    # 这是真正的关联冲突 (NVDX vs NVDA)
                    logger.critical(log_msg) # 使用 Critical 级别，因为这是严重的策略重叠
                
                return # <--- 核心：直接返回，不执行后续任何逻辑
            
            # =========================================================
            # 1. 黑名单熔断拦截 (Priority 0)
            # =========================================================
            if symbol in self.intraday_blacklist:
                if random.random() > 0.95:
                    logger.warning(f"🚫 [熔断拦截] {symbol} 位于当日黑名单中（曾触发舆情/止损），拒绝开仓信号。")
                return
            
            # =========================================================
            # 2. 日内反向回购限制 (T+0 Cost Guard)
            # =========================================================
            # last_sell_price = self.intraday_trade_history.get(symbol)
            # # 这里务必使用 candidate 里的 trigger_price，保证决策时效性
            # current_price = candidate.get('trigger_price') or self.get_current_price(symbol)
        
            # if last_sell_price and current_price:
            #     # 增加一点点容错空间 (例如 0.1%)，或者严格执行
            #     # 这里严格执行：买入价必须 <= 卖出价
            #     if current_price > (last_sell_price + 0.001):
            #         logger.warning(
            #             f"🛑 [成本风控拦截] {symbol} 触发日内反向回购限制！"
            #             f"现价({current_price}) > 上次卖出价({last_sell_price})。"
            #             f"拒绝当韭菜，等待价格回落。"
            #         )
            #         # 既然价格太高，我们不把它加入黑名单（因为可能跌回来），
            #         # 但本次处理直接 return，不执行后续逻辑。
            #         # 也不从 pending_cache 移除（如果你的逻辑是移除的话），让它下一轮再试？
            #         # 不，process_buy_signal 是消费者，如果不执行，通常该信号就被消耗了。
            #         # 如果你想让它“等价格降下来”，应该由上层 pending_monitor 决定，
            #         # 但在这里拦截是最安全的。
            #         return # 触发 finally 解锁

            # =========================================================
            # 3. 舆情门禁 (Sentiment Gate)
            # =========================================================
            # is_fresh, sentiment_data = self.sentiment_analysis.is_cache_fresh(symbol, ttl_minutes=60)
            # sentiment_status = sentiment_data.get('sentiment', '未知')

            # # 2. 检查：是否为明确的负向？
            # if sentiment_status == '负向':
            #     logger.critical(f"🛑 [舆情门禁拦截] {symbol} 存在负向舆情，买入指令已销毁！(缓存时间: {sentiment_data.get('timestamp')})")
            #     # 将其加入黑名单，防止今天再次骚扰
            #     self.intraday_blacklist.add(symbol)
            #     self._save_blacklist()
            #     return

            # 3. 检查：数据是否过期？(TTL Check)
            # if not is_fresh:
            #     # 数据太旧了！或者是空的。
            #     # 这种情况下买入就是赌博。作为全球第二的极客，我们不赌博。
            #     logger.warning(f"⏳ [舆情门禁拦截] {symbol} 舆情数据过期或缺失，拒绝盲目开仓。已触发加急刷新。")
                
            #     # A. 立即触发一次加急刷新 (虽然在run_strategy_loop预热过，但可能还没跑完，或者这是漏网之鱼)
            #     # 注意：这里不能阻塞等待，因为 analyze 很慢。
            #     self.sentiment_analysis.trigger_async_refresh(symbol, self.task_executor)
                
            #     # B. 策略选择：
            #     # 选项1 (激进): 既然前面已经预热了，这里为了防止死锁，如果没拿到最新数据，暂时放弃本次tick，
            #     #              让 pending_signal_monitor_loop 下一轮循环（几秒后）再试。
            #     #              只要 pending_cache 里还有它，就会不断重试，直到数据变新鲜。
                
            #     # 我们选择选项1，直接 Return。等待数据刷新后，下一轮 process_buy_signal 自然会通过 is_fresh 检查。
            #     return

            # 如果通过了以上两关，说明：舆情是新鲜的，且不是负向的。
            # logger.info(f"✅ [舆情门禁通过] {symbol} 舆情状态: {sentiment_status} (新鲜度校验通过)")

            # 获取市场类型
            market = get_market_type(symbol)
        
            try:
                # 盈利达标熔断机制 (Profit Target Kill Switch)
                if (market == MarketType.US and self.daily_us_profit_target_hit) or \
                   (market == MarketType.HK and self.daily_hk_profit_target_hit):
                    
                    # 检查是否为做空对冲 (豁免逻辑)
                    is_bearish = candidate.get('is_bearish_trade', False)
                    if not is_bearish:
                        if random.random() > 0.8: # 降低日志频率，避免刷屏
                            logger.info(f"🛡️ [盈利熔断] 今日{market.value}盈利目标已达成，拦截 {symbol} 开仓信号。")
                        return
            except Exception:
                pass # 防御性编程：防止极个别 symbol 解析 market 失败导致崩溃
        
            # =========================================================
            # 4. 锁内检查 (防重入) - 再次确认
            # =========================================================
            with self.position_lock, self.pending_orders_lock:
                if symbol in self.positions:
                    logger.debug(f"{symbol} 已有持仓，买入信号忽略。")
                    return
                if symbol in self.pending_orders:
                    logger.debug(f"{symbol} 已存在待处理的开仓订单，忽略新的买入信号。")
                    return
        
            # =========================================================
            # 5. 执行具体的建仓逻辑
            # =========================================================
            # 检查持仓上限
            if self.get_current_positions_count(market) >= self.get_max_positions(market):
                logger.error(f"{market.value}市场达到最大持仓数({self.get_max_positions(market)})，无法买入 {symbol}")
                return
            
            current_price = self.get_current_price(symbol)
            if current_price is None: return

            self._handle_initial_buy(symbol, market, current_price, candidate)
        except Exception as e:
            logger.error(f"处理买入信号时出错 {symbol}: {e}", exc_info=True)
        
        finally:
            # =========================================================
            # ▼▼▼ 飞行锁释放协议 (Release The Lock) ▼▼▼
            # =========================================================
            # 只有当订单没有成功进入 pending_orders 时，才释放锁。
            # 如果进入了 pending_orders，锁的职责移交给了订单管理器。
            has_pending_order = False
            with self.pending_orders_lock:
                if symbol in self.pending_orders:
                    has_pending_order = True
            
            # 2. 如果没有生成订单（说明被风控拦截了，或者出错了），必须释放锁！
            if not has_pending_order:
                with self.signals_in_flight_lock:
                    if symbol in self.signals_in_flight:
                        self.signals_in_flight.discard(symbol)
                        logger.info(f"🔒 [风控拦截/失败] 强制释放飞行锁: {symbol}")

    def _handle_initial_buy(self, symbol: str, market: MarketType, current_price: float,candidate: dict):
        """
        处理首次买入 - 实现“试探性建仓”逻辑 (第一阶段)
        1. 根据风险计算出【总计划仓位】。
        2. 如果启用分批建仓策略，则只买入计划仓位的一部分作为“侦察仓”。
        3. 将完整的建仓计划（总仓位、每股风险）存入待处理订单，以便成交后创建 Position 对象。
        4.  此版本实现了“风险取优”策略：以固定-4%作为风险上限，但如果ATR能提供更小的风险，则采用ATR，从而完美兼容两种模式的需求。
        """
        strategy_name = candidate.get('strategy_name', 'ConservativeMA20Breakout')
        exchange_rate = getattr(self.config, 'exchange_rate_usd_to_hkd', 7.8)
        account_value_hkd = self.get_net_equity_value(strict=True)

        if account_value_hkd <= 0:
            logger.critical(f"[{symbol}] 无法获取账户净值，拒绝开仓。")
            return

        current_exposure_hkd = self._get_total_stock_exposure_hkd(include_pending=True)
        max_total_exposure_hkd = account_value_hkd * 0.70
        min_total_exposure_hkd = account_value_hkd * 0.60
        max_single_position_hkd = account_value_hkd * 0.085

        if current_exposure_hkd >= max_total_exposure_hkd:
            logger.warning(
                f"[{symbol}] 当前总仓位已达上限: {current_exposure_hkd/account_value_hkd:.2%} >= 70.00%，拒绝开仓。"
            )
            return

        available_cash = self.get_available_cash(market)
        available_cash_hkd = available_cash * exchange_rate if market == MarketType.US else available_cash
        remaining_total_room_hkd = max_total_exposure_hkd - current_exposure_hkd
        target_trade_value_hkd = min(max_single_position_hkd, remaining_total_room_hkd, available_cash_hkd)

        if target_trade_value_hkd <= 0:
            logger.warning(f"[{symbol}] 无可用仓位或现金，无法开仓。")
            return

        target_trade_value_native = target_trade_value_hkd / exchange_rate if market == MarketType.US else target_trade_value_hkd
        symbol_info = self.get_cached_stock_static_info(symbol)
        actual_lot_size = 1 if market == MarketType.US else symbol_info.get('lot_size', self.config.lot_size)
        quantity_to_buy = int(target_trade_value_native / current_price)
        quantity_to_buy = self._adjust_quantity(quantity_to_buy, market, lot_size=actual_lot_size)

        if quantity_to_buy <= 0:
            logger.warning(
                f"[{symbol}] 目标金额不足以买入最小交易单位: 目标={target_trade_value_native:.2f}, 价格={current_price:.3f}"
            )
            return

        stop_loss_price = round(current_price * 0.93, 3)
        per_share_risk = round(current_price * 0.07, 3)
        estimated_trade_value_hkd = quantity_to_buy * current_price * (exchange_rate if market == MarketType.US else 1.0)

        logger.warning(
            f"[{symbol}] 保守开仓 sizing | 当前总仓位={current_exposure_hkd/account_value_hkd:.2%}, "
            f"目标区间=60%-70%, 单票上限=8.50%, 本次金额={estimated_trade_value_hkd/account_value_hkd:.2%}, "
            f"数量={quantity_to_buy}, 止损={stop_loss_price:.3f}"
        )

        order_id = self.submit_order(symbol, quantity_to_buy, OrderSide.Buy)
        if not order_id:
            return

        try:
            symbol_name = symbol_info.get('name_cn') or symbol_info.get('name_en') or symbol
            currency = "HKD" if market == MarketType.HK else "USD"
            buy_logger.info(
                f"symbol:{symbol},name:{symbol_name},action:INITIAL_BUY,price:{current_price:.3f},"
                f"quantity:{quantity_to_buy},trade_cost:{current_price * quantity_to_buy:.3f},"
                f"currency:{currency},strategy_name:{strategy_name},initial_stop_loss:{stop_loss_price:.3f},"
                f"per_share_risk:{per_share_risk:.3f},portfolio_before:{current_exposure_hkd/account_value_hkd:.4f},"
                f"order_id:{order_id}"
            )
        except Exception as e:
            logger.error(f"[{symbol}] 记录保守买入日志失败: {e}", exc_info=True)

        strategy_params = candidate.get('strategy_params') or {}
        state = strategy_params.setdefault('conservative_exit_state', {})
        state.setdefault('highest_price', current_price)
        state.setdefault('stage_10_taken', False)
        state.setdefault('stage_15_taken', False)

        with self.pending_orders_lock:
            self.pending_orders[symbol] = {
                "order_id": order_id,
                "plan_info": {
                    "planned_total_quantity": quantity_to_buy,
                    "initial_risk_per_share": per_share_risk,
                    "initial_stop_loss_price": stop_loss_price,
                    "triggering_strategy": strategy_name,
                    "strategy_class_name": candidate.get('strategy_class_name', 'ConservativeMA20Breakout'),
                    "strategy_params": strategy_params,
                    "confirmation_add_done": True,
                    "building_stage": 2,
                    "estimated_trade_value_hkd": round(estimated_trade_value_hkd, 2),
                }
            }

        self._save_pending_orders()
        logger.info(
            f"[{symbol}] 保守策略买入订单已提交 | ID={order_id} | "
            f"组合仓位买前={current_exposure_hkd/account_value_hkd:.2%}, 买后估算={(current_exposure_hkd + estimated_trade_value_hkd)/account_value_hkd:.2%}"
        )
        return
        
        strategy_name = candidate['strategy_name']
        is_sharp_knife = (candidate.get('mode') == 'SHARP_KNIFE')
        building_stage = 1 if self.config.micro_building_config['enabled'] else 2
        target_initial_ratio = self.config.initial_position_scale_ratio # 0.4
        micro_enabled = self.config.micro_building_config.get('enabled', False)
        MIN_POS_THRESHOLD = 0.001

        if not is_sharp_knife and micro_enabled:
            # 🕵️ 微观侦察模式：强制使用配置的侦察比例，无视信号建议
            target_ratio = self.config.micro_building_config.get('scout_ratio', 0.15)
            log_tag = "🕵️ [侦察仓]"
            
            # 在这种模式下，直接锁定比例，禁止下方逻辑篡改
            base_buy_percentage = target_ratio
        else:
            # 💥 一键重仓模式 / 尖刀模式：允许信号覆盖默认比例
            target_initial_ratio = self.config.initial_position_scale_ratio
            log_tag = "💥 [一键重仓]"
            
            # 优先取信号里的 buy_percentage，取不到才用配置的初始比例
            base_buy_percentage = candidate.get('buy_percentage', target_initial_ratio)

        # 数据清洗：处理信号传来的非法值（如负数或极小值）
        if base_buy_percentage < MIN_POS_THRESHOLD:
            # 如果信号给的值不靠谱，回退到当前模式的默认配置（而不是盲目回退到 real_buy_ratio 这个中间变量）
            # 注意：这里的回退逻辑要根据你的业务需求定，这里我假设回退到配置值
            logger.warning(f"{log_tag} 信号建议比例过小 ({base_buy_percentage})，已修正。")
            base_buy_percentage = target_ratio if (not is_sharp_knife and micro_enabled) else self.config.initial_position_scale_ratio

        # >>> 调用核心决策引擎 <<<
        buy_percentage = self._calculate_dynamic_position_ratio(symbol, base_buy_percentage)

        # 熔断检查
        if buy_percentage <= MIN_POS_THRESHOLD:
            logger.warning(f"🛑 [风险拦截] {symbol} 最终仓位 {buy_percentage} 触发熔断 (可能位于 Risky Pool)，放弃建仓。")
            return

        # is_day_trade_only = (strategy_name in self.config.day_trade_only_strategies)
        per_share_risk = 0.0 # 我们的风险单位 R
        # ==============================================================================
        # ▼▼▼ [尖刀队 强制风控注入 (-2% 硬止损) ▼▼▼
        # ==============================================================================
        if is_sharp_knife:
            # 强制 -2% 止损，无视 ATR
            stop_loss_price = current_price * 0.98
            logger.warning(f"🔪 [{symbol}] 尖刀队风控注入: 强制硬止损 -2% (止损价: {stop_loss_price:.3f})")
        else:
            # 常规逻辑
            stop_loss_price = self.adaptive_stop_loss.calculate_stop_loss(symbol, current_price, 'long')
        
        # 4. 根据最终的止损价，反算出真实的每股风险 (R)
        per_share_risk = current_price - stop_loss_price

        if current_price <= stop_loss_price or per_share_risk <= 0:
            logger.error(f"无法开仓 {symbol}: 当前价({current_price})已低于或等于计算出的止损价({stop_loss_price:.3f})。风险: {per_share_risk}")
            return
        
        # === 1. 风险计算阶段: 基于“总净值”确定理想的头寸规模 ===
    
        # 风险计算的唯一基石是账户总净值
        total_capital_base = self.get_total_account_value_in_hkd()
        # 根据市场，将总资本转换为用于计算的币种
        if market == MarketType.US:
            exchange_rate = getattr(self.config, 'exchange_rate_usd_to_hkd', 7.8)
            capital_for_sizing = total_capital_base / exchange_rate
        else:
            capital_for_sizing = total_capital_base

        # 根据市场温度动态调整风险比例
        risk_per_trade_ratio = self.config.risk_per_trade_ratio # 获取基础比例
        #【WRP 注入点】
        if self._is_entering_weekend_risk_for_symbol(symbol):
            risk_per_trade_ratio *= self.config.wrp_risk_ratio_multiplier
            logger.warning(f"[WRP] {symbol} 处于周末风险期，新仓位单笔风险已临时下调至: {risk_per_trade_ratio:.3%}")

        current_regime = self.market_regime_engine.get_marget_regime(market)
        risk_multiplier = self.config.risk_multiplier_map.get(current_regime, 0.20) # 默认为0，不开仓
        # 动态调整后的单笔风险比例
        adjusted_risk_per_trade_ratio = risk_per_trade_ratio * risk_multiplier

        logger.warning(f"市场状态: {current_regime.value}, 风险调节系数: {risk_multiplier}x. "f"最终单笔风险比例: {adjusted_risk_per_trade_ratio:.3%}")
        
        # 基于总净值，计算单笔交易的最大可承受亏损额 (1R 的价值)
        max_loss_per_trade = capital_for_sizing * adjusted_risk_per_trade_ratio
        planned_total_quantity = int(max_loss_per_trade / per_share_risk)

        # ==============================================================================
        # ▼▼▼【核心升级：双重漏斗模型 (Dual Funnel Sizing)】▼▼▼
        # ------------------------------------------------------------------------------
        # 漏斗 1 (Risk Funnel): 上面计算出的 planned_total_quantity，基于止损距离和风险敞口。
        # 漏斗 2 (Capital Funnel): 基于“资金均摊”原则，计算单只股票的资金配额上限。
        # ------------------------------------------------------------------------------
        
        # 1. 获取最大持仓限制 (分母)
        max_pos_limit = self.get_max_positions(market)
        if max_pos_limit < 1: max_pos_limit = 1 # 防止除零防御
        
        # 2. 计算单只股票的“理想资金配额” (Capital Cap)
        # 逻辑：总净值 / 设定的最大持仓数。例如 100万 / 10 = 10万。
        capital_allocation_per_stock = capital_for_sizing / max_pos_limit
        
        # 3. 将资金配额转换为股数
        quantity_capital_cap = int(capital_allocation_per_stock / current_price)
        
        # 4. 记录决策过程 (透明化)
        risk_based_qty = planned_total_quantity # 暂存原值用于日志
        
        # 5. 【终极裁决】 取两者的最小值
        # 含义：既不允许超过风险限额（防爆仓），也不允许超过资金配额（防梭哈）。
        planned_total_quantity = min(risk_based_qty, quantity_capital_cap)
        # planned_total_quantity = (risk_based_qty + quantity_capital_cap)/2

        # 根据当前市场状态对基础仓位做缩放，严禁硬编码 STRONG_BULL。
        planned_total_quantity = int(
            planned_total_quantity * self.config.risk_multiplier_map.get(current_regime, 1.0)
        )
        is_strong_bull = self.get_strong_bull(market)
        is_super_weak = self.get_super_weak(market)
        if is_strong_bull or is_super_weak: # 当大盘适合做多或者做空增加1.5倍
            planned_total_quantity = int(planned_total_quantity * 1.5)
        
        logger.warning(
            f"⚖️ [双重漏斗决策] {symbol} | "
            f"漏斗A(风险): {risk_based_qty}股 (R={adjusted_risk_per_trade_ratio:.1%}) | "
            f"漏斗B(资金): {quantity_capital_cap}股 (Avg={100/max_pos_limit:.1f}%) | "
            f"-> 最终基准: {planned_total_quantity}股"
        )
        

        current_status = get_trading_window_status(symbol)
        buy_ratio = 1.0  # 默认买入比例为100%
                    
        if current_status == TradingWindowStatus.MORNING_DIP_BUY:
            # 早盘下跌买入，市场不确定性高，降低买入比例以控制风险
            buy_ratio = 0.85
            logger.warning(f"[{symbol}] 处于 [早盘下跌买入] 窗口，买入比例调整为 {buy_ratio:.0%}")
        # elif current_status == TradingWindowStatus.AFTERNOON_GOLDEN_PIT:
        #     # 2) 午后黄金坑，通常是日内确定性较高的低点，是宝贵的建仓机会，应适当增加买入比例
        #     buy_ratio = 1.0
        #     logger.warning(f"[{symbol}] 处于 [午后黄金坑] 窗口，买入比例放大至 {buy_ratio:.0%}")
        # elif current_status == TradingWindowStatus.MIDDAY_LULL_PROBE:
        #     # 3) 午间沉寂，是“试探性”的体现，因此降低买入比例
        #     buy_ratio = 0.85
        #     logger.warning(f"[{symbol}] 处于 [午间沉寂] 窗口触发买入，买入比例调整为 {buy_ratio:.0%}")
        elif current_status == TradingWindowStatus.FINAL_MINUTES_GAMBLE:
            # 尾盘时段买入，需防范隔夜风险，因此降低买入比例，极度保守，只买 20%
            buy_ratio = 0.4
            logger.warning(f"[{symbol}] 处于 [尾盘] 窗口触发买入，买入比例调整为 {buy_ratio:.0%}")
        elif current_status == TradingWindowStatus.SHORT_MORNING_GAP_OPEN:
            # 赌高开低走
            buy_ratio = 0.70
            logger.warning(f"[{symbol}] 处于 [赌高开低走] 窗口触发做空买入，买入比例调整为 {buy_ratio:.0%}")
        # else:
        #     buy_ratio = 0.80
        #     logger.warning(f"[{symbol}] 处于 [不利于交易] 窗口触发买入，买入比例调整为 {buy_ratio:.0%}")
            
        # 应用最终计算出的买入比例
        if buy_ratio != 1.0:
            planned_total_quantity = int(planned_total_quantity * buy_ratio)

        ## 增加特殊股票购买比例&完成每日盈利后降低购买比例
        is_target_hit = False
        if market == MarketType.HK and self.daily_hk_profit_target_hit:
            is_target_hit = True
        elif market == MarketType.US and self.daily_us_profit_target_hit:
            is_target_hit = True
        
        # 如果盈利达标且防御模式开启，应用打折
        if is_target_hit and self.config.defensive_mode_enabled:
            logger.warning(f"🛡️ [{symbol}] 盈利达标，防御模式激活！买入比例限制为 {self.config.defensive_buy_ratio:.0%}")
            buy_ratio_cap = self.config.defensive_buy_ratio
        else:
            buy_ratio_cap = 1.0 # 正常模式不限制上限

        planned_total_quantity = int(planned_total_quantity * self.config.position_scaling_factors.get(symbol,self.config.position_scaling_factors['default']) * buy_ratio_cap)
        
        # a. 获取真实lot_size
        symbol_info = get_stock_static_info(self.quote_ctx, symbol)
        actual_lot_size = symbol_info.get('lot_size', self.config.lot_size) # 提供一个默认值以防API没有
        planned_total_quantity = self._adjust_quantity(planned_total_quantity,market,lot_size=actual_lot_size)
        if planned_total_quantity <= 0:
            logger.error(f"计算股数为0，无法买入 {symbol}。最大亏损额={max_loss_per_trade:.2f}, 每股风险={per_share_risk:.2f}")
            
            planned_total_quantity = symbol_info.get('lot_size', self.config.lot_size)
            if market == MarketType.US:
                if current_price>=100:
                    planned_total_quantity = planned_total_quantity * 2
                else:
                    planned_total_quantity = planned_total_quantity * self.config.us_lot_size
        
        # 确定本次“侦察仓”要买入的数量
        # 确定本次“第一枪”的数量 (3-3-4模型的第一步)
        if self.config.enable_scale_in_strategy:
            # quantity_to_buy = int(planned_total_quantity * self.config.initial_position_scale_ratio)
            quantity_to_buy = int(planned_total_quantity * buy_percentage)
            # quantity_to_buy = int(planned_total_quantity * buy_percentage * self.config.global_risk_multiplier)
            log_msg_prefix = "分批建仓-首次侦察"
        else:
            # 如果是一次性建仓，也应该受系数影响 (比如 Risky 股即使一次性买也得是 0)
            # 但这里我们假设一次性建仓就是满仓干，或者你可以应用一个系数
            # 极客建议：保持一致性，使用系数打折
            # 系数 = buy_percentage / base_config_ratio
            scaling_factor = buy_percentage / self.config.initial_position_scale_ratio
            quantity_to_buy = int(planned_total_quantity * scaling_factor)
            # quantity_to_buy = planned_total_quantity
            log_msg_prefix = "一次性建仓"
        
        quantity_to_buy = self._adjust_quantity(quantity_to_buy, market, lot_size=actual_lot_size)
        
        # 如果风险模型计算出的数量小于等于0，进行一次最小数量的保底尝试
        if quantity_to_buy <= 0:
            logger.warning(f"风险模型计算股数为0，尝试使用最小手数进行保底买入 {symbol}")
            quantity_to_buy = planned_total_quantity * self.config.initial_position_scale_ratio * 0.80
            # if market == MarketType.US:
            #     if current_price >= 100:
            #         quantity_to_buy = 2  # 对于高价美股，保底买2股
            #     else:
            #         quantity_to_buy = self.config.us_lot_size

        ### ▼▼▼【核心修改：双漏斗约束 + 确定性成本下单】▼▼▼ ###
        # 这是解决问题的关键。我们不再猜测市价单的成本，而是主动控制它。

        # --- 漏斗1: 风险漏斗 ---
        # risk_based_quantity 就是我们刚刚计算出的 quantity_to_buy，代表我们“想”买多少。
        risk_based_quantity = quantity_to_buy
        logger.info(f"[{symbol}] 风险漏斗计算结果 (理想仓位): {risk_based_quantity} 股")

        # --- 漏斗2: 资金漏斗 ---
        # 我们必须检查钱包，看看到底“能”买多少。
        available_cash = self.get_available_cash(market)
        
        # 复用 reserve_ratio (0.1%) 作为最小成本的购买力安全垫，避免100%满仓下单
        # 这是为了满足你“不引入新变量”的要求，一个优雅的复用。
        buffered_buying_power = available_cash * (1 - self.config.reserve_ratio)
        
        # 设定一个略微上浮的限价（+0.3%），确保成交率的同时锁定最大成本
        limit_price = round(current_price * 1.003, 3)

        affordable_quantity = 0
        if limit_price > 0:
            affordable_quantity = int(buffered_buying_power / limit_price)
            affordable_quantity = self._adjust_quantity(affordable_quantity, market, lot_size=actual_lot_size)
        
        logger.info(f"[{symbol}] 资金漏斗计算结果 (最大可买): {affordable_quantity} 股 (基于可用资金 {available_cash:.2f})")

        # --- 最终决策：取两者的最小值 ---
        # 最终下单数量，既不能超出风险预算，也不能超出钱包厚度。
        quantity_to_buy = min(risk_based_quantity, affordable_quantity)
        
        if quantity_to_buy <= 0:
            logger.error(f"[{symbol}] 无法开仓，最终计算数量为0。风险理想: {risk_based_quantity}, 资金可买: {affordable_quantity}。")
            return
            
        logger.warning(f"[{symbol}] 双漏斗决策完成，最终买入数量: {quantity_to_buy} 股")

        # === 3. 最终风控检查: 检查总组合风险敞口 ===
        total_capital_base = self.get_total_account_value_in_hkd()
        if total_capital_base <= 1.0:
            logger.critical(f"🛑 [{symbol}] 严重异常：获取账户总资产为 {total_capital_base}！计算出的仓位将强制为 0！请检查 API 连接。")
            return

        if not self.risk_manager.can_open_new_position(self.positions, market, total_capital_base, per_share_risk, quantity_to_buy):
            logger.error(f"无法开仓 {symbol}，总投资组合风险敞口超限。")
            # send_email(subject=f'开仓/加仓被总风险策略拒绝-{symbol}', content=f"无法开仓 {symbol}，总投资组合风险敞口超限。")
            # return

            quantity_to_buy = quantity_to_buy * 0.5
            if not self.risk_manager.can_open_new_position(self.positions, market, total_capital_base, per_share_risk, quantity_to_buy):
                quantity_to_buy = symbol_info.get('lot_size', self.config.lot_size)

                if market == MarketType.US:
                    if current_price>=100:
                        quantity_to_buy = 2
                    else:
                        quantity_to_buy = quantity_to_buy * self.config.us_lot_size
           
        logger.info(f"{log_msg_prefix} {symbol}: 止损价={stop_loss_price:.3f}, 计划总仓位={planned_total_quantity}, 本次买入={quantity_to_buy}")

        if quantity_to_buy <= 0:
            logger.error(f"按首次建仓比例计算后股数为0，无法买入 {symbol}")
            return
        order_id = self.submit_order(symbol, quantity_to_buy, OrderSide.Buy)
        if not order_id: return
        
        # 更封装了完整的“决策快照”，为未来的策略复盘和AI训练提供最高质量的养料。
        try:
            symbol_info = self.get_cached_stock_static_info(symbol)
            symbol_name = symbol_info.get('name_cn') or symbol_info.get('name_en')
            trade_cost = current_price * quantity_to_buy
            currency = "HKD" if market == MarketType.HK else "USD"

            # 构造一个无懈可击的、专为数据分析而生的结构化日志。
            # 每个字段都经过深思熟虑，缺一不可。这才是专业。
            log_payload = (
                f"symbol:{symbol},"
                f"name:{symbol_name},"
                f"action:INITIAL_BUY,"  # 行为：明确这是建仓操作
                f"price:{current_price:.3f},"  # 价格：触发时的市场价格
                f"quantity:{quantity_to_buy},"  # 数量：本次实际下单数量
                f"trade_cost:{trade_cost:,.3f},"  # 金额：本次交易名义成本
                f"currency:{currency},"  # 币种
                f"strategy_name:{strategy_name},"  # 策略：哪个策略产生的信号
                f"initial_stop_loss:{stop_loss_price:.3f},"  # 【风控核心】初始止损价：这笔交易的安全底线
                f"per_share_risk:{per_share_risk:.3f},"  # 【风控核心】每股风险(R)：衡量这笔交易风险的基本单位
                f"planned_total_quantity:{planned_total_quantity},"  # 计划：按风险模型计算的总计划仓位
                f"risk_ratio:{adjusted_risk_per_trade_ratio:.4f},"  # 风险敞口：动用了总资金多大比例的风险
                f"market_regime:{current_regime.value},"  # 【宏观环境】大盘状态：这决定了我们是在顺风还是逆风操作
                f"trading_window:{current_status.name},"  # 【微观环境】交易窗口：我们是在日内哪个时间节点扣下的扳机
                f"order_id:{order_id}",  # 订单ID：与券商对账的唯一凭证，无可辩驳
                f"is_sharp_knife:{is_sharp_knife}"
            )
            buy_logger.info(log_payload)
        except Exception as e:
            # 即使日志记录失败，也绝不能影响交易主流程。这是系统的鲁棒性。
            logger.error(f"[{symbol}] 记录买入日志时发生致命错误: {e}", exc_info=True)

        with self.pending_orders_lock:
            # 将计算好的止损价和风险也存入待处理订单
            self.pending_orders[symbol] = {
                "order_id": order_id,
                "plan_info": {
                    "planned_total_quantity": planned_total_quantity,
                    "initial_risk_per_share": round(per_share_risk,3),
                    "initial_stop_loss_price": round(stop_loss_price,3), # 传递止损价
                    "triggering_strategy": strategy_name,
                    "strategy_class_name": candidate.get('strategy_class_name'),
                    "strategy_params": candidate.get('strategy_params'),
                    "confirmation_add_done":candidate.get('final_confirmation',True),
                    "building_stage": building_stage,
                }
            }

        self._save_pending_orders()
        logger.info(f"买入订单已提交并加入待处理队列 {symbol} | ID: {order_id}")

        # ==============================================================================
        # ▼▼▼【核心权力交接】▼▼▼
        # 在此处，当且仅当订单提交成功、待处理记录创建完毕后，才从缓存中移除该信号。
        # 这完美地遵循了“行为确认原则”，确保了操作的原子性和最终一致性。
        # 任何在此之前的失败（如风控检查、价格计算错误等）都绝不会触达这里，从而保留了信号的有效性。
        # ------------------------------------------------------------------------------
        self.pending_buy_cache.remove_signal(symbol)
        # ▼▼▼【解除飞行锁定】▼▼▼
        with self.signals_in_flight_lock:
            self.signals_in_flight.discard(symbol) # 使用 discard 更安全
        # ▲▲▲【解除飞行锁定】▲▲▲

        logger.info(f"信号 {symbol} 已成功转化为待处理订单，从待买入缓存中消费完毕。")
        # ==============================================================================

    def _check_trade_safety_gate(self, symbol: str,  candidate: dict = None,is_regular_open:bool = True) -> Tuple[bool, str]:
        """
        [极简风控-5大场景自适应版]
        
        核心哲学：
        只拦截真正危险的“高位逼空”，对于“水下反转”和“低开修复”给予最大宽容。
        构建立体防御网：
        1. 策略意图 (Strategy): 决定基准水位。
        2. 相对位置 (Rank): 决定当前高度。
        3. 涨幅 (ROI): 决定压制力度。
        4. 防飞刀机制：跌幅过大(-8%)、急跌(5min -2.5%)、底部未确认(0.25%)，统统拦截。
        """
        
        quote_data = None
        
        if is_regular_open:
            q = get_raw_quote(self.quote_ctx, symbol)
            if q:
                quote_data = {
                    'last': float(q.last_done),
                    'high': float(q.high),
                    'low': float(q.low),
                    'open': float(q.open),
                    'prev_close': float(q.prev_close),
                    'volume': float(q.volume),
                    'turnover': float(q.turnover) 
                }
        else:
            q = self.hs_data_provider.get_smart_quote(symbol)
            if q:
                quote_data = {
                    'last': float(q.get('last_price', 0)),
                    'high': float(q.get('high_price', 0)),
                    'low': float(q.get('low_price', 0)),
                    'open': float(q.get('open_price', 0)),
                    'prev_close': float(q.get('prev_close_price', 0)),
                    'volume': float(q.get('volume', 0)),
                    'turnover': float(q.get('turnover', 0))
                }
        
        if not quote_data or quote_data['last'] <= 0:
            return True, "无有效行情数据"

        current_price = quote_data['last']
        high_price = quote_data['high']
        low_price = quote_data['low']
        open_price = quote_data['open']
        prev_close_price = quote_data['prev_close']
        volume = quote_data['volume']
        turnover = quote_data['turnover']

        # === 场景 1: 大盘震荡/织布机 (Volatility Check) ===
        # 逻辑：如果全天振幅连 0.5% 都不到，所谓的“高点”就是噪音。
        day_range_pct = (high_price - low_price) / prev_close_price
        if day_range_pct < 0.005: # 0.5% 的振幅忽略不计
            return False, f"波动极小({day_range_pct:.2%})，豁免所有压制"

        # --- 核心坐标计算 ---
        # A. 相对位置 Rank (0.0 - 1.0)
        day_range = high_price - low_price
        if day_range <= 1e-6: position_val = 1.0 # 极小波动防除零
        else: position_val = (current_price - low_price) / day_range
        
        # B. 当日涨幅 ROI
        daily_roi = (current_price - prev_close_price) / prev_close_price if prev_close_price > 0 else 0
        current_roi = (current_price - open_price) / open_price if open_price > 0 else daily_roi

        # ==============================================================================
        # ▼▼▼ 防接飞刀三道防线 (The Safety Gates) ▼▼▼
        # ==============================================================================

        # --- 第一道防线：人气/趋势熔断 (The Popularity Gate) ---
        # 逻辑：跌幅超过8%，视为趋势崩坏或基本面暴雷，此时大概率是下跌中继，禁止接盘。
        # 除非策略名明确带有“超跌”字样，否则一刀切。
        if daily_roi < -0.08:
            # 你可以在这里加白名单逻辑，但根据你的要求，先严格执行
            return True, f"人气熔断: 跌幅({daily_roi:.2%})过深(<-8%)，趋势已崩坏，拒绝接盘"

        # [子逻辑 B] 暴涨/过热熔断 (Overheat Check)
        overheat_threshold = 0.05
        # 动态设定阈值：妖股/高波票给予 8% 空间，普通票锁死 5%
        if symbol in self.config.high_vol_symbols:
            overheat_threshold = 0.08

        if current_roi > overheat_threshold:
            return True, f"人气过热: 涨幅({current_roi:.2%}) > 阈值({overheat_threshold:.2%})，趋势涨幅过大，拒绝接盘"

        # --- 第二道防线：动能刹车 (The Velocity Brake) ---
        # 逻辑：检查5分钟内是否发生急跌(>2.5%)。如果正在瀑布流中，强制冷却。
        # 优化：为了节省API资源，只有当日已经跌了超过 1.5% 时，才去检查是不是刚刚发生的急跌。
        if daily_roi < 0.018: # 从 -0.015 改为 0.018
            try:
                klines = get_klines_data(self.quote_ctx, symbol, 6, Period.Min_5, AdjustType.NoAdjust)
                if klines is not None and len(klines) > 0:
                    # 获取区间内的最高点（可能是5分钟前的价格）
                    recent_high = klines['high'].max()
                    if recent_high > 0:
                        velocity_drop = (current_price - recent_high) / recent_high
                        # 如果5分钟内瞬时跌幅超过 2.5%
                        if velocity_drop < -0.025:
                            return True, f"动能刹车: 5分钟内闪崩({velocity_drop:.2%})，正在接飞刀，强制冷却"
            except Exception as e:
                # 容错处理：如果K线获取失败，不阻塞流程，但记录警告
                logger.warning(f"[{symbol}] 动能刹车检查跳过 (数据源异常): {e}")

        strategy_name = str(candidate.get('strategy_name', '')).lower() if candidate else ''
        # --- 第三道防线：J型钩/底部确认 (The J-Hook Confirmation) ---
        # 逻辑：拒绝买在当日绝对最低点。强制要求现价必须从最低点反弹一定幅度 (0.25% - 0.3%)。
        if low_price > 0:
            # 1. 动态确定反弹阈值 (Threshold Selection)
            # 默认使用普通股阈值 (0.25%)
            rebound_threshold = self.config.j_hook_threshold_map.get('normal', 0.0025)
            
            # 如果是“高波动/杠杆名单”中的股票，强制提升门槛 (0.32%)
            if symbol in self.config.high_vol_symbols:
                rebound_threshold = self.config.j_hook_threshold_map.get('high_vol', 0.0032)
                # logger.debug(f"[{symbol}] 识别为高波动标的，J-Hook 确认阈值提升至 {rebound_threshold:.2%}")

            # 2. 计算及格线 (Required Price)
            # 现价必须 > 最低价 * (1 + 阈值)
            required_price = low_price * (1 + rebound_threshold)

            # 3. 执行检查 (Execution)
            is_underwater = current_price < open_price
            force_jhook_keywords = [
                '低点', '企稳', 'stall',      # IntradayLowStall
                'reversal', '反转', 'macd',  # MacdReversal
                'w底', 'w_bottom',           # NarrativeWBottom
                'ambush', '伏击',            # Ghostblade
                'secondary', '二次', '回调',  # SecondaryStrike
                # '雷霆确认','闪电突袭'          # QuantumBlitzStrategy
                'dip', '抄底'                # 通用关键词
            ]
            is_force_jhook_strategy = any(k in strategy_name for k in force_jhook_keywords)
            # 仅当股票处于“水下”(ROI < 0) 时执行
            if is_underwater or is_force_jhook_strategy:
                # 情况 A: 价格还在及格线之下 -> 拒绝 + 重置计时器
                if current_price < required_price:
                    # 如果之前开始计时了，说明反弹夭折，必须重置计时器
                    if candidate and 'jhook_conf_start_ts' in candidate:
                        del candidate['jhook_conf_start_ts']
                        if random.random() > 0.8:
                            logger.warning(f"[{symbol}] J-Hook反弹夭折，计时器已重置。")

                    actual_rebound = (current_price - low_price) / low_price
                    return True, f"[J-HOOK]底部未确认: 距最低点仅反弹 {actual_rebound:.2%} (<{rebound_threshold:.1%})，拒绝左侧博弈"
                
                # 情况 B: 价格已经站上及格线 -> 进入时间确认流程 (Time Debounce)
                else:
                    if candidate is not None:
                        now_ts = time.time()
                        # 获取配置的确认时间，默认120秒
                        required_duration = self.config.night_config.get('verify_seconds', 120) if not is_regular_open else 120
                        
                        # 1. 如果是第一次站上，初始化计时器
                        if 'jhook_conf_start_ts' not in candidate:
                            candidate['jhook_conf_start_ts'] = now_ts
                            return True, f"[J-HOOK] 价格达标，开始{required_duration}秒时间确认..."
                        
                        # 2. 检查持续时间
                        elapsed = now_ts - candidate['jhook_conf_start_ts']
                        if elapsed < required_duration:
                            return True, f"[J-HOOK] 底部确认中: 已维持 {elapsed:.0f}/{required_duration}s"
                        
                        # 3. 时间达标，放行！
                        if elapsed >= required_duration:
                            if random.random() > 0.85:
                                logger.warning(f"⏳✅ [{symbol}] J-Hook时间防抖通过: 价格{current_price:.3f} 已站稳 {elapsed:.0f}s (>{required_duration}s)，允许开仓。")
                            pass
                        
                    
            # 仅当股票处于“水下”(ROI < 0) 或“探底”(Close < Open) 状态时，才严格执行此逻辑。
            # 如果股票已经翻红且涨势如虹 (daily_roi > 0)，则 Day Low 可能是很久以前的事了，此时不应受此限制。
            # if daily_roi < 0 and current_price < required_price:
            #     actual_rebound = (current_price - low_price) / low_price
            #     return True, f"[J-HOOK]底部未确认: 距最低点仅反弹 {actual_rebound:.2%} (<{rebound_threshold:.1%})，转入动态寻底"
            #     # return True, f"底部未确认: 距最低点仅反弹 {actual_rebound:.2%} (<{rebound_threshold:.1%})，拒绝左侧博弈"

        # --- 3. 动态阈值矩阵  ---
        
        # 关键词匹配
        dip_keywords = ['低点', '企稳', '兜底', '抄底', '回调', 'dip', 'bottom','伏击','闪电','雷霆']
        trend_keywords = ['突破', '新高', '趋势', 'macd', 'breakout', 'trend']

        is_dip_strategy = any(k in strategy_name for k in dip_keywords)
        is_trend_strategy = any(k in strategy_name for k in trend_keywords)

        # [A. 基准水位设定]
        if is_dip_strategy:
            # 抄底策略：严控在日内下半区。0.45 给了 5% 的容错空间 (相比 0.40)
            allowed_max_rank = 0.45
            log_tag = "左侧基准"
        elif is_trend_strategy:
            # 趋势策略：允许追击日内高点，防止踏空
            allowed_max_rank = 0.92
            log_tag = "右侧基准"
        else:
            # 中性策略
            allowed_max_rank = 0.75
            log_tag = "中性基准"

        # [B. ROI 暴力压制]
        # 即使是趋势策略，如果当天已经涨飞了，也不允许在天花板接盘
        if daily_roi > 0.05: # 涨超 5%
            allowed_max_rank = min(allowed_max_rank, 0.50) # 强制要求回撤到中轴
            log_tag += "/ROI暴涨压制"
        elif daily_roi > 0.03: # 涨超 3%
            allowed_max_rank = min(allowed_max_rank, 0.70) # 只能接7成以下
            log_tag += "/ROI大涨压制"

        # --- 天条一：VWAP 成本豁免权 (The Safe Haven) ---
        # 计算 VWAP (Volume Weighted Average Price)
        vwap = 0.0
        try:
            # 强制转 float 防止类型问题，防御 0 成交量
            vol_float = float(volume)
            to_float = float(turnover)
            if vol_float > 0 and to_float > 0:
                vwap = to_float / vol_float
        except Exception:
            vwap = 0.0 # 计算异常时保守处理，不触发豁免
        
        # 逻辑：现价 < 均价，意味着你的买入成本优于今天市场上大多数人。
        # 在这种物理状态下，“追高”是一个伪命题。你是在“低吸”或者博弈“均值回归”。
        # 单边阴跌的股票永远在 VWAP 下方，不能因此豁免。
        # 必须加一个条件：不能是深跌阴线 (比如跌幅超过 -1.5% 且 Current < Open)。
        is_crashing = daily_roi < -0.015 and current_price < open_price

        # 不仅要低于 VWAP，还要剔除单边下跌的形态
        # 1. 既然在均价下方，说明趋势偏弱，此时绝对不能是正在下跌的阴线（Close < Open）
        # 2. 或者要求它至少比 N 分钟前的价格高（有反弹力度）
        is_falling_knife = False
        if is_regular_open: # 仅在常规交易时间做微观判定
             try:
                 # 获取最近1根1分钟K线，看是不是正在杀跌
                 last_kline = get_klines_data(self.quote_ctx, symbol, 1, Period.Min_1, AdjustType.NoAdjust)
                 if last_kline is not None and not last_kline.empty:
                     k_close = last_kline['close'].iloc[-1]
                     k_open = last_kline['open'].iloc[-1]
                     # 如果当前这1分钟是阴线，且跌幅明显，视为飞刀
                     if k_close < k_open:
                         is_falling_knife = True
             except:
                 pass
             
        if vwap > 0 and current_price < vwap and not is_crashing and not is_falling_knife:
            # 锁：如果已经涨超 3%，或者位置在 0.8 以上，VWAP 失效
            # 只有 Dip 策略需要额外检查是不是假摔 (Rank > 0.6 依然危险)
            if daily_roi < 0.03 and not (is_dip_strategy and position_val > 0.60):
                 # 只有满足这些条件，VWAP 才是有效的护身符
                 # 返回 False 表示不拦截 (Accept)
                 return False, f"VWAP豁免: 现价({current_price:.3f}) < 均价"
            # 否则：虽然低于 VWAP 但风险高，继续向下执行 Rank 检查
        
        # --- 天条二：ATR 头部空间约束---
        atr_value = get_historical_atr(self.quote_ctx, symbol)
        # 如果是抄底策略，将 ATR 空间要求从 0.2 降低到 0.01【0.005】，或者直接跳过
        min_atr_factor = 0.1 if is_dip_strategy else 0.2
        min_headroom = atr_value * min_atr_factor if (atr_value and atr_value > 0) else (high_price - low_price) * min_atr_factor
        actual_headroom = high_price - current_price
        
        # 逻辑：如果上方空间已经被吃得只剩渣了，哪怕形态再好也不做“接盘侠”。
        # 注意：只有在波动率正常(>0.5%)时才拦截，避免织布行情误杀。
        # 只有在波动率正常时才拦截
        if day_range_pct > 0.005:
            if actual_headroom < min_headroom:
                # 构造更详细的拒绝理由
                atr_info = f"{atr_value:.3f}" if atr_value else "N/A"
                return True, f"ATR空间拦截: 距日内高点空间({actual_headroom:.3f}) < 0.2ATR({min_headroom:.3f}, ATR={atr_info})，盈亏比极差"
        
        # === 核心计算：相对位置 ===
        # 我们在哪里？ 0.0 = 最低点, 1.0 = 最高点
        if high_price - low_price <= 1e-6: position_val = 1.0 # 极小波动防除零
        else: position_val = (current_price - low_price) / (high_price - low_price)
        
        # 离高点的距离 (百分比)
        dist_to_high_pct = (high_price - current_price) / current_price
        is_opportunistic_bottom_probe = candidate.get('is_opportunistic_bottom_probe',False)

        # ==============================================================================
        # ▼▼▼ 5大场景 智能分流 (Smart Routing) ▼▼▼
        # ==============================================================================

        # === 场景 3 & 5: 低走 (Intraday Bearish / Underwater) ===
        # 无论是低开低走，还是高开低走，只要 现价 < 开盘价 (阴线)
        # 说明我们在“水下”博反弹。此时离日内高点肯定很远，或者高点没有压制意义。
        if current_price < open_price:
            # 只有 ROI 处于安全区 (比如 < 1.5%) 才直接放行
            if daily_roi < 0.015:
                return False, "水下阴线策略，无视高点压制"
            else:
                # 涨幅过大，即使是阴线也不安全，继续向下检查
                logger.debug(f"[{symbol}] 高位阴线 (ROI {daily_roi:.2%})，取消直接豁免")

        # === 场景 4: 低开高走 (Gap Down & Recovery) ===
        is_gap_down = open_price < prev_close_price * 0.998 # 低开超过 0.2%
        if is_gap_down and current_price >= open_price:
            # 1. 检查是否已经翻红 (收复失地)
            is_turned_green = current_price > prev_close_price
            
            # 2. 如果还在水下 (Current < Prev Close)，我们视为"弱势修复"，给予宽容度
            if not is_turned_green:
                # 在水下，我们允许 Rank 稍微高一点 (放宽到 0.75)，因为绝对价格便宜
                # 同时也检查是否贴死高点 (dist < 0.3%)
                if position_val <= 0.75 and dist_to_high_pct >= 0.003:
                    # 只有在这里，才给予特权放行
                    return False, "低开水下修复，放宽压制"
                
                # 如果在水下但 Rank > 0.75，说明是水下震荡的高点，也不接，让它落入下方常规检查被拦截
                logger.debug(f"[{symbol}] 水下反弹位置偏高 (Rank {position_val:.2f}), 失去Gap特权")
            
            else:
                # 3. 如果已翻红 (TNA/MP 情况)，特权立即失效！
                # 这种情况下，它必须像普通股票一样，接受下方"常规形态"的严格检查 (Rank < 0.45)。
                logger.debug(f"[{symbol}] 低开但已翻红 (Gap Fill), 失去Gap特权，转入常规风控")
                pass

        # === 场景 2: 高开高走 (Gap Up / Strong Trend) ===
        # 这是最危险的。昨天收100，今天开101，现在105。
        # 全民狂欢，这时候最容易接盘。必须严格！
        is_gap_up = open_price > prev_close_price * 1.002
        if is_gap_up and current_price > open_price:
            # 只有非趋势策略，才锁死在 0.80，趋势策略跟随 dynamic rank
            limit_rank = allowed_max_rank if is_trend_strategy else min(0.80, allowed_max_rank)

            if position_val > limit_rank: # 如果处于日内极高位
                # 这里只拦截 Rank 过高，不再强制要求贴近日内高点
                logger.error(f"股票代码:{symbol},风控拦截[高位逼空]: 位置({position_val:.2f}) > 上限({limit_rank:.2f})")
                return True, f"风控拦截[高位逼空]: 位置({position_val:.2f}) > 上限({limit_rank:.2f})"

        # === 常规形态 (平开高走) ===
        # 不属于以上极端的，按标准逻辑：别买在天花板
        # 只要留出 0.2% 的空间，或者不在日内最高 5% 的位置即可
        # 修改点：常规行情波动小，更不能买在天花板。
        # 阈值调整：95%->90%, 0.2%->0.4%
        # 逻辑修复：只要位置 Rank 超标，必须拦截！不管离最高点远不远。
        # 这是为了杀 APPX 这种在 0.46 (超标) 但离最高点还有距离的垃圾时间交易。
        if position_val > allowed_max_rank:
            logger.error(f"股票代码:{symbol},位置风控拦截: 位置({position_val:.2f}) > 动态上限({allowed_max_rank:.2f}) (策略:{strategy_name})")
            return True, f"位置风控拦截: 位置({position_val:.2f}) > 动态上限({allowed_max_rank:.2f}) (策略:{strategy_name})"

        # 即使 Rank 没超标，如果离天花板太近 (<0.2%)，且不是强势趋势策略，也要拦截
        if dist_to_high_pct < 0.002 and not is_trend_strategy:
            logger.error(f"股票代码:{symbol},风控拦截[触顶风险]: 贴死天花板 ({position_val:.2f})")
            return True, f"风控拦截[触顶风险]: 贴死天花板 ({position_val:.2f})"
        
        # ==============================================================================
        # ▼▼▼ 微观N分钟结构检查 (Micro Check) ▼▼▼
        # 需求：检查N分钟内是否最高点？是否低于N分钟均值？
        # ==============================================================================
        if is_regular_open:
            try:
                # 1. 获取动态N分钟
                # 妖股/早盘看短(5m)，午盘/蓝筹看长(10m)
                # 注意：这里需要确保 get_trading_window_status 和 get_dynamic_k_minutes 已导入
                current_status = get_trading_window_status(symbol)
                market = get_market_type(symbol)
                k_mins = get_dynamic_k_minutes(current_status, market)
                
                # 2. 获取微观K线
                # 多取 2 根作为 buffer
                df_micro = get_klines_data(self.quote_ctx, symbol, count=k_mins + 2, period=Period.Min_1, adjust_type=AdjustType.NoAdjust)
                
                if df_micro is not None and not df_micro.empty:
                    # 截取最近的 k_mins 数据
                    recent_df = df_micro.tail(k_mins)
                    # 开盘冷启动防御：如果K线少于2根，无法定义"结构"，直接放行
                    if len(recent_df) >= 2:
                        # 3. 计算关键指标
                        micro_high = recent_df['high'].max()
                        micro_avg = recent_df['close'].mean()
                        
                        # --- 检查 A: 是否 N 分钟最高点? (Headroom Check) ---
                        # 逻辑：如果现价 >= N分钟最高价的 99.95%，说明正贴着天花板，风险极大
                        if current_price >= micro_high * 0.9995:
                            # 除非是明确的“趋势突破”策略，否则一律拦截
                            if not is_trend_strategy:
                                return True, f"微观拦截: 触及{k_mins}分高点({micro_high:.2f})，拒绝接盘"

                        # --- 检查 B: 是否低于 N 分钟均值? (Mean Reversion) ---
                        # 逻辑：你的需求核心。如果是抄底/低吸策略，必须买在均线下方。
                        if is_dip_strategy:
                            # 允许 0.1% 的容错 (current_price > avg * 1.001)
                            if current_price > micro_avg * 1.001:
                                return True, f"微观拦截: 抄底需在均线({micro_avg:.2f})下方，现价({current_price:.2f})过高"
            except Exception as e:
                # 微观检查出错不阻断交易，仅打印日志
                logger.error(f"[{symbol}] 微观结构检查异常: {e}")                
        
        return False, "风控通过"
    
    def _check_trade_safety_gate_mini(self, symbol: str, candidate: dict = None, is_regular_open: bool = True) -> Tuple[bool, str]:
        """
        [极简风控-跟单特供版] (The Execution Guard)
        
        全球第二的杠精工程师认证：
        此方法专为 Slave (跟随端) 设计。
        核心哲学：
        1. 绝对信任 Master (长桥) 的策略判断（形态、J-Hook、Rank等）。
        2. 既然大哥说买，小弟只负责检查“能不能买”和“会不会死”。
        3. 坚决剔除 J-Hook/寻底逻辑，防止因数据微差导致的“左右互搏”和“死循环寻底”。
        
        保留防线：
        1. 数据有效性 (Data Validity): 没有行情不买。
        2. 灾难级熔断 (Catastrophe Stop): 跌幅超过 -15% (长桥可能疯了，小弟要保命)。
        3. 停牌/流动性枯竭检查。
        """
        
        # --- 1. 获取行情数据 (复用原有逻辑) ---
        quote_data = None
        
        if is_regular_open:
            q = get_raw_quote(self.quote_ctx, symbol)
            if q:
                quote_data = {
                    'last': float(q.last_done),
                    'open': float(q.open),
                    'prev_close': float(q.prev_close),
                    'volume': float(q.volume)
                }
        else:
            q = self.hs_data_provider.get_smart_quote(symbol)
            if q:
                quote_data = {
                    'last': float(q.get('last_price', 0)),
                    'open': float(q.get('open_price', 0)),
                    'prev_close': float(q.get('prev_close_price', 0)),
                    'volume': float(q.get('volume', 0))
                }
        
        # [风控门禁 1] 数据死活检查
        if not quote_data or quote_data['last'] <= 0:
            return True, "❌ [数据异常] 无有效行情数据或价格为0"

        current_price = quote_data['last']
        # prev_close_price = quote_data['prev_close']
        volume = quote_data['volume']
        
        # [风控门禁 2] 停牌/流动性枯竭检查
        # 盘中且成交量为0，大概率停牌或数据断流
        if is_regular_open and volume <= 0:
             return True, "❌ [流动性异常] 盘中成交量为0，疑似停牌或数据中断"

        # [风控门禁 4] 价格偏离度检查 (可选，防止滑点过大)
        # 如果当前价比如 trigger_price 高出太多(>1.5%)，说明网络延迟期间已经飞了，追高有风险
        # 这里为了保证成交率，暂不开启，或者阈值设宽一点
        trigger_price = candidate.get('trigger_price', 0)
        if trigger_price > 0:
            slippage = (current_price - trigger_price) / trigger_price
            if slippage > 0.02: # 偏离超过 2%
                return True, f"⚠️ [滑点保护] 现价({current_price}) 比触发价({trigger_price}) 偏离 {slippage:.2%}，放弃追高"

        # --- 通过所有检查 ---
        # 不需要 J-Hook，不需要 Rank，不需要 VWAP
        # 尚方宝剑在此，诸邪退散！
        return False, "✅ [执行风控通过] 跟单指令确认有效"
    
    def _check_pending_initial_buy_orders(self):
        """
        检查待处理的**开仓订单**状态。
        如果订单成交，则创建持仓对象；如果失败或取消，则从待处理列表中移除。
        此方法保证了开仓事务的原子性和状态恢复能力。
        """
        # 1. 快照读取 keys，减少锁竞争
        with self.pending_orders_lock:
            if not self.pending_orders:
                return
            # 创建副本避免长时间持有锁
            pending_symbols = list(self.pending_orders.keys())
        
        # 后续处理使用副本，避免锁竞争
        if pending_symbols:
            logger.debug(f"正在检查 {len(pending_symbols)} 个待处理开仓订单...")
        
        for symbol in pending_symbols:
            try:
                # --- 获取订单元数据 ---
                order_id = None
                with self.pending_orders_lock:
                    pending_data = self.pending_orders.get(symbol)
                    if not isinstance(pending_data, dict):
                        if symbol in self.pending_orders: del self.pending_orders[symbol]
                        continue
                    order_id = pending_data.get("order_id")
                    if not order_id: 
                        continue
                
                kwargs = {
                    'symbol': symbol
                }
                order_detail = self.data_provider.get_order_detail(order_id,**kwargs)
                
                if not order_detail:
                    logger.warning(f"无法获取订单 {order_id} ({symbol}) 的详情，将在下次检查时重试。")
                    continue

                status = order_detail.get('status')

                if status == "Filled":
                    avg_price = float(order_detail.get('price', 0.0))
                    filled_quantity = int(order_detail.get('quantity', 0))

                    if avg_price <= 0 or filled_quantity <= 0:
                        logger.error(f"订单 {order_id} ({symbol}) 状态为 'Filled' 但成交价或数量无效。Price: {avg_price}, Qty: {filled_quantity}")
                        continue
                    
                    logger.info(f"检测到待开仓订单已成交 {symbol}: ID={order_id}, 均价={avg_price}, 数量={filled_quantity}")
                    
                    with self.position_lock, self.pending_orders_lock:
                        if symbol not in self.pending_orders:
                            logger.info(f"待处理订单 {symbol} 已被并发处理，跳过。")
                            continue
                        
                        if symbol in self.positions:
                            logger.error(f"严重错误：在处理待开仓订单 {symbol} 时，持仓已存在。将清理待处理订单以避免重复。")
                            del self.pending_orders[symbol]
                            self._save_pending_orders()
                            continue
                        
                        plan_info = self.pending_orders[symbol].get("plan_info", {})
                        self._finalize_position_creation(symbol, avg_price, filled_quantity, plan_info)
                        
                        del self.pending_orders[symbol]

                        self._save_positions()
                        self._save_pending_orders()

                elif status in ["Canceled", "Rejected", "Expired"]:
                    logger.error(f"待开仓订单执行失败或已失效 {symbol}: ID={order_id}, 状态={status}")
                    with self.pending_orders_lock:
                        if symbol in self.pending_orders:
                            del self.pending_orders[symbol]
                    self._save_pending_orders()
            except Exception as e:
                logger.error(f"检查待开仓订单 {symbol} 状态时出错: {e}", exc_info=True)
            time.sleep(0.5)

    # ==============================================================================
    # 仓位动态计算引擎
    # ==============================================================================
    def _calculate_dynamic_position_ratio(self, symbol: str, base_ratio: float) -> float:
        """
        【仓位裁决法庭】
        基于 静态身份(VIP) + 动态评级(AI Tier) 的正交矩阵计算最终买入比例。
        
        矩阵逻辑 (Multiplier Matrix):
        ----------------------------------------------------------------------------
        | 身份\评级    |  Risky (黑名单) |  Tier 1 (优) |  Tier 2 (良)  |  None (无)   |
        |-------------|----------------|--------------|--------------|-------------|
        | VIP         |  0.0 (熔断)     |  1.3x (激进) |  1.0x (标准)   |  0.6x (防守)|
        | Non-VIP     |  0.0 (熔断)     |  0.9x (关注) |  0.7x (普通)   |  0.5x (轻仓)|
        ------------------------------------------------------------------------------
        
        Args:
            symbol: 股票代码
            base_ratio: 策略建议的原始比例 (通常是 0.55)
            
        Returns:
            float: 最终执行的买入比例 (0.0 表示拒绝交易)
        """

        is_super_vip = symbol in self.config.super_vip_symbols
        is_vip = symbol in self.config.vip_symbols
        is_bearish = self._is_bearish_symbol(symbol)
        market = get_market_type(symbol)
        is_strong_bull = self.get_strong_bull(market)
        is_super_weak = self.get_super_weak(market)

        multiplier = 1.0
        logic_tag = "Default"

        if is_super_vip:
            multiplier = 1.2
            logic_tag = "👑 VIP + Tier 1 (Aggressive)"
        elif is_vip:
            multiplier = 1.1
            logic_tag = "👑 VIP + Tier 2 (Aggressive)"
        else:
            # --- 非 VIP 股票处理逻辑 ---
            multiplier = 0.8
            logic_tag = "🐎 Non-VIP + Tier 2"

        ## 做多和做空适配
        if is_bearish:
            if self._is_entering_weekend_risk_for_symbol(symbol):
                multiplier = 1.2
            
            if is_strong_bull:
                multiplier *= 0.8
            if is_super_weak:
                multiplier *= 1.2

            logic_tag = f"启用做空，multiplier-->{multiplier}"
        else:
            if is_strong_bull:
                multiplier *= 1.2
            if is_super_weak:
                multiplier *= 0.8
            logic_tag = f"启用做多，multiplier-->{multiplier}"

        final_ratio = base_ratio * multiplier
        
        if multiplier != 1.0:
            logger.warning(f"⚖️ [仓位裁决] {symbol} | 评级: [{logic_tag}] | 基准: {base_ratio:.2f} x 系数: {multiplier} = 最终: {final_ratio:.2f}")

        return final_ratio
        
    def _finalize_position_creation(self, symbol: str, filled_price: float, filled_quantity: int, plan_info: dict):
        try:
            symbol = normalize_symbol(symbol)
            market = get_market_type(symbol)
            actual_amount = filled_price * filled_quantity

            with self.position_lock:
                if symbol in self.positions:
                    logger.warning(f"订单成交后，仓位记录已存在 {symbol}。将不会重复创建。")
                    return

                # 从 plan_info 中提取 building_stage，默认为 2 (即直接完成)
                # 如果开启了微观建仓，这里应该是 1
                init_stage = plan_info.get("building_stage", 2)

                # 创建Position对象时，传入 initial_stop_loss_price
                position = Position(
                    symbol=symbol,
                    market=market,
                    initial_price=round(filled_price,3),
                    initial_scout_price=round(filled_price,3),
                    # overall_phase=PositionOverallPhase.RUNNING if 'NightHunterStrategy'==plan_info.get('strategy_class_name') else PositionOverallPhase.BUILDING,
                    overall_phase=PositionOverallPhase.RUNNING,
                    planned_total_quantity=plan_info.get('planned_total_quantity', filled_quantity),
                    initial_risk_per_share=round(plan_info.get('initial_risk_per_share', (filled_price * self.config.stop_loss_ratio)),3),
                    initial_stop_loss_price=plan_info.get('initial_stop_loss_price'),
                    triggering_strategy=plan_info.get('triggering_strategy'),
                    strategy_class_name=plan_info.get('strategy_class_name'),
                    strategy_params=plan_info.get('strategy_params'),
                    confirmation_add_done=plan_info.get('confirmation_add_done'),
                    building_stage=init_stage
                    )
                
                # ==============================================================================
                # ▼▼▼初始化“持续新低”风控模块的状态 ▼▼▼
                # ------------------------------------------------------------------------------
                # 1. 将成交价设为初始的“买后最低价”。
                position.post_purchase_low = filled_price
                # 2. 将当前时间设为首次检查的时间戳基准。
                position.last_consecutive_low_check_ts = datetime.now(timezone.utc)
                # ==============================================================================

                position.add_purchase_record(PurchaseActionType.INITIAL_SCOUT, filled_price, filled_quantity, actual_amount)
                self.positions[symbol] = position
                self._save_positions()

            logger.info(f"✅ 首次买入成功并创建持仓记录 {symbol}: 价格={filled_price}, 数量={filled_quantity}")
            self.notification_manager.send_trade_execution("BUY", symbol,filled_quantity, filled_price, "首次建仓信号")
        except Exception as e:
            logger.error(f"从成交订单创建持仓时失败 {symbol}: {e}", exc_info=True)

    # ==============================================================================
    # ▼▼▼ C2 指令总线与战术雷达引擎 (C2 Command & Radar Engine) ▼▼▼
    # ==============================================================================
    def _sync_shadow_tags(self):
        """【同步神经元】拉取远端指令并刷新影子内存"""
        try:
            self.shadow_tags['tactical_liquidation'] = get_custom_watchlist_group(self.quote_ctx, "tactical_liquidation")
            self.shadow_tags['profit_only_mode'] = get_custom_watchlist_group(self.quote_ctx, "profit_only_mode")
            self.shadow_tags['strategic_hold'] = get_custom_watchlist_group(self.quote_ctx, "strategic_hold")
            self.shadow_tags['macro_tactical_radar'] = get_custom_watchlist_group(self.quote_ctx, "macro_tactical_radar")
        except Exception as e:
            logger.error(f"同步影子标签总线失败: {e}")

    def _process_tactical_liquidation(self):
        """【战术清仓引擎】基于数量的动态比例异步处决"""
        liquidation_list = self.shadow_tags['tactical_liquidation']
        if not liquidation_list:
            self._last_liquidation_hash = "" # 清空列表时重置锁
            return
            
        # 计算当前列表哈希，防止每20秒无限触发清仓
        current_hash = ",".join(sorted(liquidation_list))
        if current_hash == self._last_liquidation_hash:
            return 
            
        self._last_liquidation_hash = current_hash
        count = len(liquidation_list)
        
        # 核心算法：按标的数量决定清仓烈度
        if count == 1:
            ratio = 1.0   # 1个标的：100% 屠杀
        elif count == 2:
            ratio = 0.50  # 2个标的：50% 减仓
        else:
            ratio = 0.35  # 3个及以上：35% 柔性撤退
            
        logger.critical(f"☢️ [C2总线] 接收到 Tactical_Liquidation 指令！(标的数量: {count}) -> 将异步执行 {ratio*100}% 战术清仓。")
        # 极致优雅：丢进线程池，绝不阻塞主循环！
        self.task_executor.submit(self.execute_tactical_clearance, ratio)

    def _process_clearance_commands(self):
        """【统一清仓矩阵】处理一键清仓与仅卖盈利指令 (基于数量动态算比例)"""
        list_all = self.shadow_tags['tactical_liquidation']
        list_profit = self.shadow_tags['profit_only_mode']
        
        active_list = None
        mode = 'ALL'
        
        # 优先级：无差别清仓 > 仅卖盈利
        if list_all:
            active_list = list_all
            mode = 'ALL'
        elif list_profit:
            active_list = list_profit
            mode = 'PROFIT_ONLY'
        else:
            self._last_clearance_hash = "" # 清空状态
            return
            
        current_hash = f"{mode}_{','.join(sorted(active_list))}"
        if current_hash == self._last_clearance_hash:
            return # 防抖：只要自选股列表没变，就不重复触发
            
        self._last_clearance_hash = current_hash
        count = len(active_list)
        
        # 核心算法：按自选股数量决定清仓烈度
        if count == 1: ratio = 1.0   # 1个标的：100% 全卖
        elif count == 2: ratio = 0.50  # 2个标的：50% 减仓
        else: ratio = 0.35             # 3个及以上：35% 柔性撤退
            
        logger.critical(f"☢️[C2总线] 接收到清仓指令! 模式:{mode}, 触发物数量:{count} -> 将异步执行 {ratio*100}% 清仓。")
        # 丢进线程池异步执行，绝不阻塞主循环！
        self.task_executor.submit(self.execute_tactical_clearance, ratio, mode)

    def _process_macro_tactical_radar(self):
        """【大盘战术雷达】SPY 级别心跳监控与极端拐点预警"""
        radar_list = self.shadow_tags['macro_tactical_radar']
        if not radar_list: return
        
        now = time.time()
        for symbol in radar_list:
            high_key = f"{symbol}_HIGH"
            low_key = f"{symbol}_LOW"
            
            # --- 探测高点 (做空/离场信号) ---
            # 冷却机制：同一方向30分钟内只报一次，防止连环夺命 Call
            if now - self._radar_alert_cooldowns.get(high_key, 0) > 1800:
                # 机构派发窗口：看15分钟，回撤 > 0.15% 确立顶部结构
                if check_tactical_exit_signal(self.quote_ctx, symbol, lookback_minutes=15, pullback_pct_threshold=0.15):
                    msg = f"🚨 [宏观雷达] {symbol} 确立 15分钟级 顶部结构 (回撤>0.15%)！建议: 准备大盘做空 / 规避多头！"
                    logger.critical(msg)
                    self.notification_manager.send_critical_alert(msg)
                    self._radar_alert_cooldowns[high_key] = now
                    
            # --- 探测低点 (做多/抄底信号) ---
            if now - self._radar_alert_cooldowns.get(low_key, 0) > 1800:
                # 恐慌托底窗口：看10分钟，反弹 > 0.10% 确立底部结构
                if check_tactical_entry_signal(self.quote_ctx, symbol, lookback_minutes=10, rebound_pct_threshold=0.10, mode='CONFIRMATION'):
                    msg = f"🟢 [宏观雷达] {symbol} 确立 10分钟级 底部结构 (反弹>0.10%)！建议: 准备大盘做多 / 抄底！"
                    logger.critical(msg)
                    self.notification_manager.send_critical_alert(msg)
                    self._radar_alert_cooldowns[low_key] = now
    # ==============================================================================
    # IV. 持仓管理与退出 (Position Management & Exit)
    # ==============================================================================

    def _main_monitor_loop(self):
        """
        保守策略持仓监控循环。
        只处理开仓成交、卖出成交、账户级风控和固定止盈止损规则。
        """
        logger.info("主监控循环开始...")
        while not self.stop_main_monitor.is_set():
            try:
                loop_config = self.config
                if not loop_config:
                    logger.debug("当前为非交易时间或配置未加载，主监控线程休眠中...")
                    time.sleep(30)
                    continue

                self._check_pending_initial_buy_orders()

                with self.position_lock:
                    current_symbols = list(self.positions.keys())

                if current_symbols:
                    self._check_pending_position_transactions(current_symbols)

                if self._is_account_daily_loss_limit_hit():
                    self._liquidate_all_positions("账户当日总亏损达到3%")
                    time.sleep(loop_config.check_interval)
                    continue

                if self._is_account_monthly_loss_limit_hit():
                    self._liquidate_all_positions("当月累计账户总亏损达到6%")
                    time.sleep(loop_config.check_interval)
                    continue

                for symbol in list(current_symbols): # 迭代副本以防在循环中修改字典
                    with self.position_lock:
                        pos = self.positions.get(symbol)
                    if not pos:
                        continue
                    if self._is_pure_stock(symbol):
                        self._check_position_signals(symbol)

                    time.sleep(0.5) # 防止 CPU 飙升

                time.sleep(loop_config.check_interval)
            except Exception as e:
                logger.error(f"主监控循环出错: {e}", exc_info=True)
                # 【告警集成】 7. 核心线程崩溃告警
                error_msg = f"P1级告警: 主监控线程崩溃! 系统部分功能可能已失效。错误: {e}"
                logger.error(error_msg, exc_info=True)
                self.notification_manager.send_critical_alert(error_msg)
                time.sleep(30)
        logger.info("主监控循环已停止...")
    
    def _signal_replication_monitor_loop(self):
        """
        【信号复制监控循环 (Signal Replication Monitor)】
        
        职责：
        1. 定时扫描 pending_buy_signals_ft.db (由主脑写入)。
        2. 读取到信号后，直接调用 process_buy_signal 执行买入逻辑。
        3. 验证买入成功（持仓存在）后，从 DB 中清除信号。
        4. 若买入未成功（如资金不足、API错误），保留 DB 记录以便自动重试。
        """
        logger.info("🚀 [跟随模式] 信号复制监控循环已启动，等待主脑指令...")
        
        while not self.stop_pending_monitor.is_set():
            try:
                # --- 1. 读取信号 ---
                # self.pending_buy_cache 在 ft 代码中已经指向了 pending_buy_signals_ft.db
                candidates = self.pending_buy_cache.get_all_signals()
                
                if not candidates:
                    time.sleep(5) # 极速轮询，保证跟单延迟最低
                    continue

                for candidate in candidates:
                    symbol = candidate['symbol']
                    
                    # --- 2. 防重入检查 ---
                    # 如果已经在持仓里，或者正在下的单子里，就不要再调 process_buy_signal 了
                    # 并从 DB 中清理掉（可能是上次崩了没删掉，或者是人工买了）
                    with self.position_lock:
                        if symbol in self.positions:
                            logger.info(f"♻️ [状态同步] {symbol} 已持有仓位，从待买入队列中移除。")
                            self.pending_buy_cache.remove_signal(symbol)
                            continue
                        
                        if not self.enable_bearish and \
                            self._is_entering_weekend_risk_for_symbol(symbol) and \
                            is_in_opening_window(MarketType.US, window_minutes=15) and \
                            self._is_bearish_symbol(symbol): # 当日是以做多为主&周四/五&开盘前15分钟内，将所有做空票剔除
                            
                            self.pending_buy_cache.remove_signal(symbol)
                            continue
                            
                    with self.pending_orders_lock:
                        if symbol in self.pending_orders:
                            logger.debug(f"⏳ [执行中] {symbol} 存在待处理订单，跳过本次循环。")
                            continue
                            
                    # --- 3. 执行买入逻辑 ---
                    if random.random() > 0.8:
                        logger.warning(f"⚔️ [响应跟随] 收到主脑信号 {symbol} ({candidate.get('strategy_name')})，开始执行...")
                    
                    # 获取必要的市场信息
                    try:
                        current_status = get_trading_window_status(symbol)
                        is_rth_open = is_any_market_open(symbol) # 严格的盘中判断(9:30-16:00)
                        # market = get_market_type(symbol)
                        # rth_windows = self.config.favorable_sell_windows if is_bearish else self.config.favorable_buy_windows
                        
                        # 夜盘窗口
                        ext_windows = self.config.extended_hours_buy_windows
                        strategy_name = candidate.get('strategy_name', '')

                    except Exception as e:
                        logger.error(f"[{symbol}] 获取市场状态失败: {e}")
                        continue
                    
                    # ==================================================================
                    # 轨道一：常规交易时段 (RTH)
                    # ==================================================================
                    if is_rth_open and 'NightHunter' !=strategy_name:
                        # 调用原有的处理逻辑，复用所有黑名单、风控、资金检查
                        # 注意：process_buy_signal 内部通常是异步 submit 到线程池或者直接下单
                        # 这里的调用是非阻塞的，所以我们不能立刻判断 success
                        self.process_buy_signal(candidate)
                    # ==================================================================
                    # 轨道二：扩展交易时段 (夜盘/盘前)
                    # ==================================================================
                    elif not is_rth_open and current_status in ext_windows and 'NightHunter'==strategy_name and not is_entering_weekend_risk_for_symbol(symbol, enable_wrp=True, wrp_activation_days=[2]):
                        if symbol not in self.config.night_hunter_targets or symbol in self.pending_orders:
                            continue
                        
                        trigger_price = candidate.get('trigger_price')
                        if not trigger_price:
                            logger.warning(f"[{symbol}] 信号缺少 trigger_price，移除。")
                            self.pending_buy_cache.remove_signal(symbol)
                            continue
                        
                        # 获取实时价格
                        current_price = self.get_realtime_price(symbol)
                        
                        if not current_price or current_price <= 0:
                            logger.error(f"[{symbol}] 无法获取有效价格，跳过。")

                        # 简单的价格硬约束：绝不追高
                        price_limit = trigger_price * (1.0 + self.config.limit_order_price_buffer) # 0.5% 容忍度
                        k_mins_check = self.config.tactical_k_mins_map.get('NIGHT_MARKET', 5) #夜盘5分钟最低点
                        is_low_confirmed = check_extended_hours_tactical_entry_signal(
                                self.hs_data_provider, symbol, k_mins_check, 
                                self.config.rebound_pct_threshold_map.get('default', 0.08)
                            )
                        # if random.random() > 0.8:
                        #     logger.warning(f"★★★ [{symbol}] 夜盘/盘前狙击窗口 is_low_confirmed={is_low_confirmed},price_limit={price_limit},current_price={current_price}")

                        if current_price <= price_limit and is_low_confirmed:
                            logger.warning(f"★★★ [{symbol}] 夜盘/盘前狙击窗口 ({current_status.name}) 命中! 现价 {current_price} <= 触发价 {trigger_price}")
                            
                            # 1. 锁定信号
                            with self.signals_in_flight_lock:
                                if symbol in self.signals_in_flight: continue
                                self.signals_in_flight.add(symbol)
                            
                            # 2. 执行夜盘专属买入
                            # 传入 loop_config 以便读取 night_config 等参数
                            self._execute_extended_hours_pending_buy(candidate, current_price, self.config)
                        
                        else:
                            # 价格不达标，静默等待
                            pass

                    # --- 4. 后置处理策略 ---
                    # 此时我们不立即删除 DB 记录。
                    # 只有当 下一次循环 检测到 (symbol in self.positions) 时才删除。
                    # 这天然构成了“不断重试直到成功”的逻辑。
                    # 如果 process_buy_signal 因为黑名单拒绝了，DB 里会一直有，怎么处理？
                    # 可以在 process_buy_signal 内部加入黑名单拦截的返回值，但在不改动该方法前提下，
                    # 我们可以依赖 intaday_blacklist 的逻辑。如果一直在 DB 里，会一直触发，但一直被拦截。
                    # 这符合“不断重试”的要求，直到人工干预或第二天清空。

                time.sleep(2) # 稍微休眠，给下单逻辑一点时间

            except Exception as e:
                logger.error(f"信号复制监控循环异常: {e}", exc_info=True)
                time.sleep(5)

    def _pre_market_monitor_loop(self):
        """
        【夜猎者 (Night Hunter)】
        修复点：移除了对 timestamp 的 300秒 强校验。
        原因：夜盘API返回的 timestamp 经常停留在 16:00:00，导致策略误判为数据滞后。
        我们改用 Volume > 0 和后续的 Volume Delta 来确保数据活性。
        """
        logger.warning(f"夜猎者监控循环启动，扫描目标: {self.config.night_hunter_targets}")
        
        while not self.stop_pre_market_monitor.is_set():
            try:
                # 0. 全局开关：如果是非交易日或系统配置未加载，暂歇
                if not self.config or is_us_market_open():
                    time.sleep(30)
                    continue
                
                now_ts = time.time()
                # 遍历目标股票
                for symbol in self.config.night_hunter_targets:
                    # --- A. 状态初始化 ---
                    if symbol not in self.pre_market_states:
                        self.pre_market_states[symbol] = {
                            'status': 'WATCHING',      # WATCHING | REBOUNDING | BOUGHT
                            'session_low': 99999.0,    # 本场最低价
                            'volume_at_low': 0,        # 创低时的累计成交量
                            'rebound_start_ts': 0,     # 反弹开始时间戳
                            'last_update_ts': 0,       # 上次处理时间
                            'tick_history': []         # 内存微观K线容器
                        }

                    self._save_pre_market_states()

                    state = self.pre_market_states[symbol]
                    last_update = state.get('last_update_ts', 0)
                    time_in_state = now_ts - last_update

                    # =================================================================
                    # 逻辑 A: BOUGHT -> WATCHING (三无检查)
                    # 防止幽灵状态锁定开仓
                    # =================================================================
                    if state.get('status') == 'BOUGHT':
                        # 60秒宽限期，给下单流程一点时间
                        if time_in_state > 60:
                            has_position = False
                            with self.position_lock:
                                has_position = symbol in self.positions
                            
                            has_pending = False
                            with self.pending_orders_lock:
                                has_pending = symbol in self.pending_orders
                            
                            has_signal = self.pending_buy_cache.has_signal(symbol)

                            if not has_position and not has_pending and not has_signal:
                                logger.warning(f"🧹 [{symbol}] 状态自愈: 满足[无持仓+无挂单+无信号]，重置 BOUGHT -> WATCHING")
                                state['status'] = 'WATCHING'
                                # ⚡️ 致命Bug修复: 
                                # 必须重置 session_low 为无穷大 (或当前价)，否则如果刚卖出，价格通常高于之前的 low，
                                # 下一次循环会立即误判为“强力反弹”而追高买入。
                                # 将其重置为 99999.0，强制系统必须先找到一个新的低点 (New Low) 才能开始新的反弹逻辑。
                                state['session_low'] = 99999.0
                                state['volume_at_low'] = 0     # 同时清除成交量记忆
                                state['rebound_start_ts'] = 0  # 必须重置计时器！
                                state['last_update_ts'] = now_ts
                                self._save_pre_market_states()

                    # =================================================================
                    # 逻辑 B: WATCHING -> BOUGHT (持仓同步)
                    # 防止手动买入或重启后，状态仍停留在 WATCHING 导致重复扫描
                    # =================================================================
                    elif state.get('status') == 'WATCHING':
                        # 同样给一点宽限期，或者直接检查
                        if time_in_state > 60:
                            is_holding = False
                            with self.position_lock:
                                is_holding = symbol in self.positions
                            
                            if is_holding:
                                logger.warning(f"🚑 [{symbol}] 状态自愈: 检测到实际持仓，重置 WATCHING -> BOUGHT")
                                state['status'] = 'BOUGHT'
                                state['last_update_ts'] = now_ts
                                self._save_pre_market_states()
                                continue # 既然已买入，跳过后续扫描
                    
                    # 状态锁定：如果本轮夜盘已经买入过，就不再重复操作，防止单边下跌中不断加仓
                    if state['status'] == 'BOUGHT':
                        continue
                    
                    # 检查是否已有持仓（包括常规持仓），如果有，也不在夜盘加仓，控制风险
                    with self.position_lock:
                        if symbol in self.positions:
                            continue
                    
                    # ==============================================================================
                    # ▼▼▼ 状态超时熔断机制 (Zombie State Breaker) ▼▼▼
                    # 逻辑：必须在获取行情之前执行。即使行情中断(vol=0)，时间依然在流逝。
                    # ==============================================================================
                    if state['status'] == 'REBOUNDING':
                        # 获取当前等待时长
                        elapsed_time = time.time() - state.get('rebound_start_ts', 0)
                        # 设置最大容忍时间：配置的验证时间 x 2 (例如 180s * 2 = 6分钟)
                        # 如果6分钟还没确认也没失败，说明数据断了或者逻辑死锁，必须强制复位
                        max_zombie_tolerance = self.config.night_config.get('verify_seconds', 180) * 2
                        
                        if elapsed_time > max_zombie_tolerance:
                            logger.warning(f"🚑 [自我修复] {symbol} 处于 REBOUNDING 状态过久 ({elapsed_time:.0f}s)，疑似行情中断，强制重置为 WATCHING。")
                            state['status'] = 'WATCHING'
                            state['last_update_ts'] = time.time()
                            self._save_pre_market_states()
                            continue # 重置后跳过本轮，等待下一轮重新扫描
                    # ▲▲▲ 状态超时熔断机制 (Zombie State Breaker) ▲▲▲ =========================================================

                    # --- B. 获取实时快照 ---
                    try:
                        quote = self.hs_data_provider.get_smart_quote(symbol)
                        # 如果 HS 挂了，quote 可能为 None
                    except:
                        quote = None

                    if not quote or quote.get('last_price', 0) <= 0:
                        # 降级：如果 HS 拿不到，去拿长桥的 snapshot
                        quote = get_smart_quote(self.quote_ctx, symbol)

                    if not quote:
                        # logger.debug(f"[{symbol}] 无法获取行情数据 (None)，跳过。")
                        continue
                    
                    # 再次检查延迟（双重保险）：超过5分钟的数据绝不开仓
                    # ts = quote.get('timestamp')
                    # try:
                    #     # 但我们在外部再做一次绝对时间差检查是安全的
                    #     if ts and (datetime.now() - ts).total_seconds() > 300: # 此时如果 ts 是 aware，datetime.now() 会报错
                    #         logger.warning('(datetime.now() - ts).total_seconds() > 300')
                    #         # 修正：使用 quote 里的时间戳对象，通常它由 data_provider 处理过
                    #         # 如果你不确定 ts 的时区，这里最稳妥的是：直接信赖 quote['volume'] 变化
                    #         pass 
                    # except Exception:
                    #     pass

                    # 数据解包
                    curr_price = quote.get('last_price', 0.0)
                    curr_vol = quote.get('volume', 0)
                    prev_close = quote.get('prev_close_price', 0.0)
                    high_price = quote.get('high_price', 0.0)
                    # quote_ts = quote.get('timestamp')

                    # logger.warning(f"[{symbol}] curr_price:{curr_price} curr_vol:{curr_vol} prev_close:{prev_close} high_price:{high_price}")
                    
                    # B.1 数据完整性与时效性检查
                    if curr_vol <= 0: continue
                    if curr_price <= 0 or prev_close <= 0: continue
                    
                    # 防止数据卡死：如果行情时间滞后当前时间超过5分钟，视为无效数据
                    # if quote_ts and (datetime.now() - quote_ts).total_seconds() > 300:
                    #     logger.warning(f"[{symbol}] 行情数据滞后，跳过。")
                    #     continue

                    # ==============================================================================
                    # ▼▼▼【核心植入】人工微观结构构建器 (Synthetic Micro-Structure Builder) ▼▼▼
                    # 逻辑：利用轮询切片构建最近 5分钟 的内存 K线，0 IO成本实现微观风控。
                    # ==============================================================================
                    # 1. 确保容器存在
                    if 'tick_history' not in state: state['tick_history'] = []
                    
                    # 2. 录入当前 Tick
                    state['tick_history'].append({'p': curr_price, 't': now_ts})
                    
                    # 3. 滑动窗口清洗 (只保留最近 300秒/5分钟 的数据)
                    # 列表推导式效率极高，不用担心性能
                    TIME_WINDOW = 300
                    state['tick_history'] = [
                        tick for tick in state['tick_history'] 
                        if (now_ts - tick['t']) < TIME_WINDOW
                    ]
                    
                    # --- C. 状态机流转 ---
                    
                    # 计算相对于昨收的跌幅 (e.g., -0.02)
                    drop_pct = (curr_price - prev_close) / prev_close
                    # logger.warning(f"[{symbol}] drop_pct: {drop_pct}")

                    # [状态 1: 潜伏观望 WATCHING]
                    if state['status'] == 'WATCHING':
                        # --- 1.1 动态探底 (Finding the Bottom) ---
                        # 只要价格创新低，就更新记录
                        if curr_price < state['session_low']:
                            state['session_low'] = curr_price
                            state['volume_at_low'] = curr_vol 
                            
                            # 只有真跌(跌幅超过0.5%)才刷屏，避免 SPYU 这种上涨股刷日志
                            # 这里的跌幅显示，为了兼容性，暂时还是显示相对于昨收的
                            if drop_pct < -0.005: 
                                # logger.warning(f"[{symbol}] 📉 夜盘创新低: {curr_price} (昨收跌幅: {drop_pct:.2%})")
                                state['last_update_ts'] = time.time() # 务必加上这行
                                self._save_pre_market_states() 
                        
                        # --- 1.2 多维反弹触发检查 (The Multi-Dimensional Hook) ---
                        # 杠精注：这里是核心修改点。不要写死一种逻辑，我们要同时检查两种触发条件。
                        
                        # 基础数据准备
                        session_low = state['session_low']
                        
                        # === 策略 A: 昨收超跌 (Prev Close Strategy) ===
                        # 计算最低点相对于昨收的跌幅
                        low_pct_from_close = (session_low - prev_close) / prev_close
                        # 计算策略A的反弹目标位
                        target_from_close = session_low * (1 + self.config.night_config['close_rebound_threshold'])
                        # 判定A是否成立：跌得够深 且 反弹够高
                        signal_close = (low_pct_from_close <= self.config.night_config['close_dip_threshold']) and \
                                       (curr_price > target_from_close)

                        # === 策略 B: 高点回撤 (High Price Strategy) ===
                        # 只有当high_price有效且大于0时才计算，防止数据还没推送到导致除零或误判
                        signal_high = False
                        if high_price > 0:
                            # 计算最低点相对于当日最高价的跌幅 (回撤幅度)
                            low_pct_from_high = (session_low - high_price) / high_price
                            # 计算策略B的反弹目标位
                            target_from_high = session_low * (1 + self.config.night_config['high_rebound_threshold'])
                            # 判定B是否成立：回撤够深 且 反弹够高
                            signal_high = (low_pct_from_high <= self.config.night_config['high_dip_threshold']) and \
                                          (curr_price > target_from_high)

                        # --- 1.3 综合信号触发 ---
                        # 只要满足任意一个策略，且有量能配合，就干！
                        if signal_close or signal_high:
                            
                            # 步骤 D: 成交量确认 (Volume Confirmation)
                            # 必须有增量成交，证明是主力买上去的，不是空涨
                            vol_delta = curr_vol - state['volume_at_low']
                            
                            if vol_delta > 0:
                                # ==============================================================================
                                # ▼▼▼ 微观位置风控 (Micro-Rank Gatekeeper) ▼▼▼
                                # 逻辑：如果当前价格处于过去5分钟的最高点附近(>90%)，说明是瞬间脉冲(Needle Top)。
                                # 此时追进去大概率被埋。强制要求回调或等待结构稳定。
                                # ==============================================================================
                                history = state.get('tick_history', [])
                                allow_entry = True
                                micro_msg = "N/A"
                                # 如果是高波股(High Vol)，我们要求更多的数据点(4个/40秒)来确认结构
                                min_ticks_required = 4 if symbol in self.config.high_vol_symbols else 3
                                # [冷启动防御] 如果数据不足，但涨幅相对于昨收已经超过 2%，视为"盲飞"风险，强制拦截
                                if len(history) < min_ticks_required:
                                    # 计算当前相对于昨收的涨幅
                                    if prev_close > 0 and (curr_price - prev_close) / prev_close > 0.02:
                                        allow_entry = False
                                        if random.random() > 0.8:
                                            logger.warning(f"🛡️ [{symbol}] 冷启动保护: 数据不足({len(history)})且涨幅明显，暂缓开仓。")
                                
                                # [常规微观风控]
                                elif len(history) >= min_ticks_required:
                                    # 提取价格序列
                                    prices = [t['p'] for t in history]
                                    micro_high = max(prices)
                                    micro_low = min(prices)
                                    micro_range = micro_high - micro_low
                                    # 增强型波动率过滤：既要看百分比(0.1%)，也要看绝对值(防止仙股噪音)
                                    # 假设最小有意义波动为 $0.02 (对于高价股) 或 价格的 0.1%
                                    noise_floor = max(0.02, micro_low * 0.001)
                                    
                                    # 只有波动幅度有意义(>0.1%)时才压制，防止织布机行情被误杀
                                    if micro_range > noise_floor:
                                        # 计算当前价格在微观箱体的位置 (0.0 ~ 1.0)
                                        micro_rank = (curr_price - micro_low) / micro_range
                                        
                                        # 阈值：0.92 (极高位)。如果处于前5分钟的 92% 高位，视为"针尖"。
                                        rank_threshold = 0.97 if symbol in self.config.high_vol_symbols else 0.95
                                        if micro_rank > rank_threshold:
                                            allow_entry = False
                                            micro_msg = f"位置过高(Rank {micro_rank:.2f} > {rank_threshold})"
                                            # 可选：如果这里被拦截，可以打印一个 Debug 日志
                                            if random.random() > 0.85:
                                                logger.warning(f"🛡️ [{symbol}] 微观风控拦截: {micro_msg} | 范围: {micro_low:.2f}-{micro_high:.2f}")

                                if not allow_entry:
                                    # 如果被微观风控拦截，直接跳过本次触发，等待下一次轮询 (那时价格可能会回落或High被推高)
                                    # 既然没进入 REBOUNDING，session_low 保持不变，下一次还会进这里检查
                                    continue 

                                trigger_reason = []
                                if signal_close: trigger_reason.append(f"昨收超跌(低点{low_pct_from_close:.2%})")
                                if signal_high: trigger_reason.append(f"高点回撤(低点{low_pct_from_high:.2%})")
                                reason_str = " & ".join(trigger_reason)

                                logger.warning(f"🚀 [{symbol}] 夜盘量价触底! 触发机制: [{reason_str}] | 低点:{session_low} -> 现价:{curr_price}, 增量成交:{vol_delta}")
                                
                                state['status'] = 'REBOUNDING'
                                state['rebound_start_ts'] = time.time()
                                state['last_update_ts'] = time.time()
                                self._save_pre_market_states()
                            else:
                                # 无量反弹，视为噪音，忽略
                                pass

                    # [状态 2: 时间验证 REBOUNDING]
                    elif state['status'] == 'REBOUNDING':
                        # 2.1 破位风控：如果价格重新跌破 Session Low，反弹宣告失败
                        if curr_price <= state['session_low']:
                            logger.warning(f"[{symbol}] 夜盘反弹夭折 (破前低)，重置为观望。")
                            state['status'] = 'WATCHING'
                            state['session_low'] = curr_price # 更新为更低的低点
                            state['volume_at_low'] = curr_vol
                            state['last_update_ts'] = time.time() # 务必加上这行
                            self._save_pre_market_states() # <--- 状态回滚，保存
                            continue
                        
                        # 2.2 时间熔断：检查是否稳住了 N 秒 (3分钟)
                        elapsed = time.time() - state['rebound_start_ts']
                        if elapsed >= self.config.night_config['verify_seconds']:
                            # --- D. 信号确认与分发 (Signal Dispatch) ---
                            logger.warning(f"✅ [{symbol}] 夜盘反弹通过3分钟时间验证，生成信号推入缓存！")
                            
                            # 1. 构造标准信号对象
                            # 这里我们不直接调用 _execute_night_buy，而是把意图打包
                            signal_candidate = {
                                "symbol": symbol,
                                "strategy_name": self.config.night_config['strategy_name'], # e.g. "NightHunter"
                                "strategy_class_name": "NightHunterStrategy",
                                "trigger_price": curr_price,  # 以验证通过时的价格作为触发基准
                                "reason": "Night Hunter Rebound Verified (Pre-Market/Night)",
                                "buy_percentage": self.config.night_config.get('max_pos_ratio', 0.1), # 传递配置的仓位比例
                                "timestamp": time.time()
                            }
                            llm_approved, llm_reason = self._get_llm_decision(signal_candidate, 'buy')
                            if not llm_approved:
                                logger.warning(f"[{symbol}] 非盘中交易被LLM否决: {llm_reason}")
                                state['status'] = 'WATCHING'
                                state['last_update_ts'] = time.time()
                                self._save_pre_market_states()
                                continue
                                            
                            # 2. 推入待买入缓存
                            # 这一步完成后，_pending_signal_monitor_loop 会接手处理
                            self.pending_buy_cache.add_signal(signal_candidate)
                            
                            # 3. 更新本地状态机
                            # 标记为 'BOUGHT' 是为了告诉当前循环："这个机会我已经处理过了（推给缓存了），
                            # 不要再重复扫描它了"。这能防止同一个反弹生成几百个重复信号。
                            state['status'] = 'BOUGHT'
                            state['last_update_ts'] = time.time()
                            self._save_pre_market_states()

                            # --- D. 执行买入 (Execution) ---
                            # logger.critical(f"✅ [{symbol}] 夜盘反弹通过3分钟时间验证，准备执行买入！")
                            # self._execute_night_buy(symbol, curr_price, state['session_low'])
                            # state['status'] = 'BOUGHT' # 标记本场已买，不再扫描
                            # state['last_update_ts'] = time.time() # 务必加上这行
                            # self._save_pre_market_states() # <--- 标记已买入，防止重启后重复下单

                time.sleep(10) # 轮询间隔

            except Exception as e:
                logger.error(f"夜猎者监控循环异常: {e}", exc_info=True)
                time.sleep(30)

    def _execute_extended_hours_pending_buy(self, candidate: dict, current_price: float, loop_config: 'TradingConfig'):
        """
        夜盘/盘前专属买入执行器
        
        严格复刻 _execute_night_buy 的实现逻辑：
        1. 仓位计算：不使用盘中R值模型，严格基于 config.night_config['max_pos_ratio'] (默认5%)。
        2. 止损设定：不使用自适应止损，采用 0.5% 刚性回撤止损。
        3. 下单方式：强制使用 LO (限价单)。
        """
        symbol = candidate['symbol']
        # 黑名单熔断拦截 (Priority 0)
        if symbol in self.intraday_blacklist:
            # 降低日志级别防止刷屏，或者用 debug
            if random.random() > 0.95:
                logger.warning(f"🚫 [熔断拦截] {symbol} 位于当日黑名单中（曾触发舆情/止损），拒绝开仓信号。")
            return

        # ▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼
        # TTL 门禁机制 (The TTL Gatekeeper)】
        # 这是防止“刚买入就触发舆情核按钮”的绝对防线。
        # 必须同时满足：1. 舆情非负向 2. 数据够新鲜(比如45分钟内)
        # ▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼
        
        # 1. 获取带时间戳的舆情缓存
        # 定义TTL为45分钟，对于美股/港股，超过45分钟的新闻真空期是不可接受的风险
        is_fresh, sentiment_data = self.sentiment_analysis.is_cache_fresh(symbol, ttl_minutes=45)
        sentiment_status = sentiment_data.get('sentiment', '未知')

        # 2. 检查：是否为明确的负向？
        if sentiment_status == '负向':
            logger.critical(f"🛑 [舆情门禁拦截] {symbol} 存在负向舆情，买入指令已销毁！(缓存时间: {sentiment_data.get('timestamp')})")
            # 将其加入黑名单，防止今天再次骚扰
            self.intraday_blacklist.add(symbol)
            self._save_blacklist()
            return

        # 3. 检查：数据是否过期？(TTL Check)
        # if not is_fresh:
        #     # 数据太旧了！或者是空的。
        #     # 这种情况下买入就是赌博。作为全球第二的极客，我们不赌博。
        #     logger.warning(f"⏳ [舆情门禁拦截] {symbol} 舆情数据过期或缺失，拒绝盲目开仓。已触发加急刷新。")
            
        #     # A. 立即触发一次加急刷新 (虽然在run_strategy_loop预热过，但可能还没跑完，或者这是漏网之鱼)
        #     # 注意：这里不能阻塞等待，因为 analyze 很慢。
        #     self.sentiment_analysis.trigger_async_refresh(symbol, self.task_executor)
            
        #     # B. 策略选择：
        #     # 选项1 (激进): 既然前面已经预热了，这里为了防止死锁，如果没拿到最新数据，暂时放弃本次tick，
        #     #              让 pending_signal_monitor_loop 下一轮循环（几秒后）再试。
        #     #              只要 pending_cache 里还有它，就会不断重试，直到数据变新鲜。
            
        #     # 我们选择选项1，直接 Return。等待数据刷新后，下一轮 process_buy_signal 自然会通过 is_fresh 检查。
        #     return

        # # 如果通过了以上两关，说明：舆情是新鲜的，且不是负向的。
        # logger.info(f"✅ [舆情门禁通过] {symbol} 舆情状态: {sentiment_status} (新鲜度校验通过)")

        # ▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲
        
        if symbol in self.pending_orders:
            logger.debug(f"{symbol} 已存在待处理的开仓订单，忽略新的买入信号。")
            return
        strategy_name = candidate.get('strategy_name', 'NightHunter(Pending)')
        is_rejected, reject_reason = self._check_trade_safety_gate(symbol,candidate,is_regular_open=False)
        if is_rejected:
            logger.error(f"[{strategy_name}][{symbol}] 建仓请求被否决。原因: {reject_reason}")
            # send_email(subject=f"[{strategy_name}][{symbol}] 建仓请求被否决",content=reject_reason)
            return
        
        try:
            market = get_market_type(symbol)
            
            # ==================================================================
            # 1. 资金计算 (Capital Calculation)
            # ==================================================================
            total_asset = self.get_total_account_value_in_hkd()
            if market == MarketType.US:
                exchange_rate = getattr(loop_config, 'exchange_rate_usd_to_hkd', 7.8)
                total_asset /= exchange_rate
            
            # ==================================================================
            # 2. 仓位规模计算 (Position Sizing - Night Mode)
            # ==================================================================
            # [核心修正] 严格遵循夜盘配置，使用固定仓位比例 (max_pos_ratio: 0.05)
            # 夜盘流动性差，不进行基于波动率的风险敞口计算，而是预算制控制
            pos_ratio = loop_config.night_config.get('max_pos_ratio', 0.05)
            budget = total_asset * pos_ratio
            current_status = get_trading_window_status(symbol)

            buy_ratio = 1.0
            if is_entering_weekend_risk_for_symbol(symbol, wrp_activation_days=[3]) or current_status in [TradingWindowStatus.NIGHT_ASIA_CORRELATION,TradingWindowStatus.NIGHT_LUNCH_DIP]:
                buy_ratio = 0.50
            
            # 计算股数
            quantity_to_buy = int(budget*buy_ratio / current_price)
            
            # 最小交易单位调整 (Lot Size Adjustment)
            stock_info = self.get_cached_stock_static_info(symbol)
            lot_size = stock_info.get('lot_size', 1 if market == MarketType.US else 100)
            quantity_to_buy = self._adjust_quantity(quantity_to_buy, market, lot_size)
            
            # [兜底] 如果计算出0股，但资金允许，尝试买一手/一股进行“占座”
            if quantity_to_buy <= 0:
                logger.warning(f"[{symbol}] 夜盘计算数量为0 (预算:{budget:.2f})，尝试最小手数保底。")
                quantity_to_buy = lot_size

            # ==================================================================
            # 3. 止损与风险参数 (Stop Loss & Risk)
            # ==================================================================
            # 夜盘使用刚性止损：买入价回撤 0.5%
            # 这里不使用 adaptive_stop_loss，因为夜盘数据稀疏，ATR计算可能失真
            stop_loss_price = current_price * (1-loop_config.night_config.get('stop_loss_ratio',self.config.night_rigid_stop_ratio))
            per_share_risk = current_price - stop_loss_price

            # ==================================================================
            # 4. 提交限价单 (Limit Order Submission)
            # ==================================================================
            # 挂单价格 = 现价 (或者 current_price * 1.005 以确保成交)
            # price_limit = trigger_price * (1.0 + self.config.limit_order_price_buffer)
            # 夜盘必须用 LO，不能用 MO
            if market == MarketType.US:
                # 美股规则：>= $1 保留2位，< $1 保留4位
                if current_price >= 1.0:
                    limit_price_str = "{:.2f}".format(current_price)
                else:
                    limit_price_str = "{:.4f}".format(current_price)
            elif market == MarketType.HK:
                limit_price_str = "{:.3f}".format(current_price)
            else:
                limit_price_str = "{:.2f}".format(current_price)
            
            limit_price = Decimal(limit_price_str)
            
            logger.critical(f"🚀 [夜盘执行] {symbol} | 仓位比例:{pos_ratio:.1%} | 数量:{quantity_to_buy} | 限价:{limit_price}")
            
            if not loop_config.test_mode:
                # 调用 submit_order_lo 提交限价单
                order_id = self.submit_order_lo(symbol, quantity_to_buy, OrderSide.Buy, limit_price)
                
                if order_id:
                    # 5. 注册到待处理列表 (Pending Orders)
                    # 这里的 plan_info 结构必须完整，以便主循环接管生成 Position
                    with self.pending_orders_lock:
                        self.pending_orders[symbol] = {
                            "order_id": order_id,
                            "plan_info": {
                                "planned_total_quantity": quantity_to_buy, # 夜盘通常是一次性买入，没有分批计划
                                "initial_risk_per_share": round(per_share_risk,3),
                                "initial_stop_loss_price": round(stop_loss_price,3), # 写入刚性止损
                                "triggering_strategy": loop_config.night_config.get('strategy_name', 'NightHunter'),
                                "strategy_class_name": "NightHunterStrategy", # 明确标记为夜猎者策略
                                # "overall_phase": PositionOverallPhase.RUNNING, # 直接进入运行期，跳过建仓逻辑
                                "building_stage": 1 if self.config.micro_building_config['enabled'] else 2,
                                "confirmation_add_done": False
                            }
                        }
                    self._save_pending_orders()
                    logger.info(f"[{symbol}] 夜盘订单已提交并注册 pending_orders。止损线: {stop_loss_price:.2f}")
                    
                    # 6. 任务完成，从缓存中移除该信号
                    self.pending_buy_cache.remove_signal(symbol)

                    # ▼▼▼ 通知 ▼▼▼
                    self.notification_manager.send_trade_execution(
                        action="BUY (NIGHT)",  # 标记为夜盘买入
                        symbol=symbol,
                        quantity=quantity_to_buy,
                        price=float(limit_price), # 这里的价格是挂单价
                        reason=f"夜盘狙击触发: {strategy_name}"
                    )
        
        except Exception as e:
            logger.error(f"执行夜盘买入 {symbol} 时发生错误: {e}", exc_info=True)
        
        finally:
            # [必须] 无论成功失败，都要释放飞行锁
            with self.signals_in_flight_lock:
                self.signals_in_flight.discard(symbol)

    def _execute_night_buy(self, symbol: str, price: float, stop_base: float):
        """夜盘限价买入执行逻辑 - 这里的 stop_base 是 session_low"""
        try:
            # 1. 资金计算
            market = get_market_type(symbol)
            total_asset = self.get_total_account_value_in_hkd()
            if market == MarketType.US:
                exchange_rate = getattr(self.config, 'exchange_rate_usd_to_hkd', 7.8)
                total_asset /= exchange_rate
                
            # 2. 仓位计算 (基于总资产的固定比例，例如15%)
            # 夜盘流动性差，严格控制仓位
            budget = total_asset * self.config.night_config['max_pos_ratio']
            quantity = int(budget / price)
            quantity = int(quantity * self.config.position_scaling_factors.get(symbol,self.config.position_scaling_factors['default']))
            
            # 最小股数调整
            stock_info = self.get_cached_stock_static_info(symbol)
            lot_size = stock_info.get('lot_size', 1 if market == MarketType.US else 100)
            quantity = self._adjust_quantity(quantity, market, lot_size=lot_size)
            
            if quantity <= 0:
                logger.warning(f"[{symbol}] 夜盘计算买入数量为0，放弃操作。")
                return

            # 3. 智能下单 (挂现价+0.5%的限价单，确保成交但防滑点)
            # limit_price = round(price * 1.005, 2) 
            # 强制转换为符合精度的 Decimal
            if market == MarketType.US:
                limit_price = Decimal("{:.2f}".format(price))
            else:
                limit_price = Decimal("{:.3f}".format(price))
            
            # 4. 提交订单 (使用 submit_order_lo 限价单专用方法)
            # 注意：夜盘/盘前有些券商不支持市价单(MO)，必须用限价单(LO)
            order_id = self.submit_order_lo(symbol, quantity, OrderSide.Buy, limit_price)
            
            if order_id:
                # 5. 构造特殊的 Position 计划信息
                # 关键：我们将 stop_loss 直接写入 plan，这样成交后生成的 Position 就自带止损
                stop_loss = stop_base * (1.0 - self.config.night_rigid_stop_ratio) # 跌破前低0.5%止损
                
                with self.pending_orders_lock:
                    self.pending_orders[symbol] = {
                        "order_id": order_id,
                        "plan_info": {
                            "planned_total_quantity": quantity,
                            # 风险计算：这里填入真实的每股风险
                            "initial_risk_per_share": round(price - stop_loss,3),
                            # 核心：将夜猎者的刚性止损写入初始化参数
                            "initial_stop_loss_price": round(stop_loss,3),
                            # 标记策略身份 "NightHunter"
                            "triggering_strategy": self.config.night_config['strategy_name'],
                            "strategy_class_name": "NightHunterStrategy",
                            "confirmation_add_done":False,
                            # 夜盘买入直接进入Running阶段，不需要分批建仓逻辑干扰
                            # "overall_phase": PositionOverallPhase.RUNNING
                        }
                    }
                    
                self._save_pending_orders()
                logger.info(f"[{symbol}] 夜盘待处理订单已创建。止损位: {stop_loss:.2f}")
                
        except Exception as e:
            logger.error(f"执行夜盘买入 {symbol} 时出错: {e}", exc_info=True)

    def _check_pending_position_transactions(self, symbols: list):
        """
        统一检查已持仓股票的待处理事务（加仓、卖出等）。
        这是一个状态机，确保一个仓位在同一时间只处理一种待定事务。
        """
        with self.position_lock:
            positions_to_check = [self.positions.get(symbol) for symbol in symbols]

        for pos in positions_to_check:
            if not pos:
                continue
            if pos.pending_sell_order_id:
                self._handle_pending_sell_order(pos)
            if pos.pending_pyramid_order_id:
                logger.warning(f"[{pos.symbol}] 保守策略不处理加仓订单，清理遗留加仓锁。")
                with self.position_lock:
                    current_pos = self.positions.get(pos.symbol)
                    if current_pos:
                        current_pos.pending_pyramid_order_id = None
                        current_pos.pending_add_reason_tag = None
                        self._save_positions()
        return

        symbols_to_check = symbols[:] # 保留您的原始实现，创建副本
        ORDER_TIMEOUT = 180  # 超时阈值：3分钟
        with self.position_lock:
            for symbol in symbols_to_check:
                pos = self.positions.get(symbol)
                if not pos:
                    continue
                
                # 如果是夜猎者持仓，或者是正常盘中，都允许检查订单状态
                # is_regular_open = is_any_market_open(symbol)
                # is_night_hunter = (pos.triggering_strategy == self.config.night_config.get('strategy_name', 'NightHunter'))
                
                # if not is_regular_open and not is_night_hunter:
                if get_current_market_session(MarketType.US)==TradingSession.MARKET_CLOSED:
                    # 如果既不是盘中，也不是夜盘策略股，才跳过
                    continue
                
                # =======================
                # 逻辑分支 A: 卖出订单监控
                # =======================
                if pos.pending_sell_order_id:
                    # [Review 2] 获取市场类型，判断是否为盘中
                    # 只有非盘中(Pre/Post/Night)才启用重试，绝对隔离盘中(RTH)逻辑
                    market = pos.market
                    is_rth = is_market_in_trading_hours(market)
                    
                    if not is_rth:
                        # 获取挂单时间
                        # 优先从内存字典取，取不到(如重启后)则补录当前时间，避免立即误杀
                        start_ts = self.extended_hours_order_timers.get(symbol, 0)
                        if start_ts == 0:
                            self.extended_hours_order_timers[symbol] = time.time()
                        
                        # 检查超时
                        elif (time.time() - start_ts) > ORDER_TIMEOUT:
                            logger.warning(f"⏰ [{symbol}] 盘外限价单 {pos.pending_sell_order_id} 挂单超时(>3min)，准备重置...")
                            
                            old_order_id = pos.pending_sell_order_id
                            old_reason = pos.sell_reason or "超时重发"

                            # [僵尸锁熔断]
                            # 绝对禁止拿 "SUBMITTING_GUARD" 去查询 API，这是日志刷屏的根源。
                            # 如果超时了还是这个值，说明发单线程已经挂了，必须强制重置。
                            if old_order_id == "SUBMITTING_GUARD":
                                logger.error(f"🛡️ [{symbol}] 发单线程疑似崩溃 (僵尸锁超时)，强制释放锁状态，不执行查询。")
                                with self.position_lock:
                                    if symbol in self.positions:
                                        self.positions[symbol].pending_sell_order_id = None
                                        self._save_positions()
                                continue # 直接跳过，等待下一轮循环重新触发卖出

                            # 获取订单详情
                            # 必须检查是否"部分成交"。如果已经成交了一部分，说明价格吻合，不是死单。
                            # 此时不应暴力撤单重发，而应走正常流程结算，防止数据混乱。
                            should_retry = False # 默认为False，只有明确查不到或者查到未成交才重试，安全第一
                            retry_ratio = 0.0
                            
                            try:
                                kwargs = {
                                    'symbol': symbol
                                }
                                o_detail = self.data_provider.get_order_detail(old_order_id, **kwargs)
                                
                                if o_detail is None:
                                    logger.warning(f"⚠️ [{symbol}] 无法获取订单 {old_order_id} 详情 (API返回None)，网络可能波动，暂缓处理。")
                                    continue # 既然查不到，就别瞎猜，等下一轮网络好了再查
                                
                                status = o_detail.get('status', 'Unknown')
                                # 安全获取 quantity，防止由 float 转换带来的意外
                                executed_qty_raw = o_detail.get('quantity', 0)
                                executed_qty = float(executed_qty_raw) if executed_qty_raw is not None else 0.0

                                # 逻辑分支 A: 订单已存在且有效 (部分成交或已完成) -> 不需要暴力重发，走正常结算流程
                                if executed_qty > 0 or status == 'Filled':
                                    logger.info(f"[{symbol}] 订单 {old_order_id} 状态正常 (St: {status}, Fill: {executed_qty})，跳过超时重置，等待常规结算。")
                                    should_retry = False
                                
                                # 逻辑分支 B: 订单虽在但未成交 (且已超时) -> 可能是死单，允许撤单重发
                                # 注意: status 可能为 'Unknown' 或 'Pending'
                                else:
                                    logger.warning(f"[{symbol}] 订单 {old_order_id} 挂单超时且无成交，准备重发。")
                                    should_retry = True
                                    # [除零保护]
                                    if pos.total_quantity > 0:
                                        retry_ratio = min(1.0, float(o_detail.get('quantity', 0)) / float(pos.total_quantity))
                                    else:
                                        retry_ratio = 0.0 # 仓位都没了，别发了

                            except Exception as e:
                                logger.error(f"[{symbol}] 处理超时订单逻辑发生异常: {e}", exc_info=True)
                                # 发生异常时不进行任何操作，防止逻辑错乱
                                continue

                            if should_retry:
                                # 1. 撤单 (吞掉异常，防止流程中断)
                                try:
                                    kwargs = {
                                        'symbol': symbol
                                    }
                                    self.data_provider.cancel_order(old_order_id,**kwargs)
                                    time.sleep(1) # 给券商撮合引擎一点反应时间
                                except: pass

                                # 2. 清理旧状态 (必须在锁内)
                                with self.position_lock:
                                    if symbol in self.positions:
                                        self.positions[symbol].pending_sell_order_id = None
                                        # 注意：这里不清理 sell_reason，把它传给新单
                                        self._save_positions()
                                
                                # 3. 原地重发
                                # 直接调用 execute 函数，绕过外层 _is_action_recently_taken 检查
                                # 函数内部会获取最新 current_price 并挂单
                                logger.warning(f"🔄 [{symbol}] 执行原地重发 (比例: {retry_ratio:.2f})...")
                                self._execute_extended_hours_sell(symbol, old_reason, sell_ratio=retry_ratio)
                                
                                # 4. 更新计时器 (在 _execute 内部其实已经做了，这里continue即可)
                                continue 

                    # 常规处理逻辑
                    # 处理盘中订单，或者盘外未超时/部分成交的订单
                    self._handle_pending_sell_order(pos)
                    
                    # [Review 8] 清理计时器
                    # 如果订单处理完了(ID没了)，把内存计时器删掉，防止内存泄漏
                    if not pos.pending_sell_order_id and symbol in self.extended_hours_order_timers:
                        del self.extended_hours_order_timers[symbol]

                # =======================
                # 逻辑分支 B: 加仓订单监控
                # =======================
                elif pos.pending_pyramid_order_id:
                    self._handle_pending_add_order(pos)
                    
                # if pos.pending_pyramid_order_id:
                #     self._handle_pending_add_order(pos)
                # elif pos.pending_sell_order_id:
                #     # 仅针对夜盘/盘外策略使用内存字典检查超时
                #     is_night_hunter = (pos.triggering_strategy == "NightHunter")
                    
                #     if is_night_hunter:
                #         # 从内存字典获取时间，如果没有记录(比如重启过)，则视为刚发单
                #         start_time = self.extended_hours_order_timers.get(symbol, time.time())
                        
                #         # 检查是否超过 3分钟 (180秒)
                #         if (time.time() - start_time) > 180:
                #             logger.warning(f"⏰ [{symbol}] 夜盘限价单 {pos.pending_sell_order_id} 超时，执行撤单重置...")
                            
                #             # 1. 尝试撤单
                #             try:
                #                 self.trade_ctx.cancel_order(pos.pending_sell_order_id)
                #                 time.sleep(1)
                #             except: pass
                            
                #             # 2. 清理状态
                #             with self.position_lock:
                #                 if symbol in self.positions:
                #                     self.positions[symbol].pending_sell_order_id = None
                #                     self._save_positions()
                                    
                #             # 3. 清理计时器
                #             if symbol in self.extended_hours_order_timers:
                #                 del self.extended_hours_order_timers[symbol]
                                
                #             logger.warning(f"🔄 [{symbol}] 状态重置完成，等待下一轮按最新价重挂。")
                #             continue # 跳过常规检查

                #     # 常规逻辑保持不变
                #     self._handle_pending_sell_order(pos)

    def _monitor_sentiment_risk(self, symbol: str) -> bool:
        """
        【舆情核按钮-全天候版】
        
        功能：
        全时段（盘中+盘外）监控持仓舆情。
        基于 query_only=True 读取外部定时任务生成的缓存，极低延迟。
        一旦发现“负向”，立即根据当前市场状态选择最优清仓路径。

        Returns:
            bool: True 表示触发了熔断清仓，调用方应立即终止对该股的其他检查。
        """
        # 1. 内存级节流：每 120 秒检查一次缓存
        # 既然外部是1小时更新一次，我们每分钟检查一次内存足矣，
        # 既不错过更新，也避免无意义的 IO。

        # [双重保险] 如果该股票已经不在持仓列表里了，直接清理遗留计时器并退出
        # 这防止了某些极端情况下的内存泄漏
        with self.position_lock:
            if symbol not in self.positions:
                if symbol in self.sentiment_cache_timers:
                    del self.sentiment_cache_timers[symbol]
                return False
        
        CACHE_CHECK_INTERVAL = 60*2
        now_ts = time.time()
        last_check = self.sentiment_cache_timers.get(symbol, 0)
        
        if now_ts - last_check < CACHE_CHECK_INTERVAL:
            return False
            
        try:
            # 更新检查时间
            self.sentiment_cache_timers[symbol] = now_ts
            
            # 2. 闪电读取：只读缓存，毫秒级返回
            # 注意：这里必须是 True，正如你所要求的
            # sentiment = self.sentiment_analysis.get_news_sentiment(symbol, query_only=True)
            is_fresh, sentiment_data = self.sentiment_analysis.is_cache_fresh(symbol, ttl_minutes=60)
            sentiment_status = sentiment_data.get('sentiment', '未知')
            
            # 调试日志
            # logger.debug(f"[{symbol}] 舆情巡检: {sentiment}")

            # 3. 熔断判断：只针对【负向】
            # "复杂/中性" 留给技术指标去博弈，"负向" 必须跑
            if sentiment_status == "负向":
                reason = "🔥 [舆情熔断] 侦测到负向舆情，无条件清仓止损"
                logger.critical(f"★★★ [{symbol}] 舆情核按钮触发！评级: {sentiment_status} -> 立即执行清仓 ★★★")
                
                # 4. 双通道执行机制
                is_rth = is_market_in_trading_hours(get_market_type(symbol))
                is_high_confirmed = False

                if is_rth:
                    # === 盘中通道 (RTH) ===
                    k_mins_check = 5
                    rebound_pct_threshold = self.config.rebound_pct_threshold_map['default']
                    is_high_confirmed = check_tactical_exit_signal(self.quote_ctx, symbol, 5,rebound_pct_threshold)
                    # 走标准卖出流程，通常使用市价单(MO)或激进限价单
                    # percentage=1.0 强制清仓
                    self.process_sell_signal(
                        symbol=symbol,
                        percentage=1.0,
                        reason=reason
                    )
                else:
                    # === 盘外通道 (Extended Hours) ===
                    k_mins_check = self.config.tactical_k_mins_map.get('SCALP_EXIT', 3)
                    # 死线临头，回撤阈值放宽(0.2%)；诱多陷阱，要求严格(0.5%)
                    rebound_pct_threshold = 0.005
                    is_high_confirmed = check_extended_hours_tactical_exit_signal(
                        self.hs_data_provider, symbol, k_mins_check, rebound_pct_threshold
                    )
                    if is_high_confirmed:
                        # 走夜盘/盘前流程，强制使用限价单(LO)保护
                        self._execute_extended_hours_sell(
                            symbol=symbol,
                            reason=reason,
                            sell_ratio=1.0
                        )
                if is_high_confirmed:
                    # 立即加入黑名单并持久化
                    self.intraday_blacklist.add(symbol)
                    self.intraday_blacklist.add(resolve_underlying_symbol(symbol))
                    self._save_blacklist()
                    logger.warning(f"🚫 [{symbol}] 已加入当日交易黑名单，今日禁止再次买入。")

                # 返回 True，告诉主循环：这只票完了，别再做止盈止损检查了
                return True

        except Exception as e:
            # 舆情模块的任何错误不应导致交易主线程崩溃，记录即可
            logger.error(f"[{symbol}] 舆情风控检查异常: {e}")
            
        return False
    
    def _check_position_signals(self, symbol: str):
        """
        检查单个持仓的所有信号（按优先级排序）。
        这是交易系统的核心风险控制和机会管理逻辑。

        _check_position_signals 的所有分支：
        _check_and_execute_stop_loss
        依赖： position.initial_stop_loss_price
        分析： 这个值在开仓时由 R 计算得出。因为R在两种模式下一致，所以初始硬止损价也完全一致。
        结论：行为完全一致。
        _check_and_execute_partial_sell_protection
        依赖： position.partial_sell_price, config.partial_sell_drop_ratio
        分析： partial_sell_price 是在部分止盈时记录的。因为R一致，所以止盈点位也一致，这个值也会一致。
        结论：行为完全一致。
        _check_and_execute_milestone_1R (包含 _manage_breakeven_stop 和 check_profit_taking_signals)
        依赖： position.initial_risk_per_share (R), config.breakeven_trigger_r_multiple, config.profit_take_r_multiple
        分析： 所有这些逻辑都基于R的倍数。因为R在两种模式下一致，所以1R里程碑的所有行为（保本、卖出半仓）都将在完全相同的价格点位触发。
        结论：行为完全一致。
        _manage_trailing_stop_activation
        依赖： position.initial_risk_per_share (R), config.trailing_stop_activation_multiplier
        分析： 激活追踪止损的盈利目标是R的倍数。因为R一致，所以激活点位也完全一致。
        结论：行为完全一致。
        _check_scale_in_conditions (包含下跌补仓和上涨追涨)
        下跌补仓依赖： config.dip_add_triggers_percent (基于价格百分比)。这个逻辑从不依赖R。
        上涨追涨依赖： position.initial_risk_per_share (R), config.rise_add_trigger_r_multiple。因为R一致，所以上涨追涨的点位也完全一致。
        结论：行为完全一致。
        _check_pyramid_add_condition
        依赖： position.initial_risk_per_share (R), config.pyramid_profit_multiplier
        分析： 金字塔加仓的盈利目标是R的倍数。因为R一致，所以加仓点位也完全一致。
        结论：行为完全一致。

        核心改进：
        1. 止损无时间限制（生存第一）
        2. 止盈受时间窗口控制（利润优化）
        3. 加仓受时间窗口控制（风险管理）
        """
        with self.position_lock:
            position = self.positions.get(symbol)
            if not position:
                return
            if position.pending_sell_order_id:
                return

        if self._is_account_monthly_loss_limit_hit():
            self._execute_full_sell(symbol, "当月累计账户总亏损达到6%")
            return
        if self._is_account_daily_loss_limit_hit():
            self._execute_full_sell(symbol, "账户当日总亏损达到3%")
            return

        current_price = self.get_current_price(symbol)
        if current_price is None or current_price <= 0:
            return

        cost_price = position.get_avg_cost(self.config)
        if cost_price <= 0:
            return

        now_utc = datetime.now(timezone.utc)
        roi = (current_price - cost_price) / cost_price
        holding_days = position.get_minutes_since_first_buy() / 1440.0

        with self.position_lock:
            live_pos = self.positions.get(symbol)
            if not live_pos:
                return
            if live_pos.strategy_params is None or not isinstance(live_pos.strategy_params, dict):
                live_pos.strategy_params = {}
            state = live_pos.strategy_params.setdefault('conservative_exit_state', {})
            highest_price = float(state.get('highest_price') or cost_price)
            if current_price > highest_price:
                highest_price = current_price
                state['highest_price'] = round(highest_price, 4)
                self._save_positions()

        peak_roi = (highest_price - cost_price) / cost_price if cost_price > 0 else 0.0
        drawdown_from_peak = (highest_price - current_price) / highest_price if highest_price > 0 else 0.0

        if roi <= -0.07:
            self._execute_full_sell(symbol, f"单只个股硬止损7%触发 | ROI:{roi:.2%}")
            return

        if roi >= 0.20:
            self._execute_full_sell(symbol, f"浮盈达到20%清仓 | ROI:{roi:.2%}")
            return

        if peak_roi >= 0.05 and drawdown_from_peak >= 0.02:
            self._execute_full_sell(
                symbol,
                f"盈利超过5%后从高点回撤2%清仓 | 峰值ROI:{peak_roi:.2%}, 回撤:{drawdown_from_peak:.2%}"
            )
            return

        if roi < 0.10 and holding_days > 14:
            self._execute_full_sell(symbol, f"未达10%浮盈且持仓超过2周 | ROI:{roi:.2%}, 持仓:{holding_days:.1f}天")
            return

        state_changed = False
        with self.position_lock:
            live_pos = self.positions.get(symbol)
            if not live_pos:
                return
            state = live_pos.strategy_params.setdefault('conservative_exit_state', {})

            if 0.10 <= roi < 0.15 and not state.get('entered_10_15_at'):
                state['entered_10_15_at'] = now_utc.isoformat()
                state_changed = True
            elif not (0.10 <= roi < 0.15) and state.get('entered_10_15_at') and not state.get('stage_10_taken'):
                state['entered_10_15_at'] = None
                state_changed = True

            if 0.15 <= roi < 0.20 and not state.get('entered_15_20_at'):
                state['entered_15_20_at'] = now_utc.isoformat()
                state_changed = True
            elif not (0.15 <= roi < 0.20) and state.get('entered_15_20_at') and not state.get('stage_15_taken'):
                state['entered_15_20_at'] = None
                state_changed = True

            entered_10_15_at = state.get('entered_10_15_at')
            entered_15_20_at = state.get('entered_15_20_at')
            stage_10_taken = bool(state.get('stage_10_taken'))
            stage_15_taken = bool(state.get('stage_15_taken'))

            if state_changed:
                self._save_positions()

        try:
            if 0.10 <= roi < 0.15 and entered_10_15_at:
                days_in_band = (now_utc - normalize_to_utc(entered_10_15_at)).total_seconds() / 86400.0
                if days_in_band > 7:
                    self._execute_full_sell(symbol, f"浮盈10%-15%区间持仓超过一周 | ROI:{roi:.2%}")
                    return
            if 0.15 <= roi < 0.20 and entered_15_20_at:
                days_in_band = (now_utc - normalize_to_utc(entered_15_20_at)).total_seconds() / 86400.0
                if days_in_band > 7:
                    self._execute_full_sell(symbol, f"浮盈15%-20%区间持仓超过一周 | ROI:{roi:.2%}")
                    return
        except Exception as e:
            logger.warning(f"[{symbol}] 计算分段持仓时间失败: {e}")

        if roi >= 0.15 and not stage_10_taken:
            if self.process_sell_signal(symbol, percentage=0.35, reason=f"浮盈达到10%阶段止盈 | 当前ROI:{roi:.2%}"):
                return
            return

        if roi >= 0.15 and not stage_15_taken:
            if self.process_sell_signal(symbol, percentage=0.45, reason=f"浮盈达到15%阶段止盈 | 当前ROI:{roi:.2%}"):
                return
            return

        if roi >= 0.10 and not stage_10_taken:
            if self.process_sell_signal(symbol, percentage=0.35, reason=f"浮盈达到10%阶段止盈 | 当前ROI:{roi:.2%}"):
                return
            return

        return

        # 准备阶段：获取锁，确保数据一致性
        with self.position_lock:
            position = self.positions.get(symbol)
            # ==============================================================================
            # ▼▼▼ 影子标签：底层止损免疫 (Underlying Immunity) ▼▼▼
            # ==============================================================================
            if symbol in self.shadow_tags.get('strategic_hold', []):
                # 拥有战略锁定金牌，直接跳过本轮所有的止损/止盈风控检查！
                return
            
            if self.disable_trade: # 禁止交易后，就不会执行下面代码
                return
            # 如果是豁免策略，直接跳过所有复杂的止损/止盈计算
            # if position and position.triggering_strategy in self.config.strategies_immune_to_exit:
            #     # logger.debug(f"[{symbol}] 策略 '{position.triggering_strategy}' 豁免检查，跳过。")
            #     return

            if not self._is_holding_period_satisfied(position, required_minutes=self.config.min_holding_minutes):
                return
            # ================= 【持仓风控分流器】 =================
            if position and self._is_bearish_symbol(symbol):
                # 如果是做空工具，则执行专属风控后直接返回。
                # 这样可以彻底阻断做多风控链穿透到反向ETF。
                self._manage_bearish_position_exit(position, symbol)
                return
            # ===============================================================
            # ▼▼▼ 做多仓位主干退出引擎（对称做空）▼▼▼
            # 在所有日内止盈/止损策略之前，先跑主干：
            #   快速回血 30% → 保本线上移 → 追踪止损 + ATR 自适应硬止损。
            # 一旦主干触发卖出，直接返回，避免其他策略重复开火。
            # 未触发时，继续往下跑 Guardian / ApexPredator 等增益层。
            # ===============================================================
            if position and getattr(self.config, 'enable_bullish_position_exit', False):
                try:
                    if self._manage_bullish_position_exit(position, symbol):
                        return
                except Exception as _e_bull:
                    logger.error(f"[{symbol}] 做多主干退出引擎异常: {_e_bull}", exc_info=True)
            # ===============================================================
            if not position: return


            # 提取判断变量
            is_night_hunter = (position.triggering_strategy == "NightHunter")
            market = position.market
            # 严格判断是否为美股盘中 (9:30 - 16:00)
            is_us_rth = is_market_in_trading_hours(MarketType.US)

            # 获取当前Session状态 (用于判断是否完全休市)
            current_session = get_current_market_session(market)
        
        # 前置过滤：如果完全休市，直接跳过
        if current_session == TradingSession.MARKET_CLOSED:
            return
        
        # 逻辑：如果是夜猎者策略，且当前不在盘中，且市场没关门（即处于盘前/盘后/夜盘）
        if is_night_hunter and not is_us_rth:
            self._manage_extended_hours_position(position, symbol)
            # 微观建仓二阶段检查
            if position.overall_phase == PositionOverallPhase.BUILDING and position.building_stage < 2:
                current_price = self.get_current_price(symbol)
                if current_price is None: return
                self._process_micro_building_continuation(symbol, position, current_price)
                return # 建仓未完成前，暂时不跑止盈逻辑，止损由 continuation 内部处理
        
            return # 专属逻辑接管，不再执行后续常规检查

        is_regular_open = is_any_market_open(symbol)
        # 如果不是盘中，普通策略也不该跑，直接退    
        if not is_regular_open:
            return
            
        # 2. 前置检查：避免指令冲突
        if position.pending_pyramid_order_id or position.pending_sell_order_id:
            return

        current_price = self.get_current_price(symbol)
        if current_price is None: return

        # ==============================================================================
        # ▼▼▼ 优先级：-1 闪电审判协议 (初生头寸防崩) ▼▼▼
        # ------------------------------------------------------------------------------
        # 在执行复杂的动能止损、网格止损之前，先判定买入逻辑是否被物理打脸。
        # if self._execute_infant_flash_trial(symbol, position, current_price):
        #     return  # 审判已执行，终止后续所有检查
        # ==============================================================================


        # ==============================================================================
        # ▼▼▼ 将新风控逻辑置于最高优先级 ▼▼▼
        # ------------------------------------------------------------------------------
        # 优先级 0: 持续新低动能止损 (最优先的生存法则)
        # 如果股价持续无力，说明基本面或市场情绪发生恶化，必须立刻离场。
        if self._check_continuous_low_stop_loss_pro(symbol, current_price, position):
            return  # 止损已触发，本轮检查对该股的所有后续操作（止盈/加仓等）全部终止。
        # ==============================================================================

        # ==============================================================================
        # ▼▼▼ 模块 2: 环境感知与止损路由 (Survival - Adaptive Defense) ▼▼▼
        # ==============================================================================

        # === 获取当前交易时间窗口状态 ===
        current_status = get_trading_window_status(symbol)
        # 定义有利卖出窗口
        favorable_sell_windows = self.config.favorable_sell_windows
        
        # 判断是否处于有利卖出时段
        is_favorable_sell_time = current_status in favorable_sell_windows
        
        # 环境感知 (Context Awareness)
        current_regime = self.market_regime_engine.get_marget_regime(market)
        intraday_health, _ = self.market_regime_engine.check_intraday_health(market)
        
        # 定义“恶劣环境” (Adverse Conditions)
        # 判定标准：熊市、极度风险、震荡市，或者日内红灯(R)/黄灯(Y)
        is_adverse_condition = (
            current_regime in [MarketRegime.CONFIRMED_BEAR, MarketRegime.HIGH_RISK_AVOID] or 
            intraday_health in [IntradayHealthType.R] # IntradayHealthType.Y
        )
        is_adverse_condition = False
        # 逻辑：开盘前5分钟机器算法乱战，点差极大，此时触发止损极易卖在地板。
        # 除非是灾难性暴跌(由动能止损处理)，否则暂缓常规止损。
        # is_opening_chaos = is_in_opening_window(market, window_minutes=5)

        if not is_favorable_sell_time or position.dip_adds_done > 1:#情况1是符合卖出窗口，情况2是加仓完毕，还在亏损
            # --- 核心路由 (The Router) ---
            if is_adverse_condition:
                # [逆风局] -> 开启网格防御 (Grid Stop)
                # 策略：分批撤退，保留火种，少亏当赢。
                if self._check_and_execute_stop_loss_grid(symbol, current_price, position): return
            else:
                # [顺风局] -> 开启标准宽幅止损
                # 策略：硬止损兜底 + 追踪止损奔跑，防止在上涨中继被洗盘。
                if self._check_and_execute_composite_stop(symbol, current_price, position): return
  
        # ==============================================================================
        # ▼▼▼【“标记-猎杀”智能时间止损】▼▼▼
        # ------------------------------------------------------------------------------
        # 检查此规则是否只应用于“日内了结”的策略
        if self.config.enable_intraday_time_stop and position.triggering_strategy in self.config.day_trade_only_strategies and is_opened_today(position,symbol):
            
            # 决策一：如果一个头寸已经被标记，那么我们的唯一任务就是找机会卖掉它
            if position.marked_for_liquidation:
                if is_favorable_sell_time:
                    reason = f"执行被标记的时间止损(进入有利卖出窗口: {current_status.name})"
                    logger.warning(f"⏰🏹 {reason} -> {symbol}")
                    if not self.config.test_mode:
                        self._execute_full_sell(symbol, reason)
                    return # 立即终止
            
            # 决策二：如果头寸尚未被标记，则检查是否应该标记它
            else:
                try:
                    holding_duration_minutes = position.get_minutes_since_first_buy()
                    current_profit_r = (current_price - position.avg_cost) / position.avg_cost

                    # 核心标记条件[20分钟以上，利润是亏损的，并且亏损范围在1%左右]
                    if holding_duration_minutes > self.config.intraday_time_stop_minutes and current_profit_r < 0 and abs(current_profit_r) >= self.config.intraday_time_stop_profit_threshold_r:
                        
                        # 条件满足后，我们不直接卖，而是先判断时机
                        if is_favorable_sell_time:
                            # 时机也好，直接“标记并猎杀”
                            reason = f"时间止损: 持仓超时且未盈利，在有利窗口({current_status.name})立即执行"
                            logger.warning(f"⏰🏹 {reason} -> {symbol}")
                            if not self.config.test_mode:
                                self._execute_full_sell(symbol, reason)
                            return # 立即终止
                        else:
                            # 时机不好，只“标记”，不“猎杀”，把任务留给下一次循环
                            with self.position_lock:
                                # 必须在锁内修改状态并保存
                                if symbol in self.positions:
                                    self.positions[symbol].marked_for_liquidation = True
                                    self._save_positions()
                                    reason = f"时间止损: 持仓超时且未盈利，但当前窗口({current_status.name})不利于卖出。已标记待清算"
                                    logger.warning(f"⏰ {reason} -> {symbol}")
                            # 注意：这里标记后不`return`，让后续的止盈/加仓逻辑继续跑。
                            # 因为在被最终清算前，如果它突然大幅拉升满足了止盈，我们当然也应该止盈。
                except Exception as e:
                    logger.error(f"[{symbol}] 在执行 V2.0 时间止损检查时发生错误: {e}", exc_info=True)
        # ==============================================================================

        # 微观建仓二阶段检查
        # 只要 building_stage < 2 (包含 1 和 101)，都属于“建仓未完成”
        if position.overall_phase == PositionOverallPhase.BUILDING and position.building_stage < 2:
            # 调用核心推进器，它内部会处理 1->101 和 101->2 的状态机流转
            self._process_micro_building_continuation(symbol, position, current_price)
            return # 建仓未完成前，暂时不跑止盈逻辑，止损由 continuation 内部处理
        
        # 2a. 检查是否应该主动进行部分止盈。
        if not position.r_profit_taken: # 只检查一次，避免重复卖出
            should_sell, sell_ratio, reason = self.check_profit_taking_signals(position, current_price)
            if should_sell:
                # === 只在有利时段执行止盈 ===
                if is_favorable_sell_time:
                    logger.warning(f"[{symbol}] R倍数止盈条件满足，且处于有利卖出窗口 ({current_status.name})，执行卖出。")
                    if self.process_sell_signal(symbol, percentage=sell_ratio, reason=reason):
                        with self.position_lock:
                            if symbol in self.positions:
                                # 核心状态变更：标记“已成功获利了结”
                                self.positions[symbol].r_profit_taken = True
                        self._save_positions()
                        return # 提交了卖出指令，本轮检查结束，等待成交
                else:
                    # 【关键日志】记录延迟执行的决策
                    logger.info(f"[{symbol}] R倍数止盈条件已满足，但当前时段 ({current_status.name}) 不利于卖出，等待更佳时机。")
                    # 不返回，继续执行保本止损等防御逻辑

        # 2b. 检查“已获利了结”的状态，并根据此状态设置保本止损。
        # 这是我们新的“State-Driven Defense”核心。
        if position.r_profit_taken and not position.is_breakeven_stop_set:
            with self.position_lock:
                # 再次获取最新仓位，确保线程安全
                pos = self.positions.get(symbol)
                if pos and not pos.is_breakeven_stop_set:
                    pos.trailing_stop_price = pos.avg_cost
                    pos.is_breakeven_stop_set = True
                    pos.is_trailing_stop_active = True # 立即激活追踪止损，让保本止损生效
                    self._save_positions()
                    logger.warning(f"★★★ 状态驱动保本触发 for {symbol}! 剩余仓位止损已上移至成本价 {pos.avg_cost:.3f}。")

        # === 优先级 3: 动态防御升级 (Dynamic Defense) ===
        # 这里的逻辑只有在保本止损之上，才会进一步提升止损位。
        # 注意：_manage_breakeven_stop 方法现在可以安全移除了，因为它的功能已被上面的逻辑取代。
        # 动态止盈/止损管理 (移动止损位)
        # self._manage_breakeven_stop(symbol, current_price, position)
        self._manage_trailing_stop_activation(symbol, current_price, position)
        self._manage_trailing_stop_update(symbol, current_price, position)

        # 优先级 2 中已包含部分止盈后的保护逻辑（如果你的 _check_and_execute_stop_loss 足够完善），
        # 但为清晰起见，保留显式检查。

        # 2. 部分卖出后的保护性止损
        if position.partial_sell_price and position.partial_sell_price > 0:
            
            if self.config.enable_atr_partial_exit:
                # --- 步骤A: 更新“记忆” ---
                # 兼容旧数据，如果字段不存在，则用partial_sell_price初始化
                if position.highest_price_since_partial_sell is None:
                     position.highest_price_since_partial_sell = position.partial_sell_price
                # 永远只记录更高的价格，这就是“追踪”的精髓
                position.highest_price_since_partial_sell = max(position.highest_price_since_partial_sell, current_price)

                # --- 步骤B: 基于最新的“记忆”计算动态止损位 ---
                atr_value = get_historical_atr(self.quote_ctx, symbol)
                if atr_value and atr_value > 0:
                    # 止损位现在是动态的，每一轮都基于最新的最高价重新计算
                    stop_price = position.highest_price_since_partial_sell - (atr_value * self.config.partial_sell_trailing_atr_multiplier)
                    
                    # --- 步骤C: 执行判断 ---
                    if current_price <= stop_price:
                        reason = f"部分止盈后从期间高点({position.highest_price_since_partial_sell:.2f})回撤超过{self.config.partial_sell_trailing_atr_multiplier}倍ATR"
                        if not self.config.test_mode:
                            self._execute_full_sell(symbol, reason)
                        logger.warning(f"触发动态ATR清仓信号 {symbol}: {reason}")
                        self.notification_manager.send_trade_execution('LIQUIDATE', symbol, position.total_quantity, current_price, reason)
                        return
                    
            # --- [兼容旧逻辑] ---
            else:
                partial_sell_stop_price = position.partial_sell_price * (1 - self.config.partial_sell_drop_ratio)
                if current_price > position.avg_cost and current_price <= partial_sell_stop_price:
                    reason = f"部分止盈后回撤超过{self.config.partial_sell_drop_ratio:.1%}"
                    if not self.config.test_mode:
                        self._execute_full_sell(symbol, reason)
                    log_msg = f"触发清仓信号 {symbol}: {reason}"
                    logger.warning(log_msg)
                    self.notification_manager.send_trade_execution('LIQUIDATE', symbol,position.total_quantity, current_price, reason)
                    return

        # === 优先级 4: 进攻前检查 (Pre-Offense Check) ===
        # 在考虑加仓前，确保没有顶在止盈目标上。
        # 仅在未接近止盈目标时才考虑加仓
        if self._is_approaching_profit_target(position, current_price):
            logger.warning(f"[{symbol}] 已接近部分止盈目标，本轮暂停所有加仓检查。")
            return

        # === 优先级 5: 进攻机动 (Offensive Maneuvers) ===
        # 只有在所有防御都就位后，才考虑加仓。
        # === 【优先级 6: 加仓检查】受时间窗口和周末风险控制 ===
        # 定义有利加仓窗口（通常避开剧烈波动时段）
        favorable_add_windows = self.config.favorable_add_windows
        
        is_favorable_add_time = current_status in favorable_add_windows
        is_weekend_risk = self._is_entering_weekend_risk_for_symbol(symbol)

        if position.overall_phase == PositionOverallPhase.BUILDING:
            self._check_scale_in_conditions(symbol, current_price, position)
            
            # 建仓期加仓：时间窗口 + 周末风险双重控制
            # 10:30-11:00 ET: 早盘喧嚣结束，趋势企稳，机构资金进场确认期
            # is_mid_morning_stabilization = is_in_custom_trading_window(
            #                     market=MarketType.US,
            #                     start_minutes=60,
            #                     end_minutes=90
            #                 )
            # if is_favorable_add_time or is_mid_morning_stabilization: # and not is_weekend_risk
            #     self._check_scale_in_conditions(symbol, current_price, position)
            # else:
            #     if not is_favorable_add_time:
            #         logger.debug(f"[{symbol}] 建仓期加仓被时间窗口限制 (当前: {current_status.name})")
            #     if is_weekend_risk:
            #         logger.debug(f"[{symbol}] 建仓期加仓被周末风险协议限制")
                    
        elif position.overall_phase == PositionOverallPhase.RUNNING:
            # 盈利期金字塔加仓：同样双重控制
            if is_favorable_add_time: #and not is_weekend_risk
                self._check_pyramid_add_condition(symbol, current_price, position)
            else:
                if not is_favorable_add_time:
                    logger.debug(f"[{symbol}] 盈利期加仓被时间窗口限制 (当前: {current_status.name})")
                if is_weekend_risk:
                    logger.debug(f"[{symbol}] 盈利期加仓被周末风险协议限制")

        ## 尾盘3分钟，并且是周五，选择购买期权过周末
        # 使用含费真实成本计算 ROI
        # real_cost = position.get_avg_cost(self.config)
        # # 防御：防止 real_cost 为 0
        # if real_cost <= 0: real_cost = position.avg_cost
        
        # pnl_pct = (current_price - real_cost) / real_cost
        # market_tz = get_timezone_for_symbol(symbol)
        # current_weekday = datetime.now(market_tz).weekday()
        # is_friday = current_weekday == 4
        
        # if is_friday and (-0.04 <= pnl_pct <= -0.02) and current_status in [TradingWindowStatus.FINAL_MINUTES_GAMBLE]:
            
        #     # --- 双重锁检查前置 ---
        #     # 必须先检查 option_positions，防止重复持仓
        #     for pos in self.option_positions.values():
        #         if pos.underlying_symbol == symbol:
        #             return
            
        #     with self.pending_option_orders_lock:
        #         if symbol in self.pending_option_orders:return
            

        #     # market = get_market_type(symbol)
        #     if self.get_current_option_positions_count(market) >= self.get_max_option_positions(market):
        #         logger.error(f"{market.value}市场达到最大持仓期权数({self.get_max_option_positions(market)})，无法买入 {symbol}")
        #         return
        #     is_us_option = (MarketType.US == market)
        #     if is_us_option:
        #         profile = self.gex_engine._calculate_gex_profile_vectorized(symbol,dte_threshold=14)
        #         veto_granted, _ = self.gex_engine.check_structural_veto(symbol,'C',profile, current_price)
        #         direction = 'BULLISH'
        #         if veto_granted:
        #             direction = 'BULLISH'
        #         else:
        #             direction = 'BEARISH'

        #         # 构造期权专属 candidate 对象
        #         option_candidate = {
        #             'symbol': symbol,
        #             'trigger_price': current_price,
        #             'strategy_name': f"propose_stock_replacement_{'call' if direction == 'BULLISH' else 'put'}_{position.triggering_strategy}",
        #             'strategy_class_name': position.strategy_class_name,
        #             'direction':direction,
        #             'final_score':30,
        #             'reason': f"执行策略: 清仓正股 -> 切换为【深度实值期权代偿】(Stock Replacement)。",
        #             'buy_percentage': 1.0,
        #             'is_bearish_trade': False,
        #             'pnl_pct':pnl_pct
        #         }
                
        #         default_symbols=[symbol,'SPYU.US','TQQQ.US']
        #         for tmp_symbol in default_symbols:
        #             option_candidate['symbol'] = tmp_symbol
        #             proposal = self.option_advisor.propose_hybrid_strategy(option_candidate)
        #             if not proposal or not proposal.get('legs',[]):
        #                 continue
        #             break
                
        #         if not proposal or not proposal.get('legs',[]):
        #             logger.warning(f"[{symbol}] 策略生成器未产出有效预案 (可能因流动性差或没找到合约)，跳过。")
        #             return
                    
        #         logger.critical(f"💎 [{symbol}] 策略生成成功: {proposal.get('strategy_name','未知')}")
        #         # for leg in proposal['legs']:
        #         #     logger.info(f"   -> Leg: {leg['direction']} {leg['quantity']}x {leg['option_symbol']} @ ${leg['limit_price']}")

        #         # 2. 执行阶段：将 Proposal 送入执行线程
        #         self.task_executor.submit(self._handle_option_proposal, proposal)

    # ============================================================================
    # ▼▼▼ ATR 自适应硬止损 helper ▼▼▼
    # 彻底替换"固定 4.5%"。优先使用日线历史 ATR（稳定、不漂移），
    # 失败时 fallback 到固定比例。区分普通股/高波股。
    # 同时支持"时间衰减"：持仓越久，止损越紧（防止趋势反转后吃掉全部利润）。
    # ============================================================================
    def _calc_adaptive_hard_stop_price(self, symbol: str, avg_cost: float,
                                       position: 'Position' = None,
                                       position_type: str = 'long') -> Tuple[float, float, str]:
        """
        计算自适应硬止损价。
        Returns:
            (stop_price, effective_stop_ratio, reason_tag)
            - stop_price: 最终止损价（多头为低点，空头为高点）
            - effective_stop_ratio: 换算回来的等效止损百分比（用于日志）
            - reason_tag: 简短说明，便于在 trigger 日志中输出
        """
        cfg = getattr(self.config, 'adaptive_hard_stop_config', {}) or {}
        if not cfg.get('enabled', True) or avg_cost <= 0:
            # 未启用则回退到原有全局比例
            fallback = cfg.get('fallback_ratio', self.config.stop_loss_ratio)
            px = avg_cost * (1 - fallback) if position_type == 'long' else avg_cost * (1 + fallback)
            return px, fallback, f"FALLBACK_FIXED_{fallback:.1%}"

        # 1. 获取 ATR（优先日线历史，避免盘中漂移）
        atr_value = 0.0
        atr_source = cfg.get('atr_source', 'historical')
        try:
            if atr_source == 'tactical_60m':
                # tactical 模式优先使用盘中动态 ATR（已导入 get_dynamic_atr）
                atr_value = get_dynamic_atr(self.quote_ctx, symbol) or 0.0
            if atr_value <= 0:  # 无论如何，先兜底到历史日 ATR
                atr_value = get_historical_atr(self.quote_ctx, symbol) or 0.0
        except Exception as e:
            logger.warning(f"[{symbol}] 自适应硬止损获取 ATR 失败: {e}")
            atr_value = 0.0        # 2. ATR 不可用 -> fallback 到固定比例
        if atr_value <= 0:
            fallback = cfg.get('fallback_ratio', self.config.stop_loss_ratio)
            px = avg_cost * (1 - fallback) if position_type == 'long' else avg_cost * (1 + fallback)
            return px, fallback, f"NO_ATR_FALLBACK_{fallback:.1%}"

        # 3. 分档选择 ATR 倍数（普通股 vs 高波股）
        atr_pct = atr_value / avg_cost
        hv_threshold = cfg.get('high_vol_atr_pct_threshold', 0.05)
        if atr_pct >= hv_threshold:
            base_mult = cfg.get('atr_multiplier_high_vol', 2.5)
            vol_tag = 'HV'
        else:
            base_mult = cfg.get('atr_multiplier_normal', 2.0)
            vol_tag = 'NV'

        # 4. 时间衰减：持仓越久，止损越紧
        final_mult = base_mult
        if cfg.get('time_decay_enabled', True) and position is not None:
            try:
                hold_hours = max(0.0, position.get_minutes_since_first_buy() / 60.0)
                decay = cfg.get('time_decay_per_hour', 0.05) * hold_hours
                floor = cfg.get('time_decay_floor', 0.6)
                final_mult = max(base_mult * floor, base_mult * (1 - decay))
            except Exception:
                pass

        # 5. 计算止损价并用硬顶/硬底夹紧
        if position_type == 'long':
            atr_stop = avg_cost - (atr_value * final_mult)
            raw_ratio = (avg_cost - atr_stop) / avg_cost
        else:
            atr_stop = avg_cost + (atr_value * final_mult)
            raw_ratio = (atr_stop - avg_cost) / avg_cost

        # 从 bullish_trade_config 读硬顶硬底（如果是做多），做空走自己的配置
        if position_type == 'long':
            bull_cfg = getattr(self.config, 'bullish_trade_config', {}) or {}
            max_r = bull_cfg.get('max_hard_stop_ratio', 0.055)
            min_r = bull_cfg.get('min_hard_stop_ratio', 0.020)
        else:
            max_r = getattr(self.config, 'stop_loss_ratio', 0.025) * 1.5
            min_r = 0.015

        eff_ratio = max(min_r, min(max_r, raw_ratio))
        if position_type == 'long':
            final_stop = avg_cost * (1 - eff_ratio)
        else:
            final_stop = avg_cost * (1 + eff_ratio)

        tag = f"ATR{vol_tag}_{final_mult:.2f}x_{eff_ratio:.2%}"
        return final_stop, eff_ratio, tag

    # ============================================================================
    # ▼▼▼ 做多阶梯硬止损 (Ladder Hard Stop) ▼▼▼
    # ----------------------------------------------------------------------------
    # 目标：替换原 ATR 自适应硬止损（不可预测）。
    # 规则：
    #   - 首次达 ROI ≤ -1.0% → 记录时间戳 → 持续 30s 仍低于阈值 → 卖 25%
    #   - 首次达 ROI ≤ -1.6% → 记录时间戳 → 持续 30s 仍低于阈值 → 再卖 33%（累计 ≈50%）
    #   - 首次达 ROI ≤ -2.0% → 立即清仓（不等待，兜底灾难防线）
    # 关键设计：
    #   1. 每档用独立 tag，借助 position.sell_records + has_executed_action_today 实现跨 tick 幂等，
    #      不会因为一档反复触发被重复扣仓。
    #   2. 用 self._ladder_stop_tracker 做"持续 N 秒确认"（防毛刺）；
    #      若 ROI 期间反弹回阈值之上，对应档的计时器自动清零。
    #   3. 不改 Position 类，不依赖 ATR，不调 LLM；可预测、可解释、易回测。
    # ============================================================================
    def _check_ladder_hard_stop(self, symbol: str, position: 'Position',
                                current_price: float, roi: float) -> bool:
        """
        阶梯硬止损检查。
        Args:
            symbol: 标的代码
            position: 当前持仓对象
            current_price: 最新现价（用于日志）
            roi: 含费 ROI（负数为亏损）
        Returns:
            bool: True 表示已触发卖出动作，调用方应立即返回。
        """
        # --- 0. 总开关与配置加载 ---
        if not getattr(self.config, 'enable_ladder_hard_stop', True):
            return False
        cfg = getattr(self.config, 'ladder_hard_stop_config', {}) or {}
        if not cfg.get('enabled', True):
            return False
        stages = cfg.get('stages', []) or []
        if not stages:
            return False
        confirm_sec = float(cfg.get('confirmation_seconds', 30))

        # --- 1. 懒加载 tracker（避免改 __init__） ---
        if not hasattr(self, '_ladder_stop_tracker'):
            self._ladder_stop_tracker: Dict[str, Dict[str, datetime]] = {}
        sym_tracker = self._ladder_stop_tracker.setdefault(symbol, {})

        now_utc = datetime.now(timezone.utc)

        # --- 2. 按阈值从严到宽排序（-2.0% 先于 -1.6% 先于 -1.0%） ---
        #    这样即使 ROI 一跤摔到 -2.1%，最严的 S3 兜底档会第一时间触发清仓，
        #    不会被 S1/S2 的"持续 30s"等待拖死。
        sorted_stages = sorted(stages, key=lambda s: float(s.get('roi_threshold', 0)))

        for stage in sorted_stages:
            thr = float(stage.get('roi_threshold', 0))
            tag = stage.get('tag', f"LADDER_{thr:.3f}")
            sell_ratio = float(stage.get('sell_ratio', 1.0))
            require_confirm = bool(stage.get('require_confirm', True))

            # 2.1 未触及本档阈值 → 反弹清理 tracker，跳过
            if roi > thr:
                if tag in sym_tracker:
                    logger.debug(f"[{symbol}] ✨ 阶梯止损[{tag}] 反弹清零 (ROI={roi:.2%} > {thr:.2%})")
                    sym_tracker.pop(tag, None)
                continue

            # 2.2 今天已执行过本档（跨 tick 幂等）→ 跳过
            try:
                already_done = position.has_executed_action_today(tag)
            except Exception:
                already_done = False
            if already_done:
                continue

            # 2.3 需要持续确认（S1/S2）→ 走 tracker
            if require_confirm:
                first_ts = sym_tracker.get(tag)
                if first_ts is None:
                    sym_tracker[tag] = now_utc
                    logger.info(
                        f"[{symbol}] ⏱️ 阶梯止损[{tag}] 首次触达 "
                        f"ROI={roi:.2%} ≤ {thr:.2%}，进入 {confirm_sec:.0f}s 确认窗口"
                    )
                    return False  # 本轮不动手，等下一轮复核
                elapsed = (now_utc - first_ts).total_seconds()
                if elapsed < confirm_sec:
                    logger.debug(
                        f"[{symbol}] ⏳ 阶梯止损[{tag}] 确认中 "
                        f"({elapsed:.0f}/{confirm_sec:.0f}s, ROI={roi:.2%})"
                    )
                    return False
                # 确认完成，落刀
                confirm_note = f"持续≥{confirm_sec:.0f}s"
            else:
                # 2.4 S3 兜底档：不等待，立即清仓
                confirm_note = "兜底立即"

            # 2.5 组装 reason（tag 必须出现在 reason 中，后续 has_executed_action_today 靠它）
            reason = (
                f"[做多·主干·阶梯硬止损][{tag}] 现价 {current_price:.3f} | "
                f"ROI {roi:.2%} ≤ {thr:.2%} | {confirm_note} | "
                f"卖出 {sell_ratio:.0%}"
            )
            logger.warning(f"[{symbol}] 🛑 {reason}")

            # 2.6 测试模式只打日志，不真卖
            if getattr(self.config, 'test_mode', False):
                # 测试环境也清 tracker，避免状态残留影响下一轮
                sym_tracker.pop(tag, None)
                return True

            # 2.7 执行卖出（S3 全仓走 _execute_full_sell；S1/S2 走 process_sell_signal）
            try:
                if sell_ratio >= 0.999:
                    self._execute_full_sell(symbol, reason)
                    if symbol not in self.intraday_blacklist:
                        self.intraday_blacklist.add(symbol)
                        self.intraday_blacklist.add(resolve_underlying_symbol(symbol))
                        self._save_blacklist()
                        logger.critical(f"🚫 [{symbol}] 因盘外严重负面离场 ({reason})，已加入黑名单！")
                else:
                    self.process_sell_signal(symbol, percentage=sell_ratio, reason=reason)
            except Exception as e:
                logger.error(f"[{symbol}] 阶梯止损[{tag}] 执行卖出异常: {e}", exc_info=True)
                # 异常时不清 tracker，下一轮重试
                return False

            # 2.8 清理本档 tracker（已执行，幂等由 has_executed_action_today 接管）
            sym_tracker.pop(tag, None)
            return True

        return False

    # ============================================================================
    # ▼▼▼ 做多仓位对称退出引擎 ▼▼▼
    # 镜像 _manage_bearish_position_exit：
    #   阶段 1: 快速回血（ROI ≥ 1.5% 减仓 30%）
    #   阶段 1 后: 立刻把止损线上移到 成本 × 1.002（保本 + 小缓冲）
    #   阶段 2: 追踪止损（Chandelier: High − 1.5 × ATR，兜底百分比 1.0%）
    #   硬止损: 【S1 简化版】阶梯百分比 -1.0% / -1.6% / -2.0%（见 _check_ladder_hard_stop）
    #           ATR 自适应保留为可回滚开关（use_adaptive_hard_stop=True 时启用）
    # 返回 True 表示已触发卖出，外层应终止后续检查。
    # ============================================================================
    def _manage_bullish_position_exit(self, position: 'Position', symbol: str) -> bool:
        """
        做多对称退出引擎（方案 B 主干）。
        当且仅当 config.enable_bullish_position_exit 为 True 时被外层路由调用。
        """
        if not getattr(self.config, 'enable_bullish_position_exit', False):
            return False

        bull_cfg = getattr(self.config, 'bullish_trade_config', {}) or {}
        current_price = self.get_current_price(symbol)
        if current_price is None or current_price <= 0:
            return False

        # 使用含费真实成本计算 ROI
        real_cost = position.get_avg_cost(self.config) if hasattr(position, 'get_avg_cost') else position.avg_cost
        if real_cost <= 0: real_cost = position.avg_cost
        if real_cost <= 0: return False

        roi = (current_price - real_cost) / real_cost

        # ==================================================================
        # 策略 1: 硬止损（默认下沉给外层 _check_and_execute_composite_stop 接管）
        # ------------------------------------------------------------------
        # 【方案 A 重构】此分支与外层主干路径完全重叠，曾出现 ladder 被跑两遍的隐患
        # （仅靠 has_executed_action_today 幂等防住，纯属侥幸）。
        # 现默认关闭：bullish_trade_config['enable_internal_hard_stop'] = False
        # 需要做"做多专属硬止损实验"时，把开关打开即可启用本分支。
        # ==================================================================
        enable_internal_hard_stop = bool(bull_cfg.get('enable_internal_hard_stop', False))
        if enable_internal_hard_stop and not position.is_trailing_stop_active:
            use_adaptive = bull_cfg.get('use_adaptive_hard_stop', False)
            if not use_adaptive:
                # --- [S1] 阶梯硬止损（主方案）---
                if self._check_ladder_hard_stop(symbol, position, current_price, roi):
                    return True
            else:
                # --- [回滚方案] ATR 自适应硬止损 ---
                stop_price, eff_ratio, tag = self._calc_adaptive_hard_stop_price(
                    symbol, real_cost, position=position, position_type='long'
                )
                if current_price <= stop_price:
                    reason = (f"[做多·主干] 硬止损触发[{tag}]: 现价 {current_price:.3f} "
                              f"≤ 止损线 {stop_price:.3f} (等效 -{eff_ratio:.2%})")
                    logger.warning(f"[{symbol}] 🛑 {reason}")
                    if not self.config.test_mode:
                        self._execute_full_sell(symbol, reason)
                    return True

        # ==================================================================
        # 策略 2: 两阶段动态止盈 (Dual-Stage Profit Taking)
        # ==================================================================
        # --- [阶段 1]: 快速回血 (Scalping) ---
        if not position.r_profit_taken:
            scalp_threshold = bull_cfg.get('scalp_target_pct', 0.015)
            scalp_ratio = bull_cfg.get('scalp_sell_ratio', 0.30)
            dedup_tag = f"BULL_SCALP_ROI_{scalp_threshold*100:.1f}"

            if roi >= scalp_threshold:
                if position.has_executed_action_today(dedup_tag):
                    return False

                reason = (f"[做多·主干] 快速回血[{dedup_tag}]: ROI {roi:.2%} ≥ "
                          f"{scalp_threshold:.1%}，减仓 {scalp_ratio:.0%}")
                logger.warning(f"[{symbol}] 💰 {reason}")

                if not self.config.test_mode:
                    if self.process_sell_signal(symbol, percentage=scalp_ratio, reason=reason):
                        # 状态机更新：保本线上移 + 激活追踪止损
                        with self.position_lock:
                            if symbol in self.positions:
                                pos = self.positions[symbol]
                                pos.r_profit_taken = True
                                buffer = bull_cfg.get('breakeven_buffer_ratio', 0.002)
                                pos.trailing_stop_price = pos.avg_cost * (1 + buffer)
                                pos.is_trailing_stop_active = True
                                pos.highest_price_since_partial_sell = current_price
                        self._save_positions()
                return True

        # --- [阶段 2]: 追踪止损 (Trailing) ---
        elif position.r_profit_taken and position.is_trailing_stop_active:
            # A. 更新高点
            if not position.highest_price_since_partial_sell:
                position.highest_price_since_partial_sell = current_price
            if current_price > position.highest_price_since_partial_sell:
                position.highest_price_since_partial_sell = current_price

                # B. 计算新的追踪止损线：取 max(Chandelier ATR, 百分比兜底)
                atr_mult = bull_cfg.get('trailing_atr_multiplier', 1.5)
                pct_callback = bull_cfg.get('trailing_callback_pct', 0.010)
                try:
                    atr_value = get_historical_atr(self.quote_ctx, symbol) or 0.0
                except Exception:
                    atr_value = 0.0

                candidate_atr = (current_price - atr_value * atr_mult) if atr_value > 0 else 0.0
                candidate_pct = current_price * (1 - pct_callback)
                # 取更高者作为新止损（更保守、锁更多利润）
                new_stop = max(candidate_atr, candidate_pct)

                # 止损只能上移（棘轮）
                if new_stop > (position.trailing_stop_price or 0):
                    position.trailing_stop_price = new_stop
                    with self.position_lock:
                        if symbol in self.positions:
                            self.positions[symbol].trailing_stop_price = new_stop
                    self._save_positions()

            # C. 检查触发
            if position.trailing_stop_price and current_price <= position.trailing_stop_price:
                reason = (f"[做多·主干] 追踪止盈: 现价 {current_price:.3f} 跌破动态止损线 "
                          f"{position.trailing_stop_price:.3f} (高点 "
                          f"{position.highest_price_since_partial_sell:.3f})")
                logger.warning(f"[{symbol}] 📉 {reason}")
                if not self.config.test_mode:
                    self._execute_full_sell(symbol, reason)
                return True

        return False

    def _manage_bearish_position_exit(self, position: Position, symbol: str) -> bool:
        """
        【双轨制核心】做空工具的专属持仓与退出管理。
        
        升级说明：
        引入"快速回血 + 动态追踪"两阶段止盈机制。
        1. 阶段一(Scalping): 也就是"回血"。微利即落袋为安(卖一半)，降低持仓心理压力。
        2. 阶段二(Trailing): 也就是"奔跑"。剩余仓位设移动止损，博取单边暴跌带来的超额收益。
        
        Returns:
            bool: True 表示已触发卖出指令，终止后续检查。
        """

        bearish_cfg = self.config.bearish_trade_config
        current_price = self.get_current_price(symbol)
        if current_price is None: return False

        # --- 退出策略1: 市场状态反转 (最高优先级) ---
        # market = get_market_type(symbol)
        # is_strong_bull = self.get_strong_bull(position.market)
        # if is_strong_bull:
        #     holding_duration_minutes = position.get_minutes_since_first_buy()
        #     if holding_duration_minutes < 10 and current_price <= position.avg_cost:
        #         return False
        #     reason = f"市场风险解除，做空对冲任务完成"
        #     logger.critical(f"★★★ [{symbol}] 触发做空工具强制平仓: {reason} ★★★")
        #     self._execute_full_sell(symbol, reason)
        #     return True

        # ==================================================================
        # 策略 1: 时间止损 (Time Stop)
        # ==================================================================
        # try:
        #     days_held = position.get_minutes_since_first_buy()
        #     max_hold_days = bearish_cfg.get('max_hold_days', 5)
        #     if days_held >= max_hold_days:
        #         reason = f"持仓达到 {max_hold_days} 天上限，强制平仓"
        #         logger.critical(f"★★★ [{symbol}] 触发做空工具强制平仓: {reason} ★★★")
        #         self._execute_full_sell(symbol, reason)
        #         return True
        # except (TypeError, IndexError, KeyError, ValueError) as e:
        #     logger.warning(f"[{symbol}] 计算做空工具持仓天数时出错: {e}")

        # ==================================================================
        # 策略 1.5: 轻量版动能止损 (Bearish Kinetic Stop)
        # ==================================================================
        # 做空工具(SQQQ等)价格持续创新低 = 大盘涨你亏钱，动能恶化需提前预警
        # 复用 position 上的计数器字段，但跳过 SAAIS/tech_exit 等做多专属逻辑
        if self._check_kinetic_exit(symbol, current_price, position, direction='short'):
            return True

        # ==================================================================
        # 策略 2: 硬止损 (Hard Stop Loss)
        # ==================================================================
        # 如果已经激活了追踪止损，则由追踪止损接管，这里只作为未激活时的“灾难防线”
        if not position.is_trailing_stop_active:
            stop_loss_ratio = bearish_cfg.get('stop_loss_ratio', self.config.stop_loss_ratio) #0.025
            stop_loss_price = position.avg_cost * (1 - stop_loss_ratio)
            
            if current_price <= stop_loss_price:
                reason = f"做空工具触发 -{stop_loss_ratio:.1%} 硬止损 (现价 {current_price:.3f} <= 线 {stop_loss_price:.3f})"
                logger.warning(f"[{symbol}] 🛑 触发做空工具止损: {reason}")
                self._execute_full_sell(symbol, reason)
                return True

        # ==================================================================
        # 策略 3: 两阶段动态止盈 (Dual-Stage Profit Taking)
        # ==================================================================
        # --- [阶段 1]: 快速回血 (Scalping) ---
        # 逻辑：如果不贪，先赚一点就跑一半。
        # 触发条件：尚未部分止盈 且 收益率 >= 1.5% (参照夜盘逻辑)
        if not position.r_profit_taken:
            # 使用含费真实成本计算 ROI
            real_cost = position.get_avg_cost(self.config)
            # 防御：防止 real_cost 为 0
            if real_cost <= 0: real_cost = position.avg_cost
            
            roi = (current_price - real_cost) / real_cost

            # 设定第一目标阈值：默认 1.5% (0.015)。
            # 做空工具波动大，1.5% 是一个很容易触达的安全垫。
            scalp_threshold = bearish_cfg.get('profit_target_pct', 0.016) # 0.015

            is_super_weak = self.get_super_weak(position.market)
            if is_super_weak:
                scalp_threshold *=2
            
            # 将阈值写入特征码 (例如: BEAR_SCALP_ROI_1.5)，确保该比例的止盈每天只做一次
            dedup_tag = f"BEAR_SCALP_ROI_{scalp_threshold*100:.1f}"
            
            if roi >= scalp_threshold:
                if position.has_executed_action_today(dedup_tag):
                    # logger.warning(f"[{symbol}] 做空回血拦截: {dedup_tag} 今日已执行，拒绝重复操作。")
                    return False
                reason = f"[做空策略] 快速回血[{dedup_tag}]: 收益达标 {roi:.2%} (>= {scalp_threshold:.1%}), 减仓50%"
                logger.warning(f"[{symbol}] 💰 触发做空第一目标止盈! {reason}")
                
                if not self.config.test_mode:
                    # 1. 执行减仓 (40%)
                    # 使用 process_sell_signal 自动处理 LO/MO 路由和锁逻辑
                    if self.process_sell_signal(symbol, percentage=bearish_cfg.get('scalp_sell_ratio', 0.40), reason=reason):
                        
                        # 2. 立即更新状态机 (State Update)
                        with self.position_lock:
                            # 重新获取对象以防并发修改
                            if symbol in self.positions:
                                pos = self.positions[symbol]
                                pos.r_profit_taken = True
                                
                                # 关键：为剩余仓位设置“微利保本”止损
                                # 成本价上浮 0.2% 作为保本线，防止倒亏
                                pos.trailing_stop_price = pos.avg_cost * 1.002
                                pos.is_trailing_stop_active = True
                                
                                # 初始化最高价记录，准备进入阶段二
                                pos.highest_price_since_partial_sell = current_price
                                
                        self._save_positions()
                return True # 触发了动作，返回 True

        # --- [阶段 2]: 动态追踪 (Trailing) ---
        # 逻辑：剩下的仓位博取大趋势。价格创新高就抬高止损，回撤就离场。
        # 触发条件：已经部分止盈 且 追踪止损已激活
        elif position.r_profit_taken and position.is_trailing_stop_active:
            
            # A. 更新最高水位线 (High Water Mark)
            if not position.highest_price_since_partial_sell:
                position.highest_price_since_partial_sell = current_price
            
            if current_price > position.highest_price_since_partial_sell:
                position.highest_price_since_partial_sell = current_price
                
                # B. 动态计算止损位 (Trailing Stop)
                # 做空工具(如3倍杠杆)波动极大，回撤阈值不能太窄，否则容易被洗。
                # 这里设定为 0.8% (0.008)，比夜盘的 0.5% 稍宽一点，增加容错。
                trailing_callback = 0.008 
                new_stop = current_price * (1 - trailing_callback)
                
                # 止损位只能上移 (Ratchet Up)
                if new_stop > position.trailing_stop_price:
                    position.trailing_stop_price = new_stop
                    with self.position_lock:
                        if symbol in self.positions:
                            self.positions[symbol].trailing_stop_price = new_stop
                    self._save_positions()
                    # logger.debug(f"[{symbol}] 做空追踪止损上移至: {new_stop:.3f}")
            
            # C. 检查触发 (Execution)
            if current_price <= position.trailing_stop_price:
                reason = (f"[做空策略] 追踪止盈: 现价 {current_price:.3f} 跌破动态止损线 {position.trailing_stop_price:.3f} "
                          f"(最高 {position.highest_price_since_partial_sell:.3f}, 回撤>0.8%)")
                
                logger.warning(f"[{symbol}] 📉 触发做空追踪离场! {reason}")
                
                if not self.config.test_mode:
                    self._execute_full_sell(symbol, reason)
                return True
            
        return False # 未触发任何退出条件
    
    def _check_bearish_kinetic_stop(self, symbol: str, current_price: float, position: 'Position') -> bool:
        """
        【做空工具轻量版动能止损 (Bearish Kinetic Stop)】
        
        设计目标：
        复用 _check_continuous_low_stop_loss_pro 的核心"创新低计数"逻辑，
        但去掉所有做多专属组件 (SAAIS/GEX/tech_exit/probation) 以降低复杂度。
        
        语义对齐：
        SQQQ 价格下跌 = 大盘上涨 = 做空亏钱 → 与做多持仓跌相同，需要止损。
        
        触发条件：
        每 check_interval 分钟检查一次，若 current_price < post_purchase_low * (1 - break_pct)
        则计数器 +1。连续达标 n_periods_threshold 次 → 卖出。
        
        衰减机制：
        如果价格反弹超过 post_purchase_low 的 0.8%，计数器回退 1。

        Returns:
            bool: True 表示已触发卖出
        """
        from datetime import datetime, timezone, timedelta
        
        # ─── 配置参数  ───
        ksl_cfg = self.config.kinetic_stop_loss_config
        if not ksl_cfg.get('enabled', True):
            return False
        
        check_interval_min = ksl_cfg.get('fallback_check_interval_minutes', 2)
        n_periods_threshold = ksl_cfg.get('fallback_n_periods_threshold', 3)
        effective_break_pct = ksl_cfg.get('fallback_effective_break_pct', 0.003)
        enable_decay = ksl_cfg.get('enable_counter_decay', True)
        
        now_utc = datetime.now(timezone.utc)
        
        # ─── 免疫期：建仓后 5 分钟内不检测，防止买入即触发 ───
        try:
            first_buy_time = position.get_first_buy_time()
            if first_buy_time:
                if first_buy_time.tzinfo is None:
                    first_buy_time = first_buy_time.replace(tzinfo=timezone.utc)
                if (now_utc - first_buy_time).total_seconds() < 300:
                    return False
        except Exception:
            pass
        
        # ─── 初始化 post_purchase_low（做空场景：追踪价格下限） ───
        if position.post_purchase_low <= 0 or position.post_purchase_low > position.avg_cost:
            position.post_purchase_low = position.avg_cost
        
        # ─── 检查间隔控制 ───
        if position.last_consecutive_low_check_ts is None:
            position.last_consecutive_low_check_ts = now_utc
        elif isinstance(position.last_consecutive_low_check_ts, str):
            try:
                position.last_consecutive_low_check_ts = datetime.fromisoformat(position.last_consecutive_low_check_ts)
            except ValueError:
                position.last_consecutive_low_check_ts = now_utc

        last_check = position.last_consecutive_low_check_ts
        if last_check:
            elapsed = (now_utc - last_check).total_seconds() / 60.0
            if elapsed < check_interval_min:
                return False
        
        # ─── 更新检查时间戳 ───
        position.last_consecutive_low_check_ts = now_utc
        
        # ─── 核心判定：是否有效击穿前低 ───
        break_threshold = position.post_purchase_low * (1 - effective_break_pct)
        
        if current_price < break_threshold:
            # 创新低 → 计数器 +1，更新前低
            position.consecutive_new_low_periods += 1
            position.post_purchase_low = current_price
            count = position.consecutive_new_low_periods
            
            logger.info(
                f"📉 [{symbol}] 做空动能恶化 ({count}/{n_periods_threshold}) | "
                f"现价 {current_price:.3f} < 前低线 {break_threshold:.3f}"
            )
            
            # ─── 达标 → 直接卖出 (无 SAAIS/tech_exit/probation) ───
            if count >= n_periods_threshold:
                roi_pct = (current_price - position.avg_cost) / position.avg_cost if position.avg_cost > 0 else 0
                reason = (
                    f"做空动能止损: 连续{count}次创新低 (≥{n_periods_threshold}) | "
                    f"ROI {roi_pct:.2%} | 买入价{position.avg_cost:.3f} → 现价{current_price:.3f}"
                )
                logger.critical(f"⚡ [{symbol}] {reason}")
                if not self.config.test_mode:
                    self._execute_full_sell(symbol, reason)
                # 重置计数器防止残影
                position.consecutive_new_low_periods = 0
                self._save_positions()
                return True
            
            self._save_positions()
        
        elif enable_decay and current_price > position.post_purchase_low * 1.008:
            # ─── 衰减：反弹超过 0.8% → 计数器回退 1 ───
            if position.consecutive_new_low_periods > 0:
                old = position.consecutive_new_low_periods
                position.consecutive_new_low_periods -= 1
                logger.info(
                    f"🔄 [{symbol}] 做空动能恢复: 价格反弹 > 前低+0.8% | "
                    f"计数 {old} → {position.consecutive_new_low_periods}"
                )
                self._save_positions()
        
        return False

    # ============================================================================
    # ▼▼▼ 统一动能止损调度器 (Unified Kinetic Exit Router) ▼▼▼
    # ----------------------------------------------------------------------------
    # 设计目的：消除做多/做空两套调用方式重复维护的问题，对外提供统一入口；
    # 内部按 direction 分流到对应的具体实现：
    #   - 'long'  → _check_continuous_low_stop_loss_pro（含 SAAIS/技术面/probation）
    #   - 'short' → _check_bearish_kinetic_stop（轻量版，无 SAAIS，避开做空标的稀薄期权数据）
    #
    # 对外建议：所有调用方都改用本调度器；将来若要再细分（如 'long_options'），
    # 在此处增加分支即可，业务调用点无需变更。
    # ============================================================================
    def _check_kinetic_exit(self, symbol: str, current_price: float,
                            position: 'Position', direction: str = 'long') -> bool:
        """
        统一动能止损调度入口。
        Args:
            direction: 'long' 或 'short'，决定走哪套实现
        Returns:
            bool: True 表示已触发卖出动作
        """
        try:
            if direction == 'short':
                return self._check_bearish_kinetic_stop(symbol, current_price, position)
            # 默认走做多动能止损
            return self._check_continuous_low_stop_loss_pro(symbol, current_price, position)
        except Exception as e:
            logger.error(f"[{symbol}] 动能止损调度异常 (direction={direction}): {e}", exc_info=True)
            return False

    def _manage_extended_hours_position(self, position: Position, symbol: str):
        """
        【全时段通用持仓管理 (Extended Hours Manager)】
        
        适用场景：夜盘(-3)、盘前(-1)、盘后(-2) 的持仓风控。
        核心逻辑：
        1. 刚性止损 (Hard Stop): 必须守住底线。
        2. 快速回血 (Scalping): 这种时段不谈格局，赚了就跑一半。
        3. 动态追踪 (Trailing): 剩下的仓位博取波动。
        4. 时间死线 (Deadline): 开盘前/收盘前强制清场。
        """
        current_price = self.get_realtime_price(symbol)
        if current_price <= 0: return

        # 获取策略名称，用于日志
        strategy_name = position.triggering_strategy or "ExtendedHours"
        
        # --- 0. 刚性止损 (Hard Stop) ---
        # initial_stop_loss_price 是底线
        if position.initial_stop_loss_price > 0 and current_price <= position.initial_stop_loss_price:
            reason = f"[{strategy_name}] 刚性止损: 现价 {current_price:.2f} <= 保护价 {position.initial_stop_loss_price:.2f}"
            if not self.config.test_mode:
                self._execute_extended_hours_sell(symbol, reason, sell_ratio=1.0)
                logger.warning(f"[{symbol}] 触发夜盘刚性止损! {reason}")
                # current_status = get_trading_window_status(symbol)
                # if current_status in self.config.extended_hours_sell_windows:
                #     self._execute_extended_hours_sell(symbol, reason, sell_ratio=1.0)
                #     logger.warning(f"[{symbol}] 触发夜盘刚性止损! {reason}")
            return

        real_cost = position.get_avg_cost(self.config)
        if real_cost <= 0: real_cost = position.avg_cost
        
        # 计算收益率 (现价 - 含费成本) / 含费成本
        roi = (current_price - real_cost) / real_cost

        # --- 1. 第一目标位：快速回血 (Target 1 - Scalping) ---
        # 如果尚未部分止盈，且收益 > 1.5% -> 卖一半
        if not position.r_profit_taken:

            # 明确标识这是"盘外"的"1.2%回血"操作
            dedup_tag_a = "EXT_SCALP_ROI_1.2"
            dedup_tag_b = "EXT_SCALP_TECH_1.0"

            # 场景 A: 收益达标 1.2%
            if roi >= 0.012:
                # 如果今天已经执行过 EXT_SCALP_ROI_1.2，直接跳过，防止价格在 1.2% 上下反复横跳
                if position.has_executed_action_today(dedup_tag_a):
                    # logger.debug(f"[{symbol}] 策略拦截: {dedup_tag_a} 今日已触发，跳过重复卖出。")
                    return
                
                # 防抖检查：防止因 r_profit_taken 状态延迟更新导致的重复卖出
                if self._is_action_recently_taken(position, "快速回血", lookback_minutes=15):
                    return
                
                reason = f"[{strategy_name}] 快速回血[{dedup_tag_a}]: 收益达标 {roi:.2%}, 减仓50%"
                logger.warning(f"[{symbol}] 💰 触发盘外止盈第一目标! {reason}")
                if not self.config.test_mode:
                    self._execute_extended_hours_sell(symbol, reason, sell_ratio=0.5)
                    with self.position_lock:
                        if symbol in self.positions:
                            self.positions[symbol].r_profit_taken = True
                            # 剩下仓位设置微利保本
                            self.positions[symbol].trailing_stop_price = position.avg_cost * 1.002
                            self.positions[symbol].is_trailing_stop_active = True
                            # self.positions[symbol].add_sell_record(current_price, int(0.3*position.total_quantity), reason)
                    self._save_positions()
                return
            
            # 场景 B: 收益达标 1.0% + 技术指标
            if roi >= 0.010 and len(position.sell_records)==0: # 确保没卖过
                # 核心特征码拦截
                if position.has_executed_action_today(dedup_tag_b):
                    return
                
                k_mins_check = self.config.tactical_k_mins_map.get('SCALP_EXIT', 3)
                rebound_pct_threshold = 0.002 #0.2%
                is_high_confirmed = check_extended_hours_tactical_exit_signal(
                                self.hs_data_provider, symbol, k_mins_check,rebound_pct_threshold)
                if is_high_confirmed:
                    # 防抖检查
                    if self._is_action_recently_taken(position, "快速回血", lookback_minutes=15):
                        return
                    
                    if not self.config.test_mode:
                        reason = f"[{strategy_name}] 快速回血[{dedup_tag_b}](技术面): {k_mins_check}分高点回撤, 收益 {roi:.2%}, 减仓30%"
                        logger.warning(f"[{symbol}] 💰 触发盘外技术止盈! {reason}")
                        self._execute_extended_hours_sell(symbol, reason, sell_ratio=0.4)
                    return

        # --- 2. 第二目标位：动态追踪 (Target 2 - Trailing) ---
        # 如果已经减过仓，剩下的走严格追踪止损
        if position.r_profit_taken and position.is_trailing_stop_active:
            # 更新最高价记录
            if not position.highest_price_since_partial_sell:
                position.highest_price_since_partial_sell = current_price
            
            if current_price > position.highest_price_since_partial_sell:
                position.highest_price_since_partial_sell = current_price
                # 更新止损位：从最高点回撤 0.5% 离场
                new_stop = current_price * (1.0 - self.config.night_rigid_stop_ratio)
                # 止损位只能上移，不能下移
                if new_stop > position.trailing_stop_price:
                    position.trailing_stop_price = new_stop
                    with self.position_lock:
                        self.positions[symbol].trailing_stop_price = new_stop
                    self._save_positions()
            
            # 检查是否击穿追踪止损
            if current_price <= position.trailing_stop_price:
                reason = f"[{strategy_name}] 追踪止盈: 止损位 {position.trailing_stop_price:.2f} 触发"
                if not self.config.test_mode:
                    # current_status = get_trading_window_status(symbol)
                    # if current_status in self.config.extended_hours_sell_windows:
                    self._execute_extended_hours_sell(symbol, reason, sell_ratio=1.0)
                    ## 清仓后安全更新状态（防止 KeyError）
                    if symbol in self.pre_market_states:
                        self.pre_market_states[symbol]['status'] = 'WATCHING'
                        self.pre_market_states[symbol]['last_update_ts'] = time.time()
                        self._save_pre_market_states()

                logger.warning(f"[{symbol}] 📉 触发盘外追踪离场! {reason}")
                return

        # --- 3. 灰姑娘死线 (The 09:25 Rule) ---
        # 在美股开盘前5分钟 (09:25 ET)，强制清场，不留仓过夜赌开盘
        # 这是一个通用规则：任何盘前持仓都不应该带进 9:30 的开盘波动中
        try:
            from utils.market_time_utils import get_market_open_time
            open_time = get_market_open_time(symbol)
            if open_time:
                # 获取带时区的当前时间
                now = datetime.now(open_time.tzinfo)
                # 死线时间：开盘前5分钟
                deadline = open_time - timedelta(minutes=5)
                # 只有在 (死线 <= 当前 < 开盘) 的区间内才触发
                if deadline <= now < open_time and roi>=0.005:
                    reason = f"[{strategy_name}] 灰姑娘死线: 09:25 ET 强制清场"
                    logger.critical(f"[{symbol}] ⏰ 触发盘前死线清仓! {reason}")
                    if not self.config.test_mode:
                        self._execute_extended_hours_sell(symbol, reason, sell_ratio=1.0)
                        ## 清仓后安全更新状态（防止 KeyError）
                        if symbol in self.pre_market_states:
                            self.pre_market_states[symbol]['status'] = 'WATCHING'
                            self.pre_market_states[symbol]['last_update_ts'] = time.time()
                            self._save_pre_market_states()
                    return
        except Exception as e:
            # 日志降级，防止频繁报错刷屏
            logger.error(f"检查夜盘死线时间出错: {e}")
            pass
            
        # --- 4. 当前建仓股票在夜盘前先走了 ---
        try:
            current_status = get_trading_window_status(symbol)
            if current_status in [TradingWindowStatus.NIGHT_MIDNIGHT_PUMP,TradingWindowStatus.NIGHT_EURO_OPEN_RUSH]: ## 03:00 - 03:30: 欧盘抢跑
                k_mins_check = self.config.tactical_k_mins_map.get('SCALP_EXIT', 3)
                rebound_pct_threshold = 0.002 #0.2%
                is_high_confirmed = check_extended_hours_tactical_exit_signal(self.hs_data_provider, symbol, k_mins_check,rebound_pct_threshold)
                
                if current_price > position.avg_cost and roi>=0.010 and is_opened_today(position, symbol) and is_high_confirmed:
                    reason = f"[{strategy_name}] 夜盘前[欧盘抢跑]盈利部分走人"
                    logger.critical(f"[{symbol}] ⏰ 触发夜盘前[欧盘抢跑]! {reason}")
                    if not self.config.test_mode:
                        self._execute_extended_hours_sell(symbol, reason, sell_ratio=0.8)
                    return
        except Exception as e:
            # 日志降级，防止频繁报错刷屏
            logger.error(f"检查夜盘死线时间出错: {e}")
            pass
    
    def _process_micro_building_continuation(self, symbol: str, position: Position, current_price: float):
        """
        【微观建仓推进器】
        
        重构说明：采用“漏斗式过滤”架构 (Funnel Filtering Architecture)
        1. 全局熔断 (Global Circuit Breakers) -> 2. 策略路由 (Router) -> 3. 信号触发 (Trigger) 
        -> 4. 多维审计 (Audit) -> 5. 执行 (Execution)
        """
        # --- 环境与配置预加载 ---
        if not position.is_opened_today(symbol): return
        micro_cfg = self.config.micro_building_config
        if not micro_cfg.get('enabled', False): return

        cost_price = position.avg_cost
        if cost_price <= 0: return

        # 核心状态数据
        scout_pnl_pct = (current_price - cost_price) / cost_price
        mins_since_first_buy = position.get_minutes_since_first_buy()
        
        # 市场环境快照
        is_rth = is_any_market_open(symbol)
        health_type, _ = self.market_regime_engine.check_intraday_health(position.market)
        is_strong_bull = self.get_strong_bull(position.market)
        is_bearish_asset = self._is_bearish_symbol(symbol)
        
        # 宽松因子计算
        relax_factor = 1.0
        if is_strong_bull: relax_factor = 0.5
        if is_bearish_asset: relax_factor = 1.6

        # ==============================================================================
        # 1. 全局硬风控 (Global Hard Constraints) - 此时不满足直接退出，节省算力
        # ==============================================================================
        
        # [需求1] 硬顶风控：涨幅超过 1.5% (或其他配置值) 绝对不追
        price_cap_pct = micro_cfg.get('price_cap_pct', 0.015)
        if current_price >= cost_price * (1 + price_cap_pct):
            # if int(time.time()) % 300 == 0:
            #     logger.debug(f"🔭 [{symbol}] 触及硬顶 (+{price_cap_pct:.1%})，暂停加仓。当前: {(current_price/cost_price)-1:.2%}")
            
            log_msg=f"🔭 [{symbol}] 触及硬顶 (+{price_cap_pct:.1%})，暂停加仓。当前: {(current_price/cost_price)-1:.2%}"
            logger.warning(log_msg)
            self._abort_micro_building(position, reason=log_msg)
            return

        # 止损风控：跌幅过大直接转入止损流程
        cancel_threshold = micro_cfg.get('cancel_threshold_pct', -0.015)
        if scout_pnl_pct < cancel_threshold:
            logger.warning(f"🛑 [{symbol}] 触发止损熔断 (PNL: {scout_pnl_pct:.2%})，终止建仓。")
            self._abort_micro_building(position, reason=f"浮亏超标({scout_pnl_pct:.2%})")
            return

        # 动能衰竭风控 或 卖出一次后
        sell_records = getattr(position, 'sell_records', [])
        if position.consecutive_new_low_periods >= 3 or len(sell_records) > 0:
            self._abort_micro_building(position, reason=f"动能衰竭(NewLow={position.consecutive_new_low_periods})")
            return

        # 宏观熔断 (非做空标的)
        if (health_type == IntradayHealthType.R) and not is_bearish_asset:
            self._abort_micro_building(position, reason=f"大盘熔断({health_type.name})")
            return

        # ==============================================================================
        # 2. 策略路由 (Strategy Router) - 确定下一步目标
        # ==============================================================================
        steps = micro_cfg.get('steps', [])
        target_step = None
        
        # 状态机映射
        if position.building_stage == 1:
            target_step = steps[0] if steps else {'threshold': 0.001, 'ratio': 0.50, 'next_stage': 2}
        elif position.building_stage == 101:
            if len(steps) > 1:
                target_step = steps[1]
            else:
                logger.warning(f"⚠️ [{symbol}] Stage 101 无后续配置，强制结束。")
                self._abort_micro_building(position, reason="配置缺失")
                return
        
        # 如果没有目标阶段，直接退出
        if not target_step: return

        # ==============================================================================
        # 3. 信号触发器 (Signal Trigger) - 判断是否满足进攻条件
        # ==============================================================================
        threshold = target_step['threshold'] * relax_factor
        
        # 信号A: 利润达标 (进攻)
        # is_profit_hit = scout_pnl_pct > threshold
        
        # 信号B: 超时防御 (企稳) - 你的需求2
        monitor_window = micro_cfg.get('monitor_window_mins', 10)
        # 逻辑：时间够久 + 没大跌(> -0.1%) + 还在Stage1
        is_timeout_mature = (mins_since_first_buy > monitor_window) and \
                            (scout_pnl_pct > -0.001) and \
                            (position.building_stage == 1)

        # 既不达标，也未超时 -> 检查是否彻底超时需放弃
        # if not (is_profit_hit): # or is_timeout_mature
        #     max_scout_time = monitor_window * 3
        #     if mins_since_first_buy > max_scout_time:
        #         logger.info(f"💤 [{symbol}] 侦察期超时({mins_since_first_buy}m)且无建树，放弃增援。")
        #         self._abort_micro_building(position, reason="超时放弃")
        #     return
        
        # ==============================================================================
        # 3.5 高位缺氧检测 (High Altitude Check) - 专治高位接盘
        # 逻辑：如果是盈利加仓(Averaging Up)，严禁在微观箱体的"天花板"买入，必须等回踩。
        # ==============================================================================
        if current_price > cost_price:  # 仅针对“浮盈加仓”场景生效
            return
            try:
                # 获取过去 10分钟 的K线 (足以覆盖 AAPU 这种日内波段)
                # 为什么是10分钟？太短(1-3分)看不出箱体，太长(30分)会错过主升浪
                lookback_k = 10
                micro_klines = get_klines_data(self.quote_ctx, symbol, lookback_k, Period.Min_1, AdjustType.NoAdjust)
                
                if micro_klines is not None and not micro_klines.empty and len(micro_klines) >= 5:
                    # 包含当前实时价格，构建真实的高低点范围
                    m_high = max(micro_klines['high'].max(), current_price)
                    m_low = min(micro_klines['low'].min(), current_price)
                    m_range = m_high - m_low
                    
                    needle_limit_pct = 0.006  if (is_strong_bull or symbol in self.config.high_vol_symbols) else 0.003 
                    micro_pos_limit = 0.94  if (is_strong_bull or symbol in self.config.high_vol_symbols) else 0.92
                    
                    # 只有当波动幅度有意义时 (>0.3%) 才启动过滤，死鱼盘不需要过滤
                    if m_range > (current_price * needle_limit_pct):
                        # 计算当前价格在箱体中的位置 (Rank: 0~1)
                        # 1.0 = 最高点，0.0 = 最低点
                        price_rank = (current_price - m_low) / m_range
                        
                        # 【核心阈值】：如果价格处于过去10分钟的 92% 高位以上，且没有巨量突破迹象
                        if price_rank > micro_pos_limit:
                            # 进一步检查：是否是日内最高点附近 (双重确认)
                            quote = get_smart_quote(self.quote_ctx,symbol)
                            day_high = quote.get('high_price', 0.0) if quote else m_high
                            dist_to_day_high = abs(day_high - current_price) / current_price
                            
                            # 如果既是局部高位，又是日内最高点(距离<0.2%)，坚决不追
                            if dist_to_day_high < 0.002:
                                if int(time.time()) % 60 == 0:
                                    logger.warning(f"🧗 [{symbol}] 高位缺氧拦截: Rank {price_rank:.2f} (区间{m_low:.2f}-{m_high:.2f})，等待回踩。")
                                return
            except Exception as e:
                logger.error(f"High altitude check error: {e}")

        # ==============================================================================
        # 4. 多维审计
        # ==============================================================================
        
        # --- 4.1 微观结构审计 ---
        is_micro_stable = False
        is_trend_stable = False
        
        # 封装：计算 K 线窗口
        k_mins = 3
        if is_rth:
            # 动态调整K线窗口逻辑
            current_regime = self.market_regime_engine.get_marget_regime(position.market)
            stress_factor = self.config.regime_stress_map.get(current_regime, 1.0) * \
                            self.config.health_stress_map.get(health_type, 1.0)
            k_mins = min(3, int(3 * stress_factor))
        
        rebound_thr = 0.20 if symbol in self.config.high_vol_symbols else self.config.rebound_pct_threshold_map['default']
        
        # 检查微观形态
        if is_rth:
            is_micro_stable = check_tactical_entry_signal(self.quote_ctx, symbol, k_mins, rebound_thr)
            # 趋势检查
            is_trend_stable = self._check_trend_breakout(symbol, current_price)
        else:
            # 夜盘逻辑
            is_micro_stable = check_extended_hours_tactical_entry_signal(self.hs_data_provider, symbol, 5, rebound_thr)

        # --- 4.2 场景定性与定额 ---
        scenario_multiplier = 0.0
        scenario_name = ""

        if is_micro_stable:
            scenario_multiplier = 1.0
            if current_price >= cost_price: # 不能增加成本
                scenario_multiplier = 0.3
            scenario_name = "结构确认"
        elif is_trend_stable:
            scenario_multiplier = 1.0
            if current_price >= cost_price: # 不能增加成本
                scenario_multiplier = 0.4
            scenario_name = "趋势确认"
        elif is_timeout_mature:
            # 超时防御，需要更宽松的二次确认
            if self._check_loose_stability(symbol, is_rth):
                scenario_multiplier = 0.6
                scenario_name = "超时企稳(0.6x)"
        elif scout_pnl_pct <= -0.004 and position.building_stage == 1:
            scenario_multiplier = 0.5
            scenario_name = "微利兜底(0.5x)"
        
        if scenario_multiplier <= 0: return

        # --- 4.3 物理与语义精英审计 (Elite Gate) ---
        # 你的需求3核心：大盘不好时多检查
        
        phys_msg = "Physical OK"
        if is_rth and not is_bearish_asset:
            # A. 物理网关
            is_phys_pass, phys_msg = self._verify_trade_quality_gate(symbol, current_price, mode='CONTINUATION')
            
            # 强牛市豁免物理拦截
            if not is_phys_pass and not is_strong_bull:
                if int(time.time()) % 60 == 0: logger.info(f"🛡️ [{symbol}] 物理网关拦截: {phys_msg}")
                return # 拦截

            # B. LLM 语义网关 (大盘不好时强制检查)
            # 逻辑：如果不是强牛，或者刚才物理没过(靠强牛硬撑)，都建议跑一遍LLM
            should_check_llm = (not is_strong_bull) or (not is_phys_pass)
            
            if should_check_llm:
                check_payload = {
                    'symbol': symbol, 
                    'avg_cost': cost_price, 
                    'pnl': scout_pnl_pct,
                    'reason': f"加仓审计: {scenario_name}"
                }
                is_llm_pass, llm_reason = self.stock_tech_advisor.re_check_buy_smart(check_payload)
                
                if not is_llm_pass:
                    # 仅在超时防御且有结构支撑时，做最后的宽容，否则直接拒
                    if is_timeout_mature:
                        if (is_micro_stable or is_trend_stable) and current_price < cost_price:
                            scenario_multiplier = 0.6 # 再次提高，从0.6-->0.8
                            logger.warning(f"⚠️ 超时买入 [{symbol}] LLM否决但结构完好，降权通过: {llm_reason}")
                    else:
                        logger.warning(f"🤖 [{symbol}] LLM否决加仓: {llm_reason}")
                        return

        # ==============================================================================
        # 5. 执行层 (Execution)
        # ==============================================================================
        quantity = int(position.planned_total_quantity * target_step['ratio'] * scenario_multiplier)
        if quantity <= 0: return

        logger.warning(f"🚀 [{symbol}] 执行加仓 | 场景:{scenario_name} | 审计:{phys_msg} | 数量:{quantity}")
        
        # 执行下单
        reason_tag = "main_force_add"
        success = False
        if is_rth:
            success = self._execute_add_position(symbol, quantity, position, reason_tag)
        else:
            success = self._execute_extended_hours_add_position(symbol, quantity, position, reason_tag)

        # 状态更新 (无论成功与否，都要处理状态，失败则终止以防死循环)
        if success:
            self._update_position_stage(position, target_step['next_stage'])
            # ==============================================================================
            # ▼▼▼ 主力进场联动期权买入 (延迟执行) ▼▼▼
            # 逻辑：正股加仓成功确认后，才触发期权信号。置于最后，防止阻塞正股流程。
            # ==============================================================================
            # if position.building_stage == 101 and (symbol in self.config.vip_symbols or symbol in self.clean_first_tier_stocks) and position.market == MarketType.US:
            #     try:
            #         # 构造期权专属 candidate 对象
            #         option_candidate = {
            #             'symbol': symbol,
            #             'trigger_price': current_price,
            #             # 生成一个独立的策略名，方便日志追踪
            #             'strategy_name': f"Opt_Link_{position.triggering_strategy}",
            #             'strategy_class_name': position.strategy_class_name,
            #             'direction':'BULLISH',
            #             'final_score':30,
            #             'reason': f"主力进场联动: 正股加仓确认",
            #             'buy_percentage': 1.0, # 期权通常按张数预算，这里给满额信号
            #             'is_bearish_trade': False # 默认为做多联动
            #         }
            #         logger.info(f"⚡ [{symbol}] 正股加仓完成，正在触发期权联动信号...")
            #         # 调用现有的期权处理入口
            #         self.process_alpha_trade_signal(option_candidate)
            #     except Exception as e:
            #         logger.error(f"[{symbol}] 期权联动信号生成失败 (不影响正股): {e}")
        else:
            logger.error(f"❌ [{symbol}] 加仓下单失败，终止流程。")
            self._abort_micro_building(position, reason="下单失败")

    def _abort_micro_building(self, position, reason: str):
        """统一的终止逻辑，防止代码重复"""
        with self.position_lock:
            position.building_stage = 2
            position.overall_phase = PositionOverallPhase.RUNNING
            self._save_positions()
        logger.warning(f"⏹️ [{position.symbol}] 建仓终止: {reason}")

    def _update_position_stage(self, position, next_stage):
        """统一的状态更新"""
        with self.position_lock:
            position.building_stage = next_stage
            if next_stage == 2:
                position.overall_phase = PositionOverallPhase.RUNNING
                logger.warning(f"✅ [{position.symbol}] 主力建仓完成 (Stage 2) -> RUNNING")
            else:
                logger.info(f"✨ [{position.symbol}] 进入下一阶段: {next_stage}")
            self._save_positions()

    def _check_trend_breakout(self, symbol, current_price):
        """封装趋势突破检查"""
        df_1m = get_klines_data(self.quote_ctx, symbol, 5, Period.Min_1, AdjustType.NoAdjust)
        if df_1m is not None and len(df_1m) >= 4:
            # 过去3分钟最高价 (不含当前)
            recent_high = df_1m['high'].iloc[-4:-1].max()
            return current_price > recent_high
        return False

    def _check_loose_stability(self, symbol, is_rth):
        """封装宽松的企稳检查"""
        if is_rth:
            return check_tactical_entry_signal(self.quote_ctx, symbol, 3, 0.10)
        return check_extended_hours_tactical_entry_signal(self.hs_data_provider, symbol, 5, 0.10)
    
    def process_sell_signal(self, symbol: str, percentage: float = 0.5, reason: str = "策略主动卖出"):
        # 1. 清洗 reason，防止空字符串导致锁失效
        safe_reason = reason if reason and str(reason).strip() else "Unknown_Strategy"
        lock_key = (symbol, safe_reason)
        if lock_key in self.sell_locks and time.time() - self.sell_locks[lock_key] < self.sell_lock_duration_seconds:
            logger.warning(f"卖出信号被锁定 {symbol}: 原因 '{safe_reason}' 在有效期内已被触发，忽略。")
            return False

        with self.position_lock:
            position = self.positions.get(symbol)
            if not position:
                logger.warning(f"没有持仓，无法卖出 {symbol}")
                return False
            if position.pending_sell_order_id:
                logger.warning(f"{symbol} 已存在待处理的卖出订单 {position.pending_sell_order_id}，忽略新信号。")
                return False

        try:
            final_target_ratio = min(1.0, max(0.0, float(percentage)))
            current_price = self.get_realtime_price(symbol)
            if current_price is None or current_price <= 0:
                logger.error(f"[{symbol}] 无法获取有效卖出价格，取消卖出。")
                return False

            sell_quantity, adjust_msg = self._calculate_smart_sell_quantity(
                symbol, position, final_target_ratio, current_price
            )
            if adjust_msg:
                logger.warning(f"[{symbol}] 卖出数量修正: {adjust_msg} -> {sell_quantity}")
            if sell_quantity <= 0:
                return False

            order_id = self.submit_order(symbol, sell_quantity, OrderSide.Sell)
            if not order_id:
                return False

            self.sell_locks[lock_key] = time.time()
            self._save_sell_locks()

            with self.position_lock:
                if symbol in self.positions:
                    pos = self.positions[symbol]
                    pos.pending_sell_order_id = order_id
                    pos.sell_reason = safe_reason
                    pos.partial_sell_price = current_price
                    self._save_positions()

            logger.warning(
                f"[{symbol}] 保守策略卖出指令已提交 | 比例={final_target_ratio:.0%}, "
                f"数量={sell_quantity}, 价格={current_price:.3f}, 原因={safe_reason}, 订单ID={order_id}"
            )
            return True
        except Exception as e:
            logger.error(f"处理保守策略卖出信号时出错 {symbol}: {e}", exc_info=True)
            return False

        with self.position_lock:
            # ==============================================================================
            # ▼▼▼ 影子标签无损拦截网关 (Shadow Gates) ▼▼▼
            # ==============================================================================
            # 1. 战略锁定盾牌 [Strategic_Hold] (钻石手，死也不卖)

            if symbol in self.shadow_tags.get('strategic_hold', []):
                if random.random() > 0.95:
                    logger.warning(f"🛡️[神圣庇护] {symbol} 拥有 [Strategic_Hold] 免死金牌，拒绝执行任何卖出指令！(意图: {reason})")
                    return False

            position = self.positions.get(symbol)
            if not position:
                logger.warning(f"没有持仓，无法卖出 {symbol}")
                return False
            
            # 核心检查：如果策略在豁免名单里，直接拒绝！
            # 即使天塌下来(舆情熔断)，只要你在名单里，我就不卖。
            if position.triggering_strategy in self.config.strategies_immune_to_exit:
                logger.warning(
                    f"🛡️ [卖出豁免] 策略 '{position.triggering_strategy}' 拥有免死金牌，"
                    f"拒绝执行卖出 {symbol} (意图: {safe_reason})。"
                )
                return False
            
            if not self._is_holding_period_satisfied(position, required_minutes=self.config.min_holding_minutes):
                return
            if position.pending_sell_order_id:
                logger.warning(f"{symbol} 已存在待处理的卖出订单 {position.pending_sell_order_id}，忽略新信号。")
                return False
        

        # 1. 计算理论卖出量
        # 引入 WRP 和 窗口系数
        try:
            wrp_sell_proportion_multiplier = 1.0
            if is_entering_weekend_risk_for_symbol(symbol,wrp_activation_days=[2,3,4]):
                wrp_sell_proportion_multiplier = self.config.wrp_sell_proportion_multiplier
            # 早盘冲高加大卖出比例
            # 基于交易窗口的卖出比例动态调整逻辑
            sell_ratio_multiplier = 1.0
            current_status = get_trading_window_status(symbol)
            # 根据节奏，定义不同的“侵略性系数”
            if current_status == TradingWindowStatus.MORNING_RUSH_SELL:
                # 早盘冲高是市场情绪驱动的绝佳卖点，应果断加大卖出比例，锁定更多利润
                sell_ratio_multiplier = 1.5
                logger.warning(f"[{symbol}] 处于 [早盘冲高卖出] 窗口，卖出比例放大 {sell_ratio_multiplier} 倍。")
            elif current_status == TradingWindowStatus.FINAL_HOUR_PROFIT_TAKE:
                # 尾盘是锁定日内利润的最后机会，同样需要更果断。
                sell_ratio_multiplier = 1.2 # 侵略性系数 120%
                logger.warning(f"[{symbol}] 处于 [尾盘核心止盈] 窗口，卖出比例放大 {sell_ratio_multiplier} 倍。")

            final_target_ratio = min(1.0, percentage * wrp_sell_proportion_multiplier * sell_ratio_multiplier)
            
            current_price = self.get_realtime_price(symbol)
            if current_price <= 0: return
            
            sell_quantity, adjust_msg = self._calculate_smart_sell_quantity(
                symbol, position, final_target_ratio, current_price
            )
            if adjust_msg:
                logger.warning(f"🧹 [{symbol}] 触发碎股清洗: {adjust_msg} -> 修正为全仓卖出 ({sell_quantity})")
            
            # --- 提交订单 ---
            if sell_quantity <= 0:
                return False
            
            order_id = self.submit_order(symbol, sell_quantity, OrderSide.Sell)
            if not order_id: return False

            logger.warning(f"卖出指令已提交 {symbol} | 原因: {safe_reason} | 订单ID: {order_id}")
            # ==============================================================================
            # ▼▼▼ 更新日内卖出记忆 ▼▼▼
            # 只要提交了卖单，就记录当前的触发价格（或最新价）
            # ==============================================================================
            try:
                # 优先使用触发本次卖出的价格（如果有），没有则取现价
                # 这里为了简单，重新获取一次现价，确保数据新鲜
                exec_price = self.get_current_price(symbol)
                if exec_price:
                    self.intraday_trade_history[symbol] = exec_price
                    # logger.debug(f"[{symbol}] 日内记忆已更新: Last Sell = {exec_price}")
            except Exception:
                pass # 更新记忆失败不应阻塞交易

            self.sell_locks[lock_key] = time.time()
            self._save_sell_locks()

            with self.position_lock:
                if symbol in self.positions:
                    pos = self.positions[symbol]
                    pos.pending_sell_order_id = order_id
                    pos.sell_reason = safe_reason
                    current_price = self.get_current_price(symbol)
                    if current_price: pos.partial_sell_price = current_price

                    pos.add_sell_record(current_price if current_price else 0.0, sell_quantity, safe_reason)
                    self._save_positions()
            
            # try:
                # current_price = self.get_current_price(symbol) or 0.0
                # symbol_info = self.get_cached_stock_static_info(symbol)
                # symbol_name = symbol_info.get('name_cn') or symbol_info.get('name_en')
                # 这里的 reason 就是上游传进来的，可能是"止盈"、"逃跑"等
                # sell_logger.info(f"symbol:{symbol},name:{symbol_name},price:{current_price},strategy_reason:{reason},llm_reason:Check_Strategy_Config")
            # except Exception as e:
            #     logger.error(f"[{symbol}] 通用卖出记录日志失败: {e}")
            
            return True
        except Exception as e:
            logger.error(f"处理卖出信号时出错 {symbol}: {e}", exc_info=True)
            return False

    def _check_and_execute_stop_loss_grid(self, symbol: str, current_price: float, position: Position) -> bool:
        """
        【不死鸟·动态网格止损 V3.0 (Phoenix Grid Stop)】
        
        功能：
        1. 动态计算三级防线 (L1/L2/L3)，支持跳空缺口处理。
        2. 触发减仓时，自动激活防御模式并记录回补种子 (Resurrection Seed)。
        3. 融合 WRP (周末风控) 进行动态收紧。

        核心升级：基于 ATR% 自动压缩网格倍数。
        目标：无论股票波动率是 2% 还是 12%，强制将 L1 防线锚定在 -2.8% 左右，
             确保在 -4% 硬止损触发前，一定有机会先执行防御性减仓。
        """
        # ▼▼▼【豁免名单检查】▼▼▼
        if position.triggering_strategy in self.config.strategies_immune_to_exit:
            return False
        
        # --- 1. 数据准备 ---
        base_price = position.get_avg_cost(self.config)
        
        # 获取ATR (日线级别，稳定)
        atr = get_historical_atr(self.quote_ctx, symbol)
        
        # [降级保护] 如果ATR获取失败，使用固定百分比模拟ATR (假设波动率为3%)
        if not atr or atr <= 0:
            atr = base_price * 0.03 # 默认3%波动率
            atr_pct = 0.03
        else:
            atr_pct = atr / base_price

        # --- 2. [核心算法] 动态压缩系数计算 ---
        # 我们的战略目标：L1 必须在 -2.8% 左右触发 (避开补仓-1.8%，早于硬止损-4%)
        TARGET_L1_PCT = 0.028 

        # 计算理论需要的倍数
        # 例如 NVDX (6.8%): ideal = 2.8 / 6.8 = 0.41
        # 例如 AAPL (1.8%): ideal = 2.8 / 1.8 = 1.55
        ideal_l1_multiplier = TARGET_L1_PCT / atr_pct

        # [安全约束] 倍数不能太离谱
        # 上限 1.5 (防止低波股止损太宽，超过 -4% 硬止损)
        # 下限 0.25 (防止超高波股止损太窄，噪音误杀)
        effective_l1_mult = max(0.25, min(1.5, ideal_l1_multiplier))

        # 级联计算 L2 和 L3
        # L2 设为 L1 的 1.5 倍 (约 -4.2%，可能略超硬止损，由硬止损兜底)
        # L3 设为 L1 的 2.0 倍
        effective_l2_mult = effective_l1_mult * 1.5
        effective_l3_mult = effective_l1_mult * 2.0

        # [WRP 周末风控叠加]
        if self._is_entering_weekend_risk_for_symbol(symbol):
            effective_l1_mult *= 0.8
            effective_l2_mult *= 0.8
            effective_l3_mult *= 0.8


        # --- 3. 计算价格线 ---
        level_1_price = base_price - (atr * effective_l1_mult)
        level_2_price = base_price - (atr * effective_l2_mult)
        level_3_price = base_price - (atr * effective_l3_mult) # 这里的L3通常会比硬止损还低


        # 硬止损兜底 (取最高值，即最先触发的价格)
        hard_stop_line = max(level_3_price, position.initial_stop_loss_price if position.initial_stop_loss_price > 0 else 0.0)

        # --- 4. 场景裁决 (Scenario Judgement) ---
        triggered_action = False
        action_qty = 0
        grid_level_hit = 0
        reason = ""

        # [场景 A: 触及死亡线 L3] -> 清仓
        if current_price <= hard_stop_line:
            reason = f"💀 触及死亡防线 (L3/HardStop) | 现价 {current_price:.2f} <= 线 {hard_stop_line:.2f} (ATR x {effective_l3_mult:.2f})"
            logger.critical(f"[{symbol}] 不死鸟协议终止: {reason}")
            if not self.config.test_mode:
                k_mins_check = 5
                rebound_pct_threshold = self.config.rebound_pct_threshold_map['default']
                is_high_confirmed = check_tactical_exit_signal(self.quote_ctx, symbol, k_mins_check, rebound_pct_threshold)
                
                if not is_high_confirmed:
                    # 价格虽然破位，但还没确认是离场时机（可能正在急跌中或者正在反抽中），暂缓
                    # logger.debug(f"[{symbol}] 触及 L2 但技术面未确认离场信号，暂缓操作。")
                    return False
                self._execute_full_sell(symbol, reason)
            return True

        # [场景 B: 触及危险线 L2] -> 减仓 33% (累积)
        # 逻辑：当前没到L2，或者我是直接跳空跌穿L1直达L2
        elif current_price <= level_2_price and position.current_grid_level < 2:
            # 卖出剩余仓位的 33%
            # 注意：如果之前L1没触发(跳空)，这里只卖33%是不够的吗？
            # 策略：我们不追溯L1，直接按L2的定义卖出当前手头的33%。这符合"保留火种"的逻辑。

            k_mins_check = 5
            rebound_pct_threshold = self.config.rebound_pct_threshold_map['default']
            is_high_confirmed = check_tactical_exit_signal(self.quote_ctx, symbol, k_mins_check, rebound_pct_threshold)
            
            if not is_high_confirmed:
                # 价格虽然破位，但还没确认是离场时机（可能正在急跌中或者正在反抽中），暂缓
                # logger.debug(f"[{symbol}] 触及 L2 但技术面未确认离场信号，暂缓操作。")
                return False
            
            # ▼▼▼【跳空补偿计算】▼▼▼
            # 逻辑：如果直接从 L0 跳到 L2，必须卖出足够多的量，以达到 L2 应有的仓位水位 (50%)。
            if position.current_grid_level == 0:
                # 越级打击 (Gap Down): L0 -> L2
                # 目标是累计卖出 50% (25% + 33% of remaining ≈ 50% total)
                # 简单点：直接卖 50%
                action_qty = int(position.total_quantity * 0.50)
                reason = f"📉 严重跳空! 直击 L2 (ATR x {effective_l2_mult:.2f}) | 越级防御: 减仓 50%"
            else:
                # 正常升级: L1 -> L2
                # 卖出剩余的 33%
                action_qty = int(position.total_quantity * self.config.grid_level_2_sell_ratio)
                reason = f"🛡️ 触及防御网格 L2 (ATR x {effective_l2_mult:.2f}) | 升级防御: 减仓 33%"

            # action_qty = int(position.total_quantity * self.config.grid_level_2_sell_ratio)
            # reason = f"🛡️ 触及防御网格 L2 (ATR x {l2_atr:.2f}) | 现价 {current_price:.2f} <= 线 {level_2_price:.2f}"
            grid_level_hit = 2
            triggered_action = True

        # [场景 C: 触及警戒线 L1] -> 减仓 25%
        elif current_price <= level_1_price and position.current_grid_level < 1:
            k_mins_check = 5
            rebound_pct_threshold = self.config.rebound_pct_threshold_map['default']
            is_high_confirmed = check_tactical_exit_signal(self.quote_ctx, symbol, k_mins_check, rebound_pct_threshold)
            
            if not is_high_confirmed:
                return False
            
            action_qty = int(position.total_quantity * self.config.grid_level_1_sell_ratio)
            reason = f"🛡️ 触及防御网格 L1 (ATR x {effective_l1_mult:.2f}) | 波动率适配: {atr_pct:.1%} | 线 {level_1_price:.2f}"
            grid_level_hit = 1
            triggered_action = True

        # --- 4. 执行与状态锁定 (Execution) ---
        if triggered_action and action_qty > 0:
            # 最小手数调整
            stock_info = self.get_cached_stock_static_info(symbol)
            lot_size = stock_info.get('lot_size', 1 if position.market == MarketType.US else 100)
            action_qty = self._adjust_quantity(action_qty, position.market, lot_size)

            # 防御小股数：如果计算出来是0 (比如剩10股，卖25%=2股)，直接跳过，等待更深跌幅
            if action_qty <= 0:
                return False

            # ==============================================================================
            # ▼▼▼【SAAIS 上帝视角拦截器】▼▼▼
            # 逻辑：即使触网，如果主力在强力护盘，也暂缓减仓，避免丢失筹码。
            # ==============================================================================
            strategy_params = getattr(position, 'strategy_params', {}) or {}
            is_sharp_knife = strategy_params.get('is_sharp_knife',False)
            if not is_sharp_knife:

                saais_action, saais_reason = self.advanced_kinetic_intervention(symbol, current_price)
                
                if saais_action == "VETO":
                    logger.warning(f"🛡️ [{symbol}] 网格防御被上帝视角(VETO)拦截: {saais_reason}")
                    return False
                elif saais_action == "EXTEND":
                    logger.info(f"⏳ [{symbol}] 网格防御进入观察(EXTEND): {saais_reason}")
                    return False
                elif saais_action == "KILL":
                    reason += " [SAAIS核准:KILL]"
                # HOLD 则继续
            # ==============================================================================
            if not self.config.test_mode:
                # 提交卖单
                # 这里必须使用 process_sell_signal 的变体或者直接 submit_order 
                # 为了不破坏 process_sell_signal 的签名，我们直接调用底层，但要手动记录 sell_record
                order_id = self.submit_order(symbol, action_qty, OrderSide.Sell)
                
                if order_id:
                    with self.position_lock:
                        # 重新获取对象引用
                        pos = self.positions.get(symbol)
                        if pos:
                            pos.pending_sell_order_id = order_id
                            pos.sell_reason = reason
                            # 【核心状态变更】
                            pos.current_grid_level = grid_level_hit 
                            pos.is_defense_mode_active = True # 开启防御模式，熔断补仓
                            
                            # 【核心：埋下复活种子】
                            # 记录这一笔是为了将来T+0接回
                            pos.resurrection_cache.append({
                                'qty': action_qty,
                                'sell_price': current_price,
                                'level': grid_level_hit,
                                'status': 'WAITING',
                                'timestamp': time.time()
                            })
                            self._save_positions()
                            
                    logger.warning(f"[{symbol}] 网格防御执行成功: {reason} | 减仓: {action_qty}")

                    try:
                        symbol_info = self.get_cached_stock_static_info(symbol)
                        symbol_name = symbol_info.get('name_cn') or symbol_info.get('name_en')
                        sell_logger.info(f"symbol:{symbol},name:{symbol_name},price:{current_price},strategy_reason:{reason},llm_reason:N/A(GridDefense)")
                    except Exception as e:
                        logger.error(f"[{symbol}] 网格止损记录日志失败: {e}")

                    return True
        
        return False
    
    def _check_and_execute_composite_stop(self, symbol: str, current_price: float, position: Position) -> bool:
        """
        【复合防御协议 (Composite Defense Protocol)】
        
        功能：
        在顺风局（Normal/Bull Regime）下运行的主力风控逻辑。
        它融合了三层防御网，取【最高价格】（即最严格标准）作为有效止损线。
        
        三层防线：
        1. L1 基准层: 初始硬止损 或 追踪止损 (Trailing Stop)。
        2. L2 波动层: 基于 ATR 的动态止损，防止波动率突然放大。
        3. L3 熔断层: 基于短时急跌 (Flash Crash) 的紧急离场。
        
        叠加逻辑: WRP (周末风险协议) 会动态收紧上述防线。
        
        Returns:
            bool: True 表示止损已触发并执行，False 表示持仓安全。
        """
        # ▼▼▼【豁免名单检查】▼▼▼
        if position.triggering_strategy in self.config.strategies_immune_to_exit:
            return False
        
        # === 第一层：常规止损价 ===
        regular_stop_loss_price = 0.0
        if position.is_trailing_stop_active:
            regular_stop_loss_price = position.trailing_stop_price
        elif position.initial_stop_loss_price > 0:
            regular_stop_loss_price = position.initial_stop_loss_price

        # === 第二层：波动率调整止损 ===
        volatility_stop_loss_price = 0.0
        try:
            current_atr = get_dynamic_atr(self.quote_ctx, symbol)
            if current_atr and current_atr > 0:
                # 使用1.5倍当前ATR作为动态缓冲
                volatility_stop_loss_price = current_price - (current_atr * self.config.volatility_stop_atr_multiplier) # 1.5
                logger.debug(f"[{symbol}] 波动率止损价: {volatility_stop_loss_price:.3f} (ATR={current_atr:.3f})")
        except Exception as e:
            logger.warning(f"[{symbol}] 计算波动率止损失败: {e}")

        # === 第三层：紧急止损- 防闪崩 ===
        emergency_stop_loss_price = 0.0
        try:
            df_5m = get_klines_data(self.quote_ctx, symbol, 6, Period.Min_5, AdjustType.NoAdjust)
            if df_5m is not None and len(df_5m) >= 5:
                five_min_ago_high = df_5m['high'].iloc[-5]
                drop_ratio = (five_min_ago_high - current_price) / five_min_ago_high
                
                if drop_ratio >= 0.03:  # 5分钟内跌超3%
                    emergency_stop_loss_price = current_price * 0.998  # 立即止损，仅留0.2%缓冲
                    logger.critical(f"[{symbol}] ⚠️ 触发紧急止损！5分钟跌幅{drop_ratio:.2%}")
        except Exception as e:
            logger.warning(f"[{symbol}] 紧急止损检查失败: {e}")

        # === 【核心融合逻辑】取三层中最严格的止损价 ===
        effective_stop_loss_price = regular_stop_loss_price
        
        if volatility_stop_loss_price > 0:
            effective_stop_loss_price = max(effective_stop_loss_price, volatility_stop_loss_price)
        
        if emergency_stop_loss_price > 0:
            effective_stop_loss_price = max(effective_stop_loss_price, emergency_stop_loss_price)
            reason = "紧急止损(闪崩保护)"
        elif volatility_stop_loss_price > regular_stop_loss_price:
            reason = "波动率调整止损"
        elif position.is_trailing_stop_active:
            reason = "追踪止损"
        else:
            reason = "初始止损"

        # === WRP逻辑===
        wrp_stop_loss_price = 0.0
        if self._is_entering_weekend_risk_for_symbol(symbol):
            if position.is_trailing_stop_active:
                trailing_ratio = self.config.trailing_stop_ratio * self.config.wrp_trailing_stop_multiplier
                wrp_stop_loss_price = current_price * (1 - trailing_ratio)
            elif current_price > position.avg_cost:
                profit_protect_stop = current_price * (1 - self.config.wrp_hard_stop_profit_protect_ratio)
                wrp_stop_loss_price = max(profit_protect_stop, position.avg_cost)

        if wrp_stop_loss_price > effective_stop_loss_price:
            effective_stop_loss_price = wrp_stop_loss_price
            reason = f"WRP-{reason}"

        # === 执行判断 ===
        if effective_stop_loss_price > 0 and current_price <= effective_stop_loss_price:
            # 给市场一点喘息时间，也防止因数据毛刺导致的连续止损。
            if self._is_action_recently_taken(position, "止损", lookback_minutes=5):
                logger.info(f"[{symbol}] 止损信号频繁触发，处于冷却期，跳过。")
                return False
            
            # ==============================================================================
            # ▼▼▼【SAAIS 上帝视角拦截器】(God View Interceptor) ▼▼▼
            # 逻辑：在扣动扳机前，最后问一次法官。
            # ==============================================================================
            strategy_params = getattr(position, 'strategy_params', {}) or {}
            is_sharp_knife = strategy_params.get('is_sharp_knife',False)
            max_stop_loss_ratio = float(getattr(self.config, 'stop_loss_ratio', 0.025))
            max_stop_loss_price = round(position.avg_cost*(1-max_stop_loss_ratio),3)
            if not is_sharp_knife:
                saais_action, saais_reason = self.advanced_kinetic_intervention(symbol, current_price)
                
                # 尚未跌穿最终硬止损时，才允许上帝视角暂缓止损。
                if saais_action == "VETO" and current_price > max_stop_loss_price:
                    logger.warning(f"🛡️ [{symbol}] 复合止损被上帝视角(VETO)驳回: {saais_reason}")
                    return False # 刀下留人
                
                elif saais_action == "EXTEND" and current_price > max_stop_loss_price:
                    logger.info(f"⏳ [{symbol}] 复合止损进入死缓观察(EXTEND): {saais_reason}")
                    return False # 暂缓执行
                
                elif saais_action == "KILL":
                    reason = f"{reason} | ⚡SAAIS核准(KILL)"
                # HOLD 状态则维持原判，继续向下执行
                # ==============================================================================

            logger.warning(f"🛑 触发止损 {symbol}: 现价={current_price:.2f} <= 止损价={effective_stop_loss_price:.2f} (原因: {reason})")
            if not self.config.test_mode:
                self._execute_full_sell(symbol, reason)
                self.intraday_blacklist.add(symbol)
                self.intraday_blacklist.add(resolve_underlying_symbol(symbol))
                self._save_blacklist()
                logger.warning(f"🚫 [{symbol}] 已加入当日交易黑名单，今日禁止再次买入。")
                
                # 止损后期权补偿逻辑
                # if reason=='初始止损' and self.config.enable_option_trading:
                #     market = get_market_type(symbol)
                #     is_us_option = (MarketType.US == market)
                #     current_market_regime = self.market_regime_engine.get_marget_regime(market)
                #     if current_market_regime in [MarketRegime.STRONG_BULL, MarketRegime.CAUTIOUS_BULL]:
                #         candidate = {'symbol':symbol}
                #         if symbol in self.config.mag_seven_symbols and is_us_option:
                #             candidate['is_us_option'] = True
                #             candidate['us_lot_size'] = 1
                #             candidate['reason'] = f'{reason}'
                #             proposal = self.option_advisor.generate_trade_proposal(candidate)
                #             if proposal:
                #                 self.task_executor.submit(self._handle_option_proposal, proposal)
            return True
                
        return False

    # ==============================================================================
    # ▼▼▼ 持续新低动能止损  ▼▼▼
    # ------------------------------------------------------------------------------
    def _check_continuous_low_stop_loss(self, symbol: str, current_price: float, position: Position) -> bool:
        """
        检查持仓的持续下跌动能，以最低成本修复所有已知缺陷。
        - 修正时序漏洞，防止重试风暴。
        - [健壮性] 全面采用.get()访问配置，杜绝因配置错误导致的崩溃。
        - [效率优化] 前置时间检查，避免不必要的ATR计算。
        - [日志增强] 关键日志包含决策上下文，便于复盘。
        
        返回:
            bool: True 表示触发了止损并已执行卖出，外部调用应立即终止。
        """
        # --- 步骤 0: 安全加载配置 ---
        kinetic_cfg = self.config.kinetic_stop_loss_config
        # 使用 .get() 确保安全，即使 cfg 不是字典或 enabled 键不存在也不会崩溃
        if not isinstance(kinetic_cfg, dict) or not kinetic_cfg.get('enabled', False):
            return False

        # ==============================================================================
        # ▼▼▼ 补仓后免疫机制 (Dip Add Immunity) ▼▼▼
        # ------------------------------------------------------------------------------
        # 逻辑说明：
        # 1. 如果 position.dip_adds_done == 0 (未补仓)：
        #    处于“试探期”，严格执行动能止损，发现趋势不对立即撤退。
        # 2. 如果 position.dip_adds_done > 0 (已补仓)：
        #    处于“持仓运作期”，补仓意味着确认了支撑或拉低了成本。此时应给予更多波动空间，
        #    避免因短期惯性下跌导致刚补完仓就被动能止损扫地出门。
        # position.dip_adds_done == 1
        # ------------------------------------------------------------------------------
        # if position.dip_adds_done > 0:
        #     # logger.debug(f"🛡️ [{symbol}] 已补仓({position.dip_adds_done}次)，激活补仓保护，跳过持续新低止损。")
        #     return False
        # ==============================================================================

        # if position.triggering_strategy not in self.config.day_trade_only_strategies:
        #     return False

        # --------------------------------------------------------------------------
        # 1. 核心策略选择：Scheme D (The King's Choice)
        # --------------------------------------------------------------------------
        base_threshold = kinetic_cfg.get('fallback_n_periods_threshold', 3)
        
        if position.dip_adds_done > 0:
            # 【关键逻辑】已补仓，进入"宽容模式"
            # 杠杆ETF在补仓点位附近通常会有剧烈震荡洗盘。
            # 必须将阈值拉大，防止倒在黎明前。
            # 如果配置里的 base 是 3，这里直接给 5。
            n_periods_threshold = 5  
            mode_log = f"[宽容模式-已补仓{position.dip_adds_done}次]"
        else:
            # 【关键逻辑】未补仓，进入"猎杀模式"
            # 底仓试错，一旦势头不对，3个周期直接斩断，不留情面。
            n_periods_threshold = base_threshold
            mode_log = "[侦察模式]"
        # --------------------------------------------------------------------------
        
        # 2. 时间门槛检查 (防止高频Tick造成的伪信号)
        now_utc = datetime.now(timezone.utc)

        # 数据自愈：修复可能损坏的时间戳
        if position.last_consecutive_low_check_ts is None:
            position.last_consecutive_low_check_ts = now_utc
        elif isinstance(position.last_consecutive_low_check_ts, str):
            try:
                position.last_consecutive_low_check_ts = datetime.fromisoformat(position.last_consecutive_low_check_ts)
            except ValueError:
                position.last_consecutive_low_check_ts = now_utc

        # 检查间隔 (默认3分钟)
        check_interval_minutes = kinetic_cfg.get('fallback_check_interval_minutes', 3)
        if (now_utc - position.last_consecutive_low_check_ts).total_seconds() < check_interval_minutes * 60:
            return False
        
        # 更新检查时间
        position.last_consecutive_low_check_ts = now_utc
        
        # 3. 动态波动率适配 (可选的高级增强，进一步强化方案D)
        # 如果开启ATR，在极端行情下(如VIX>30)，阈值甚至可以进一步放宽
        effective_break_pct = kinetic_cfg.get('fallback_effective_break_pct', 0.003)
        try:
            enable_atr = kinetic_cfg.get('enable_atr', False)
            if enable_atr:
                atr = get_historical_atr(self.quote_ctx, symbol)
                if atr and current_price > 0:
                    vol_pct = (atr / current_price) * 100
                    # 如果波动率炸裂(>2.5%)，再给1次机会
                    if vol_pct > 2.5:
                        n_periods_threshold += 1
                        mode_log += "(ATR加成+1)"
                    
                    # 动态调整击穿幅度，波动越大，要求击穿越深才算有效
                    effective_break_pct = kinetic_cfg.get('atr_multiplier_for_break_pct', 1.2) * vol_pct / 100
                
        except Exception as e:
            logger.error(f"[{symbol}] 动能止损: 动态参数计算异常: {e}，回退到静态后备参数。")
            check_interval_minutes = kinetic_cfg.get('fallback_check_interval_minutes', 3)
            n_periods_threshold = kinetic_cfg.get('fallback_n_periods_threshold', 3)
            effective_break_pct = kinetic_cfg.get('fallback_effective_break_pct', 0.003)
        
        # 4. 审判时刻
        # 计算有效新低阈值 (必须跌破 前低 * (1 - 0.3%) 才算新低，过滤微小噪音)
        effective_low_threshold = position.post_purchase_low * (1 - effective_break_pct)

        if current_price < effective_low_threshold:
            old_low = position.post_purchase_low
            position.post_purchase_low = current_price # 更新前低
            position.consecutive_new_low_periods += 1 # 计数器 +1

            # 状态变更立即保存！
            # 否则一旦重启，这个 consecutive_new_low_periods 就会变回 0
            self._save_positions()
            
            logger.warning(
                f"🚨 {symbol} 动能恶化 {mode_log}: 现价{current_price:.3f} < 前低{old_low:.3f} | "
                f"连跌计数: {position.consecutive_new_low_periods}/{n_periods_threshold}"
            )

            if position.consecutive_new_low_periods >= n_periods_threshold:
                reason = f"动能衰竭止损: {mode_log} 连续 {position.consecutive_new_low_periods} 个周期创新低"
                logger.critical(f"🛑 确认止损信号: {symbol} -> {reason}")
                
                if not self.config.test_mode:
                    self._execute_full_sell(symbol, reason)
                return True
        else:
            # === 动能衰减 (Smart Decay) ===
            # 只有当计数器 > 0 时才需要处理
            if position.consecutive_new_low_periods > 0:
                decay_msg = ""
                # 计算相对于“前低”的反弹幅度
                # 注意：post_purchase_low 是这一轮连续下跌中记录的最低点
                if position.post_purchase_low > 0.0001:
                    rebound_from_low_pct = (current_price - position.post_purchase_low) / position.post_purchase_low
                else:
                    rebound_from_low_pct = 0.0 # 避免除零，视为无反弹
                
                # [分级大赦逻辑]

                # 说明多头强势介入，之前的阴跌趋势已断
                if rebound_from_low_pct > 0.012:
                    position.consecutive_new_low_periods = 0
                    decay_msg = f"🚀 强力反转(>{rebound_from_low_pct:.2%})，计数器彻底归零"
                
                # 等级 2: 弱势抵抗 (0.5% < 反弹 <= 1.2%) -> 计数器减 1
                # 说明有抵抗但不够强，给一点喘息空间，但不完全解除警报
                elif rebound_from_low_pct > 0.005:
                    old_count = position.consecutive_new_low_periods
                    position.consecutive_new_low_periods = max(0, position.consecutive_new_low_periods - 1)
                    decay_msg = f"💪 弱势抵抗({rebound_from_low_pct:.2%})，计数器衰减 ({old_count}->{position.consecutive_new_low_periods})"
                
                # 等级 3: 无力反弹 (反弹 <= 0.5%) -> 保持现状 (No Mercy)
                # 这种微弱反弹通常是下跌中继，不应衰减计数器，保持高压状态
                else:
                    pass
                    # if kinetic_cfg.get('enable_counter_decay', True):
                    #     position.consecutive_new_low_periods = max(0, position.consecutive_new_low_periods - 1)
                    #     decay_msg = f"📉 {symbol} 弱势震荡，计数器自然衰减至 {position.consecutive_new_low_periods}"
                    # else:
                    #     # 如果你配置了不衰减，那就保持原样（保持高压状态），这是最激进的"不耗着"
                    #     # 为了符合你的"不耗着"目标，这里应该是不变，或者最多减1。
                    #     # 这里我们保持减1的逻辑，作为兜底。
                    #     position.consecutive_new_low_periods = max(0, position.consecutive_new_low_periods - 1)
                    #     decay_msg = f"📉 {symbol} 震荡(无配置)，计数器衰减至 {position.consecutive_new_low_periods}"

                # 只有发生了状态变更才保存和打印
                if decay_msg:
                    self._save_positions()
                    logger.warning(f"♻️ [{symbol}] 动能修复: 现价 {current_price:.3f} (前低 {position.post_purchase_low:.3f}) | {decay_msg}")
            
        return False


    # ==============================================================================
    # ▼▼▼ SAAIS: 态势感知与自适应干预系统 (Situational Awareness & Adaptive Intervention) ▼▼▼
    # ==============================================================================
    def advanced_kinetic_intervention(self, symbol: str, current_price: float) -> Tuple[str, str]:
        """
        【上帝视角裁决】
        当动能止损即将触发（进入死缓期）时，调用此方法进行多维裁决。
        融合 [资金矢量] + [Gamma 地形] 做出最终生死判决。
        
        Returns:
            action (str): 
                - 'KILL': 立即处决 (雪崩/阴跌)
                - 'VETO': 一票否决/大赦 (诱空/护盘)
                - 'EXTEND': 延长死缓 (结构博弈)
                - 'HOLD': 维持原判 (无特殊信号)
            reason (str): 裁决理由
        """
        try:
            # --- 1. 获取情报 (Intelligence Gathering) ---
            
            # A. 资金情报 (Capital Will)
            # 获取机构净流入比例
            infr = get_institutional_net_flow_ratio(self.quote_ctx, resolve_underlying_symbol(symbol))
            # 防御：如果接口返回 None，设为 0
            if infr is None: infr = 0.0

            # 2. 获取均价线 VWAP (防卖飞的核心指标)
            vwap = 0.0
            quote = get_raw_quote(self.quote_ctx, symbol)
            if quote and quote.turnover and quote.volume and float(quote.volume) > 0:
                vwap = float(quote.turnover) / float(quote.volume)

            market = get_market_type(symbol)
            # B. 结构情报 (Structural Terrain)
            profile = None
            if market==MarketType.US:
                profile = self.gex_engine._calculate_gex_profile_vectorized(symbol)
            
            # [降级模式] 无期权数据
            if not profile:
                # 只有资金流极度恶化 (-8%) 且 价格已经跌破 VWAP 2% 以上才杀
                if infr < -0.10 and (vwap > 0 and current_price < vwap * 0.98):
                    return "KILL", f"🩸 [盲视模式] 资金溃逃({infr:.2%}) + 破位"
                return "HOLD", "无结构数据，且未破位"
    
            # [满血模式] 资金 + 结构
            # 解析关键点位
            total_gex = profile.get('total_gex', 0.0)
            zero_gamma = profile.get('zero_gamma_level', float('nan'))

            # 提取 Put Wall (支撑)
            put_wall_price = 0.0
            # put_wall_strength = 0.0
            gamma_walls = profile.get("gamma_walls", [])
            if gamma_walls:
                # 筛选出 GEX < 0 的墙
                put_walls = [w for w in gamma_walls if w['gex'] < 0]
                if put_walls:
                    # 找绝对值最大的 Put Wall
                    strongest = max(put_walls, key=lambda x: abs(x['gex']))
                    put_wall_price = strongest.get('price', 0.0)
                    # put_wall_strength = abs(strongest.get('gex', 0.0))

            # --- 2. 地形判定 (Terrain Mapping) ---
            
            # 判定 A: 宏观环境是否恶劣 (Is the Regime Unstable?)
            # 如果 Total GEX 为负，或者价格低于 Zero Gamma，都视为负伽马区域
            is_negative_regime = False
            if total_gex < 0:
                is_negative_regime = True
                regime_reason = "Total GEX Negative"
            elif not np.isnan(zero_gamma) and current_price < zero_gamma:
                is_negative_regime = True
                regime_reason = f"Price < Zero Gamma ({zero_gamma})"
            
            # 判定 B: 是否处于 Put Wall 强支撑区 (Wall Proximity)
            # 优化：距离小于 1.5% 且 Put Wall 在下方或附近
            is_at_put_wall = False
            if put_wall_price > 0:
                dist_to_wall = (current_price - put_wall_price) / put_wall_price
                # 只有当价格 仅仅略高于 Wall (0% ~ +1.5%) 或者 稍微跌破 Wall (-0.5% ~ 0%) 时才算支撑
                # 如果跌破太多，Wall 就变成了压力盖子
                if -0.005 < dist_to_wall < 0.015:
                    is_at_put_wall = True

            # --- 3. 矩阵裁决 (The Matrix Decision) ---
            
            # 【场景 1: 雪崩 (Avalanche)】 -> 立即处决 (KILL)
            # 条件: (负伽马环境) + (资金明显流出 OR 资金微弱但环境极差)
            if is_negative_regime:
                # 但如果价格依然站稳 VWAP (或偏离极小)，说明主力在【吸筹】或【护盘】！
                # 负伽马区域的吸筹往往会导致随后的暴力拉升。
                # 阈值：价格 > VWAP * 0.995 (允许 0.5% 的瞬间刺穿)
                if vwap > 0 and current_price > vwap * 0.995:
                    return "VETO", f"🛡️ [负伽马博弈] 价格站稳VWAP({vwap:.2f})，警惕轧空反转，暂缓死刑。"

                # 只有当：负伽马 + 资金流出 + 【价格真跌破了】 才杀
                if infr < -0.10 and (vwap > 0 and current_price < vwap * 0.985):
                    return "KILL", f"⚡ [雪崩确认] 负伽马 + 跌破均线({vwap:.2f}) + 资金溃逃"
                
                # 如果环境不稳定，且没人护盘 (infr < 0)，甚至不需要大流出，直接杀
                # if infr < -0.02:
                #     log_msg = f"⚡ [核按钮] 雪崩协议: 不稳定环境({regime_reason}) + 资金流出({infr:.2%}) -> 立即清仓"
                #     return "KILL", log_msg
                # 极度恶劣环境 (GEX < -10w) + 资金微弱流出 -> 避险
                # if total_gex < -100000 and infr < 0.0:
                #     return "KILL", f"⚡ [核按钮] 极度负伽马({total_gex:.0f}) + 资金疲软 -> 避险清仓"

            # 【场景 2: 诱空 (Bear Trap)】 -> 强力护盘 (VETO)
            # 条件: 处于支撑位 + 资金强力吸筹
            if is_at_put_wall and infr >0.00:
                log_msg = f"🛡️ [上帝视角] 诱空识别: Put Wall({put_wall_price}) + 机构吸筹({infr:.2%}) -> 否决止损"
                return "VETO", log_msg
            
            # 【场景 3: 结构博弈 (Structural Bet)】 -> 延长死缓 (EXTEND)
            # 条件: 踩在 Put Wall 上 + 资金虽然流出但不恐慌 (> -5%)
            if is_at_put_wall and infr > -0.05:
                log_msg = f"🧱 [结构博弈] 测试 Put Wall({put_wall_price}) + 资金惜售 -> 延长观察"
                return "EXTEND", log_msg
            
            # 【场景 4: 强力接飞刀 (Flying Dagger)】 -> VETO
            # 即使不在 Wall 附近，如果出现巨大的买盘 (INFR > 0.00)，
            # 说明可能是一个没被 GEX 捕捉到的消息面利好，或者是主力暴力洗盘
            if infr > 0.00:
                return "VETO", f"🛡️ [暴力承接] 资金出现罕见大额净买入({infr:.2%})"
            
            # 【场景 5: 阴跌 (Slow Bleed)】 -> 维持原判 (HOLD -> Let it die)
            # 既无险可守，资金也在流出，不需要额外干预，让动能止损自然触发
            if infr < -0.05:
                log_msg = f"🩸 [阴跌确认] 无结构支撑 + 持续流出({infr:.2%}) -> 维持原判(允许止损)"
                return "HOLD", log_msg

            # 默认
            return "HOLD", "无显著干预信号"

        except Exception as e:
            logger.error(f"[{symbol}] SAAIS 干预系统异常: {e}")
            return "HOLD", "系统异常维持原判"

    def _check_continuous_low_stop_loss_pro(self, symbol: str, current_price: float, position: Position) -> bool:
        """
        【动能止损 - 死缓验证版】(Probation Verification Protocol)
        
        逻辑重构：Check -> Verify -> Sell
        1. [侦察期] (Count < Threshold): 
           - 每 N 分钟检查一次，创新低则计数+1。
        2. [死缓期] (Count == Threshold): 
           - 触发瞬间不卖，进入"刀下留人"观察模式。
           - 实时监控(无视间隔)：
             a. 加速下跌 (-0.3%) -> 立即斩 (Count+1)。
             b. 强力反弹 (+0.5%) -> 大赦 (Count-1)。
             c. 时间耗尽 (180s)  -> 确认衰竭，斩 (Count+1)。
        3. [处决期] (Count > Threshold):
           - 执行卖出。
        """
        # --- 步骤 0: 安全加载配置 ---
        kinetic_cfg = self.config.kinetic_stop_loss_config
        if not isinstance(kinetic_cfg, dict) or not kinetic_cfg.get('enabled', False):
            return False
        # ▼▼▼【豁免名单检查】▼▼▼
        if position.triggering_strategy in self.config.strategies_immune_to_exit or self._is_bearish_symbol(symbol):
            return False
        
        # --------------------------------------------------------------------------
        # 1. 确定当前阈值 (The Threshold)
        # --------------------------------------------------------------------------
        base_threshold = kinetic_cfg.get('fallback_n_periods_threshold', 3)
        
        # [宽容模式]：如果已补仓，阈值拉大，防止倒在黎明前
        if position.dip_adds_done > 0:
            n_periods_threshold = 5 
            mode_log = f"[宽容模式-已补{position.dip_adds_done}]"
        else:
            n_periods_threshold = base_threshold
            mode_log = "[侦察模式]"

        # 数据自愈：修复可能损坏的时间戳
        now_utc = datetime.now(timezone.utc)
        if position.last_consecutive_low_check_ts is None:
            position.last_consecutive_low_check_ts = now_utc
        elif isinstance(position.last_consecutive_low_check_ts, str):
            try:
                position.last_consecutive_low_check_ts = datetime.fromisoformat(position.last_consecutive_low_check_ts)
            except ValueError:
                position.last_consecutive_low_check_ts = now_utc

        # ==============================================================================
        # ▼▼▼ 阶段 A: [死缓/处决] 逻辑 (Probation & Execution Phase) ▼▼▼
        # 逻辑：当计数器达到或超过阈值时，接管控制权，进行高频复核
        # ==============================================================================
        if position.consecutive_new_low_periods >= n_periods_threshold:
            # --- 【核心植入】 SAAIS 上帝视角干预 ---
            # 只有在刚刚进入死缓期，或者处于延长观察期时调用
            saais_action, saais_reason = self.advanced_kinetic_intervention(symbol, current_price)
            
            if saais_action == "VETO":
                # 大赦天下：重置计数器
                logger.warning(f"👑 {symbol} {saais_reason}，计数器重置 (Count 0)。")
                position.consecutive_new_low_periods = 0
                self._save_positions()
                return False
                
            elif saais_action == "KILL":
                # 立即处决：不需要等待时间耗尽
                logger.critical(f"💀 {symbol} {saais_reason}，无视死缓立即执行！")
                self._execute_kinetic_sell(symbol, saais_reason)
                return True
                
            elif saais_action == "EXTEND":
                # 延长死缓：将观察时间戳重置为当前，相当于再给一轮时间
                # 只有当还没有超时太久时才延长，防止无限循环
                time_in_probation = (now_utc - position.last_consecutive_low_check_ts).total_seconds()
                if time_in_probation > 60: # 每分钟允许延长一次
                    logger.info(f"⏳ {symbol} {saais_reason}，死缓时间重置。")
                    position.last_consecutive_low_check_ts = now_utc
                    self._save_positions()
                return False
                # 继续往下走，接受常规检查
            
            # elif saais_action == "HOLD":
            #     pass # 继续向下执行常规死缓逻辑
            # ----------------------------------------

            # 1. [处决期]：如果计数器已经爆表，说明已经确认死亡，直接卖出
            if position.consecutive_new_low_periods > n_periods_threshold:
                reason = f"动能衰竭止损: {mode_log} 最终确认死亡 (Count {position.consecutive_new_low_periods} > {n_periods_threshold})"
                self._execute_kinetic_sell(symbol, reason) # 封装下方的卖出逻辑
                return True

            # 2. [死缓期]：计数器刚好等于阈值，进行最后验证
            # 计算进入死缓期的时间 (即上一次更新计数器的时间)
            time_in_probation = (now_utc - position.last_consecutive_low_check_ts).total_seconds()
            
            # --- 验证参数 (可配置化) ---
            # 默认：死缓3分钟，加速跌0.3%杀，反弹0.5%救
            probation_timeout = kinetic_cfg.get('probation_timeout_seconds', 180) # 观察窗口 3分钟
            acceleration_kill_pct = kinetic_cfg.get('probation_acceleration_kill_pct', 0.003) # 加速下跌 0.3% 直接杀
            rescue_rebound_pct = kinetic_cfg.get('probation_rescue_rebound_pct', 0.005) # 反弹 0.5% 救回
            
            # 基准价格：就是触发死缓那一刻的最低价 (position.post_purchase_low)
            trigger_price = position.post_purchase_low
            if trigger_price <= 0: trigger_price = current_price # 防御除零

            # [分支 A]: 加速暴跌 -> 立即处决
            if current_price < trigger_price * (1 - acceleration_kill_pct):
                position.consecutive_new_low_periods += 1 # 3 -> 4 (确立死亡)
                self._save_positions()
                
                cost_price = position.get_avg_cost(self.config)
                roi_pct = (current_price - cost_price) / cost_price if cost_price > 0 else 0
                current_status = get_trading_window_status(symbol)
                # 定义有利卖出窗口
                favorable_sell_windows = self.config.favorable_sell_windows
                is_favorable_sell_time = current_status in favorable_sell_windows
                is_strong_bull = self.get_strong_bull(position.market)

                should_sell = False
                sell_logic_msg = ""
                # ============================================================
                # 【方案 A 重构】职责切干净：
                #   - 轻伤场景：仍由本函数处置，按 config 的 light_kinetic_sell_ratio 部分卖出；
                #   - 重伤场景：交回外层 _check_and_execute_composite_stop / _check_ladder_hard_stop
                #     接管（S2 -1.6% / S3 -2.0% 已经在覆盖，原重伤 else 分支属死代码已删除）。
                # 阈值来源：ladder_hard_stop_config.light_kinetic_threshold(_strong_bull)
                # ============================================================
                ladder_cfg = getattr(self.config, 'ladder_hard_stop_config', {}) or {}
                if is_strong_bull:
                    OPTIMAL_LIGHT_THRESHOLD = float(ladder_cfg.get(
                        'light_kinetic_threshold_strong_bull', -0.022))
                else:
                    OPTIMAL_LIGHT_THRESHOLD = float(ladder_cfg.get(
                        'light_kinetic_threshold', -0.018))

                # 情况 1: 轻伤 (ROI > OPTIMAL_LIGHT_THRESHOLD)
                # 逻辑：动能崩了，亏损还没扩大 → 部分减仓快跑，避免继续放大。
                if roi_pct > OPTIMAL_LIGHT_THRESHOLD:
                    is_tech_exit = check_tactical_exit_signal(
                        self.quote_ctx, symbol, 5,
                        self.config.rebound_pct_threshold_map['default']
                    )
                    if is_tech_exit:
                        should_sell = True
                        sell_logic_msg = f"轻伤快跑 (ROI {roi_pct:.2%})"
                # 情况 2: 重伤 (ROI ≤ OPTIMAL_LIGHT_THRESHOLD)
                # → 不在本函数处置，下一轮主干路由会落到 ladder 硬止损。
                else:
                    if position.consecutive_new_low_periods % 5 == 0:
                        logger.warning(
                            f"🛑 {symbol} 动能崩塌但 ROI {roi_pct:.2%} ≤ "
                            f"{OPTIMAL_LIGHT_THRESHOLD:.2%}，移交阶梯硬止损接管..."
                        )
                    return False

                # ─── 最终判决 ───
                if should_sell:
                    reason = f"动能止损: {sell_logic_msg} | 触发价{trigger_price:.3f} -> 现价{current_price:.3f}"
                    logger.critical(f"⚡ {symbol} {reason}")
                    self._execute_kinetic_sell(symbol, reason)
                    return True

                return False

            # [分支 B]: 强力反转 -> 大赦天下
            elif current_price > trigger_price * (1 + rescue_rebound_pct):
                old_count = position.consecutive_new_low_periods
                # 计数器回退 1 格，退回侦察期，并更新时间戳防止立即再次触发
                position.consecutive_new_low_periods = max(0, n_periods_threshold - 1)
                position.last_consecutive_low_check_ts = now_utc 
                self._save_positions()
                logger.warning(f"🛡️ [{symbol}] 动能止损: 诱空反转成功！解除死缓 (Count {old_count}->{position.consecutive_new_low_periods}) | 反弹 > 0.5%")
                return False

            # [分支 C]: 时间耗尽 -> 确认衰竭
            # elif time_in_probation > probation_timeout:
            #     position.consecutive_new_low_periods += 1 # 3 -> 4 (确立死亡)
            #     self._save_positions()
            #     reason = f"动能止损: 死缓期观察结束，无有效反弹 (耗时 {time_in_probation:.0f}s > {probation_timeout}s)"
            #     logger.critical(f"⌛ {symbol} {reason}")
            #     self._execute_kinetic_sell(symbol, reason)
            #     return True

            # [分支 D]: 僵持中 -> 继续观察
            else:
                # 可以选择每隔几十秒打印一次心跳，避免日志刷屏
                if int(time_in_probation) % 30 == 0:
                    logger.info(f"👀 [{symbol}] 动能死缓观察中... 耗时: {time_in_probation:.0f}s | 现价: {current_price:.3f} vs 触发价: {trigger_price:.3f}")
                return False

        # ==============================================================================
        # ▼▼▼ 阶段 B: [侦察期] 逻辑 (Normal Scouting Phase) ▼▼▼
        # 逻辑：正常的间隔检查，用于累积计数器
        # ==============================================================================
        
        # 1. 时间门槛检查 (仅在侦察期有效)
        # 检查间隔 (默认3分钟)
        check_interval_minutes = kinetic_cfg.get('fallback_check_interval_minutes', 3)
        if (now_utc - position.last_consecutive_low_check_ts).total_seconds() < check_interval_minutes * 60:
            return False
        
        # 更新检查时间 (注意：只有执行了下面的检查逻辑才更新)
        # position.last_consecutive_low_check_ts = now_utc <--- 移到下方 update 处

        # 2. 动态波动率适配
        effective_break_pct = kinetic_cfg.get('fallback_effective_break_pct', 0.003)
        try:
            enable_atr = kinetic_cfg.get('enable_atr', False)
            if enable_atr:
                atr = get_historical_atr(self.quote_ctx, symbol)
                if atr and current_price > 0:
                    vol_pct = (atr / current_price) * 100
                    if vol_pct > 2.5: # 波动大，给机会
                        n_periods_threshold += 1
                    effective_break_pct = kinetic_cfg.get('atr_multiplier_for_break_pct', 1.2) * vol_pct / 100
        except Exception:
            pass # 保持默认

        # 3. 创新低检查
        # 计算有效新低阈值 (必须跌破 前低 * (1 - 0.3%))
        effective_low_threshold = position.post_purchase_low * (1 - effective_break_pct)
        position.last_consecutive_low_check_ts = now_utc # 否则如果不创新低，下一次循环会立即再次进入，导致CPU空转。

        if current_price < effective_low_threshold:
            old_low = position.post_purchase_low
            
            # === 状态更新 ===
            position.post_purchase_low = current_price # 更新前低
            position.consecutive_new_low_periods += 1 # 计数器 +1
            position.last_consecutive_low_check_ts = now_utc # 重置计时器 (既是检查间隔，也是死缓开始时间)
            self._save_positions()
            
            logger.warning(
                f"🚨 {symbol} 动能恶化 {mode_log}: 现价{current_price:.3f} < 前低{old_low:.3f} | "
                f"连跌计数: {position.consecutive_new_low_periods}/{n_periods_threshold}"
            )

            # === 【关键修改】 触发阈值不卖，而是进入死缓 ===
            if position.consecutive_new_low_periods >= n_periods_threshold:
                logger.critical(f"⚠️ [{symbol}] 动能计数达标 ({position.consecutive_new_low_periods})，进入 [死缓观察期]！3分钟内若无反弹将执行卖出。")
                # 这里返回 False，把生杀大权留给下一轮循环的“阶段 A”
                return False 
                
        else:
            # === 动能自然衰减 (Smart Decay) ===
            # 只有当计数器 > 0 且不在死缓期时才处理
            # 这里的逻辑保持你原有的即可，只是更新了时间戳
            
            if position.consecutive_new_low_periods > 0:
                decay_msg = ""
                rebound_from_low_pct = 0.0
                if position.post_purchase_low > 0.0001:
                    rebound_from_low_pct = (current_price - position.post_purchase_low) / position.post_purchase_low
                
                # 强力反转归零
                if rebound_from_low_pct > 0.012:
                    position.consecutive_new_low_periods = 0
                    decay_msg = f"🚀 强力反转(>{rebound_from_low_pct:.2%})，计数器归零"
                # 弱势抵抗减1
                elif rebound_from_low_pct > 0.005:
                    old_count = position.consecutive_new_low_periods
                    position.consecutive_new_low_periods = max(0, position.consecutive_new_low_periods - 1)
                    decay_msg = f"💪 弱势抵抗({rebound_from_low_pct:.2%})，计数器衰减 ({old_count}->{position.consecutive_new_low_periods})"
                
                if decay_msg:
                    self._save_positions()
                    logger.warning(f"♻️ [{symbol}] 动能修复: {decay_msg}")

        return False

    def _execute_kinetic_sell(self, symbol: str, reason: str):
        """辅助方法：执行动能止损卖出"""
        logger.critical(f"🛑 确认动能止损信号: {symbol} -> {reason}")
        if not self.config.test_mode:
            if '轻伤快跑' in reason:
                # ─── 【方案 A 重构】单次配置化部分卖出 ───
                # 原 3 段递进 (0.5/0.7/1.0) 已废弃：动能函数只负责"轻伤减仓"，
                # 重伤交回外层 _check_ladder_hard_stop（S2/S3）兜底。
                pos = self.positions.get(symbol)
                if not pos:
                    return
                ladder_cfg = getattr(self.config, 'ladder_hard_stop_config', {}) or {}
                light_sell_ratio = float(ladder_cfg.get('light_kinetic_sell_ratio', 0.33))
                dedup_tag = "EXT_RUN_FAST_LIGHT"
                if pos.has_executed_action_today(dedup_tag):
                    logger.info(f"[{symbol}] 轻伤快跑[{dedup_tag}] 今日已执行，跳过重复减仓。")
                    return
                full_reason = f"{reason}[{dedup_tag}], 减仓{light_sell_ratio:.0%}"
                self._execute_extended_hours_sell(symbol, full_reason, sell_ratio=light_sell_ratio)
            else:
                self._execute_full_sell(symbol, reason)
                if '负伽马' in reason:
                    self.intraday_blacklist.add(symbol)
                    self.intraday_blacklist.add(resolve_underlying_symbol(symbol))
                    self._save_blacklist()
                    logger.warning(f"🚫 [{symbol}] 已加入当日交易黑名单，今日禁止再次买入。")

    def _execute_kinetic_sell_v1(self, symbol: str, reason: str):
        """辅助方法：执行动能止损卖出"""
        logger.critical(f"🛑 确认动能止损信号: {symbol} -> {reason}")
        if not self.config.test_mode:
            if '轻伤快跑' in reason:
                # ─── 递进式减仓：第1次减50%，第2次减剩余的70%，第3次全卖 ───
                # 每次用不同 tag，允许当日多次触发（防止只减一次后剩余仓位失控）
                pos = self.positions.get(symbol)
                if not pos:
                    return
                sell_ratio = 0.5
                for attempt in range(1, 4):  # 最多3次递进
                    dedup_tag = f"EXT_RUN_FAST_{attempt}"
                    if not pos.has_executed_action_today(dedup_tag):
                        # 递进比例：50% → 70% → 100%
                        sell_ratio = [0.5, 0.7, 1.0][attempt - 1]
                        reason = f"{reason}[{dedup_tag}], 减仓{sell_ratio:.0%}"
                        self._execute_extended_hours_sell(symbol, reason, sell_ratio=sell_ratio)
                        return
                # 3次都用过了 → 直接全卖
                logger.warning(f"[{symbol}] 轻伤快跑已递进3次，直接全卖")
                self._execute_full_sell(symbol, reason)
            elif '灾难兜底' in reason:
                # 灾难兜底：不走减仓，直接全仓清除
                self._execute_full_sell(symbol, reason)
                self.intraday_blacklist.add(symbol)
                self.intraday_blacklist.add(resolve_underlying_symbol(symbol))
                self._save_blacklist()
                logger.warning(f"🚫 [{symbol}] 灾难兜底已加入黑名单，今日禁止再买入。")
            else:
                self._execute_full_sell(symbol, reason)
                if '负伽马' in reason:
                    self.intraday_blacklist.add(symbol)
                    self.intraday_blacklist.add(resolve_underlying_symbol(symbol))
                    self._save_blacklist()
                    logger.warning(f"🚫 [{symbol}] 已加入当日交易黑名单，今日禁止再次买入。")

    # ==============================================================================
    # ▼▼▼ 闪电审判协议 (Flash Trial Protocol) ▼▼▼
    # 逻辑：在入场3-15分钟黄金期，通过物理事实直接判定买入逻辑是否证伪。
    # ==============================================================================
    def _execute_infant_flash_trial(self, symbol: str, position: Position, current_price: float) -> bool:
        """
        对刚出生(3-15min)的头寸执行“证伪审计”。
        """
        # 0. 豁免策略名单
        if position.triggering_strategy in self.config.strategies_immune_to_exit:
            return False

        # 做空免检
        if self._is_bearish_symbol(symbol):
            return False
        
        # 1. 时间窗口校验：只在入场3-30分钟执行
        holding_minutes = position.get_minutes_since_first_buy()
        if not (3 <= holding_minutes <= 30):
            return False
            
        # ==============================================================================
        # 2. 盈利保护盾 (Profit Shield)
        # 既然是“闪电审判”，针对的是“刚买入就失败”的单子。
        # 如果当前价格高于成本价（哪怕只是微利），说明市场暂时认可该交易。
        # 此时任何基于 VWAP/Flow 的看空指标都可能是洗盘，绝不应在浮盈状态下因“预测”而止损。
        # ==============================================================================
        if current_price >= position.avg_cost:
            return False

        # 3. 获取战场环境
        # market = position.market
        # regime = self.market_regime_engine.get_marget_regime(market)
        
        # 判定强牛环境：大趋势为牛
        is_strong_bull = self.get_strong_bull(position.market)
        # if is_strong_bull: return False

        # 4. 动态设定基础门槛
        roi = (current_price / position.avg_cost) - 1
        
        # 基础门槛：如果连 -0.4% 都没跌破，根本无需浪费算力去查证伪条件
        # 注意：这里只做初步筛选，不做决策
        if roi > -0.004:
            return False

        # --- 5. 证伪维度审计 (Trinity of Disproof) ---
        disproof_count = 0
        kill_reasons = []
                
        # B. 资金意志证伪：主力撤退
        symbol = resolve_underlying_symbol(symbol)
        infr = get_institutional_net_flow_ratio(self.quote_ctx, symbol) or 0.0
        # [优化] 资金流阈值分级：强牛市下，需要更显著的流出才计入坏账 (-2.5% vs -1.5%)
        infr_threshold = -0.15 if is_strong_bull else -0.10
        
        if infr < infr_threshold:
            disproof_count += 1
            kill_reasons.append(f"资金背离(NetFlow:{infr:.2%})")

        # C. 期权结构压制
        if position.market==MarketType.US:
            gex_profile = self.gex_engine._calculate_gex_profile_vectorized(symbol)
            if gex_profile:
                zero_gamma = gex_profile.get('zero_gamma_level')
                if zero_gamma and current_price < zero_gamma:
                    disproof_count += 1
                    kill_reasons.append("GEX负反馈区(Below Zero Gamma)")

        # --- 6. 最终裁决 (Final Judgment) ---
        should_kill = False
        final_reason = ""

        # 设定处决线 (Hard Death Line)
        # 强牛市：-1.1% (给予极高宽容度，防止被洗)
        # 普通市：-0.8% (严防死守)
        hard_death_line = -0.015 if is_strong_bull else -0.011
        
        # 场景一：非强牛市下的常规审判
        if not is_strong_bull:
            # 1. 触碰死线，直接杀
            if roi <= hard_death_line and disproof_count >= 2:
                should_kill = True
                final_reason = f"⚡ [闪电审判] 触碰处决线({roi:.2%}) | 理由: 逻辑崩塌(非强牛),{', '.join(kill_reasons)}"
            # 2. 弱势震荡中，跌破 -0.5% 且有 2 个坏消息，杀
            # elif roi <= -0.010 and disproof_count >= 2:
            #     should_kill = True
            #     final_reason = f"⚡ [闪电审判] 证伪共振({roi:.2%}) | 理由: {', '.join(kill_reasons)}"
        
        # 场景二：强牛市下的特赦逻辑 (Strong Bull Immunity)
        else:
            # 强牛市下，即便跌破 hard_death_line，我们也要看一眼是否有坏消息支撑
            # 如果只是单纯的价格下跌但没有任何证伪信号（disproof_count == 0），可能是主力洗盘，不杀！
            if roi <= hard_death_line:
                if disproof_count >= 1: # 至少有一个结构性坏消息才动手
                    should_kill = True
                    final_reason = f"⚡ [闪电审判-强牛] 破位止损({roi:.2%}) | 理由: 深度回撤且{kill_reasons[0]}"
                else:
                    # 这里就是你要的逻辑：大盘好的时候，不要在死线上无脑卖
                    # 记录日志但不执行
                    if random.random() > 0.9:
                        logger.info(f"🛡️ [强牛豁免] {symbol} 触及死线({roi:.2%})但结构完好，暂缓处决")
            
            # 强牛市下的“共振”要求极高：跌幅必须更深(-1.2%) 且 坏消息更多
            elif roi <= -0.012 and disproof_count >= 2:
                 should_kill = True
                 final_reason = f"⚡ [闪电审判-强牛] 证伪共振({roi:.2%}) | 理由: {', '.join(kill_reasons)}"

        # 记录关键日志用于调试（随机采样）
        if random.random() > 0.8:
            log_msg = f"FlashTrial check: {symbol}, roi:{roi:.2%}, is_strong_bull:{is_strong_bull}, disproof:{disproof_count}, kill:{should_kill}"
            logger.debug(log_msg)

        if should_kill:
            # 执行前最后问一次LLM（可选，为了速度可以移除，但针对非强牛市建议保留）
            if not is_strong_bull:
                signal_candidate = {'symbol': symbol, 'avg_cost': position.avg_cost, 'reason': f"闪电审判预警: {final_reason}",'strategy_name':position.triggering_strategy}
                is_confirmed, llm_msg = self._get_llm_decision(signal_candidate, 'sell')
                if not is_confirmed:
                    logger.warning(f"🛡️ [闪电审判] LLM大赦: {symbol} 虽证伪但LLM建议观察, 理由: {llm_msg}")
                    return False

            # 执行前再次校验价格（防止数据延迟导致杀在反弹上）
            # 如果这微秒间价格拉回成本上方，再次触发豁免
            latest_price = self.get_current_price(symbol)
            if latest_price and latest_price > position.avg_cost:
                logger.warning(f"🛡️ [闪电审判] 枪下留人: {symbol} 最后一刻价格回升至保本线之上")
                return False

            logger.critical(f"💀处决指令: {symbol} | {final_reason}")
            if not self.config.test_mode:
                self._execute_full_sell(symbol, final_reason)
                if '深度回撤且结构破位' in final_reason or '深度回撤且GEX负反馈区' in final_reason:#  加入黑名单
                    self.intraday_blacklist.add(symbol)
                    self.intraday_blacklist.add(resolve_underlying_symbol(symbol))
                    self._save_blacklist()
                    logger.critical(f"🚫 [{symbol}] ({final_reason})，已加入黑名单！")

            return True

        return False
    

    def check_profit_taking_signals(self, position: Position, current_price: float) -> Tuple[bool, float, str]:
        """
        检查主动止盈信号。
        当基于R倍数或固定百分比的任一止盈条件满足时，触发卖出。
        """
        # 如果已经部分止盈过，则不再触发此逻辑
        if position.r_profit_taken:
            return False, 0.0, ""

        # --- 触发器1: 基于风险回报 (R倍数) ---
        market_key = position.market
        r_multiple_rule = self.config.profit_take_r_by_market.get(market_key)
        
        if r_multiple_rule and r_multiple_rule.get('enable', False):
            if position.initial_risk_per_share > 0 and position.initial_scout_price > 0:
                # 获取趋势强度指标（ADX）
                adx_value = get_adx(self.quote_ctx, position.symbol, period=14)
                # 根据ADX动态调整止盈目标
                base_r_multiple = r_multiple_rule['profit_take_r_multiple']
                
                if adx_value is not None:
                    if adx_value > 40:
                        # 强趋势：提高止盈目标，让利润奔跑
                        adjusted_r_multiple = base_r_multiple * 1.5
                        trend_context = "强趋势"
                    elif adx_value > 25:
                        # 标准趋势：使用配置值
                        adjusted_r_multiple = base_r_multiple
                        trend_context = "标准趋势"
                    else:
                        # 弱趋势/震荡：降低止盈目标，快速落袋
                        adjusted_r_multiple = base_r_multiple * 0.7
                        trend_context = "弱势震荡"
                    
                    logger.info(
                        f"[{position.symbol}] 趋势分析：ADX={adx_value:.1f}({trend_context})，"
                        f"止盈目标调整为 {adjusted_r_multiple:.2f}R"
                    )
                else:
                    # ADX获取失败，使用默认值
                    adjusted_r_multiple = base_r_multiple
                    logger.debug(f"[{position.symbol}] ADX数据不可用，使用默认止盈目标")
                

                # 计算并检查止盈条件
                target_profit = adjusted_r_multiple * position.initial_risk_per_share
                current_profit = current_price - position.initial_scout_price
                
                if current_profit >= target_profit:
                    # 即使价格一直高于目标价，我们也只允许每 60 分钟收割一次利润。
                    # 这就实现了“在趋势中分批减仓”，而不是“一到价位瞬间清空”。
                    if self._is_action_recently_taken(position, "止盈", lookback_minutes=60):
                        return False, 0.0, ""
                    
                    reason = f"{r_multiple_rule['profit_take_r_multiple']}R倍数止盈"
                    logger.warning(f"触发信号 ({reason}) for {position.symbol}: 盈利({current_profit:.2f}) >= 目标({target_profit:.2f})")
                    return True, r_multiple_rule.get('sell_ratio', 0.5), reason

        # --- 触发器2: 基于固定百分比 (你的安全网) ---
        fixed_percentage_rule = self.config.fixed_profit_take_by_market.get(market_key)

        if fixed_percentage_rule and fixed_percentage_rule.get('enable', False):

            # ===【您的核心策略实现】===
            base_price = 0.0
            if position.has_dip_added():
                # 如果发生过下跌补仓，使用平均成本价作为基准
                base_price = position.avg_cost
                logger.debug(f"[{position.symbol}] 检测到下跌补仓记录，止盈基准切换为 avg_cost: {base_price:.3f}")
            else:
                # 否则，使用初始建仓价作为基准
                base_price = position.initial_scout_price
                logger.debug(f"[{position.symbol}] 无下跌补仓记录，止盈基准维持 initial_scout_price: {base_price:.3f}")

            # ==========================
            if base_price > 0:
            # 使用 initial_scout_price 作为基准，更稳定
            # if position.initial_scout_price > 0:
                target_price = base_price * (1 + fixed_percentage_rule['percentage'])
                
                if current_price >= target_price:
                    # ===【“更妙”的策略实现】===
                    sell_ratio = 0.0
                    if position.has_dip_added():
                        # 如果发生过下跌补仓，使用更小的、专属的卖出比例
                        sell_ratio = r_multiple_rule.get('dip_add_sell_ratio', 0.3) # 读取新配置，提供默认值
                    else:
                        # 否则，使用常规的卖出比例
                        sell_ratio = r_multiple_rule.get('sell_ratio', 0.5) # 读取原配置
                    # ==========================

                    reason = f"固定{fixed_percentage_rule['percentage']:.0%}止盈 (基于 {'平均成本' if position.has_dip_added() else '初始价格'})"
                    logger.warning(f"触发信号 ({reason}) for {position.symbol}: 现价({current_price:.2f}) >= 目标价({target_price:.2f})")
                    
                    return True, sell_ratio, reason
                    
        # 如果两个条件都不满足
        return False, 0.0, ""
    
    def _check_and_execute_daily_profit_target(self):
        """
        【每日盈利治理引擎】
        
        逻辑分层:
        1. Stage 2 (水位 100%): 收割 (Harvest). 
           - 目标达成。亏损股清仓，盈利股止盈一半+剩余严管。
        2. Stage 1 (水位 80%): 净化 (Purge). 
           - 半程维护。亏损股清仓，盈利股收紧止损(套紧箍咒)。
        """

        self._check_date_change_and_reset() # 每日重置检查

        if not self.config.daily_profit_target_enabled:
            return

        # 如果今日双市场都已经完美达标，进入残余清理扫描
        if self.daily_hk_profit_target_hit or self.daily_us_profit_target_hit:
            target_markets = []
            if self.daily_hk_profit_target_hit: target_markets.append(MarketType.HK)
            if self.daily_us_profit_target_hit: target_markets.append(MarketType.US)
            
            if target_markets:
                self._scan_and_trim_leftovers(target_markets)
            

        # ===========================================================
        # [Phase A: 达标计算模式] (Calculation Track)
        # ===========================================================

        positions_copy = []
        realized_hk = 0.0
        realized_us = 0.0
        with self.position_lock:
            # 再次检查状态，防止在等待锁的过程中状态突变
            if self.daily_hk_profit_target_hit and self.daily_us_profit_target_hit:
                return
            # 读取已实现盈亏
            realized_hk = self.daily_realized_pnl_hk
            realized_us = self.daily_realized_pnl_us
            
            # 复制持仓列表
            positions_copy = list(self.positions.values())

        # 2. 准备计算变量
        float_pnl_hk = 0.0
        float_pnl_us = 0.0
        candidates_hk = []
        candidates_us = []
        exchange_rate = getattr(self.config, 'exchange_rate_usd_to_hkd', 7.8)

        for pos in positions_copy:
            symbol = pos.symbol
            # 如果该市场已经达标，跳过计算，节省资源
            if pos.market == MarketType.HK and self.daily_hk_profit_target_hit: continue
            if pos.market == MarketType.US and self.daily_us_profit_target_hit: continue
            
            # 如果这只票已经处理完了，跳过
            if pos.daily_profit_trimmed: continue

            current_price = self.get_current_price(symbol)
            if not current_price or current_price <= 0: continue

            # 判断基准价 (隔夜仓用昨收，新仓用成本)
            is_opened_today = pos.is_opened_today(symbol)
            base_price = pos.avg_cost
            is_overnight = False # 默认为 False，安全第一
            
            # 判断是否为当日新仓
            if not is_opened_today:
                is_overnight = True

            if is_opened_today:
                base_price = pos.avg_cost
            elif not is_opened_today or is_overnight:
                prev_close = get_yesterday_close_price(self.quote_ctx, symbol)
                base_price = prev_close
            
            # 计算当日浮动盈亏
            pnl = (current_price - base_price) * pos.total_quantity
            if pos.market == MarketType.HK:
                float_pnl_hk += pnl
                candidates_hk.append(pos)
            elif pos.market == MarketType.US:
                # 美股 PnL 转换为 HKD 累计
                float_pnl_us += (pnl * exchange_rate)
                candidates_us.append(pos)

        # --- 汇总判定与执行 ---
        total_hk = float_pnl_hk + realized_hk
        total_us_hkd = float_pnl_us + (realized_us * exchange_rate)
        # 计算全账户当前总盈亏 (HKD)
        current_total_pnl = total_hk + total_us_hkd
        
        # 获取资产
        total_asset = self.get_net_equity_value(strict=True)
        if total_asset <= 0:
            logger.warning("无法获取有效总资产，跳过每日止盈检查")
            return

        target_amount = total_asset * self.config.daily_profit_target_asset_ratio # 100% 目标
        half_target = target_amount * self.config.daily_profit_half_target_ratio # 80% 目标
        
        # ==============================================================================
        # ▼▼▼ 核心分级治理 ▼▼▼
        # ==============================================================================
        # --- Stage 2: 完美达标 (100%) ---
        if current_total_pnl >= target_amount:
            if not (self.daily_hk_profit_target_hit and self.daily_us_profit_target_hit):
                logger.critical(f"🎉 [每日达标] 总盈利 {current_total_pnl:.0f} > 目标 {target_amount:.0f} (100%)！启动全员收割！")
                
                # 执行 Stage 2 收割：锁利 + 补刀
                self._execute_portfolio_maintenance(stage=2, reason="Stage2_Harvest(100%)")
                
                # 标记达标，防止重复触发 Stage 2
                self.daily_hk_profit_target_hit = True
                self.daily_us_profit_target_hit = True
                # 同时也标记半程触发，确保补仓逻辑被熔断
                self.half_target_escape_triggered = True
                self._save_daily_pnl_state()
                # self.notification_manager.send_email_direct(f"[{self.script_name}]每日盈利达标", f"总盈利 {current_total_pnl:.0f} HKD，已执行锁定策略。")
                self.notification_manager.send_feishu(f"[{self.script_name}]每日盈利达标", f"总盈利 {current_total_pnl:.0f} HKD，已执行锁定策略。")

        # --- Stage 1: 半程净化 (80%) ---
        elif current_total_pnl >= half_target:
            # 只有在未达标 100% 时才检查 80%
            if random.random() > 0.7:
                # 执行 Stage 1 净化：去弱留强 + 紧箍咒
                self._execute_portfolio_maintenance(stage=1, reason=f"Stage1_Purge({self.config.daily_profit_half_target_ratio*100}%)")
                self.half_target_escape_triggered = True
                logger.warning(f"🛡️ [半程标记] 盈利达标 80%，已激活补仓熔断机制。")
                # self.notification_manager.send_email_direct(f"[{self.script_name}]半程达标-净化启动", f"总盈利 {current_total_pnl:.0f} HKD，已执行去弱留强。")
                self.notification_manager.send_feishu(f"[{self.script_name}]半程达标-净化启动", f"总盈利 {current_total_pnl:.0f} HKD，已执行去弱留强。")

    def _execute_portfolio_maintenance(self, stage: int, reason: str):
        """
        【组合维护执行器】
        根据治理阶段 (Stage) 执行差异化操作。
        
        Stage 1 (80%): 
          - Losers: 杀无赦 (全仓卖出)。
          - Winners: 紧箍咒 (收紧止损至 99%，不卖出)。
          - New: 豁免。
          
        Stage 2 (100%):
          - Losers: 杀无赦 (补刀)。
          - Winners: 半仓止盈 (卖 50%) + 剩余仓位铁底保护 (止损提至 98%)。
          - New: 豁免。
        """
        # 1. 获取所有持仓快照 (Lock)
        with self.position_lock:
            positions_snapshot = list(self.positions.values())
        
        for pos in positions_snapshot:
            symbol = pos.symbol
            # ▼▼▼【豁免名单检查】▼▼▼
            if pos.triggering_strategy in self.config.strategies_immune_to_exit:
                return False
            
            if not self._is_pure_stock(symbol): continue
            
            # 0. 【新兵豁免权】
            # 无论哪个阶段，刚买入 30 分钟内的股票，拥有绝对豁免权。
            if not self._is_position_mature(pos, min_minutes=30):
                continue

            # 1. 盈亏判定
            current_price = self.get_current_price(symbol)
            if not current_price: continue
            real_cost = pos.get_avg_cost(self.config)
            if real_cost <= 0: continue
            roi = (current_price - real_cost) / real_cost
            
            # 判定属性
            is_winner = roi > 0
            is_loser = roi <= -0.005 # 亏损超过 0.5% 定义为劣质资产
            
            # ================= Stage 1: 净化 (80%) =================
            if stage == 1:
                # A. 亏损股 -> 全卖
                if is_loser:
                    if not pos.daily_profit_trimmed: # 防止重复提交
                        logger.warning(f"🛡️ [Stage 1] 清理负资产 {symbol} (ROI {roi:.2%}) | 原因: {reason}")
                        self._execute_full_sell(symbol, f"{reason}-清理拖油瓶")
                        with self.position_lock:
                            if symbol in self.positions: self.positions[symbol].daily_profit_trimmed = True
                
                 # B. 盈利股 -> 卖50% + 紧箍咒
                elif is_winner:
                    # 如果今天还没针对利润做过修剪
                    if not pos.r_profit_taken:
                        # 执行：卖出 50% (读取配置 daily_profit_escape_sell_ratio，建议设为 0.5)
                        sell_ratio = getattr(self.config, 'daily_profit_escape_sell_ratio', 0.5)
                        
                        logger.warning(f"💰 [Stage 1] 锁定胜果 {symbol} (ROI {roi:.2%}) | 卖出 {sell_ratio:.0%} | 原因: {reason}")
                    
                        # 发送卖单
                        success = self.process_sell_signal(symbol, percentage=sell_ratio, reason=f"{reason}-锁定利润")
                        
                        if success:
                            # ß处理剩下的股票：设置极度安全的止损线
                            if symbol in self.positions:
                                p = self.positions[symbol]
                                
                                # 目标：将止损线强制提升到 [现价 * 0.99] 和 [原止损] 的较大值
                                tight_stop = current_price * 0.995
                                # 只有当新止损线比现在的更高时才更新，防止反而把止损放宽了
                                if tight_stop > pos.trailing_stop_price:
                                    p.trailing_stop_price = tight_stop
                                
                                p.is_trailing_stop_active = True # 强制激活追踪止损，否则主循环不会执行卖出！# [必须] 强制激活追踪止损，否则主循环不会执行卖出！
                                # 标记今日已处理，防止 Stage 1 反复对同一只股开刀
                                p.daily_profit_trimmed = True
                                self._save_positions()
                                logger.warning(f"🔒 [Stage 1] 盈利股 {symbol} 止损收紧至 {tight_stop:.2f} (现价 {current_price:.2f})")

            # ================= Stage 2: 收割 (100%) =================
            elif stage == 2:
                # A. 处理亏损股 -> 补刀 (如果在 Stage 1 没死透)
                if is_loser:
                    logger.warning(f"🛡️ [Stage 2] 终极清算 {symbol} (ROI {roi:.2%}) | 原因: {reason}")
                    self._execute_full_sell(symbol, f"{reason}-终极清算")
                    with self.position_lock:
                        if symbol in self.positions: self.positions[symbol].daily_profit_trimmed = True
                
                # B. 处理盈利股 -> 锁利一半 + 剩余严管
                elif is_winner:
                    # 如果已经执行过利润收割，就不再重复卖了，只检查止损是否够紧
                    if pos.r_profit_taken:
                        # 检查剩余仓位的止损是否足够安全 (99%)
                        safe_stop = current_price * 0.99
                        if safe_stop > pos.trailing_stop_price:
                            with self.position_lock:
                                if symbol in self.positions:
                                    self.positions[symbol].trailing_stop_price = safe_stop
                                    self.positions[symbol].is_trailing_stop_active = True
                                    self._save_positions()
                            logger.info(f"🔒 [Stage 2] {symbol} 剩余仓位止损升级至 {safe_stop:.2f}")
                    
                    # 如果之前完全没卖过（直接冲到Stage 2），则执行卖出
                    else:
                        # 执行：卖出 50% (读取配置 daily_profit_sell_ratio，建议设为 0.5)
                        sell_ratio = getattr(self.config, 'daily_profit_sell_ratio', 0.5)
                        
                        logger.warning(f"💰 [Stage 2] 锁定胜果 {symbol} (ROI {roi:.2%}) | 卖出 {sell_ratio:.0%} | 原因: {reason}")
                        
                        # 发送卖单
                        success = self.process_sell_signal(symbol, percentage=sell_ratio, reason=f"{reason}-锁定利润")
                        
                        if success:
                            # 【核心】处理剩下的股票：设置极度安全的止损线
                            with self.position_lock:
                                if symbol in self.positions:
                                    p = self.positions[symbol]
                                    p.r_profit_taken = True # 标记已收割，防止重复卖
                                    p.is_trailing_stop_active = True
                                    # 剩余仓位止损线提至现价 98%，确保存活就是赚
                                    p.trailing_stop_price = max(p.trailing_stop_price, current_price * 0.99)
                                    p.daily_profit_trimmed = True
                                    self._save_positions()
                            logger.info(f"🔒 [Stage 2] {symbol} 剩余仓位进入严管模式 (止损线 {current_price * 0.99:.2f})")

    def _is_position_mature(self, position: Position, min_minutes: int = 30) -> bool:
        """
        【成熟度过滤器】判断持仓是否足够“成熟”，可以被治理逻辑处理。
        
        标准：
        1. 持仓时间必须超过 N 分钟 (默认30分钟)。
        2. 必须有有效的建仓时间戳。
        """
        try:
            duration_minutes = position.get_minutes_since_first_buy()
            return duration_minutes >= min_minutes
            
        except Exception as e:
            logger.warning(f"[{position.symbol}] 成熟度检查异常: {e}，默认豁免。")
            return False
    
    def _execute_batch_daily_trim(self, positions: List[Position], reason: str) -> bool:
        """
        批量执行每日止盈削减。
        Returns:
            bool: 如果至少成功提交了一个卖出订单，返回 True；否则返回 False。
        """
        if not positions:
            logger.info("虽达标，但无符合条件的盈利持仓可卖。")
            return False
        sell_ratio_profit = 1.0 # 盈利单：建议全走或按高比例走
        sell_ratio = self.config.daily_profit_sell_ratio
        any_success = False
        allow_loss_selling = getattr(self.config, 'daily_trim_allow_loss_selling', False)
        
        for pos in positions:
            symbol = pos.symbol
            # 实时获取价格，保证决策准确性
            current_price = self.get_current_price(symbol)
            if not current_price: continue

            real_cost = pos.get_avg_cost(self.config)
            # 计算 ROI
            roi = (current_price - real_cost) / real_cost
            
            # --- 分支 A: 盈利持仓 (Profit) ---
            if current_price > real_cost:
                logger.warning(f"💰 [{symbol}] 每日止盈(盈利): ROI {roi:.2%}, 执行锁定...")
                success = self.process_sell_signal(pos.symbol, percentage=sell_ratio_profit, reason=f"{reason}-锁定利润")
                if success:
                    pos.daily_profit_trimmed = True # ✅ 处理完毕
                    any_success = True

            # --- 分支 B: 亏损持仓 (Loss) ---
            else:
                # 1. 意愿锁检查: Config 说不许卖亏损的？
                if not allow_loss_selling:
                    # 只有当它还没被标记时，才打印日志并标记
                    if not pos.daily_profit_trimmed:
                        logger.info(f"🛡️ [{symbol}] 每日止盈: 亏损持仓(ROI {roi:.2%})，配置禁止卖出，标记忽略。")
                        pos.daily_profit_trimmed = True # ✅ 标记为已处理(放过它)
                        any_success = True # 状态变了，需要保存
                    continue

                # 2. 深度检查: 亏损太小不值得操作？(比如只亏了 0.5%)
                if roi > self.config.daily_trim_loss_threshold: # 亏损小于 1%，忽略不卖
                    if not pos.daily_profit_trimmed:
                        pos.daily_profit_trimmed = True # ✅ 微亏视为震荡，标记忽略
                        any_success = True
                    continue

                # 3. 技术锁检查: 是否是反弹高点？
                # 这里的逻辑是：既然已经亏了，就别在低位割肉，等一个 5分钟 K 线的高点确认
                k_mins_check = 5
                rebound_pct_threshold = self.config.rebound_pct_threshold_map['default']
                is_high_confirmed = check_tactical_exit_signal(self.quote_ctx, symbol, k_mins_check, rebound_pct_threshold)

                if is_high_confirmed:
                    # 信号确认 -> 动手！
                    logger.warning(f"📉 [{symbol}] 每日止盈(亏损): 反弹确认 (ROI {roi:.2%}), 顺势减亏 {sell_ratio:.0%}...")
                    success = self.process_sell_signal(pos.symbol, percentage=sell_ratio, reason=f"{reason}-战术减亏")
                    if success:
                        pos.daily_profit_trimmed = True # ✅ 卖出成功，标记完成
                        any_success = True
                # else:
                #     # 信号未确认 -> 等待！
                #     # 关键：不设置 trimmed = True，保持 False
                #     # 也不打印 info 日志防止刷屏，只在 debug 打印
                #     # 这样下一轮 _scan_and_trim_leftovers 会再次进来
                #     pass
        
        # 批量操作后保存一次即可，减少IO
        if any_success:
            self._save_positions()
            
        return any_success

    def _scan_and_trim_leftovers(self, target_markets: List[MarketType]):
        """
        扫描已达标市场中，尚未处理完毕(Trimmed=False)的持仓。
        """
        leftovers = []
        with self.position_lock:
            for pos in self.positions.values():
                # 1. 已经处理过的跳过
                if pos.daily_profit_trimmed: continue
                # 2. 必须是已达标市场
                if pos.market not in target_markets: continue
                
                leftovers.append(pos)
        
        if leftovers:
            # [节流] 并不是每次循环都必须去问技术指标，给API喘息机会
            # 约 20% 的概率执行检查，或者你可以用时间戳控制
            if random.random() > 0.5:
                # 复用批量执行逻辑，Reason 标记为后续清理
                self._execute_batch_daily_trim(leftovers, "每日止盈(战术补刀)")

    def _execute_escape_plan(self):
        """
        【逃跑计划执行器】
        功能：当倒计时结束且未达全额目标时，强制卖出所有**浮盈**持仓的 50%。
        特点：
        1. 只卖赚钱的，不卖亏钱的。
        2. 简单粗暴，不看技术指标，只为锁定胜果。
        """
        candidates = []
        # 获取倒计时开始的时间戳（作为分界线）
        # 如果意外为空（虽然理论上不会），则设为当前时间，相当于不卖出任何新仓
        cutoff_timestamp = self.daily_profit_countdown_start_ts or time.time()

        with self.position_lock:
            for pos in self.positions.values():
                # 过滤掉非股票资产
                if not self._is_pure_stock(pos.symbol): continue
                
                # 获取现价
                current_price = self.get_current_price(pos.symbol)
                if not current_price: continue
                
                # ==========================================================
                # ▼▼▼【时间围栏豁免】▼▼▼
                # ==========================================================
                if pos.get_minutes_since_first_buy() > int(cutoff_timestamp/60):
                    logger.info(f"🛡️ [逃跑豁免] {pos.symbol} 是预警后新开仓位，不参与本次逃跑计划。")
                    continue

                # 获取真实成本
                real_cost = pos.get_avg_cost(self.config)
                
                # 核心筛选：只卖出【当前价格 > 成本价】的盈利票
                if current_price > real_cost:
                    # 计算ROI仅用于日志
                    roi = (current_price - real_cost) / real_cost
                    candidates.append((pos, roi))
        
        if not candidates:
            logger.info("🤷‍♂️ [逃跑计划] 触发，但当前无盈利持仓可卖。")
            return

        logger.warning(f"⚡ [逃跑计划] 正在对 {len(candidates)} 个盈利持仓执行 50% 强制止盈...")
        
        for pos, roi in candidates:
            symbol = pos.symbol
            reason = f"[逃跑计划] 倒计时结束-锁定半程利润 (ROI {roi:.2%})"
            
            # percentage=0.5 (卖出一半)
            # 这里的 process_sell_signal 会自动处理挂单锁、日志记录等
            success = self.process_sell_signal(symbol, percentage=self.config.daily_profit_escape_sell_ratio, reason=reason)
            
            if success:
                logger.warning(f"✅ {symbol} 逃跑成功，已提交卖单。")
            else:
                logger.error(f"❌ {symbol} 逃跑失败 (可能被锁或状态异常)。")

# ==============================================================================
# 每日状态重置检查
# ==============================================================================
    def _check_date_change_and_reset(self):
        """
        日期变更与状态重置管理器。
        职责:
        1. 每日午夜(00:00): 重置当日本地PnL记录，归零达标状态。
        2. 美股开盘(09:30 ET): 强制执行 'RTH Reset Protocol'。
           - 目的: 清洗盘前(Pre-market)的盈亏波动。
           - 效果: 确保 'daily_realized_pnl_us' 仅反映常规交易时段的战绩。
        """
        need_save = False
        tz_hk = pytz.timezone(MARKET_TRADING_HOURS["HK"]["timezone"])
        tz_us = pytz.timezone(MARKET_TRADING_HOURS["US"]["timezone"])

        now_hk = datetime.now(tz_hk)
        now_us = datetime.now(tz_us)
        
        today_hk_str = str(now_hk.date())
        today_us_str = str(now_us.date())

        with self.position_lock:
            # ==================================================================
            # 1. 检查 HK 市场 (常规日切)
            # ==================================================================
            if today_hk_str != self.current_trading_date_hk:
                logger.warning(f"🌏 [HK] 日期变更 ({self.current_trading_date_hk} -> {today_hk_str})，执行清零...")
                self.daily_realized_pnl_hk = 0.0
                self.daily_hk_profit_target_hit = False
                self.current_trading_date_hk = today_hk_str
                # 重置削减标记
                for pos in self.positions.values():
                    if pos.market == MarketType.HK: pos.daily_profit_trimmed = False
                need_save = True

            # ==================================================================
            # 2. 检查 US 市场 (午夜日切 00:00)
            # ==================================================================
            # 即使有RTH重置，午夜重置依然必要，用于清除隔夜(Pre-market前)的陈旧数据
            # if today_us_str != self.current_trading_date_us:
            #     logger.warning(f"🗽 [US] 日期变更 ({self.current_trading_date_us} -> {today_us_str})，执行午夜清零...")
            #     self.daily_realized_pnl_us = 0.0
            #     self.daily_us_profit_target_hit = False
            #     self.current_trading_date_us = today_us_str

            #     # 重置逃跑计划
            #     self.daily_profit_countdown_start_ts = None
            #     self.half_target_escape_triggered = False
            #     self.daily_pnl_high_water_mark = 0.0
            #     self.daily_pnl_last_reset_base = 0.0
                
            #     # 重置削减标记
            #     for pos in self.positions.values():
            #         if pos.market == MarketType.US: pos.daily_profit_trimmed = False
                
            #     # [关键] 日期变了，说明新的一天开始了，之前的 RTH 锁失效了
            #     # 即使昨晚重置过，今天还没重置，所以这里理论上不用手动置 None，
            #     # 但为了数据一致性，确保加载时逻辑正确。
            #     # self.daily_pnl_us_rth_last_reset_date 保持不变即可，靠比对 today_us_str 判定失效
                
            #     need_save = True

            # ==================================================================
            # 3. 美股 RTH 开盘脉冲重置 (09:30 ET) - 核心修复
            # ==================================================================
            # 逻辑：当前时间 >= 09:30 且 锁记录的日期 != 今天

            if is_in_opening_window(MarketType.US, window_minutes=1) and self.daily_pnl_us_rth_last_reset_date != today_us_str:
                logger.warning(f"🔔 [美股开盘] 09:30 ET 触发 PnL 归零重启协议! ({today_us_str})")
                # --- A. 资金与状态重置 ---
                self.daily_realized_pnl_us = 0.0
                self.daily_us_profit_target_hit = False

                # 重置逃跑计划 (确保盘中从零开始累计)
                self.daily_profit_countdown_start_ts = None
                self.half_target_escape_triggered = False
                self.daily_pnl_high_water_mark = 0.0
                self.daily_pnl_last_reset_base = 0.0
                
                # 重置美股持仓的削减标记
                for pos in self.positions.values():
                    if pos.market == MarketType.US: pos.daily_profit_trimmed = False
                
                # --- B. 记忆体清洗 ---
                self.intraday_trade_history.clear()
                logger.warning("🧹 [开盘重置] 日内交易价格记忆(T+0限制)已清空。")

                # 黑名单重置：只在每天第一次开盘时做
                self.intraday_blacklist.clear() # 跨日清空
                # self._save_blacklist()
                logger.warning("🧹 [跨日重置] 当日交易黑名单已清空。")

                if self.extended_hours_risk_engine:
                    self.extended_hours_risk_engine.reset_all_states()
                    logger.info("🧹 [系统维护] 已强制重置夜魅风控引擎状态 (新交易日)。")
                
                self._refresh_candidate_pools()

                self.pending_buy_cache.cleanup_all_signals() ##删除所有信号

                # --- C. 更新持久化锁 (Locking) ---
                # 这行代码是防止重启后重复执行的关键
                self.daily_pnl_us_rth_last_reset_date = today_us_str
                
                need_save = True

            if need_save:
                self._save_positions()
                self._save_daily_pnl_state()
                logger.info(f"✅ 状态检查与重置完成 (Save Triggered). RTH Lock: {self.daily_pnl_us_rth_last_reset_date}")
                

    def _execute_full_sell(self, symbol: str, reason: str):
        """执行全仓卖出操作 (非阻塞)"""
        with self.position_lock:
            position = self.positions.get(symbol)
            if not position: return
            if position.pending_sell_order_id:
                logger.info(f"{symbol} 已存在待处理的卖出订单，忽略新的全仓卖出指令。")
                return
            quantity_to_sell = position.total_quantity
        
        symbol_info = self.get_cached_stock_static_info(symbol)
        quantity_to_sell = self._adjust_quantity(quantity_to_sell, position.market, lot_size=symbol_info.get('lot_size', 100))
        if quantity_to_sell <= 0:
            self._cleanup_position(symbol, f"仓位为0或计算后为0，直接清理 ({reason})")
            return

        logger.warning(f"准备全仓卖出 {symbol} | 原因: {reason} | 数量: {quantity_to_sell}")
        try:
            current_price = self.get_current_price(symbol) or 0.0
            symbol_info = self.get_cached_stock_static_info(symbol)
            symbol_name = symbol_info.get('name_cn') or symbol_info.get('name_en')
            sell_logger.info(f"symbol:{symbol},name:{symbol_name},price:{current_price:.3f},quantity:{quantity_to_sell},reason:全仓卖出 - {reason}")
        except Exception as e:
            logger.error(f"记录卖出日志时发生异常: {e}")

        order_id = self.submit_order(symbol, quantity_to_sell, OrderSide.Sell)
        if order_id:
            with self.position_lock:
                if symbol in self.positions:
                    pos = self.positions[symbol]
                    pos.pending_sell_order_id = order_id
                    pos.sell_reason = "清仓: " + reason
                    self._save_positions()

    def _execute_extended_hours_sell(self, symbol: str, reason: str, sell_ratio: float = 1.0):
        """
        【全时段通用卖出执行器】
        
        修正说明:
        针对盘前/盘后/夜盘交易的特殊性（低流动性、必须限价单），增加了“撤单重发”机制。
        解决了因旧挂单未成交导致 pending_sell_order_id 长期占用，进而阻塞新策略执行的死锁问题。
        
        适用场景：盘后(-2)、夜盘(-3)、盘前(-1)。
        """
        # 1. 先在锁内获取必要信息，然后立刻释放锁
        with self.position_lock:
            # 重新获取最新状态
            position = self.positions.get(symbol)
            if not position: return
            
            # ▼▼▼【豁免名单检查】▼▼▼
            if position.triggering_strategy in self.config.strategies_immune_to_exit:
                logger.warning(
                    f"🛡️ [盘外卖出豁免] 策略 '{position.triggering_strategy}' 拥有免死金牌，"
                    f"拒绝执行盘外卖出 {symbol} (原因: {reason})。"
                )
                return
            
            if position.pending_sell_order_id:
                logger.warning(
                    f"🔒 [{symbol}] 并发锁拦截 | "
                    f"当前状态: 有在途订单/正在提交 ({position.pending_sell_order_id}) | "
                    f"被拒操作: {reason}"
                )
                return

            # 占位日志 (Debug级别即可，证明开始抢锁)
            logger.warning(f"🔒 [{symbol}] 正在获取发单锁 (Setting GUARD)...")
            position.pending_sell_order_id = "SUBMITTING_GUARD"

        # 2. 在锁外处理撤单 (IO 和 Sleep)
        # if pending_order_id:
        #     logger.warning(f"[{symbol}] 盘外交易检测到挂单 {pending_order_id}，尝试撤单重发...")
        #     try:
        #         self.trade_ctx.cancel_order(pending_order_id)
        #         time.sleep(0.5)
                
        #         # 检查订单状态 (IO)
        #         check_order = self.trade_ctx.order_detail(pending_order_id)
        #         if check_order.status == OrderStatus.Filled:
        #             logger.warning(f"[{symbol}] 撤单失败，订单已成交，跳过重发。")
        #             return
        #     except Exception as e:
        #         logger.error(f"[{symbol}] 撤销盘外旧挂单失败: {e}。终止本次操作以防风险。")

        #     # 3. 再次获取锁，更新状态 (清理ID)
        #     with self.position_lock:
        #         position = self.positions.get(symbol)
        #         # 必须确认 ID 没变 (防止并发修改)
        #         if position and position.pending_sell_order_id == pending_order_id:
        #             position.pending_sell_order_id = None
        #             position.sell_reason = None
        #             self._save_positions()
                
        try:
            # --- 步骤 2: 获取全时段智能报价 ---
            current_price = self.get_realtime_price(symbol)
            # --- 调用智能计算器 ---
            # 这里的逻辑被完全替换了，不再手动算 raw_sell_quantity
            sell_quantity, adjust_msg = self._calculate_smart_sell_quantity(
                symbol, position, sell_ratio, current_price
            )
            
            # if sell_quantity <= 0:
            #     logger.warning(f"[{symbol}] 盘外卖出计算量为0 (原因: {adjust_msg})，跳过。")
            #     self._rollback_pending_status(symbol) # 记得回滚锁！
            #     return

            if adjust_msg:
                logger.warning(f"🌙 [{symbol}] 盘外卖出修正: {adjust_msg} -> 全平")
                # reason = f"{reason} [修正:全平]"
            
            # --- 步骤 3: 挂单价格策略 ---
            # 盘外交易流动性差，为了确保卖出，通常挂“最新价”或略低。
            # 这里使用最新价作为限价，这是最稳妥的策略。
            market = get_market_type(symbol)
            if market == MarketType.US:
                # 美股绝大多数情况是 2位小数 ($0.01)
                # 使用 format 确保四舍五入并转为字符串，然后再转 Decimal
                price_str = "{:.2f}".format(current_price)
            elif market == MarketType.HK:
                # 港股不同价位 tick 不同，但保留 3位小数 通常是 API 的安全边界
                price_str = "{:.3f}".format(current_price)
            else:
                # 其他市场默认 2位
                price_str = "{:.2f}".format(current_price)

            limit_price = Decimal(price_str)
            
            # --- 步骤 4: 确定操作标签与数量 ---
            if sell_ratio >= 1.0:
                action_tag = "全仓清算"
                sell_ratio = 1.0
            else:
                action_tag = f"减仓{int(sell_ratio * 100)}%"

            # --- [风控] 小数额边界处理 ---
            if sell_quantity <= 0:
                if sell_ratio < 1.0:
                    logger.warning(f"[{symbol}] 仓位过小({position.total_quantity})无法执行{action_tag}，跳过。")
                    # ✅ 必须在这里回滚锁状态，否则死锁！
                    self._rollback_pending_status(symbol)
                    return
                else:
                    # 全仓清算但算出来是0，说明也就是碎股或数据误差，清理掉
                    self._cleanup_position(symbol, f"仓位计算后为0，清理记录 ({reason})")
                    return

            logger.warning(f"准备执行【{action_tag}】 {symbol} | 数量: {sell_quantity} | 限价: {limit_price} | 原因: {reason}")
            
            # --- 步骤 6: 提交订单 (LO 限价单是盘外交易的唯一选择) ---
            order_id = self.submit_order_lo(symbol, sell_quantity, OrderSide.Sell, limit_price)
            
            if order_id:
                with self.position_lock:
                    # 重新获取对象，防止并发修改（虽然上面加了锁，但 submit_order_lo 是耗时操作，期间锁是释放的）
                    pos = self.positions.get(symbol)
                    if pos:
                        pos.pending_sell_order_id = order_id
                        pos.sell_reason = f"盘外[{action_tag}]: {reason}"
                        pos.add_sell_record(float(current_price), sell_quantity, f"盘外[{action_tag}]: {reason}")
                        self._save_positions()
                        self.extended_hours_order_timers[symbol] = time.time()
                        # 锁升级成功
                        logger.warning(f"✅ [{symbol}] 卖单提交成功，ID: {order_id}")
                
                # ▼▼▼ 最小成本插入通知 (注意：在锁释放后发送) ▼▼▼
                self.notification_manager.send_trade_execution(
                    action="SELL (NIGHT)",
                    symbol=symbol,
                    quantity=sell_quantity,
                    price=float(limit_price),
                    reason=reason
                )

                try:
                    symbol_info = self.get_cached_stock_static_info(symbol)
                    symbol_name = symbol_info.get('name_cn') or symbol_info.get('name_en')
                    sell_logger.info(f"symbol:{symbol},name:{symbol_name},price:{current_price},strategy_reason:{reason},llm_reason:N/A(ExtendedHours)")
                except Exception as log_e:
                    logger.error(f"[{symbol}] 盘外卖出日志记录失败: {log_e}")

            else:
                # 提交失败
                logger.error(f"❌ [{symbol}] 卖单提交返回空ID，准备回滚锁...")
                self._rollback_pending_status(symbol)
        except Exception as e:
            # 异常回滚
            logger.error(f"💥 [{symbol}] 卖出流程发生异常: {e}，正在强制回滚锁状态！", exc_info=True)
            self._rollback_pending_status(symbol)
    
        # 智能黑名单触发逻辑
        # 检查卖出原因，如果是负面原因（舆情、刚性止损、崩盘），则拉黑
        # 避免正常的“止盈”操作导致拉黑
        critical_keywords = ["舆情","崩盘","闪崩"] # "刚性止损", "死线","强制清仓",
        if any(kw in reason for kw in critical_keywords):
            if symbol not in self.intraday_blacklist:
                self.intraday_blacklist.add(symbol)
                self.intraday_blacklist.add(resolve_underlying_symbol(symbol))
                self._save_blacklist()
                logger.critical(f"🚫 [{symbol}] 因盘外严重负面离场 ({reason})，已加入黑名单！")

    def _rollback_pending_status(self, symbol: str):
        with self.position_lock:
            pos = self.positions.get(symbol)
            if pos and pos.pending_sell_order_id == "SUBMITTING_GUARD":
                pos.pending_sell_order_id = None
                logger.warning(f"🔄 [{symbol}] 状态锁已回滚 (GUARD -> None)，允许后续重试。")

    def execute_tactical_clearance(self, sell_ratio: float = 1.0, mode: str = 'ALL'):
        """
        【统一清仓矩阵】(Global Clearance Matrix)
        支持无差别清仓 (ALL) 和 仅卖盈利 (PROFIT_ONLY)。
        """
        targets =[]
        with self.position_lock:
            if not self.positions: return
            targets = list(self.positions.keys())

        logger.critical(f"☢️[统一清仓执行] 模式: {mode} | 比例: {sell_ratio*100}% | 目标扫描数: {len(targets)}")

        for symbol in targets:
            try:
                if not self._is_pure_stock(symbol): continue
                
                # --- 如果是“仅卖盈利”模式，检查是否盈利 ---
                if mode == 'PROFIT_ONLY':
                    current_price = self.get_current_price(symbol)
                    pos = self.positions.get(symbol)
                    if not current_price or not pos: continue
                    
                    real_cost = pos.get_avg_cost(self.config)
                    # 如果现价 <= 成本，说明是亏损的，直接跳过！
                    if current_price <= real_cost:
                        logger.info(f"⏭️ [一键清仓-仅盈利] {symbol} 处于亏损(现价{current_price:.2f} <= 成本{real_cost:.2f})，跳过。")
                        continue
                
                market = get_market_type(symbol)
                is_us_rth = (market == MarketType.US and is_market_in_trading_hours(MarketType.US))
                
                reason_tag = f"【指令清仓({mode})】"
                
                if is_us_rth:
                    # 盘中调用
                    self.process_sell_signal(symbol, percentage=sell_ratio, reason=f"{reason_tag}(盘中)")
                    self.notification_manager.send_critical_alert(f'股票:{symbol},{reason_tag}(盘中),卖出比例:{sell_ratio}')
                else:
                    # 盘外调用
                    self._execute_extended_hours_sell(symbol, reason=f"{reason_tag}(盘外)", sell_ratio=sell_ratio)
                    self.notification_manager.send_critical_alert(f'股票:{symbol},{reason_tag}(盘外),卖出比例:{sell_ratio}')
                    
            except Exception as e:
                logger.error(f"❌ [清仓执行] {symbol} 失败: {e}", exc_info=True)

        
    def _handle_pending_sell_order(self, pos: Position):
        """
        处理待卖出订单的成交回报。
        修正：严格分离 锁内状态读取 -> 锁外IO操作 -> 锁内状态更新。杜绝死锁与阻塞。
        """
        # --- 阶段 1: 快速读取状态 (锁内) ---
        # 这里的 pos 引用虽然在手，但为了线程安全，最好还是短暂加锁读取 ID
        # (或者假设 pos 对象本身由 position_lock 保护，外部调用时通常是从 copy 列表来的)
        order_id = pos.pending_sell_order_id
        symbol = pos.symbol
        
        if not order_id:
            return

        # --- 阶段 2: 执行网络 I/O (锁外 - 关键！！！) ---
        # 此时不持有锁，其他线程可以自由读写 positions
        try:
            logger.debug(f"正在检查 {symbol} 的待处理卖出订单: {order_id}")
            kwargs = {
                'symbol': symbol
            }
            o_detail = self.data_provider.get_order_detail(order_id, **kwargs)
            status = o_detail.get('status')
        except Exception as e:
            logger.error(f"查询订单 {order_id} 失败: {e}")
            return  # 网络错误直接跳过，下一轮再试

        # --- 阶段 3: 更新状态 (锁内) ---
        # 必须重新获取锁，并再次确认状态未被其他线程改变
        is_terminal = status in ['Filled','Canceled', 'Rejected', 'Expired']
        
        if is_terminal:
            with self.position_lock:
                # 再次从主字典获取最新的对象，防止 pos 引用过期
                current_pos = self.positions.get(symbol)
                
                # 双重检查：确保 ID 没变 (防止并发撤单/重发导致 ID 变更)
                if not current_pos or current_pos.pending_sell_order_id != order_id:
                    logger.warning(f"[{symbol}] 订单状态已变更或被并发处理，跳过本次更新。")
                    return

                if status == 'Filled':
                    filled_quantity = int(o_detail.get('quantity', 0))
                    filled_price = float(o_detail.get('price', 0.0))
                    exit_reason = current_pos.sell_reason or "策略卖出"

                    # 1. 更新持仓状态 (内存)
                    current_pos.pending_sell_order_id = None
                    current_pos.sell_reason = None
                    current_pos.total_quantity -= filled_quantity
                    current_pos.add_sell_record(filled_price, filled_quantity, exit_reason)
                    if current_pos.strategy_params is None or not isinstance(current_pos.strategy_params, dict):
                        current_pos.strategy_params = {}
                    conservative_state = current_pos.strategy_params.setdefault('conservative_exit_state', {})
                    if "浮盈达到10%" in exit_reason:
                        conservative_state['stage_10_taken'] = True
                    if "浮盈达到15%" in exit_reason:
                        conservative_state['stage_15_taken'] = True

                    # 判断是清仓还是部分卖出
                    action = "LIQUIDATE" if current_pos.total_quantity <= 0 else "PARTIAL SELL"
                    
                    if action == "LIQUIDATE":
                        logger.warning(f"✅ {symbol} 清仓完成。")
                        self._archive_completed_trade(current_pos, filled_price, filled_quantity, exit_reason)
                        self._cleanup_position(symbol, f"成交清仓 ({exit_reason})")
                    else:
                        logger.warning(f"✅ {symbol} 部分卖出 {filled_quantity} 股。")
                        current_pos.partial_sell_price = filled_price
                        current_pos.highest_price_since_partial_sell = filled_price
                        self._save_positions() # 保存持仓变动
                    
                    # 2. 计算已实现盈亏 (Realized PnL)
                    # 统一使用 (卖出价 - 成本价) 计算
                    trade_pnl = (filled_price - current_pos.avg_cost) * filled_quantity
                    
                    if current_pos.market == MarketType.HK:
                        self.daily_realized_pnl_hk += trade_pnl
                        self.daily_realized_pnl_hk -= getattr(self.config, 'hk_roundtrip_fixed_fee', 30.0)
                    elif current_pos.market == MarketType.US:
                        self.daily_realized_pnl_us += trade_pnl
                        self.daily_realized_pnl_us -= getattr(self.config, 'us_roundtrip_fixed_fee', 1.21)* getattr(self.config, 'exchange_rate_usd_to_hkd','7.8')
                    
                    logger.info(f"💰 [落袋为安] {symbol} 卖出 {filled_quantity}股，单笔盈亏: {trade_pnl:.2f}")
                    # 明确打印锁释放
                    logger.warning(
                        f"🔓 [{symbol}] 卖单完全成交 | "
                        f"ID: {order_id} | "
                        f"操作: 释放 pending_sell_order_id 锁 | "
                        f"盈亏: {trade_pnl:.2f}"
                    )

                    # 3. ★★★ 状态持久化 (最关键的一步) ★★★
                    # 必须在发通知之前保存，防止通知失败导致数据丢失
                    self._save_daily_pnl_state()

                    # 4. 发送通知 (非阻塞)
                    self.notification_manager.send_trade_execution(action, symbol, filled_quantity, filled_price, exit_reason)

                else:
                    # 处理失败/取消/拒绝
                    logger.warning(f"❌ {symbol} 卖出订单异常结束: {status}")
                    current_pos.pending_sell_order_id = None
                    current_pos.sell_reason = None
                    self._save_positions()
                    # 明确打印异常状态下的锁释放
                    logger.warning(
                        f"🔓 [{symbol}] 卖单异常终结 ({status}) | "
                        f"ID: {order_id} | "
                        f"操作: 强制释放锁，等待下一次机会"
                    )

    def _cleanup_position(self, symbol: str, reason: str):
        """从系统中移除一个持仓"""
        with self.position_lock, self.notification_lock: 
            if symbol in self.positions:
                del self.positions[symbol]
                self._save_positions()
                logger.info(f"已清理持仓记录: {symbol} | 原因: {reason}")

                # ==============================================================================
                # ▼▼▼【核心植入 1/2】夜盘状态联动重置 ▼▼▼
                # 逻辑：一旦仓位离场，必须立即解除夜盘的'BOUGHT'锁定，否则今晚无法再次捕捉机会。
                # ==============================================================================
                if symbol in self.pre_market_states:
                    state = self.pre_market_states[symbol]
                    # 仅当状态为 BOUGHT 时才重置，避免干扰 REBOUNDING 中的信号
                    if state.get('status') == 'BOUGHT':
                        logger.warning(f"🔄 [{symbol}] 仓位清理联动触发: 夜盘状态机 BOUGHT -> WATCHING")
                        state['status'] = 'WATCHING'
                        # 注意：我们保留 session_low 不重置，防止在同一低点反复震荡止损
                        # 但必须更新时间戳，确保状态活跃
                        state['last_update_ts'] = time.time()
                        self._save_pre_market_states() # 立即持久化，防止断电丢失
                # ▲▲▲【核心植入 1/2】结束 ▲▲▲

                # 在这里重置该股票的通知计数器
                if symbol in self.buy_signal_notifications:
                    del self.buy_signal_notifications[symbol]
                    logger.info(f"已重置 {symbol} 的买入信号通知计数器，允许未来再次触发。")
    
                # =======================================================
                # ▼▼▼ [舆情风控计时器 GC (垃圾回收) ▼▼▼
                # 必须清理！否则如果几小时后又买回该股，会继承脏数据。
                # =======================================================
                if symbol in self.sentiment_cache_timers:
                    del self.sentiment_cache_timers[symbol]
                    logger.debug(f"🧹 已销毁 {symbol} 的舆情风控计时器，内存状态归零。")

    # ==============================================================================
    # V. 仓位调整与加仓 (Position Sizing & Scaling)
    # ==============================================================================
    def _check_scale_in_conditions(self, symbol: str, current_price: float, position: Position):
        """检查并执行“建仓期”的加仓逻辑 (3-3-4 模型核心)。"""
        logger.info(f"[{symbol}] 保守策略模式：补仓检查已禁用。")
        return

        if symbol in self.risky_second_tier_stocks:
            return
        # ==============================================================================
        # ▼▼▼【空头/波动率产品 补仓物理熔断】▼▼▼
        # 逻辑：做空工具(SQQQ)和恐慌指数(UVIX)具有极高的损耗性和爆发性。
        # 它们的交易逻辑是“一击必杀”或“快进快出”，绝对禁止在亏损时进行金字塔补仓。
        # 如果方向错了，直接止损，不允许摊薄成本。
        # ==============================================================================
        # if self._is_bearish_symbol(symbol):
        #     return
        
        # ==============================================================================

        # ==============================================================================
        # ▼▼▼【盈利达标熔断】▼▼▼
        # 逻辑：一旦触发了 Stage 1 (80%净化) 或 Stage 2 (100%收割)，
        # 说明账户进入“防守/收尾”阶段，绝对禁止对亏损股进行逆势补仓！
        # ==============================================================================
        is_hk_hit = (position.market == MarketType.HK and self.daily_hk_profit_target_hit)
        is_us_hit = (position.market == MarketType.US and self.daily_us_profit_target_hit)
        
        # 如果 [半程预警已触发] 或 [当日目标已达成]，直接熔断补仓逻辑
        if self.half_target_escape_triggered or is_hk_hit or is_us_hit:
            # logger.debug(f"[{symbol}] 补仓被熔断：账户已进入盈利治理阶段 (Stage 1/2)，禁止逆势加仓。")
            return

        # ==============================================================================
        # ▼▼▼【趋势矛盾锁 (Momentum Contradiction Lock)】▼▼▼
        # 逻辑：如果动能止损模块已经侦测到"持续新低"（哪怕只有1次），
        # 说明当前正处于加速下跌的刀口上。此时逆势补仓是送死行为。
        # 必须等待计数器归零（自然衰减或V反）后，才能进场。
        # ==============================================================================
        kinetic_cfg = self.config.kinetic_stop_loss_config
        if kinetic_cfg.get('enabled', False) and  position.consecutive_new_low_periods > 0:
            # 只有 debug 级别，或者极低频 warning，防止刷屏
            # 既然是全球第二的系统，我们就用 warning 提醒你它在保护你
            if random.random() > 0.9:
                logger.warning(f"🛡️ [{symbol}] 补仓被动能锁拦截：正处于连续新低周期 (Count={position.consecutive_new_low_periods})，拒绝接飞刀。")
            return
        
        # ==============================================================================
        # ▼▼▼【相位锁定机制】冲突熔断 ▼▼▼
        # ------------------------------------------------------------------------------
        # 疑问1解答：如果进入了防御模式（即跌破过 ATR 1.0），说明趋势转弱。
        # 此时绝对禁止“越跌越买”的常规补仓，防止并在伤口上撒盐。
        if position.is_defense_mode_active:
            # 唯一的解锁机会：价格V反，重新站上成本价
            if current_price > position.get_avg_cost(self.config):
                logger.info(f"🎉 [{symbol}] 价格修复重回成本之上，解除防御模式，恢复常规补仓功能。")
                position.is_defense_mode_active = False
                position.current_grid_level = 0
                self._save_positions()
            else:
                # logger.debug(f"[{symbol}] 处于防御相位中，常规补仓(Dip Add)被熔断。")
                return
        
        # 检查当前仓位的触发策略是否在“补仓白名单”中
        # allowlist = self.config.strategy_scale_in_allowlist
        # if not self.config.enable_scale_in_strategy or not allowlist.get(position.triggering_strategy, False): return
        if not self.config.enable_scale_in_strategy: return
        if position.overall_phase != PositionOverallPhase.BUILDING: return # 正在有挂单，直接退出
        if position.pending_pyramid_order_id or position.pending_sell_order_id: return
        
        # 获取该仓位的特定限制
        # max_dip_adds = position.strategy_params.get('max_dip_adds', 999) # 默认 999 不限制
        # if position.dip_adds_done >= max_dip_adds:
        #     # 这是一个静默拦截，不需要 error log，debug 即可
        #     # logger.debug(f"[{symbol}] 达到最大补仓次数限制 ({max_dip_adds})，停止补仓。")
        #     return
        
        # 检查是否已满仓
        if position.total_quantity >= position.planned_total_quantity:
            logger.warning(f"仓位 {symbol} 已满，但阶段仍是BUILDING。自动切换至RUNNING。")
            position.overall_phase = PositionOverallPhase.RUNNING
            self._save_positions()
            return
        # +++ 3-3-4 下跌补仓逻辑 (Dip Add) +++
        if self.config.scale_in_on_dip_enabled:
            # +++  ATR自适应补仓逻辑 +++
            if self.config.enable_atr_dip_add:
                # 只支持一次ATR补仓，更符合“抓住黄金坑”的原则，避免连续下跌连续补。
                if position.dip_adds_done == 0:
                    atr_value = get_historical_atr(self.quote_ctx, symbol)
                    if atr_value and atr_value > 0:
                        trigger_price = position.initial_price * (1 - self.config.dip_add_trigger_atr_multiple * (atr_value / position.initial_price))
                        if current_price <= trigger_price:
                            current_status = get_trading_window_status(symbol)
                            favorable_buy_windows = self.config.favorable_buy_windows
                            if current_status in favorable_buy_windows:
                                logger.warning(f"触发ATR下跌补仓 for {symbol}: 现价({current_price:.2f}) <= ATR触发价({trigger_price:.2f})")
                                if self._handle_dip_add(symbol, position): return
                    else:
                        logger.warning(f"无法获取 {symbol} 的ATR值，跳过ATR补仓检查。")
            # --- [改良版固定百分比 + J-Hook 右侧确认] ---
            else:
                # 获取配置的阈值列表，例如 [0.012, 0.022]
                triggers = self.config.dip_add_triggers_percent
                num_triggers = len(triggers)
                
                # 当前完成了几次补仓？
                # 0次 -> 准备打第二枪 (Reinforce 30%)
                # 1次 -> 准备打第三枪 (Sniper 40%)
                current_step_index = position.dip_adds_done

                if current_step_index < num_triggers:
                    # 1. 计算触发价格
                    # 注意：这里是基于 initial_price (首仓价格) 计算跌幅，这符合你的“总浮亏”逻辑
                    trigger_drop_pct = triggers[current_step_index]
                    trigger_price = position.initial_price * (1 - trigger_drop_pct)

                    # --- 初始化运行时状态 (Runtime State Injection) ---
                    # 如果状态为空，或者状态里的层级与当前实际层级不符(说明刚补完上一枪)，强制初始化为 None
                    if position.dip_pending_state:
                        if position.dip_pending_state.get('step_index') != current_step_index:
                            position.dip_pending_state = None
                            self._save_positions() # 状态清理落盘

                    # =================================================================
                    # 阶段 I: IDLE -> PENDING (触价监听)
                    # =================================================================
                    if position.dip_pending_state is None:
                        # 只有价格跌破触发线，才初始化状态字典
                        if current_price <= trigger_price:
                            position.dip_pending_state = {
                                'status': 'PENDING',
                                'session_low': current_price,
                                'step_index': current_step_index,
                                'trigger_ts': time.time()
                            }
                            self._save_positions() # 立即保存，防止重启丢失观测状态
                            
                            logger.warning(f"👀 [{symbol}] 触及补仓线({trigger_price:.3f})，激活 Pending_Scale_In。当前低点: {current_price:.3f}")
                            # 不 return，直接流转到下方进行 PENDING 判断

                    # =================================================================
                    # 阶段 II: PENDING (探底、熔断与确认)
                    # =================================================================
                    if position.dip_pending_state and position.dip_pending_state['status'] == 'PENDING':
                        p_state = position.dip_pending_state
                        # A. 实时更新最低点 (Bottom Finding)
                        # 只要还在创新低，就更新记录，并且绝对不买
                        if current_price < p_state['session_low']:
                            p_state['session_low'] = current_price
                            # 创新低，之前的任何确认计时都作废，必须重置！
                            if p_state.pop('confirm_start_ts', None):
                                self._save_positions()
                            # if 'confirm_start_ts' in p_state:
                            #     del p_state['confirm_start_ts'] 
                            #     self._save_positions() # 状态变更需保存
                            
                            return 

                        # 2. 安全阀检查 (Safety Valve - 动能熔断)
                        # 逻辑：如果 5分钟内跌幅过大 (>2%)，视为瀑布流，强制挂起
                        is_crashing = False
                        try:
                            # 获取最近 2 根 5分钟 K线 (足以覆盖过去 5-10分钟)
                            klines_5m = get_klines_data(self.quote_ctx, symbol, 2, Period.Min_5, AdjustType.NoAdjust)
                            if klines_5m is not None and not klines_5m.empty:
                                # 简单粗暴：最高价 vs 现价
                                recent_high = klines_5m['high'].max()
                                if recent_high > 0:
                                    drop_velocity = (recent_high - current_price) / recent_high
                                    if drop_velocity > 0.02: # 2% 熔断阈值
                                        is_crashing = True
                                        if random.random() > 0.9: # 降低日志频率
                                            logger.warning(f"🛑 [{symbol}] 动能熔断: 5分钟急跌 {drop_velocity:.2%} (>2%)，暂停补仓。")
                        except Exception:
                            pass # 数据获取失败暂不熔断，依赖 J-Hook 兜底

                        if is_crashing:
                            return # 熔断生效，本轮不买

                        # 3. 执行扳机 (Trigger Execution - J-Hook)
                        # 需求: Current > Low + (ATR * 0.1)
                        
                        # 获取 ATR
                        # atr_val = get_historical_atr(self.quote_ctx, symbol)
                        atr_val = None
                        
                        # 计算反弹阈值 (Delta)
                        if atr_val and atr_val > 0:
                            rebound_delta = atr_val * 0.1 # 方案A的核心参数
                        else:
                            # 兜底：如果没 ATR，要求反弹 0.3% (普通股) 或 0.5% (高波股)
                            fallback_ratio = 0.005 if symbol in self.config.high_vol_symbols else 0.003
                            rebound_delta = p_state['session_low'] * fallback_ratio
                        
                        # 目标确认价
                        confirmation_price = p_state['session_low'] + rebound_delta
                        
                        # === 最终裁决 ===
                        # --- 统一执行 (Execution) ---
                        if current_price > confirmation_price:
                            # 补仓不仅要看反弹，还要看是不是踩在“钢板”上
                            is_dip_safe, dip_msg = self._verify_trade_quality_gate(symbol, current_price, mode='DIP_ADD')
                            
                            if not is_dip_safe:
                                if random.random() > 0.8:
                                    logger.warning(f"🚫 [{symbol}] 下跌补仓被物理支撑拦截: {dip_msg}")
                                return # 撤退，等真正踩到 Put Wall 再说
                            
                            now_ts = time.time()
                            REQUIRED_CONFIRM_SECONDS = 120 # 硬编码 2分钟，或从 config 读取

                            # 1. 首次触发，记录时间
                            if 'confirm_start_ts' not in p_state:
                                p_state['confirm_start_ts'] = now_ts
                                self._save_positions() # 必须保存，防止重启丢失
                                logger.warning(f"⏳ [{symbol}] 补仓形态初现，开始 {REQUIRED_CONFIRM_SECONDS}s 确认计时... (现价 {current_price:.3f} > 线 {confirmation_price:.3f})")
                                return # 等待下一轮

                            # 2. 检查时间是否达标
                            elapsed = now_ts - p_state['confirm_start_ts']
                            if elapsed < REQUIRED_CONFIRM_SECONDS:
                                # logger.debug(f"[{symbol}] 补仓确认中: {elapsed:.0f}/{REQUIRED_CONFIRM_SECONDS}s")
                                return # 继续等待

                            # ==============================================================================
                            # ▼▼▼【核心卫兵：水位熔断 (High Water Mark Fuse)】▼▼▼
                            # 逻辑：J-Hook 虽然确认了，但如果此时价格暴力拉升，已经涨回了初始建仓价(成本区)，
                            # 那么所谓的"Dip(坑)"已经被填平了。此时买入不仅不是"低吸"，反而是"追高"。
                            # 必须立即熔断，将控制权交还给"上涨追涨(Rise Add)"模块。
                            # ==============================================================================
                            if current_price >= position.initial_price:
                                logger.warning(
                                    f"🛑 [{symbol}] 补仓熔断: 价格V反过猛，已修复跌幅! "
                                    f"现价 {current_price:.3f} >= 初始价 {position.initial_price:.3f}。取消补仓，重置状态。"
                                )
                                # 销毁补仓状态，视为本次下探结束
                                position.dip_pending_state = None
                                self._save_positions()
                                return

                            # 5. 风控检查：是否跌破硬止损？
                            # 如果当前价已经低于初始止损价，禁止补仓，并在主循环的止损逻辑中处理
                            if position.initial_stop_loss_price and current_price <= position.initial_stop_loss_price:
                                logger.warning(f"❌ [{symbol}] 满足补仓反弹，但价格已破止损线，放弃救援。重置状态。")
                                # logger.warning(f"[{symbol}] 虽然满足补仓形态，但已跌破硬止损线，禁止补仓！")
                                position.dip_pending_state = None
                                self._save_positions()
                                return

                            # 从配置中读取比例列表，防止越界默认取最后一个
                            ratios = self.config.scale_in_step_ratios
                            if current_step_index < len(ratios):
                                ratio = ratios[current_step_index]
                            else:
                                ratio = ratios[-1] # 兜底
                            
                            is_mid_morning_stabilization = is_in_custom_trading_window(
                                    market=MarketType.US,
                                    start_minutes=60,
                                    end_minutes=90
                                )
                            favorable_add_windows = self.config.favorable_add_windows
                            current_status = get_trading_window_status(symbol)
                            is_favorable_add_time = current_status in favorable_add_windows
                            if is_favorable_add_time or is_mid_morning_stabilization:
                                ratio = ratio * 1.02
                            else:
                                ratio = ratio * 0.80
                            
                            if symbol in self.clean_second_tier_stocks:
                                ratio = ratio * 0.60

                            # 6. 计算补仓数量 (核心 3-3-4 比例)
                            # 计划总数 * 比例
                            if current_step_index == 0:
                                # 第二枪: 30%
                                quantity_to_add = int(position.planned_total_quantity * ratio)
                                action_name = "第二枪(Reinforce)"
                            else:
                                # 第三枪: 40% (或者剩余所有)
                                quantity_to_add = int(position.planned_total_quantity * ratio)
                                action_name = "第三枪(Sniper)"
                            
                            # 7. 调整手数并下单
                            symbol_info = self.get_cached_stock_static_info(symbol)
                            quantity_to_add = self._adjust_quantity(
                                quantity_to_add, position.market, lot_size=symbol_info.get('lot_size', 100)
                            )
                            
                            if quantity_to_add > 0:
                                log_msg = (
                                    f"🚀 [{symbol}] J-Hook确认! {action_name} 触发 | "
                                    f"低点:{p_state['session_low']:.3f} -> 现价:{current_price:.3f} "
                                    f"(反弹 > {rebound_delta:.3f})"
                                )
                                logger.warning(log_msg)

                                # 调用 execute，reason 标记为 dip_add_1 或 dip_add_2
                                success = self._execute_add_position(
                                    symbol,
                                    quantity_to_add,
                                    position, 
                                    f"dip_add_{current_step_index + 1}"
                                )
                                if success:
                                    # 成功后重置状态机
                                    position.dip_pending_state = None
                                    self._save_positions()
                                return
                        
                        # [情况 B]: 价格跌回确认线之下 (但在新低之上)
                        # 这意味着反弹是骗线，或者还在震荡。必须重置计时器！
                        elif current_price <= confirmation_price:
                            if p_state.pop('confirm_start_ts', None):
                                logger.info(f"💨 [{symbol}] 补仓反弹夭折 (现价 {current_price:.3f} <= 线 {confirmation_price:.3f})，计时器重置。")
                                self._save_positions()

                            # if 'confirm_start_ts' in p_state:
                            #     logger.info(f"💨 [{symbol}] 补仓反弹夭折 (现价 {current_price:.3f} <= 线 {confirmation_price:.3f})，计时器重置。")
                            #     del p_state['confirm_start_ts']
                            #     self._save_positions()
                            return
                        
        if self.config.scale_in_on_rise_enabled and not position.rise_add_done:
            if position.initial_risk_per_share > 0 and position.initial_scout_price > 0:
                # 条件1：必须已盈利至少0.8R（约+3.2%）
                required_profit = position.initial_risk_per_share * self.config.rise_add_trigger_r_multiple
                current_profit = current_price - position.initial_scout_price
                
                if current_profit >= required_profit:
                    # 条件2：股价必须创近5日新高（趋势延续确认）
                    is_trend_confirmed = is_making_new_high(self.quote_ctx, symbol, days=5)
                    
                    # 条件3：未进入超买区（RSI < 70）
                    rsi = get_rsi(self.quote_ctx, symbol, period=14)
                    
                    if is_trend_confirmed and (rsi is None or rsi < 70):
                        rsi_display = f"{rsi:.1f}" if rsi is not None else "N/A"
                        
                        logger.warning(
                            f"触发'赢家确认'加仓 for {symbol}: "
                            f"盈利({current_profit:.2f}) >= 目标({required_profit:.2f}), "
                            f"创新高✓, RSI={rsi_display}"
                        )
                        
                        if self._handle_rise_add(symbol, position):
                            return
                    else:
                        if not is_trend_confirmed:
                            logger.debug(f"[{symbol}] 虽已盈利，但未创新高，暂不加仓。")
                        if rsi and rsi >= 70:
                            logger.debug(f"[{symbol}] RSI超买({rsi:.1f})，暂不加仓。")
                    
                    # logger.warning(f"触发上涨追涨 for {symbol}: 盈利({current_profit:.2f}) >= 目标({required_profit:.2f})")
                    # if self._handle_rise_add(symbol, position): return
    
    def _check_resurrection_opportunity(self, symbol: str, current_price: float, position: Position):
        """
        【回补机制 (Resurrection)】 - T+0 降本策略
        
        核心逻辑：
        利用网格止损释放的现金，在更低位接回筹码，实现成本摊薄。
        
        严厉纪律：
        1. 禁止追高接回 (No Chasing High): 必须比卖出价便宜。
        2. 价差护城河 (Spread Buffer): 必须有足够的利润空间。
        3. 技术确认 (Tech Confirm): 必须在低位企稳。
        """
        if not position.is_defense_mode_active or not position.resurrection_cache:
            return

        # 遍历缓存，寻找复活机会
        # 使用倒序，优先处理最近一次的止损，这样资金效率最高
        for record in reversed(position.resurrection_cache):
            if record['status'] != 'WAITING':
                continue
            
            sell_price = record['sell_price']
            qty_to_buy = record['qty']
            level = record['level']
            
             # --- 纪律 1: 反向追高熔断 (The Anti-Whipsaw Rule) ---
            # 逻辑：如果现价已经接近或高于当初的卖出价，说明那是错误的离场，或者V反太快。
            # 此时绝不追高接回，承认踏空，防止遭受“双重耳光”。
            # 阈值：卖出价的 99%。如果现价 > 卖出价 * 0.99，直接忽略。
            if current_price >= sell_price * 0.99:
                # logger.debug(f"[{symbol}] 回补被拒: 现价({current_price}) 接近/高于 卖出价({sell_price})，拒绝做负T。")
                continue 

            # --- 纪律 2: 价差护城河 (Hysteresis Buffer) ---
            # 逻辑：除了不追高，还必须跌得够深。
            # 读取配置中的回补缓冲 (例如 1.5%)
            required_dip_price = sell_price * (1 - 0.015)
            
            if current_price > required_dip_price:
                continue # 跌幅不够，不予理会
            
            # --- 过滤器 2: 技术结构确认 (Technical Confirmation) ---
            # 只有在低位出现反弹信号（如底分型）时才接回，绝不接飞刀
            is_stabilized = check_tactical_entry_signal(self.quote_ctx, symbol, 5, 0.005)
            
            if is_stabilized:
                logger.warning(
                    f"🧟 [不死鸟回补] {symbol} 触发! "
                    f"曾于 {sell_price:.3f} 减仓(L{level}), 现价 {current_price:.3f} 企稳. "
                    f"价差收益: {(sell_price/current_price - 1):.2%}. 接回 {qty_to_buy} 股."
                )
                
                # 执行买入
                # 注意：这里 reason 标记为 resurrection，不占用 dip_add 次数
                if self._execute_add_position(symbol, qty_to_buy, position, f"resurrection_L{level}"):
                    # 更新缓存状态
                    record['status'] = 'RESURRECTED'
                    record['resurrect_price'] = current_price
                    record['resurrect_ts'] = time.time()
                    
                    # ▼▼▼【层级降级】▼▼▼
                    # 既然买回了 L1 卖出的筹码，那我的风险敞口又回到了 L0 的水平。
                    # 我必须把层级降下来，否则如果价格再跌，系统不会再次触发 L1 止损！
                    with self.position_lock:
                        # 只有当买回的是最高层级的筹码时才降级
                        # 例如：在 L1 买回，Level 1 -> 0
                        if position.current_grid_level == level:
                            position.current_grid_level = max(0, level - 1)
                            logger.info(f"🧟 [{symbol}] 回补成功，网格防御等级下调: L{level} -> L{position.current_grid_level}")
                            
                            # 如果降到了0，是否解除防御模式？
                            # 建议：如果价格还在 avg_cost 以下，保持防御模式 (True)，防止常规补仓(Dip Add)干扰。
                            # 只有价格涨回成本，由 check_scale_in_conditions 去解除。

                    self._save_positions()
                    
                    # 发送专属通知
                    self.notification_manager.send_trade_execution(
                        "T+0回补", symbol, qty_to_buy, current_price, 
                        f"不死鸟回补成功 (L{level}), 锁定价差 {(sell_price - current_price)*qty_to_buy:.2f}"
                    )
                    return # 一次循环只处理一笔，防止并发资金问题
                
    def _handle_dip_add(self, symbol: str, position: Position):
        """处理下跌补仓的下单逻辑"""
        remaining_qty = position.planned_total_quantity - position.total_quantity
        if remaining_qty <= 0: return False
        quantity_to_add = int(remaining_qty * self.config.dip_add_scale_ratio)
        symbol_info = self.get_cached_stock_static_info(symbol)
        quantity_to_add = self._adjust_quantity(quantity_to_add, position.market, lot_size=symbol_info.get('lot_size', 100))
        if quantity_to_add <= 0: 
            return False
        return self._execute_add_position(symbol, quantity_to_add, position, f"dip_add_{position.dip_adds_done + 1}")

    def _handle_rise_add(self, symbol: str, position: Position):
        """处理上涨追涨的下单逻辑"""
        initial_scout_qty = int(position.planned_total_quantity * self.config.initial_position_scale_ratio)
        quantity_to_add = int(initial_scout_qty * self.config.rise_add_scale_ratio)
        symbol_info = self.get_cached_stock_static_info(symbol)
        quantity_to_add = self._adjust_quantity(quantity_to_add, position.market, lot_size=symbol_info.get('lot_size', 100))
        if quantity_to_add <= 0: 
            return False
        return self._execute_add_position(symbol, quantity_to_add, position, "rise_add")
    
    def _execute_add_position(self, symbol: str, quantity: int, position: Position, reason: str):
        """
        统一的加仓执行函数 (非阻塞)
        修复：引入购买力动态检查，防止因资金不足导致的订单拒绝。
        """
        # 所有前置检查都移到新的辅助函数中
        if not self._is_add_on_allowed(symbol, reason):
            return False # 如果检查不通过，直接返回

        additional_risk = position.initial_risk_per_share * quantity
        if position.market == MarketType.US:
            additional_risk *= getattr(self.config, 'exchange_rate_usd_to_hkd', 7.8)
        market = get_market_type(symbol)
        # total_capital_base = self.get_available_cash(market)
        total_capital_base = self.get_total_account_value_in_hkd()
        if not self.risk_manager.can_open_new_position(self.positions, market, total_capital_base, position.initial_risk_per_share, quantity):
            logger.warning(f"加仓 ({reason}) 被总风险策略拒绝 {symbol}。")
            logger.error(f'开仓/加仓被总风险策略拒绝-{symbol},加仓 ({reason}) 被总风险策略拒绝 {symbol}。')
            return False

        # ==============================================================================
        # ▼▼▼【购买力双重漏斗】▼▼▼
        # 逻辑：策略想买的数量 vs 钱包能买的数量，谁小听谁的。
        # ==============================================================================
        current_price = self.get_realtime_price(symbol)
        if not current_price:
            logger.error(f"[{symbol}] 无法获取价格，无法计算购买力，取消加仓。")
            return False

        # 1. 获取当前市场的可用现金 (Buying Power)
        available_cash = self.get_available_cash(market)
        
        # 2. 计算最大可买数量 (资金漏斗)
        # 预留 1% 的 buffer 防止市价单滑点导致资金透支
        
        # [仅预留 0.5% 防止市价单滑点]
        safe_buying_power = available_cash * 0.995

        estimated_cost_per_share = current_price * 1.01 # 假设会有1%的滑点或费用
        
        if estimated_cost_per_share <= 0: return False
        
        max_affordable_qty = int(safe_buying_power / estimated_cost_per_share)
        
        # 3. 调整手数 (Lot Size Adjustment)
        symbol_info = self.get_cached_stock_static_info(symbol)
        lot_size = symbol_info.get('lot_size', 100)
        
        # 对“钱包允许的最大数量”也做一次取整
        max_affordable_qty = self._adjust_quantity(max_affordable_qty, market, lot_size=lot_size)

        # 4. 最终裁决：取 策略计划量 和 钱包最大量 的最小值
        final_quantity = min(quantity, max_affordable_qty)

        # 5. 极小值过滤
        if final_quantity <= 0:
            logger.warning(f"[{symbol}] 资金枯竭拦截: 策略需 {quantity} 股, 但资金仅够 0 股 (LotSize限制)。建议暂停该股加仓。")
            # logger.error(f"[{symbol}] 加仓失败：购买力不足。策略需: {quantity}, 钱包可买: {max_affordable_qty}, 资金: {available_cash:.2f}")
            return False
            
        if final_quantity < quantity:
            logger.warning(f"⚠️ [{symbol}] 资金不足，加仓自动降级: 计划 {quantity} -> 实际 {final_quantity} (可用资金: {available_cash:.2f})")
        
        logger.warning(f"准备执行加仓 ({reason}) for {symbol}, 数量: {final_quantity}")
        order_id = self.submit_order(symbol, final_quantity, OrderSide.Buy)
        if not order_id: return False

        with self.position_lock:
            if symbol in self.positions:
                self.positions[symbol].pending_pyramid_order_id = order_id
                self.positions[symbol].pending_add_reason_tag = reason
                self._save_positions()
        return True
    
    def _execute_extended_hours_add_position(self, symbol: str, quantity: int, position: Position, reason: str):
        """
        夜盘/盘外加仓执行器
        修复：引入购买力动态检查。
        """
        logger.warning(f"准备执行加仓 ({reason}) for {symbol}, 数量: {quantity}")
        current_price = self.get_realtime_price(symbol)
        if not current_price: return False
        
        # ==============================================================================
        # ▼▼▼【购买力漏斗 (夜盘版)】▼▼▼
        # ==============================================================================
        market = get_market_type(symbol)
        available_cash = self.get_available_cash(market)
        
        # 夜盘通常挂限价单，滑点风险较小，但仍需预留手续费
        safe_buying_power = available_cash * 0.99
        
        # 计算最大可买
        max_affordable_qty = int(safe_buying_power / current_price)
        
        # 调整手数
        symbol_info = self.get_cached_stock_static_info(symbol)
        lot_size = symbol_info.get('lot_size', 1 if market == MarketType.US else 100)
        max_affordable_qty = self._adjust_quantity(max_affordable_qty, market, lot_size=lot_size)
        
        # 取交集
        final_quantity = min(quantity, max_affordable_qty)
        
        if final_quantity <= 0:
            logger.error(f"[{symbol}] 夜盘加仓失败：购买力不足。可用资金: {available_cash:.2f}")
            return False
            
        if final_quantity < quantity:
             logger.warning(f"⚠️ [{symbol}] 夜盘资金不足，加仓降级: {quantity} -> {final_quantity}")

        order_id = self.submit_order_lo(symbol, final_quantity, OrderSide.Buy, Decimal(str(current_price)))
        if not order_id: return False

        with self.position_lock:
            if symbol in self.positions:
                self.positions[symbol].pending_pyramid_order_id = order_id
                self.positions[symbol].pending_add_reason_tag = reason
                self._save_positions()
        return True
    
    def _handle_pending_add_order(self, pos: Position):
        """
        【已适配富途/长桥】处理所有类型的待处理加仓订单的成交回报。
        此方法通过统一的 DataProvider 接口获取订单详情，实现了券商无关性。
        """
        logger.info(f"[{pos.symbol}] 保守策略模式：加仓订单处理已禁用，清理加仓锁。")
        with self.position_lock:
            current_pos = self.positions.get(pos.symbol)
            if current_pos:
                current_pos.pending_pyramid_order_id = None
                current_pos.pending_add_reason_tag = None
                self._save_positions()
        return

        order_id = pos.pending_pyramid_order_id
        symbol = pos.symbol
        logger.debug(f"正在检查 {symbol} 的待处理加仓订单: {order_id}")
        
        try:
            # 1. 调用统一的 DataProvider 接口，而不是特定券商的 trade_ctx
            # 我们传入 symbol 作为市场提示，确保能查询到正确的账户。
            kwargs = {
                'symbol': symbol
            }
            order_details = self.data_provider.get_order_detail(order_id, **kwargs)

            # 2. 健壮性检查：如果API未能返回订单详情，则直接跳过本次检查
            if not order_details:
                logger.warning(f"无法获取加仓订单 {order_id} ({symbol}) 的详情，将在下次检查时重试。")
                return

            # 3. 使用字典键访问，并与我们定义的标准状态字符串进行比较
            status = order_details.get('status')
            
            if status == "Filled":
                filled_price = float(order_details.get('price', 0.0))
                filled_quantity = int(order_details.get('quantity', 0))

                # 4. 确保成交数据有效，防止因API返回异常数据而污染仓位
                if filled_price <= 0 or filled_quantity <= 0:
                    logger.error(f"加仓订单 {order_id} ({symbol}) 状态为 'Filled' 但成交价或数量无效。Price: {filled_price}, Qty: {filled_quantity}")
                    # 清理掉这个有问题的待处理订单，以防无限循环
                    pos.pending_pyramid_order_id = None
                    pos.pending_add_reason_tag = None
                    self._save_positions()
                    return

                actual_amount = filled_price * filled_quantity
                logger.warning(f"加仓订单已成交 {symbol}: 价格={filled_price}, 数量={filled_quantity}")

                # --- 后续的所有业务逻辑都保持不变，因为它们与券商无关 ---
                reason_tag = pos.pending_add_reason_tag or ""
                action_type_map = {
                    "dip_add": PurchaseActionType.DIP_ADD,
                    "rise_add": PurchaseActionType.RISE_ADD,
                    "pyramid_add": PurchaseActionType.PYRAMID_ADD,
                    "main_force_add": PurchaseActionType.MAIN_FORCE_ADD
                }
                action_type = next((v for k, v in action_type_map.items() if k in reason_tag), None)

                if action_type:
                    
                    # 分离标题与内容逻辑，不再硬编码 "ADD"
                    # 定义不同加仓类型对应的【邮件标题关键词】
                    title_map = {
                        PurchaseActionType.DIP_ADD: "下跌补仓",
                        PurchaseActionType.RISE_ADD: "盈利加仓",
                        PurchaseActionType.PYRAMID_ADD: "金字塔加仓",
                        PurchaseActionType.MAIN_FORCE_ADD: "主力部队进场加仓"
                    }
                    
                    # 定义不同加仓类型对应的【邮件正文原因】
                    reason_map = {
                        PurchaseActionType.DIP_ADD: f"下跌补仓 (第 {pos.dip_adds_done + 1} 次)",
                        PurchaseActionType.RISE_ADD: "上涨追涨 (建仓期)",
                        PurchaseActionType.MAIN_FORCE_ADD: "主力部队进场加仓",
                        PurchaseActionType.PYRAMID_ADD: f"金字塔加仓 (第 {pos.pyramid_level + 1} 次)"
                    }

                    # 获取动态标题和原因
                    action_title = title_map.get(action_type, "加仓")
                    action_reason = reason_map.get(action_type, "常规加仓")
                    self.notification_manager.send_trade_execution(action_title, symbol, filled_quantity, filled_price, action_reason)
                    pos.add_purchase_record(action_type, filled_price, filled_quantity, actual_amount)
                    
                    if pos.overall_phase == PositionOverallPhase.BUILDING and pos.total_quantity >= pos.planned_total_quantity:
                        pos.overall_phase = PositionOverallPhase.RUNNING
                        logger.warning(f"✅ 仓位 {pos.symbol} 已达到计划规模，转入盈利奔跑期。")
                        # --- 【核心风控分歧点】---
                        # 必须区分加仓的性质，执行正确的风险策略
                        if 'dip_add' in reason_tag:
                            # =======================================================
                            # Part 1: 更新计数器与阶段状态
                            # =======================================================
                            pos.dip_adds_done += 1
                            logger.info(f"[{symbol}] 下跌补仓完成，当前补仓阶段: {pos.dip_adds_done}")
                            
                            # 检查是否全部打完 (例如配置了2个阈值，现在 dip_adds_done 到了2)
                            num_triggers = len(self.config.dip_add_triggers_percent)
                            if pos.dip_adds_done >= num_triggers:
                                logger.warning(f"[{symbol}] 3-3-4 建仓计划全部完成，强制转入 RUNNING 阶段。")
                                pos.overall_phase = PositionOverallPhase.RUNNING

                            # =======================================================
                            # Part 2: 动态调整止损与风控
                            # =======================================================
                            # **防御性操作**：下跌补仓后，仓位仍处于水下。
                            # 绝对不能激活追踪止损，而是应该更新并收紧初始硬止损。
                            
                            # 使用最新的 avg_cost (加权平均后) 重新计算 4% 止损线
                            # 这一点至关重要，因为成本降低了，止损线也应该跟着下移，给你更多空间
                            # new_stop_loss = pos.avg_cost * (1 - self.config.stop_loss_ratio)
                            new_stop_loss = self.adaptive_stop_loss.calculate_stop_loss(
                                pos.symbol,
                                pos.avg_cost,
                                'long'
                            )
                            pos.initial_stop_loss_price = new_stop_loss
                            
                            # 确保追踪止损是关闭状态 (因为还在水下)
                            pos.is_trailing_stop_active = False 
                            logger.warning(f"★★★ 防御性补仓完成，止损重算 for {pos.symbol} | 新均价: {pos.avg_cost:.3f} | 新止损: {new_stop_loss:.3f}")
                        else:
                            # **进攻性操作**：上涨追涨或金字塔加仓后，仓位处于盈利状态。
                            # 此时激活追踪止损是合理的，以保护浮盈。
                            current_price = self.get_current_price(symbol) or filled_price
                            if not pos.is_trailing_stop_active and current_price > round(pos.avg_cost * 1.002,3):
                                pos.is_trailing_stop_active = True
                                # 使用当前价格计算追踪止损，而不是成交价，因为市场可能又变动了
                                new_trailing_stop = current_price * (1 - self.config.trailing_stop_ratio)
                                pos.trailing_stop_price = max(pos.avg_cost, new_trailing_stop) # 止损至少不能低于新成本
                                logger.warning(f"★★★ 进攻性加仓完成，激活追踪止损 for {pos.symbol}! 止损位: {pos.trailing_stop_price:.3f}")

                        # if not pos.is_trailing_stop_active:
                        #     pos.is_trailing_stop_active = True
                        #     new_trailing_stop = filled_price * (1 - self.config.trailing_stop_ratio)
                        #     pos.trailing_stop_price = max(pos.avg_cost, new_trailing_stop)
                        #     logger.warning(f"★★★ 建仓完成，立即激活追踪止损 for {pos.symbol}! 止损位: {pos.trailing_stop_price:.3f}")
                
                pos.pending_pyramid_order_id = None
                pos.pending_add_reason_tag = None
                self._save_positions()

            # 5. 与标准化的失败状态字符串进行比较
            elif status in ["Canceled", "Rejected", "Expired"]:
                logger.error(f"加仓订单执行失败或已失效 {symbol}: ID={order_id}, 状态={status}")
                # 清理待处理订单状态
                pos.pending_pyramid_order_id = None
                pos.pending_add_reason_tag = None
                self._save_positions()
        except Exception as e:
            logger.error(f"检查加仓订单 {symbol} 状态时出错: {e}", exc_info=True)

    
    def _check_pyramid_add_condition(self, symbol: str, current_price: float, position: Position):
        # ==========================================================
        # ▼▼▼【适配修改：加仓熔断器】▼▼▼
        # ----------------------------------------------------------
        if position.marked_for_liquidation:
            logger.warning(f"[{symbol}] 金字塔加仓被否决：该头寸已被时间止损标记为待清算。")
            return
        # ==========================================================

        # === 【“顺风vs逆风”策略应用】 ===
        if position.has_dip_added():
            # 如果是经历过下跌补仓的“逆风股”，则永久禁止对它进行盈利加仓。
            # 我们的策略是分批卖出，而不是继续加码。
            return 
        # =================================

        """检查并执行金字塔式盈利加仓。"""
        if position.overall_phase != PositionOverallPhase.RUNNING: return
        if position.pending_pyramid_order_id or position.pending_sell_order_id: return
        if position.is_trailing_stop_active: return
        if position.initial_risk_per_share <= 0: return
        
        if position.initial_risk_per_share > 0 and position.initial_scout_price > 0:
            required_profit = (position.pyramid_level + 1) * position.initial_risk_per_share * self.config.pyramid_profit_multiplier
            current_profit = current_price - position.initial_scout_price
            if current_profit < required_profit: return

        # 使用真实成本计算安全垫
        real_cost = position.get_avg_cost(self.config)
        # 利润空间 = (现价 / 真实成本) - 1
        profit_headroom_ratio = (current_price / real_cost) - 1

        if profit_headroom_ratio < self.config.pyramid_min_profit_headroom_ratio:
            logger.warning(f"[{symbol}] 金字塔加仓被安全垫过滤：盈利空间不足 ({profit_headroom_ratio:.2%})。")
            return
    
        logger.warning(f"触发盈利加仓条件 {symbol}: 当前盈利/股({current_profit:.2f}) >= 目标盈利/股({required_profit:.2f})")
        self._handle_pyramid_add(symbol, current_price, position)
    
    def _handle_pyramid_add(self, symbol: str, current_price: float, position: Position):
        """处理金字塔加仓 (非阻塞)"""
        try:
            initial_scout_records = position.phase_records.get(PurchaseActionType.INITIAL_SCOUT.value)
            if not initial_scout_records:
                logger.error(f"无法计算金字塔加仓数量，未找到 {symbol} 的初始侦察仓记录。")
                return
            initial_scout_quantity = initial_scout_records[0].get('quantity', 0)
            if initial_scout_quantity <= 0:
                logger.error(f"初始侦察仓数量为0，无法计算金字塔加仓规模 for {symbol}。")
                return

            quantity_to_add = int(initial_scout_quantity * (self.config.pyramid_decay_ratio ** (position.pyramid_level + 1)))
            symbol_info = self.get_cached_stock_static_info(symbol)
            quantity_to_add = self._adjust_quantity(quantity_to_add, position.market, lot_size=symbol_info.get('lot_size', 100))
            if quantity_to_add <= 0:
                logger.warning(f"金字塔加仓计算数量为0，不再加仓 {symbol}")
                return
                
            reason_tag = f"pyramid_add_{position.pyramid_level + 1}"
            self._execute_add_position(symbol, quantity_to_add, position, reason_tag)
        except Exception as e:
            logger.error(f"处理盈利加仓时出错 {symbol}: {e}", exc_info=True)

    
    def _is_add_on_allowed(self, symbol: str, reason: str) -> bool:
        """
        统一的加仓前置条件检查器。
        检查交易时段和大盘健康状况。
        """
        return True
        market = get_market_type(symbol)
        current_regime = self.market_regime_engine.get_marget_regime(market)

        # 2. 定义哪些状态对于“加仓”这个行为是“健康”的
        healthy_regimes_for_add_on = [MarketRegime.STRONG_BULL, MarketRegime.CAUTIOUS_BULL, MarketRegime.RANGE_BOUND]

        # 3. 基于当前状态进行判断
        if current_regime not in healthy_regimes_for_add_on:
            # 大盘不好，可以做空加仓
            is_bearish_asset = self._is_bearish_symbol(symbol)
            if is_bearish_asset: return True

            daily_msg = f"日线级别大盘不健康，当前状态为: {current_regime.value}"
            logger.warning(f"加仓被否决 ({reason}) for {symbol}：{daily_msg}")
            logger.error(f'加仓被大盘环境否决-{symbol},加仓 ({reason}) 被大盘环境否决，原因: {daily_msg}')
            # send_email(subject=f'加仓被大盘环境否决-{symbol}', content=f"加仓 ({reason}) 被大盘环境否决，原因: {daily_msg}")
            return False

        return True # 所有检查通过


    # ==============================================================================
    # VI. 券商API交互层 (Broker Interaction Layer)
    # ==============================================================================
    
    def submit_order(self, symbol: str, quantity: int, side: OrderSide) -> Optional[str]:
        """
        【已改造】通过统一的 data_provider 接口提交订单。
        """
        side_str = '买入' if side == OrderSide.Buy else '卖出'
        try:
            #调用 data_provider 的方法
            order_id = self.data_provider.submit_order(symbol=symbol, quantity=quantity, side=side)
            
            if order_id:
                logger.warning(f"{side_str}单提交成功 {symbol}: 数量={quantity}, 订单ID={order_id}")
                return order_id
            else:
                # submit_order 内部已经记录了详细日志，这里只触发告警
                error_msg = f"P1级告警: {side_str}单提交失败({symbol})，请检查日志获取详细错误信息。"
                # send_email(subject=f'交易失败提醒: {side_str}', content=error_msg)
                self.notification_manager.send_critical_alert(error_msg)
                return None
        except Exception as e:
            # 兜底异常处理
            error_msg = f"P1级告警: {side_str}单提交时发生系统级异常({symbol})! 错误: {e}"
            logger.error(error_msg, exc_info=True)
            # send_email(subject=f'交易失败提醒: {side_str}', content=error_msg)
            self.notification_manager.send_critical_alert(error_msg)
            return None

    def submit_order_lo(self, symbol: str, quantity: int, side: OrderSide, limit_price: Decimal) -> Optional[str]:
        """
        以限价单（Limit Order）方式提交期权或其他证券的买卖订单。
        该方法旨在精确控制成交价格，避免市价单的滑点问题。

        Args:
            symbol (str): 交易标的，格式为 'ticker.region'。
            quantity (int): 下单数量。
            side (OrderSide): 买卖方向 (OrderSide.Buy 或 OrderSide.Sell)。
            limit_price (Decimal): 期望的成交价格。买入时，为最高可接受买价；卖出时，为最低可接受卖价。

        Returns:
            Optional[str]: 成功则返回订单ID，失败则返回 None。
        """
        side_str = '买入' if side == OrderSide.Buy else '卖出'
        try:
            # ===================================================================
            # 🔥 同样使用健壮的重试机制
            # ==================================================================
            fill_outside_rth=True
            market_session = get_current_market_session(get_market_type(symbol))
            if market_session in [TradingSession.PRE_MARKET,TradingSession.AFTER_MARKET_EXTENDED]:
                fill_outside_rth = True
                logger.info(f"[{symbol}] 当前处于非标时段 ({market_session.name})，已开启 Extended Hours 标志。")
            elif market_session in [TradingSession.REGULAR_TRADING]:
                fill_outside_rth=False
            
            # buffer = getattr(self.config, 'limit_order_price_buffer', 0.005)
            # limit_price_val = limit_price * (1.0 + buffer)
            limit_price = self._normalize_limit_price(symbol, limit_price)

            logger.info(f"正在提交限价单: {symbol} | {side_str} | 数量:{quantity} | 盘外成交:{fill_outside_rth}")
            order_id = self.data_provider.submit_order(symbol=symbol, quantity=quantity, side=side,price=float(limit_price),order_type=OrderType.LO,fill_outside_rth=fill_outside_rth)

            if order_id:
                logger.warning(f"{side_str}单提交成功 {symbol}: 数量={quantity}, 订单ID={order_id}")
                return order_id
            else:
                # submit_order 内部已经记录了详细日志，这里只触发告警
                error_msg = f"P1级告警: {side_str}单提交失败({symbol})，请检查日志获取详细错误信息。"
                # send_email(subject=f'交易失败提醒: {side_str}', content=error_msg)
                self.notification_manager.send_critical_alert(error_msg)
                return None
        
        except Exception as e:
            error_msg = f"P1级告警: 限价{side_str}单提交失败({symbol}) at price {limit_price}，可能与券商API失联! 错误: {e}"
            # 错误日志中也包含价格信息，便于排查
            logger.error(f"限价{side_str}单提交失败 {symbol} at price {limit_price}: {e}", exc_info=True)
            self.notification_manager.send_email_direct(subject=f'交易失败提醒: 限价{side_str}', content=error_msg)
            self.notification_manager.send_critical_alert(error_msg)
            return None
        
    # ==============================================================================
    # VII. 状态持久化 (State Persistence)
    # ==============================================================================
    def _save_all_states(self):
        """统一保存所有需要持久化的状态。"""
        logger.info("正在将所有系统状态写入文件...")
        self._save_positions()
        self._save_pending_orders()
        self._save_sell_locks()
        self._save_daily_pnl_state()

    def _save_positions(self):
        """
        将当前持仓信息保存到文件。
        采用 Atomic Write (写临时文件 -> Fsync -> Rename) 机制，
        彻底杜绝因断电、崩溃导致 positions.json 变为空文件引发的灾难性后果。
        """
        with self.position_lock:
            # 1. 定义临时文件路径 (在同一文件系统下)
            temp_file = self.position_file + ".tmp"
            
            try:
                # 2. 写入临时文件
                with open(temp_file, 'w', encoding='utf-8') as f:
                    positions_dict = {symbol: pos.to_dict() for symbol, pos in self.positions.items()}
                    json.dump(positions_dict, f, indent=4, ensure_ascii=False, cls=EnhancedJSONEncoder)
                    
                    # 3. [关键] 强制将缓冲区数据写入物理磁盘
                    f.flush()
                    os.fsync(f.fileno())
                
                # 4. [关键] 原子替换 (Atomic Replace)
                # 这一步在操作系统层面是原子的，要么成功，要么失败，不会保留中间状态
                os.replace(temp_file, self.position_file)
                
            except Exception as e:
                logger.critical(f"P0级严重错误: 持久化持仓文件失败! {e}", exc_info=True)
                # 尝试删除可能损坏的临时文件
                if os.path.exists(temp_file):
                    try: os.remove(temp_file)
                    except: pass

    def _save_daily_pnl_state(self):
        """
        持久化当日已实现盈亏及达标状态。
        """
        # 获取当前时间用于记录
        # 注意：这里的 date 应该是内存中 self.current_trading_date_xx，
        # 如果内存中为空(刚启动未Load)，则取实时时间
        tz_hk = pytz.timezone(MARKET_TRADING_HOURS["HK"]["timezone"])
        tz_us = pytz.timezone(MARKET_TRADING_HOURS["US"]["timezone"])
        
        hk_date_str = self.current_trading_date_hk or str(datetime.now(tz_hk).date())
        us_date_str = self.current_trading_date_us or str(datetime.now(tz_us).date())

        data = {
            "hk_date": hk_date_str,
            "us_date": us_date_str,
            "hk_pnl": self.daily_realized_pnl_hk,
            "us_pnl": self.daily_realized_pnl_us,
            "hk_target_hit": self.daily_hk_profit_target_hit,
            "us_target_hit": self.daily_us_profit_target_hit,
            "countdown_start_ts": self.daily_profit_countdown_start_ts,
            "escape_triggered": self.half_target_escape_triggered,
            "pnl_high_water_mark": self.daily_pnl_high_water_mark,
            "pnl_last_reset_base": self.daily_pnl_last_reset_base,
            "us_rth_last_reset_date": self.daily_pnl_us_rth_last_reset_date,
            "daily_equity_baseline_hkd": self.daily_equity_baseline_hkd,
            "monthly_equity_baseline_hkd": self.monthly_equity_baseline_hkd,
            "monthly_equity_baseline_key": self.monthly_equity_baseline_key,
            "daily_loss_freeze_date": self.daily_loss_freeze_date,
            "conservative_paused_month": self.conservative_paused_month,
            "last_update": datetime.now().isoformat()
        }
        try:
            # temp_file = self.daily_pnl_file + ".tmp"
            # with open(temp_file, 'w', encoding='utf-8') as f:
            #     json.dump(data, f, indent=4)
            # os.replace(temp_file, self.daily_pnl_file)
            with open(self.daily_pnl_file, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=4)
        except Exception as e:
            logger.error(f"保存每日盈亏状态失败: {e}", exc_info=True)

    # 黑名单保存方法
    def _save_blacklist(self):
        try:
            with open(self.blacklist_file, 'w', encoding='utf-8') as f:
                # set 不可序列化，转 list
                json.dump(list(self.intraday_blacklist), f)
        except Exception as e:
            logger.error(f"保存黑名单失败: {e}")

    def _load_positions(self):
        """从文件加载持仓信息 (已增加对新字段的向后兼容性处理)。"""
        try:
            with open(self.position_file, 'r', encoding='utf-8') as f:
                positions_dict = json.load(f)
                with self.position_lock:
                    for symbol, data in positions_dict.items():
                        # --- [核心兼容性修复] ---
                        # 检查并修复老数据中缺失 initial_stop_loss_price 的问题
                        if 'initial_stop_loss_price' not in data or data['initial_stop_loss_price'] == 0.0:
                            logger.warning(f"为旧持仓 {symbol} 补充 initial_stop_loss_price...")
                            # 为旧仓位按固定比例规则，重新计算并填入止损价
                            initial_price = data.get('initial_price', 0.0)
                            if initial_price > 0:
                                stop_loss_ratio = getattr(self.config, 'stop_loss_ratio', 0.04) # 从配置读取
                                data['initial_stop_loss_price'] = initial_price * (1 - stop_loss_ratio) # 修正：应该是 1-ratio
                            else:
                                data['initial_stop_loss_price'] = 0.0 # 无法计算则设为0

                        # 检查并修复 initial_risk_per_share
                        if 'initial_risk_per_share' not in data or data['initial_risk_per_share'] == 0.0:
                            initial_price = data.get('initial_price', 0.0)
                            if initial_price > 0 and data['initial_stop_loss_price'] > 0:
                                data['initial_risk_per_share'] = initial_price - data['initial_stop_loss_price']
                            else:
                                data['initial_risk_per_share'] = 0.0

                        # for key, default_value in [('pending_pyramid_order_id', None), ('pending_sell_order_id', None), ('sell_reason', None), ('is_breakeven_stop_set', False), ('r_profit_taken', False), ('partial_sell_price', None), ('pending_add_reason_tag', None)]:
                        #     data.setdefault(key, default_value)
                        
                        if 'overall_phase' not in data or data['overall_phase']==None:
                            data.setdefault('overall_phase', PositionOverallPhase.RUNNING.value)
                        
                        if 'initial_scout_price' not in data or data['initial_scout_price']==0.0:
                            data.setdefault('initial_scout_price', data.get('initial_price', 0.0))
                        
                        if 'sell_records' not in data:
                            data.setdefault('sell_records', []) # 为旧数据添加一个空的卖出记录列表
                        # 为旧的持仓记录安全地添加 triggering_strategy 字段，默认值为 None
                        if 'triggering_strategy' not in data or data['triggering_strategy']==None:
                            data.setdefault('triggering_strategy', 'BrokerSync')
                        
                        if 'strategy_class_name' not in data or data['strategy_class_name']==None:
                            data.setdefault('strategy_class_name', 'Manual/External')
                        
                        if 'strategy_params' not in data or data['strategy_params']==None:
                            data.setdefault('strategy_params', {})
                        
                        if 'confirmation_add_done' not in data:
                            data.setdefault('confirmation_add_done', True) # 旧持仓默认为已完成或不适用
                        # -----------------------
                        if 'market' not in data:
                            data['market'] = MarketType.US.value
                        
                        data['market'] = MarketType(data['market'])
                        data['overall_phase'] = PositionOverallPhase(data['overall_phase'])

                        if 'highest_price_since_partial_sell' not in data:
                            data['highest_price_since_partial_sell'] = None
                        
                        try:
                            self.positions[symbol] = Position(**data)
                        except TypeError as e:
                            logger.error(f"为 {symbol} 创建Position对象时失败，字段不匹配: {e}. 数据: {data}")
                            continue
                    logger.info(f"成功从 {self.position_file} 加载并兼容了 {len(self.positions)} 个持仓。")
        except FileNotFoundError:
            logger.info("未找到持仓文件，将从零开始。")
        except Exception as e:
            logger.error(f"加载持仓文件时发生未知错误: {e}", exc_info=True)

    def _load_daily_pnl_state(self):
        """
        从文件恢复当日已实现盈亏。
        关键逻辑：通过比对日期，自动判断是“恢复现场”还是“开启新一天”。
        """
        if not os.path.exists(self.daily_pnl_file):
            logger.info("未找到每日盈亏状态文件，初始化为 0。")
            return

        try:
            with open(self.daily_pnl_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
                
            # 1. 获取市场当前的交易日期 (使用工具类)
            # 注意：这里需要确保 datetime.now() 带时区，以便 .date() 正确
            tz_hk = pytz.timezone(MARKET_TRADING_HOURS["HK"]["timezone"])
            tz_us = pytz.timezone(MARKET_TRADING_HOURS["US"]["timezone"])
            
            today_hk_str = str(datetime.now(tz_hk).date())
            today_us_str = str(datetime.now(tz_us).date())

            # 2. 恢复 HK 状态
            saved_hk_date = data.get('hk_date', '')
            if saved_hk_date == today_hk_str:
                self.daily_realized_pnl_hk = data.get('hk_pnl', 0.0)
                self.daily_hk_profit_target_hit = data.get('hk_target_hit', False)
                self.current_trading_date_hk = saved_hk_date
                logger.info(f"✅ [状态恢复] HK ({saved_hk_date}) PnL: {self.daily_realized_pnl_hk}, TargetHit: {self.daily_hk_profit_target_hit}")
            else:
                logger.warning(f"📅 [日切重置] HK 日期变更 ({saved_hk_date} -> {today_hk_str})，重置状态。")
                self.daily_realized_pnl_hk = 0.0
                self.daily_hk_profit_target_hit = False
                self.current_trading_date_hk = today_hk_str

            # 3. 恢复 US 状态 (解决北京时间12点重置问题)
            saved_us_date = data.get('us_date', '')
            if saved_us_date == today_us_str:
                self.daily_realized_pnl_us = data.get('us_pnl', 0.0)
                self.daily_us_profit_target_hit = data.get('us_target_hit', False)
                self.current_trading_date_us = saved_us_date
                logger.info(f"✅ [状态恢复] US ({saved_us_date}) PnL: {self.daily_realized_pnl_us}, TargetHit: {self.daily_us_profit_target_hit}")

                self.daily_profit_countdown_start_ts = data.get('countdown_start_ts')
                self.half_target_escape_triggered = data.get('escape_triggered', False)
                self.daily_pnl_high_water_mark = data.get('pnl_high_water_mark', 0.0)
                self.daily_pnl_last_reset_base = data.get('pnl_last_reset_base', 0.0)
                self.daily_pnl_us_rth_last_reset_date = data.get('us_rth_last_reset_date')
                self.daily_equity_baseline_hkd = float(data.get('daily_equity_baseline_hkd', 0.0) or 0.0)
                self.daily_loss_freeze_date = data.get('daily_loss_freeze_date')
                logger.info(f"🔒 [状态恢复] 美股RTH上次重置日期: {self.daily_pnl_us_rth_last_reset_date}")
                
                if self.daily_profit_countdown_start_ts:
                    logger.warning(f"⚡ [状态恢复] 恢复逃跑倒计时，开始时间: {datetime.fromtimestamp(self.daily_profit_countdown_start_ts)}")

            else:
                logger.warning(f"📅 [日切重置] US 日期变更 ({saved_us_date} -> {today_us_str})，重置状态。")
                self.daily_realized_pnl_us = 0.0
                self.daily_us_profit_target_hit = False
                self.current_trading_date_us = today_us_str
                self.daily_pnl_us_rth_last_reset_date = None
                self.daily_equity_baseline_hkd = 0.0
                self.daily_loss_freeze_date = None

            current_month_key = today_us_str[:7]
            saved_month_key = data.get('monthly_equity_baseline_key')
            if saved_month_key == current_month_key:
                self.monthly_equity_baseline_key = saved_month_key
                self.monthly_equity_baseline_hkd = float(data.get('monthly_equity_baseline_hkd', 0.0) or 0.0)
                paused_month = data.get('conservative_paused_month')
                self.conservative_paused_month = paused_month if paused_month == current_month_key else None
            else:
                self.monthly_equity_baseline_key = current_month_key
                self.monthly_equity_baseline_hkd = 0.0
                self.conservative_paused_month = None
	                
            # 加载后立即保存一次，确保文件格式同步
            self._save_daily_pnl_state()

        except Exception as e:
            logger.error(f"加载每日盈亏状态失败: {e}，将重置为 0 以保安全。", exc_info=True)
            self.daily_realized_pnl_hk = 0.0
            self.daily_realized_pnl_us = 0.0
            self.daily_hk_profit_target_hit = False
            self.daily_us_profit_target_hit = False

    # 黑名单加载方法
    def _load_blacklist(self):
        if not os.path.exists(self.blacklist_file): return
        try:
            with open(self.blacklist_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
                self.intraday_blacklist = set(data)
                # [关键] 检查黑名单是否过期（比如是昨天的），如果是则清空
                # 这里简单处理：如果文件修改时间不是今天，就清空
                mtime = os.path.getmtime(self.blacklist_file)
                if datetime.fromtimestamp(mtime).date() < date.today():
                    self.intraday_blacklist.clear()
                    logger.info("检测到黑名单文件为旧数据，已自动重置。")
                else:
                    logger.info(f"已恢复当日黑名单，共 {len(self.intraday_blacklist)} 个标的。")
        except Exception as e:
            logger.error(f"加载黑名单失败: {e}")
            self.intraday_blacklist = set()

    def _refresh_candidate_pools(self):
        try:
            # 加载数据
            file_path = os.path.join(getattr(self.config,'master_project_path','/home/nexus/project/trader'),'data/us_alpha_hunter_watchlist.json')
            us_alpha_hunter_watchlist = load_json_data(file_path)

            # ------------------------------------------------------------------
            # 场景 0: 第二梯队 + 含风险词
            # ------------------------------------------------------------------
            self.risky_second_tier_stocks = extract_candidates_pools(
                us_alpha_hunter_watchlist,
                target_rank="第二梯队",
                match_keywords=True # 包含关键词
            )

            # ------------------------------------------------------------------
            # 场景 1: 第二梯队 + 【不含】风险词 (Clean Second Tier)
            # ------------------------------------------------------------------
            # 既然reason里没有“不确定性”或“风险”，说明这是第二梯队里的优等生，值得一看
            self.clean_second_tier_stocks = extract_candidates_pools(
                us_alpha_hunter_watchlist,
                target_rank="第二梯队",
                match_keywords=False # 注意这里：False 表示排除含风险词的
            )

            # ------------------------------------------------------------------
            # 场景 2: 第一梯队 + 【不含】风险词 (Clean Second Tier)
            # ------------------------------------------------------------------
            self.clean_first_tier_stocks = extract_candidates_pools(
                us_alpha_hunter_watchlist,
                target_rank="第一梯队",
                match_keywords=False # 注意这里：False 表示排除含风险词的
            )

            logger.info(f"筛选完成: 高风二梯队[{len(self.risky_second_tier_stocks)}] | "
                        f"纯净二梯队[{len(self.clean_second_tier_stocks)}] | "
                        f"纯净一梯队[{len(self.clean_first_tier_stocks)}]")

        except Exception as e:
            logger.error(f"加载watchlist数据并筛选失败，你最好去检查一下JSON格式: {e}", exc_info=True)
            # 出错了也要优雅地兜底
            self.risky_second_tier_stocks = []
            self.clean_second_tier_stocks = []
            self.clean_first_tier_stocks = []

    # --------------------------------------------------------------------------
    #  夜猎者状态持久化模块
    # --------------------------------------------------------------------------
    def _load_pre_market_states(self):
        try:
            if not os.path.exists(self.pre_market_states_file):
                self.pre_market_states = {}
                return

            with open(self.pre_market_states_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
                
            # 数据清洗：剔除过期的状态（不是今天的）
            # 假设我们用文件修改时间，或者在 state 里加一个 'date' 字段会更稳健
            # 这里简单起见，如果状态里的 last_update_ts 距离现在超过 12小时，就丢弃
            current_ts = time.time()
            valid_states = {}
            for sym, state in data.items():
                last_ts = state.get('last_update_ts', 0)
                # 18小时 = 64800秒，足以跨越夜盘到盘前
                if (current_ts - last_ts) < 64800: 
                    valid_states[sym] = state
                else:
                    logger.info(f"清理过期夜盘状态: {sym}")
            
            self.pre_market_states = valid_states
            logger.info(f"✅ 成功恢复 {len(self.pre_market_states)} 个有效夜盘状态。")
            
        except Exception as e:
            logger.error(f"加载夜盘状态失败: {e}")
            self.pre_market_states = {}

    def _recover_intraday_history_from_logs(self):
        """
        【灾难恢复】从 sell_signals.log 中恢复当日的卖出价格记忆。
        解决程序重启后，内存中丢失“上次卖出价”导致再次追高买入的问题。
        """
        import re
        log_path = os.path.join(project_path, 'logs/sell_signals.log')
        # 如果日志在当前目录
        if not os.path.exists(log_path):
            log_path = 'sell_signals.log'
            
        if not os.path.exists(log_path):
            logger.warning("未找到 sell_signals.log，无法恢复日内交易记忆，将以空白状态启动。")
            return

        logger.info("正在从日志恢复日内交易记忆...")
        
        # 1. 获取当前时间
        now = datetime.now()
        
        # 2. 构造关键时间点用于区间判断
        today_start = now.replace(hour=0, minute=0, second=0, microsecond=0)
        t_06_00 = now.replace(hour=6, minute=0, second=0, microsecond=0)
        t_21_30_30 = now.replace(hour=21, minute=30, second=30, microsecond=0)
        t_23_59_59 = now.replace(hour=23, minute=59, second=59, microsecond=0)

        # 3. 核心逻辑：根据你的奇葩需求设定 cutoff_time (早于这个时间的日志统统不要)
        # 默认策略：只看今天的 (兜底逻辑)
        cutoff_time = today_start

        if today_start <= now <= t_06_00:
            # 需求：如果是凌晨 [00:00, 06:00]，那就都要
            # 这里设为 min 表示不设限，只要文件里有的，就算是昨天的我也给你读进来（防止跨夜交易数据丢失）
            cutoff_time = datetime.min
            logger.info(f"当前是凌晨交易时段 ({now.strftime('%H:%M')})，策略：全量恢复日志。")
            
        elif t_21_30_30 <= now <= t_23_59_59:
            # 需求：如果是美股开盘后 [21:30:30, 23:59:59]，只要最近4小时
            cutoff_time = now - timedelta(hours=4)
            logger.info(f"当前是美股开盘时段 ({now.strftime('%H:%M')})，策略：仅恢复最近4小时 ({cutoff_time.strftime('%H:%M')}之后)。")
        
        else:
            # 其他时间段（比如盘前），保持默认只看今天的，避免读到几年前的老皇历
            logger.info(f"当前是休整/盘前时段 ({now.strftime('%H:%M')})，策略：恢复今日所有记录。")
        
        # current_date_str = datetime.now().strftime('%Y-%m-%d') # 北京时间日期，用于简单过滤
        
        # 正则表达式：精准提取 symbol 和 price
        # 匹配格式: ... symbol:TSLL.US,... price:23.085,...
        pattern = re.compile(r"symbol:([A-Z]+\.[A-Z]+).*?price:(\d+\.?\d*)")
        
        try:
            with open(log_path, 'r', encoding='utf-8') as f:
                # 读取所有行，倒序遍历（因为我们通常关心最后一次卖出价，或者最高价）
                # 这里我们采取“覆盖策略”，日志后面的记录会覆盖前面的，符合“最近一次卖出”的逻辑
                for line in f:
                    # 快速提取日志时间 (日志格式固定为 "2026-02-25 00:06:44,...")
                    # 直接切片前19位，比正则快10倍，别问为什么，问就是经验。
                    if len(line) < 19: 
                        continue
                        
                    try:
                        log_time_str = line[:19]
                        # 将字符串转为 datetime 对象以便比较
                        log_dt = datetime.strptime(log_time_str, '%Y-%m-%d %H:%M:%S')
                        
                        # 只有时间晚于(大于) cutoff_time 的才要
                        if log_dt < cutoff_time:
                            continue
                    except ValueError:
                        # 遇到非标准日志行（比如报错堆栈），直接跳过，别报错崩溃
                        continue
                        
                    match = pattern.search(line)
                    if match:
                        symbol = match.group(1)
                        price = float(match.group(2))
                        
                        # 逻辑：记录该股票今天卖出过的价格。
                        # 策略选择：记录【最后一次】卖出价。
                        self.intraday_trade_history[symbol] = price
            
            if self.intraday_trade_history:
                logger.warning(f"✅ 已恢复 {len(self.intraday_trade_history)} 条日内卖出记录: {self.intraday_trade_history}")
        except Exception as e:
            logger.error(f"恢复日内记忆失败: {e}", exc_info=True)

    def _save_pre_market_states(self):
        """将夜盘策略的实时状态保存到文件。"""
        try:
            # 这是一个高频操作，但数据量很小，直接dump即可
            # 如果担心性能，可以加一个 dirty 标志位，但这属于过度优化
            with open(self.pre_market_states_file, 'w', encoding='utf-8') as f:
                json.dump(self.pre_market_states, f, indent=4, ensure_ascii=False)
        except Exception as e:
            logger.error(f"保存夜盘状态文件失败: {e}", exc_info=True)

    def _archive_completed_trade(self, position: Position, exit_price: float, exit_quantity: int, exit_reason: str):
        """
        将一个完整的、已平仓的交易记录归档到历史文件。
        这是交易复盘的基石。
        """
        try:
            position.add_sell_record(exit_price, exit_quantity, f"[归档结算] {exit_reason}")

            # try:
            #     symbol_info = self.get_cached_stock_static_info(position.symbol)
            #     symbol_name = symbol_info.get('name_cn') or symbol_info.get('name_en')
            #     log_reason = f"外部平仓同步({exit_reason})"
            #     sell_logger.info(f"symbol:{position.symbol},name:{symbol_name},price:{exit_price:.3f},strategy_reason:{log_reason},llm_reason:N/A(BrokerSync)")
            # except Exception as log_e:
            #     logger.error(f"[{position.symbol}] 归档日志记录失败: {log_e}")

            trade_record = position.to_dict()
            
            # 计算并添加最终的交易结果
            pnl_per_share = exit_price - position.avg_cost
            total_pnl = pnl_per_share * exit_quantity
            roi_percent = (pnl_per_share / position.avg_cost) * 100 if position.avg_cost > 0 else 0
            
            # 添加退出信息到记录中
            trade_record['exit_info'] = {
                'exit_timestamp': datetime.now(timezone.utc).isoformat(), # <<< BUG修复：使用带时区的UTC时间
                'exit_reason': exit_reason,
                'exit_price': exit_price,
                'exit_quantity': exit_quantity,
                'pnl_per_share': round(pnl_per_share, 4),
                'total_pnl': round(total_pnl, 3),
                'roi_percent': round(roi_percent, 3),
                # 冗余落盘 entry_market_regime / entry_signal_quality，
                # 即使后续 Position.to_dict 字段调整，离线校准脚本也能稳定读取。
                'entry_market_regime': getattr(position, 'entry_market_regime', None),
                'entry_signal_quality': getattr(position, 'entry_signal_quality', None),
            }
            # 【核心加固】将文件写入操作也包裹在内层的 try-except 中
            self._save_trade_history(trade_record)
            logger.info(f"✅ 交易记录归档成功: {position.symbol}, PnL: {total_pnl:.2f}")
            
        except Exception as e:
            # 【致命错误告警】如果归档失败，这是P1级警报！绝不能让仓位被静默删除。
            error_msg = f"P1级告警: 归档交易 {position.symbol} 时发生严重错误，仓位可能未被正确清理！错误: {e}"
            logger.critical(error_msg, exc_info=True)
            self.notification_manager.send_critical_alert(error_msg)
            # 【关键】向上抛出异常，中断后续的清理流程！
            raise

    def _save_trade_history(self, trade_record: dict):
        """ 以线程安全的方式，将单条交易记录追加到历史文件中。"""
        # 为确保线程安全，这里我们不使用 position_lock，而是针对历史文件单独加锁
        history_lock = threading.Lock()
        with history_lock:
            history = []
            try:
                if os.path.exists(self.trade_history_file):
                    with open(self.trade_history_file, 'r', encoding='utf-8') as f:
                        history = json.load(f)
            except (FileNotFoundError, json.JSONDecodeError):
                history = [] # 如果文件不存在或为空/损坏，则从新列表开始

            history.append(trade_record)

            with open(self.trade_history_file, 'w', encoding='utf-8') as f:
                json.dump(history, f, indent=4, ensure_ascii=False, cls=EnhancedJSONEncoder)

    def _save_pending_orders(self):
        """将待处理的开仓订单信息保存到文件。"""
        with self.pending_orders_lock:
            with open(self.pending_orders_file, 'w', encoding='utf-8') as f:
                json.dump(self.pending_orders, f, indent=4, ensure_ascii=False)
    
    def _load_pending_orders(self):
        """从文件加载待处理的开仓订单信息。"""
        try:
            with open(self.pending_orders_file, 'r', encoding='utf-8') as f:
                with self.pending_orders_lock:
                    self.pending_orders = json.load(f)
                    logger.info(f"成功从 {self.pending_orders_file} 加载了 {len(self.pending_orders)} 个待处理订单。")
        except FileNotFoundError:
            logger.info("未找到待处理订单文件，将从零开始。")
        except Exception as e:
            logger.error(f"加载待处理订单文件失败: {e}", exc_info=True)

    def _save_sell_locks(self):
        """将卖出锁字典保存到文件"""
        data_to_save = {f"{k[0]}|{k[1]}": v for k, v in self.sell_locks.items()}
        with open(self.sell_locks_file, 'w', encoding='utf-8') as f:
            json.dump(data_to_save, f, indent=4, ensure_ascii=False)

    def _load_sell_locks(self):
        """从文件加载卖出锁字典"""
        try:
            with open(self.sell_locks_file, 'r', encoding='utf-8') as f:
                loaded_data = json.load(f)
                for key_str, timestamp in loaded_data.items():
                    parts = key_str.split('|', 1)
                    if len(parts) == 2:
                        self.sell_locks[(parts[0], parts[1])] = timestamp
                logger.info(f"成功从 {self.sell_locks_file} 加载了 {len(self.sell_locks)} 个卖出锁。")
        except FileNotFoundError:
            logger.info("未找到卖出锁文件，将从零开始。")
        except Exception as e:
            logger.error(f"加载卖出锁文件失败: {e}", exc_info=True)

    def _save_llm_cache(self):
        """保守策略模式不使用 LLM 缓存。"""
        return
    
    def _load_llm_cache(self):
        """保守策略模式不加载 LLM 缓存。"""
        return

    # ==============================================================================
    # VIII. 账户与资金管理 (Account & Capital Management)
    # ==============================================================================
    
    def _initialize_account_info(self):
        """
        初始化并缓存账户的关键资金信息。
        此方法通过统一的 data_provider 接口获取数据，实现了对长桥和富途的兼容。
        """
        try:
            # 步骤1: 通过统一的接口 self.data_provider.get_account_info() 获取账户信息。
            # 无论是长桥还是富途的数据提供者，都会返回一个格式兼容的列表。
            all_accounts_info = self.data_provider.get_account_info(MarketType.US)
            if is_us_market_open():
                all_accounts_info = self.data_provider.get_account_info(MarketType.US)

            # 步骤2: 进行健壮性检查，确保API调用成功并返回了数据。
            if not all_accounts_info:
                # 如果 all_accounts_info 是 None 或空列表，说明API调用失败，必须中断程序。
                raise ValueError("DataProvider 未能返回任何有效的账户信息，请检查API连接或账户状态。")

            # 步骤3: 遵循原逻辑，我们只关心第一个账户的信息。
            # 我们的 get_account_info 方法已确保返回列表结构，所以 all_accounts_info[0] 的用法是安全的。
            self.account_info = all_accounts_info[0]
            logger.info(f"成功获取并缓存账户信息: 净资产={self.account_info.get('net_assets')}, 购买力={self.account_info.get('buy_power')}")

            # 步骤4: 从返回的账户信息中查找并缓存港币和美元的现金详情。
            # 这一部分的核心逻辑与原来完全兼容，无需修改，体现了封装的优势。
            if 'cash_infos' in self.account_info and self.account_info['cash_infos']:
                for cash_info in self.account_info['cash_infos']:
                    # cash_info 现在是一个标准字典, e.g., {'currency': 'HKD', 'available_cash': 10000.0, ...}
                    if cash_info.get('currency') == "HKD": 
                        self.cash_info_hk = cash_info
                        logger.info(f"已缓存港币现金信息: {self.cash_info_hk}")
                    elif cash_info.get('currency') == "USD": 
                        self.cash_info_us = cash_info
                        logger.info(f"已缓存美元现金信息: {self.cash_info_us}")
            else:
                # 如果没有 cash_infos 字段，进行警告并安全地设置为空
                logger.warning("账户信息中未找到 'cash_infos' 字段，无法初始化分币种现金信息。")
                self.cash_info_hk = None
                self.cash_info_us = None

        except Exception as e:
            logger.error(f"【严重】初始化账户信息失败: {e}", exc_info=True)
            # 【告警集成】 4. API失联告警
            error_msg = f"P1级告警: 初始化账户信息失败，可能与券商API失联! 错误详情: {e}"
            logger.error(error_msg, exc_info=True)
            
            # 调用您已有的告警发送方法
            # self._send_critical_alert(error_msg)
            
            # 向上抛出异常，这通常会中断程序的启动，是安全的做法，防止在账户信息未知的情况下运行交易。
            raise
    
    def get_available_cash(self, market: MarketType) -> float:
        """
        计算指定市场的可用风险资本 (考虑保证金、预留金和汇率)。
        通过使用字典的 .get() 方法，此函数现在与任何遵循 DataProvider 规范的券商兼容。
        """
        # self.refresh_account_info() # 确保在调用此方法前，账户信息是最新的
        risk_capital_base = 0.0

        # 检查 account_info 是否存在，这是所有计算的基础
        if not self.account_info:
            logger.warning("Account info is not initialized. Cannot calculate available cash.")
            return 0.0

        if self.config.enable_margin:
            risk_capital_base = float(self.account_info.get('buy_power', 0))
        else:
            if market == MarketType.HK and self.cash_info_hk:
                risk_capital_base = float(self.cash_info_hk.get('available_cash', 0))
            elif market == MarketType.US and self.cash_info_us:
                risk_capital_base = float(self.cash_info_us.get('available_cash', 0))
        
        # 安全检查，如果基础资本为0或负数，直接返回0
        if risk_capital_base <= 0:
            return 0.0

        # 应用资金预留比例，保留一部分现金作为缓冲
        adjusted_capital_base = risk_capital_base * (1 - self.config.reserve_ratio)

        # 处理当账户主币种为港币，但需要计算美元市场购买力时的汇率转换
        account_currency = self.account_info.get('currency')
        if market == MarketType.US and account_currency == "HKD":
            exchange_rate = getattr(self.config, 'exchange_rate_usd_to_hkd', 7.8)
            # 确保汇率不为0，防止除零错误
            if exchange_rate == 0:
                logger.error("Exchange rate (hkd_to_usd) is zero, cannot perform currency conversion.")
                return 0.0
            return max(0, adjusted_capital_base / exchange_rate)
        
        # 默认情况下，直接返回调整后的资本，并确保不为负
        return max(0, adjusted_capital_base)


    def get_total_account_value_in_hkd(self, strict: bool = False) -> float:
        """
        计算以港币计价的账户总资产净值。
        此方法现在与 DataProvider 返回的字典结构完全兼容，适用于富途和长桥。
        """
        self._initialize_account_info() # 确保在调用此方法前，账户信息是最新的
        # 必须检查 self.account_info 是否为 None
        if self.account_info is None:
            msg = "P1级警告: 无法获取账户信息对象(None)，可能是API初始化失败或网络中断。"
            logger.warning(msg)
            if strict:
                # 严格模式下，拿不到数据就是拿不到，不要返回0误导决策
                logger.error("严格模式下获取资产失败，返回 0.0 以阻断交易。")
                return 0.0
            return 0.0 # 宽松模式也只能返回0，但至少不崩
        
        # 1. 优先获取净资产 (Net Assets)
        net_assets_val = self.account_info.get('net_assets')
        if self.config.enable_margin:
            net_assets_val = self.account_info.get('buy_power')

        if net_assets_val is not None and float(net_assets_val) > 0:
            # --- 主路径：使用账户总资产净值 ---
            total_net_assets = float(net_assets_val)
            account_currency = self.account_info.get('currency')
            
            # 如果账户主币种是美元，则需要将其转换为港币
            if account_currency == "USD":
                exchange_rate = getattr(self.config, 'exchange_rate_usd_to_hkd', 7.8) # 注意这里应该是usd到hkd的汇率
                return total_net_assets * exchange_rate
            
            return total_net_assets

         # 2. 如果拿不到净资产
        if strict:
            # 【修复点】严格模式下（用于止盈），拿不到净值直接返回 0 或 None，绝不降级！
            logger.error("P1级异常: 无法获取账户 Net Assets，且处于止盈检查(Strict)模式。拒绝使用现金兜底。")
            return 0.0 
        
        # 3. 宽松模式（原有逻辑，用于风控保底）
        logger.warning("无法获取有效的账户总资产(Net Assets)，将使用各市场可用现金加总作为风险基数。")
        
        # 调用已适配的 get_available_cash 方法
        hkd_capital = self.get_available_cash(MarketType.HK)
        usd_capital = self.get_available_cash(MarketType.US)
        
        # 将美元资本转换为港币
        exchange_rate = getattr(self.config, 'exchange_rate_usd_to_hkd', 7.8)
        usd_capital_in_hkd = usd_capital * exchange_rate
        
        return hkd_capital + usd_capital_in_hkd

    def get_net_equity_value(self, strict: bool = False) -> float:
        """
        获取账户净资产 (Net Equity)。
        用途：专门用于计算“风控比例”、“止盈目标”、“回撤比例”。
        逻辑：只读取 net_assets，这是真金白银的本金。
        """
        self._initialize_account_info() # 确保数据最新
        
        if self.account_info is None:
            if strict:
                logger.error("无法获取账户净值得(None)，严格模式下返回 0.0。")
                return 0.0
            return 0.0

        # 1. 无论是否开启融资，计算盈亏比的分母永远是净资产
        net_assets_val = self.account_info.get('net_assets')
        
        if net_assets_val is not None:
            val = float(net_assets_val)
            # 汇率转换：如果账户是 USD 本位，转为 HKD
            if self.account_info.get('currency','USD') == "USD":
                exchange_rate = getattr(self.config, 'exchange_rate_usd_to_hkd', 7.8)
                return val * exchange_rate
            return val
            
        return 0.0


    def refresh_account_info(self):
        self._initialize_account_info()

    def get_current_positions_count(self, market: MarketType) -> int:
        with self.position_lock:
            return len([pos for pos in self.positions.values() if pos.market == market and self._is_pure_stock(pos.symbol)])
    
    def _is_pure_stock(self, symbol: str) -> bool:
        """
        辅助方法：根据提供的规则，判断一个symbol是否为“纯种”股票。
        将判断逻辑封装起来，是顶级工程师的基本素养。这使得代码更清晰，且易于测试。
        """
        return is_pure_stock(self.quote_ctx,symbol)
    
    def get_max_positions(self, market: MarketType) -> int:
        return self.config.max_positions_hk if market == MarketType.HK else self.config.max_positions_us

    # ==============================================================================
    # IX. 外部服务与通知 (External Services & Notifications)
    # ==============================================================================

    def _get_llm_decision(self, candidate: dict, action_type: str) -> Tuple[bool, str]:
        """保守策略模式禁用 LLM，保留方法仅兼容旧的不可达代码。"""
        return True, "保守策略模式不使用LLM"
    
    # ==============================================================================
    #  做多/做空双轨制决策辅助模块
    # ==============================================================================
    # 判断是否为波动率产品
    def _is_volatility_product(self, symbol: str) -> bool:
        return symbol in self.config.volatility_symbols

    def _is_bearish_symbol(self, symbol: str) -> bool:
        """
        【双轨制辅助】判断一个股票代码是否为做空工具。
        这是战略分流的第一步：身份识别。
        """
        return symbol in self.config.bearish_symbols or symbol in self.config.volatility_symbols

    # ==============================================================================
    # X. 通用工具与辅助方法 (Utilities & Helpers)
    # ==============================================================================

    def get_strong_bull(self,market: MarketType):
        """
        判断是否为强势牛市 (仅基于 MarketRegime)
        
        强势定义:
        - STRONG_BULL: 趋势强劲向上，低波动率，适合猛攻 ✅
        - CAUTIOUS_BULL: 趋势向上，但波动率升高，需要谨慎 ✅ (可选：如果想更严格，可以去掉这个)
        
        Returns:
            True: 当前处于强势牛市，可激进做多
            False: 非强势，不建议激进
        """
        
        try:
            current_regime = self.market_regime_engine.get_marget_regime(market)
            # 核心逻辑：只要宏观趋势是牛市（强牛或谨慎牛），就判定为强势
            # is_strong_bull = current_regime in [
            #     MarketRegime.STRONG_BULL,      # 最强牛市，全力做多
            #     MarketRegime.CAUTIOUS_BULL     # 谨慎牛市，可分批建仓
            # ]
            
            # 可选：如果你想更严格（只在超级强势时才做多），改为：
            # is_strong_bull = current_regime == MarketRegime.STRONG_BULL

            health_type, _ = self.market_regime_engine.check_intraday_health(market)
            is_strong_bull = (
                (current_regime in [MarketRegime.STRONG_BULL, MarketRegime.CAUTIOUS_BULL] and \
                 health_type in [IntradayHealthType.S_G,IntradayHealthType.LOHW])
                # or
                # (health_type in [IntradayHealthType.S_G])
            )
            return is_strong_bull
        except Exception as e:
            logger.error(f" get_strong_bull 方法异常: {e}", exc_info=True)
            return False
    
    def get_super_weak(self,market: MarketType):
        """
        判断是否为超级弱势 (仅基于 MarketRegime)
        
        弱势定义:
        - CONFIRMED_BEAR: 趋势强劲向下，只做空或空仓 ❌
        - HIGH_RISK_AVOID: VIX飙升，极端风险，保命第一 ☠️
        
        Returns:
            True: 当前处于超级弱势，必须清仓/做空
            False: 非超级弱势
        """
        try:
            current_regime = self.market_regime_engine.get_marget_regime(market)
            health_type, _ = self.market_regime_engine.check_intraday_health(market)
            
            is_super_weak = (
                        # 路径 A [顺势阴跌]: 大势已去(熊/险)，且今天毫无起色(红灯/黄灯)
                        (current_regime in [MarketRegime.CONFIRMED_BEAR, MarketRegime.HIGH_RISK_AVOID, MarketRegime.RANGE_BOUND] and
                        health_type in [IntradayHealthType.R])  #, IntradayHealthType.Y
                        or
                        # 路径 B [突发崩盘]: 不管大势如何，今天跌幅巨大(红灯)，且没有走出低开高走
                        (health_type == IntradayHealthType.R)
                    )
            return is_super_weak
        except Exception as e:
            logger.error(f" get_super_weak 方法异常: {e}", exc_info=True)
            return False
    
    def _is_price_deviation_acceptable_atr(self, symbol: str, trigger_price: float) -> bool:
        """【内置工具】使用ATR检查价格偏离度。"""

        current_price = self.get_current_price(symbol)
        if current_price is None or current_price <= 0:
            logger.error(f"[{symbol}] 无法获取有效当前价格，偏离度检查失败。")
            return False

        atr_value = get_historical_atr(self.quote_ctx, symbol)
        if atr_value is None or atr_value <= 0:
            max_dev_ratio = getattr(self.config, 'max_deviation_ratio_fallback', self.config.max_price_deviation_ratio) #0.05
            is_acceptable = abs(current_price - trigger_price) / trigger_price <= max_dev_ratio
            logger.warning(f"[{symbol}] 无法获取ATR，回退到固定百分比检查 -> {'通过' if is_acceptable else '拒绝'}")
        else:
            max_deviation_value = atr_value * getattr(self.config, 'max_deviation_atr_multiplier', 1.0)
            price_diff = abs(current_price - trigger_price)
            is_acceptable = price_diff <= max_deviation_value
            logger.info(f"[{symbol}] ATR价格偏离度检查: 触发价={trigger_price:.2f}, 现价={current_price:.2f}, "
                        f"差价={price_diff:.2f}, ATR={atr_value:.2f}, "
                        f"最大容忍差价={max_deviation_value:.2f} -> {'通过' if is_acceptable else '拒绝'}")
        return is_acceptable
    
    def _is_entering_weekend_risk_for_symbol(self, symbol: str) -> bool:
        """
        [WRP V2.0 - 上下文感知版]
        检查指定股票是否正在进入其所在市场的周末风险期。
        """
        return is_entering_weekend_risk_for_symbol(symbol,self.config.enable_wrp,self.config.wrp_activation_days)
    
    def get_current_price(self, symbol: str) -> Optional[float]:
        return self.get_realtime_price(symbol)
    
    def get_realtime_price(self, symbol: str, use_backup_for_night: bool = True) -> Optional[float]:
        """
        智能获取股票实时价格
        
        设计哲学:
        1. 场景感知: 自动识别当前是盘中(RTH)还是夜盘/盘前。
        2. 稳定性优先: 只要是长桥(LongPort)能覆盖的时段，优先用长桥。
        3. 灾备降级: 在夜盘时段，优先尝试 HS 数据源；如果 HS 挂了，自动降级回长桥兜底。
        
        Args:
            symbol: 股票代码
            use_backup_for_night: 是否在夜盘时段启用不稳定的第三方数据源 (Default: True)
            
        Returns:
            float: 最新价格，如果所有渠道都失败则返回 None
        """
        if not symbol: return None
        
        try:
            # 1. 获取市场与时段状态
            market = get_market_type(symbol)
            current_session = get_current_market_session(market)
            
            # 定义需要用到 HS 数据源的特殊时段：夜盘 (Overnight)
            # 注意：盘前(Pre)和盘后(Post)长桥通常支持得很好，不需要切源，除非你的长桥没买行情
            is_night_session = current_session in [TradingSession.OVERNIGHT_SESSION]

            # === 场景 A: 夜盘/特殊时段 (需要 HS 数据源) ===
            if is_night_session and use_backup_for_night:
                try:
                    # 优先尝试 HS 数据源
                    quote = self.hs_data_provider.get_smart_quote(symbol)
                    price = quote.get('last_price', 0.0)
                    
                    # 校验数据有效性 (防止返回0或极值)
                    if price > 0:
                        return float(price)
                    else:
                        logger.warning(f"[{symbol}] HS数据源返回无效价格: {price}，准备降级。")
                        
                except Exception as e:
                    # 捕获 HS 的不稳定异常，不让程序崩溃，并记录日志
                    logger.warning(f"[{symbol}] HS数据源调用失败 ({e})，触发熔断降级 -> 切换至 Broker 源。")

            # === 场景 B: 常规时段 或 HS失败降级 (使用 LongPort) ===
            # 且通常包含了对 quote_ctx 的封装
            quote = get_smart_quote(self.quote_ctx, symbol)
            if quote:
                price = quote.get('last_price', 0.0)
                if price > 0:
                    return float(price)
            # === 场景 C: 最后的挣扎 (尝试获取 Last Done) ===
            # 如果 snapshot 拿不到，尝试拿基础行情的 last_done
            base_quote = get_raw_quote(self.quote_ctx, symbol)
            if base_quote and float(base_quote.last_done) > 0:
                return float(base_quote.last_done)

            logger.error(f"[{symbol}] 严重: 所有数据源均无法获取有效价格！")
            return None

        except Exception as e:
            logger.error(f"[{symbol}] 统一报价网关发生未知异常: {e}", exc_info=True)
            return None

    # ==============================================================================
    # 检查某类操作在近期是否已执行
    # ==============================================================================
    def _is_action_recently_taken(self, position: Position, keyword: str, lookback_minutes: int = 20) -> bool:
        """
        检查最近 N 分钟内，是否已经执行过包含特定“关键词”的卖出操作。
        
        Args:
            position: 持仓对象
            keyword: 卖出原因中的特征词 (如 "抢收", "欧盘", "死线")
            lookback_minutes: 回溯检查的时间窗口
            
        Returns:
            bool: True 表示近期已执行过，应跳过本次操作 (防抖)。
        """
        if not position.sell_records:
            return False
            
        now_utc = datetime.now(timezone.utc)
        
        # 倒序遍历，优先检查最近的记录，效率最高
        for record in reversed(position.sell_records):
            try:
                # 1. 解析时间戳 (兼容 ISO 格式字符串)
                record_ts_str = record.get('timestamp')
                if not record_ts_str: continue
                
                record_ts = normalize_to_utc(record_ts_str)
                
                # 计算时间差 (分钟)
                delta_minutes = (now_utc - record_ts).total_seconds() / 60.0
                
                # 如果记录已经超出回溯窗口，更早的记录无需再看，直接中断
                if delta_minutes > lookback_minutes:
                    break
                
                # 2. 特征词匹配 (核心逻辑)
                # 只要历史原因中包含本次意图的关键词，即判定为重复操作
                record_reason = record.get('reason', '')
                if keyword in record_reason:
                    # logger.debug(f"[{position.symbol}] 命中防抖规则: {delta_minutes:.1f}分钟前已执行过 '{keyword}'")
                    # 高亮显示拦截动作
                    # 使用 🛡️ 图标，一眼识别是防卫系统生效
                    logger.warning(
                        f"🛡️ [{position.symbol}] 卖出防抖拦截 | "
                        f"意图: '{keyword}' | "
                        f"拦截原因: {delta_minutes:.1f}分钟前已执行过类似操作 | "
                        f"历史记录: {record_reason}"
                    )
                    return True
                    
            except Exception as e:
                logger.error(f"解析卖出记录时间戳失败: {e}")
                continue
                
        return False
    
    def _is_holding_period_satisfied(self, position: Position, required_minutes: int = 3) -> bool:
        """
        判断持仓是否满足最小冷却时间 N 分钟。
        
        改进点：
        1. 能够解析 phase_records 的嵌套列表结构 (Dict[str, List[Dict]])。
        2. 复用 utils.market_time_utils.normalize_to_utc，统一时间基准。
        3. 只要最近一次有动作（建仓、加仓），立刻重置冷却期。
        """
        if not position or not position.phase_records:
            return True # 无记录视为无需冷却（或根据风控偏好设为False）

        # 进行比较
        is_satisfied = (position.get_minutes_since_last_buy()) >= required_minutes
        
        if not is_satisfied:
            logger.info(f"🛡️ [{position.symbol}] 冷却锁生效: 最近交易于 {position.get_minutes_since_last_buy()} 分钟前，"
                        f"未满 {required_minutes} 分钟。")
                        
        return is_satisfied

    def _normalize_limit_price(self, symbol: str, raw_price: float | Decimal) -> Decimal:
        """
         价格精度强制清洗器。
        
        不管上游传进来的是 38.282 还是 38.2821234，这里统一按市场规则切断。
        防止 OpenApiException 602035 (Wrong bid size)。
        """
        if isinstance(raw_price, Decimal):
            price_val = float(raw_price)
        else:
            price_val = float(raw_price)

        market = get_market_type(symbol)
        
        # --- 核心规则库 ---
        if market == MarketType.US:
            # 美股规则：
            # 股价 >= $1.00，Tick size 通常为 0.01 (2位)
            # 股价 < $1.00，Tick size 通常为 0.0001 (4位)
            if price_val >= 1.0:
                fmt = "{:.2f}"
            else:
                fmt = "{:.4f}"
                
        elif market == MarketType.HK:
            # 港股规则复杂（仙股3位，大盘股2位或1位等）
            # 经验之谈：保留 3位小数 能够通过绝大多数长桥 API 的校验（API会自动适配）
            fmt = "{:.3f}"
            
        else:
            # A股或其他，默认2位
            fmt = "{:.2f}"

        # 格式化 -> 转字符串 -> 转Decimal (绝对安全路径)
        clean_price_str = fmt.format(price_val)
        return Decimal(clean_price_str)

    def get_current_underlying_symbol_info(self, option_symbol: str) -> Optional[dict]:
        """
        从期权代码中安全地解析标的代码并获取其当前价格。

        设计哲学：
        1.  **市场后缀驱动 (Suffix-Driven Logic):** 以 ".US", ".HK" 等市场后缀为第一识别标志，
            这是最稳定、最不可能有歧义的解析入口。
        2.  **正则表达式精确打击 (Precision Strike with Regex):** 对美股这种复杂粘合格式，
            使用正则表达式进行原子化、无差错的标的提取。
        3.  **正视复杂性 (Acknowledge Complexity):** 明确指出港股期权代码无法通过简单字符串
            操作推导出其标的，强制要求一个独立的、必须由用户实现的解析辅助函数。
            这是专业系统设计的标志：承认未知，并提供解决方案的接口。
        4.  **绝对安全 (Absolute Safety):** 任何解析路径的失败或异常，都导向唯一的、
            安全的结果：返回 None，并留下详尽的日志。
        """
        if not isinstance(option_symbol, str) or not option_symbol.strip():
            logger.warning(f"输入了无效的期权代码 (空或非字符串): '{option_symbol}'")
            return None

        try:
            # --- 步骤 1: 市场识别 ---
            # option_symbol.upper().rsplit('.', 1) 从右边分割一次，完美分离主体和后缀
            parts = option_symbol.upper().rsplit('.', 1)
            if len(parts) != 2:
                logger.error(f"无法识别期权代码 '{option_symbol}' 的市场后缀。预期格式为 'CODE.MARKET'")
                return None
            
            core_symbol, market = parts
            
            # --- 步骤 2: 分市场进行标的解析 ---
            underlying_symbol = None
            if market == MarketType.US.name:
                underlying_symbol = get_underlying_from_option_symbol(option_symbol)
                # # 美股格式: AAPU251017C27000
                # # 正则表达式解析：匹配开头的1到5个字母（可能包含点，如BRK.A）
                # import re
                # match = re.match(r'^([A-Z]{1,5}\.?[A-Z]?)\d{6}[CP]\d+', core_symbol)
                # if match:
                #     underlying_symbol = match.group(1)
                # else:
                #     logger.error(f"美股期权代码 '{core_symbol}' 格式不符合预期，无法提取标的。")
                #     return None

            elif market == MarketType.HK.name:
                # 【核心】必须调用辅助函数来获取真正的标的
                underlying_symbol = self._get_underlying_for_hk_option(option_symbol)
                if not underlying_symbol:
                    logger.error(f"缺少港股代码 {core_symbol} 到其正股的映射，无法获取价格。")
                    return None # 辅助函数已记录错误日志
            else:
                logger.warning(f"不支持的市场类型 '{market}' for symbol '{option_symbol}'")
                return None

            # --- 步骤 3: 获取价格 ---
            if underlying_symbol:
                # 对解析出的美股代码进行标准化
                price_symbol = parse_symbol(underlying_symbol) if market == MarketType.US.name else underlying_symbol
                current_price = self.get_current_price(price_symbol)
                if current_price:
                    return dict(current_price=float(current_price),underlying_symbol=price_symbol)
                else:
                    logger.error(f"get_current_price '{price_symbol}'为 None。")
                    return None
            else:
                # 这是一个兜底，正常逻辑不应走到这里
                logger.error(f"未能为 '{option_symbol}' 确定用于获取价格的标的代码。")
                return None

        except Exception as e:
            logger.error(f"为期权代码 '{option_symbol}' 获取标的价格时发生未知错误: {e}", exc_info=True)
            return None

    def _get_underlying_for_hk_option(self, hk_option_code: str) -> Optional[str]:
        """
        【关键】获取港股期权/涡轮对应的正股代码。
        这是一个占位符函数！港股期权/涡轮代码（如 '18272'）
        与其正股代码（如 '00700.HK'）之间没有直接的、通用的换算规则。
        你必须通过你的券商API或一个映射表来实现这个逻辑。
        这是一个绝对不能省略的步骤。
        """
        
        """
        # 伪代码：实际场景需要查表或API
        if hk_option_code == '18272':
            return '09626.HK'
        elif hk_option_code == '19334':
            return '03750.HK'
        elif hk_option_code == '00857':
            return '03750.HK'
        elif hk_option_code == '17969':
            return '01810.HK'
        elif hk_option_code == '19705':
            return '09992.HK'
        elif hk_option_code == '18710':
            return '02628.HK'
        elif hk_option_code == '14651':
            return '09988.HK'
        elif hk_option_code == '68009':
            return '09618.HK'
        
        # ... 其他映射
        logger.error(f"缺少港股代码 {hk_option_code} 到其正股的映射，无法获取价格。")
        """
        return resolve_underlying_symbol(hk_option_code)

    def get_cached_stock_static_info(self,symbol: str) -> dict:
        """带缓存的获取股票静态信息的方法"""
        return get_stock_static_info(self.quote_ctx, symbol)

    def _adjust_quantity(self, quantity, market: MarketType, lot_size: int = 100) -> int:
        """根据市场规则和每手股数，调整最终下单股数。"""
        if not isinstance(quantity, (int, float)) or quantity < 0:
            logger.warning(f"输入的数量无效: {quantity}，将返回0。")
            return 0
        quantity = int(quantity)
        if market == MarketType.US: return quantity
        elif market == MarketType.HK:
            if lot_size <= 0: logger.error(f"港股 lot_size 无效: {lot_size}"); return 0
            return (quantity // lot_size) * lot_size
        raise ValueError(f"不支持的市场类型: {market}")

    def _calculate_smart_sell_quantity(self, symbol: str, position: Position, target_sell_ratio: float, current_price: float) -> Tuple[int, Optional[str]]:
        """
        【智能卖出计算器】(The Smart Exit Calculator)
        
        核心逻辑：
        1. 价值锚定：只留有意义的仓位（> $300 / > 2500 HKD）。
        2. 比例清洗：拒绝保留 10% 以下的垃圾尾仓，除非它是像 BRK.A 那样的金砖。
        3. 手数对齐：非清仓场景下，严格遵守交易所 Lot Size 规则。
        
        Args:
            symbol: 股票代码
            position: 持仓对象
            target_sell_ratio: 目标卖出比例 (0.0 - 1.0)
            current_price: 当前市价
            
        Returns:
            (final_quantity, adjust_reason)
            - final_quantity: 最终决定的卖出股数
            - adjust_reason: 调整原因 (如果触发了清洗逻辑)，否则为 None
        """
        total_qty = position.total_quantity
        if total_qty <= 0: return 0, "空仓"
        
        # 1. 计算理论计划卖出量 (向下取整)
        raw_sell_qty = int(total_qty * target_sell_ratio)
        
        # 2. 预测剩余状态
        remaining_qty = total_qty - raw_sell_qty
        remaining_val = remaining_qty * current_price
        
        # 3. 动态设定价值阈值 (Hardcoded Experience)
        # 美股 $300，港股 2500 HKD。低于这个数，留着不够交电费。
        min_value_threshold = self.config.min_holding_value_threshold.get(position.market.value,200)
        
        # 4. 获取最小交易单位
        stock_info = self.get_cached_stock_static_info(symbol)
        lot_size = stock_info.get('lot_size', 100) if position.market == MarketType.HK else 1

        # === 核心判决逻辑 (The Verdict) ===
        clean_sweep_reason = None

        # [规则 A] 物理归零：如果剩下了 0 股或负数 -> 全卖
        if remaining_qty <= 0:
            return total_qty, None # 本意就是全卖，无需理由

        # [规则 B] 价值底线 (Value Floor) -> 最强规则
        # 逻辑：无论还剩多少股，只要总值太低，统统清掉。
        if remaining_val < min_value_threshold:
            clean_sweep_reason = f"剩余市值过低 ({remaining_val:.0f} < {min_value_threshold})"

        # [规则 C] 比例陷阱 (Ratio Trap) -> 尾仓清理
        # 逻辑：如果卖出后只剩不到 15%，且剩下的钱不足 5倍阈值 (非重仓)，清掉。
        # 保护：如果有 1000 股 NVDA，卖了 900 股，剩 100 股 (10%) 值 10万，这绝对不能清。
        elif (remaining_qty / total_qty) < self.config.min_residual_ratio and remaining_val < (min_value_threshold * 5):
            clean_sweep_reason = f"剩余比例过低且非重仓 ({(remaining_qty/total_qty):.1%} < 15%)"

        # [规则 D] 港股碎股 (Odd Lot)
        # 逻辑：港股如果不满一手，且价值一般，建议清掉，因为碎股卖出价格差。
        elif position.market == MarketType.HK and remaining_qty < lot_size:
             if remaining_val < (min_value_threshold * 2): # 给碎股一点容忍度，但不多
                 clean_sweep_reason = f"港股碎股清理 (剩 {remaining_qty} < 1手)"

        # === 5. 输出裁决 ===
        if clean_sweep_reason:
            # 触发清洗 -> 强制全卖
            return total_qty, clean_sweep_reason
        else:
            # 正常卖出 -> 必须进行 Lot Size 对齐
            adjusted_sell = self._adjust_quantity(raw_sell_qty, position.market, lot_size)
            
            # [最后一道防线]
            # 如果 Lot Size 取整后变成了 0 (例如持有 50 股，卖 10%，得 5 股，港股一手 100，取整为 0)
            if adjusted_sell == 0:
                # 如果总市值本身就很低，那就别耗着了，全卖
                total_val = total_qty * current_price
                if total_val < min_value_threshold:
                    return total_qty, "计算量为0且总值低 -> 强制全卖"
                else:
                    # 否则，放弃本次微不足道的卖出
                    return 0, "计算量不足一手，取消卖出"
            
            return adjusted_sell, None
    
    # ==============================================================================
    # ▼▼▼  物理动能与结构门禁 (Physical Integrity Gate) ▼▼▼
    # 逻辑：加仓必须满足“资金增量、空间开阔、VWAP站稳”三个物理事实。
    # ==============================================================================
    
    def _verify_trade_quality_gate(self, symbol: str, current_price: float, mode: str = 'CONTINUATION') -> Tuple[bool, str]:
        """
        【全球顶尖·动态准入网关】
        
        设计哲学：
        1. 弹性支撑：强牛市下，VWAP/POC 的支撑判定从“线性”变为“带状区域”，允许回踩深水区。
        2. 拒绝真空：即便是牛市，也不允许在完全脱离物理意义的真空区加仓。
        3. 乖离熔断：防止在极度亢奋时追高接盘（Anti-Climax）。
        """
        return True, "保守策略模式：交易质量网关已禁用"

        # --- 1. 战场全息数据采集 ---
        symbol = resolve_underlying_symbol(symbol)
        infr = get_institutional_net_flow_ratio(self.quote_ctx, symbol) or 0.0
        poc_price = get_intraday_poc(self.quote_ctx, symbol)
        
        # 提取市场状态：这是决策的“底色”
        market = get_market_type(symbol)

        is_strong_bull = self.get_strong_bull(market)
        # 获取 VWAP
        quote_data = get_smart_quote(self.quote_ctx, symbol)
        vwap = 0.0
        if quote_data and float(quote_data.get('volume', 0)) > 0:
            vwap = float(quote_data.get('turnover', 0)) / float(quote_data.get('volume', 1))

        # 获取 GEX 墙位
        call_wall_price = 0.0
        put_wall_price = 0.0
        profile = None
        if market==MarketType.US:
            profile = self.gex_engine._calculate_gex_profile_vectorized(symbol)
        if profile and profile.get("gamma_walls"):
            gamma_walls = profile["gamma_walls"]
            call_walls = [w for w in gamma_walls if w['gex'] > 0]
            put_walls = [w for w in gamma_walls if w['gex'] < 0]
            # 取最大GEX绝对值对应的价格
            if call_walls: call_wall_price = max(call_walls, key=lambda x: abs(x['gex'])).get('price', 0.0)
            if put_walls: put_wall_price = max(put_walls, key=lambda x: abs(x['gex'])).get('price', 0.0)

        # --- 2. 核心判决矩阵 ---

        # 场景 A: 推进加仓 (Continuation) —— 侧重于“势”
        if mode == 'CONTINUATION':
            # [硬指标1: 资金流] 
            # 强势市场豁免，只要不低于 -1.5% 就不算崩；弱势市场必须 > -0.5%
            infr_threshold = -0.015 if is_strong_bull else -0.05
            if infr < infr_threshold:
                return False, f"资金意志背离(INFR:{infr:.2%})"

            # 动态配置“有效支撑”的定义
            if is_strong_bull:
                # [强牛模式]
                # VWAP: 允许回踩下方 0.5% (洗盘常见幅度)
                # POC: 允许回踩下方 0.8% (筹码区通常较深)
                # Wall: 期权墙极其坚硬，给予 1.0% 的缓冲区
                # Max_Dev: 允许偏离均线 3.5% 追涨 (动能溢价)
                vwap_tolerance = 0.995
                poc_tolerance = 0.992
                wall_tolerance = 0.990
                max_deviation = 0.045
            else:
                # [标准模式]
                # 严防死守，哪怕跌破 0.1% 都要审视
                vwap_tolerance = 0.999
                poc_tolerance = 0.998
                wall_tolerance = 0.998
                max_deviation = 0.030

            # 检查是否踩在任意一个“宽容后的”支撑位之上
            # 这里的逻辑是：Current Price 只要大于 (KeyLevel * Tolerance)，就算踩稳。
            has_structure_support = False
            support_details = []

            if poc_price and current_price > poc_price * poc_tolerance:
                has_structure_support = True
                support_details.append(f"POC(>{poc_tolerance:.3f})")

            if not has_structure_support and put_wall_price and current_price > put_wall_price * wall_tolerance:
                has_structure_support = True
                support_details.append(f"PutWall(>{wall_tolerance:.3f})")

            if not has_structure_support and vwap > 0 and current_price > vwap * vwap_tolerance:
                has_structure_support = True
                support_details.append(f"VWAP(>{vwap_tolerance:.3f})")

            # 判决：如果连放宽后的支撑都没踩到，那是真的“悬空”
            if not has_structure_support:
                # 唯一的特例：强牛市突破创新高 (Breakout)，且动能极强，允许暂时脱离下方支撑
                is_breakout = is_strong_bull and call_wall_price and current_price > call_wall_price
                if not is_breakout:
                    return False, f"结构悬空:未踩稳任何支撑(模式:{'强牛' if is_strong_bull else '标准'})"

            # [硬指标3: 乖离率熔断] (Anti-Climax)
            # 防止在强牛市中，因为容忍度高，导致在 VWAP 上方极远处追高
            if vwap > 0:
                deviation = (current_price / vwap) - 1
                if deviation > max_deviation:
                    return False, f"动能透支:乖离率过大({deviation:.2%}>{max_deviation:.1%})，等待回归"

            # [硬指标4: 空间阻力] (可选，非强牛市检查 CallWall 压制)
            if not is_strong_bull and call_wall_price:
                dist_to_call = (call_wall_price - current_price) / current_price
                if 0 < dist_to_call < 0.008:
                    return False, f"空间压制(距CallWall仅{dist_to_call:.2%})"


        # 场景 B: 下跌补仓 (Dip Add) —— 侧重于“地基”
        elif mode == 'DIP_ADD':
            # 补仓绝不容忍资金大幅外逃
            if infr < -0.05:
                return False, f"接飞刀风险:抛压剧烈(INFR:{infr:.2%})"

            # 补仓必须有“硬地板”确认
            # 只要价格在 PutWall 或 POC 的 1.5% 范围内，视为踩稳
            floor_price = max(put_wall_price, poc_price or 0)
            if floor_price > 0:
                dist_to_floor = (current_price - floor_price) / floor_price
                if dist_to_floor < -0.01: # 有效跌破
                    return False, f"地基陷落:价格低于物理支撑({floor_price:.2f})"
                if dist_to_floor > 0.025: # 离地板太远，还在半山腰
                    return False, "悬空补仓:离支撑位过远"
            else:
                # 既无POC也无GEX数据，补仓需极其谨慎，要求 VWAP 必须在下方
                if vwap > 0 and current_price < vwap:
                    return False, "无锚点补仓且在VWAP下方"

        return True, f"物理网关通过({'强势模式' if is_strong_bull else '标准模式'})"
    
    def _is_approaching_profit_target(self, position: Position, current_price: float) -> bool:
        """
        检查是否已接近R倍数止盈目标。
        """
        if position.r_profit_taken: # 如果已经止盈过，则此检查无效
            return False

        market_key = position.market.value
        r_multiple_rule = self.config.profit_take_r_by_market.get(market_key)

        if r_multiple_rule and r_multiple_rule.get('enable', False):
            if position.initial_risk_per_share > 0 and position.initial_scout_price > 0:
                # 目标是 R 倍，我们检查是否达到了 R * 0.9 (90%)
                approaching_profit_target = (r_multiple_rule['profit_take_r_multiple'] * 0.9) * position.initial_risk_per_share
                current_profit = current_price - position.initial_scout_price
                return current_profit >= approaching_profit_target
        
        return False
    
    
    def _manage_breakeven_stop(self, symbol: str, current_price: float, position: Position):
        """管理保本止损的设置逻辑。"""
        if position.r_profit_taken: return
        if not self.config.enable_breakeven_stop or position.is_breakeven_stop_set or position.is_trailing_stop_active: return

        if position.initial_risk_per_share > 0 and position.initial_scout_price > 0:
            breakeven_trigger_profit = self.config.breakeven_trigger_r_multiple * position.initial_risk_per_share
            current_profit = current_price - position.initial_scout_price
            
            if current_profit >= breakeven_trigger_profit:
                with self.position_lock:
                    if symbol in self.positions and not self.positions[symbol].is_breakeven_stop_set:
                        pos = self.positions[symbol]
                        pos.trailing_stop_price = pos.avg_cost  # 核心动作：将追踪止损价直接设为平均成本，实现保本
                        pos.is_breakeven_stop_set = True
                        pos.is_trailing_stop_active = True # [NEW] 同时激活追踪止损，让这个保本止损立即生效
                        self._save_positions()
                        logger.warning(f"★★★ 利润保护触发 for {symbol}! 止损位上移至成本价 {pos.avg_cost:.2f}。")

    def _manage_trailing_stop_activation(self, symbol: str, current_price: float, position: Position):
        """管理利润达标，激活追踪止损。"""
        if position.is_trailing_stop_active or position.initial_risk_per_share <= 0: return
        
        activation_profit = self.config.trailing_stop_activation_multiplier * position.initial_risk_per_share
        activation_price = position.avg_cost + activation_profit
        
        if current_price >= activation_price:
            with self.position_lock:
                if symbol in self.positions and not self.positions[symbol].is_trailing_stop_active:
                    pos = self.positions[symbol]
                    pos.is_trailing_stop_active = True
                    new_trailing_stop = current_price * (1 - self.config.trailing_stop_ratio)
                    pos.trailing_stop_price = max(pos.avg_cost, new_trailing_stop)
                    self._save_positions()
                    logger.warning(f"★★★ 追踪止损已激活 for {symbol}! 初始追踪止损位: {pos.trailing_stop_price:.2f}")
   
    def _manage_trailing_stop_update(self, symbol: str, current_price: float, position: Position):
        """管理已激活追踪止损的“更新”逻辑。"""
        if not position.is_trailing_stop_active: return

        # 1. 获取基于日线的、稳定的历史ATR值
        # 这个值在一天内基本是固定的，非常适合作为战略止损的基石
        daily_atr = get_historical_atr(self.quote_ctx, symbol)
        
        # 如果无法获取ATR，安全回退到原来的固定百分比逻辑
        if daily_atr is None or daily_atr <= 0:
            new_trailing_stop = current_price * (1 - self.config.trailing_stop_ratio)
        else:
            # 2. 【核心】使用ATR来定义回撤幅度
            # 你需要在config.py中增加 trailing_stop_atr_multiplier 这个新参数，建议值为1.5到2.5之间
            trailing_stop_offset = daily_atr * self.config.trailing_stop_atr_multiplier
            new_trailing_stop = current_price - trailing_stop_offset
            if random.random() > 0.80:
                logger.warning(f'股票:{symbol}，new_trailing_stop:{new_trailing_stop}')

        # 使用智能止损更新追踪止损
        # new_trailing_stop = self.adaptive_stop_loss.update_trailing_stop(
        #     symbol, current_price, position.trailing_stop_price, 'long'
        # )

        # new_trailing_stop = current_price * (1 - self.config.trailing_stop_ratio)
        if new_trailing_stop > position.trailing_stop_price:
            with self.position_lock:
                if symbol in self.positions:
                    self.positions[symbol].trailing_stop_price = new_trailing_stop
                    self._save_positions()
                    logger.debug(f"上移追踪止损位 {symbol}: 新位置 {new_trailing_stop:.2f}")

    def _start_main_monitor(self):
        """启动核心持仓监控线程（止损、止盈、加仓等）"""
        if self.main_monitor_thread and self.main_monitor_thread.is_alive(): return
        self.stop_main_monitor.clear()
        self.main_monitor_thread = threading.Thread(target=self._main_monitor_loop, daemon=True)
        self.main_monitor_thread.start()
        logger.info("启动主监控线程...")

    def _start_sell_signal_monitor(self):
        """启动独立的策略性卖出信号监控线程"""
        # 启动双轨制卖出监控架构。
        # 增加了线程状态的严格检查，防止重复启动导致的资源竞争。

        # --- 轨道一：盘中策略监控 (Intraday) ---
        # 负责：常规交易时间的策略信号

        # if self.sell_signal_monitor_thread and self.sell_signal_monitor_thread.is_alive(): return
        # self.stop_sell_monitor.clear()
        # self.sell_signal_monitor_thread = threading.Thread(target=self._sell_signal_monitor_loop, daemon=True,name="Thread-IntradayMonitor")
        # self.sell_signal_monitor_thread.start()
        logger.info("启动策略性卖出信号监控线程...")

        # --- 轨道二：盘后智能清算 (After-Hours) ---
        # 负责：盘后35分钟内的回撤止盈与死线兜底
        if not hasattr(self, 'ah_monitor_thread') or \
           not self.ah_monitor_thread or \
           not self.ah_monitor_thread.is_alive():
            
            self.ah_monitor_thread = threading.Thread(
                target=self._after_hours_monitor_loop, 
                daemon=True, 
                name="Thread-AfterHoursCleaner"
            )
            self.ah_monitor_thread.start()
            logger.info("✅ [启动] 盘后智能清算线程 (AfterHours) 已就绪...")
    
    def _start_pending_signal_monitor(self):
        """启动独立的待买入信号监控线程"""
        if self.pending_signal_monitor_thread and self.pending_signal_monitor_thread.is_alive(): return
        self.stop_pending_monitor.clear()
        # self.pending_signal_monitor_thread = threading.Thread(target=self._pending_signal_monitor_loop, daemon=True, name="PendingSignalThread")
        # self.pending_signal_monitor_thread.start()
        # logger.info("✅ 待买入信号监控线程已启动...")
    
        # 目标函数指向 _signal_replication_monitor_loop
        self.pending_signal_monitor_thread = threading.Thread(
            target=self._signal_replication_monitor_loop,
            daemon=True,
            name="SignalReplicationThread"
        )
        self.pending_signal_monitor_thread.start()
        logger.info("✅ [线程启动] 信号复制监控线程已就绪 (Replica Mode)")

    def _start_pre_market_monitor(self):
        """启动夜盘/盘前独立监控线程"""
        if self.pre_market_monitor_thread and self.pre_market_monitor_thread.is_alive(): return
        self.stop_pre_market_monitor.clear()
        self.pre_market_monitor_thread = threading.Thread(
            target=self._pre_market_monitor_loop, 
            daemon=True, 
            name="NightHunterThread"
        )
        self.pre_market_monitor_thread.start()
        logger.info("✅ 夜盘/盘前/盘后'夜猎者'监控线程已启动...")


if __name__ == '__main__':
    setup_logging()
    logger = logging.getLogger(__name__)
    logger.info("======================================================")
    logger.info("               交易系统启动                           ")
    logger.info("======================================================")

    # logger.info("="*60 + "\n" + "交易系统启动".center(56) + "\n" + "="*60)
    cfg_file = os.path.join(project_path, 'configs/server_config_test.yaml')
    
    try:
        cfg = load_yaml2cfg(cfg_file)
        longbridge_config = get_longbridge_config(cfg)
    except FileNotFoundError:
        logger.critical(f"严重错误: 配置文件 {cfg_file} 未找到。程序无法启动。")
        sys.exit(1)
    except Exception as e:
        logger.critical(f"严重错误: 加载配置文件 {cfg_file} 时出错: {e}", exc_info=True)
        sys.exit(1)

    system = None
    try:
        system = TradingSystem(cfg_file=cfg_file, longbridge_config=longbridge_config)
        logger.info(f"系统初始化成功。当前模式: {'测试模式' if system.config.test_mode else '实盘模式'}")
        logger.info(">>> 系统进入主策略循环，监控市场信号... (按 Ctrl+C 安全退出)")
        system.run_strategy_loop()
        # system.execute_tactical_clearance()
    except KeyboardInterrupt:
        logger.info("检测到用户中断 (Ctrl+C)，正在安全关闭系统...")
    except Exception as main_exc:
        logger.critical(f"主程序发生无法恢复的严重错误: {main_exc}", exc_info=True)
        # 【告警集成】 13. 整个系统发生致命错误，程序即将退出
        error_msg = f"P1级告警: 交易系统主程序发生致命错误，程序已终止! 错误: {main_exc}"
        logger.critical(error_msg, exc_info=True)
        if system and hasattr(system, 'notification_manager'):
            system.notification_manager.send_critical_alert(error_msg)
        else:
            send_weixin_notice(error_msg)
    finally:
        if system:
            system.shutdown()
        else:
            logger.warning("系统对象未成功初始化，无需关闭。")
    logger.info("======================================================")
    logger.info("               交易系统已关闭                           ")
    logger.info("======================================================")
