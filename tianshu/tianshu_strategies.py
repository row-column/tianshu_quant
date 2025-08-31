import os, sys
project_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_path)
import pandas as pd
import pandas_ta as ta
import numpy as np
from enum import Enum
from typing import Tuple,Optional,Dict
from .event import SignalEvent, events
from datetime import datetime, timezone
from utils.market_time_utils import normalize_to_utc

# 这是一个改造后的策略基类，用于回测框架
class BacktestStrategy:
    """所有用于“天枢”回测系统的策略都应继承此类。"""
    def __init__(self, data_handler, symbol_list, **kwargs):
        self.data_handler = data_handler
        self.symbol_list = symbol_list

    @property
    def name(self):
        raise NotImplementedError

    def calculate_signals(self, event,held_symbols: list, positions: dict):
        raise NotImplementedError
    
    def calculate_atr_stop_loss(self, df_daily: pd.DataFrame,atr_stop_loss_multiplier:float = 1.8) -> Optional[float]:
        """
        100% 复刻你的 get_historical_atr 逻辑，但在回测数据上运行。
        它计算并返回基于今日买入价和昨日ATR的精确止损价。
        """
        if df_daily is None or len(df_daily) < self.atr_period + 2:
            return None
            
        try:
            # 1. 获取“昨天”的ATR值 (iloc[-2])，这与你的实盘逻辑完全一致
            yesterday_atr = df_daily.iloc[-2].get('atr')
            if pd.isna(yesterday_atr) or yesterday_atr <= 0:
                return None

            # 2. 获取“今天”的价格作为买入基准。在日线回测中，通常使用收盘价模拟成交。
            today_price = df_daily.iloc[-1]['close']

            # 3. 计算单笔风险和止损价，公式与你实盘代码完全一致
            atr_defined_risk = yesterday_atr * atr_stop_loss_multiplier
            stop_loss_price = today_price - atr_defined_risk
            return stop_loss_price
        except Exception as e:
            print(f"警告: ATR止损价计算失败. {e}")
            return None

# ==============================================================================
# === 【V2.0 修复版】“禁卫军”全天候作战平台 (回测专用版) ===
# ==============================================================================
class PraetorianSetupType(Enum):
    NONE = "无有效设置"
    APEX_PREDATOR_PULLBACK = "大势龙头·回调"
    GAP_COMMANDO = "缺口突击队"

class PraetorianStrategyForBacktest(BacktestStrategy):
    """
    - 引入了统一的 _calculate_indicators 数据预处理方法，根治了 KeyError。
    - 优化了数据流，确保所有决策都基于一份完整的、包含所有指标的数据。
    """
    def __init__(self, data_handler, symbol_list, **kwargs):
        super().__init__(data_handler, symbol_list, **kwargs)
        # --- [1] Setup 审查模块参数 (全部参数化) ---
        self.long_term_ma_period = kwargs.get('long_term_ma_period', 60)
        self.key_support_ma_period = kwargs.get('key_support_ma_period', 10)
        self.ma_distance_threshold = kwargs.get('ma_distance_threshold', 0.08)
        self.volume_spike_quantile = kwargs.get('volume_spike_quantile', 0.90)
        self.volume_shrink_ratio = kwargs.get('volume_shrink_ratio', 0.8)
        self.atr_period = kwargs.get('atr_period', 14)
        self.lookback_period = kwargs.get('lookback_period', 20)
        self.vcp_lookback_period = kwargs.get('vcp_lookback_period', 15)
        self.vcp_max_width_pct = kwargs.get('vcp_max_width_pct', 0.12)
        self.gap_min_pct = kwargs.get('gap_min_pct', 0.04)
        self.atr_multiplier_breakout = kwargs.get('atr_multiplier_breakout', 0.25)
        self.atr_stop_loss_multiplier = kwargs.get('atr_stop_loss_multiplier', 1.8) # 默认值取自你实盘config

    @property
    def name(self):
        return "禁卫军策略 (回测修复版)"

    # --- 【核心修正】 新增数据预处理中心 ---
    def _calculate_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        统一计算所有策略逻辑需要的技术指标。
        这是所有分析开始前的第一步。
        """
        if df.empty:
            return df
            
        # 使用 try...except 包裹，防止因单一指标计算失败导致整个策略崩溃。
        try:
            df['ema_long'] = ta.ema(df['close'], length=self.long_term_ma_period)
            df['ma10'] = ta.sma(df['close'], length=self.key_support_ma_period)
            df['ma20'] = ta.sma(df['close'], length=20)
            df['ma30'] = ta.sma(df['close'], length=30)
            df['avg_volume_20'] = ta.sma(df['volume'], length=20)
            df['atr'] = ta.atr(df['high'], df['low'], df['close'], length=self.atr_period)
        except Exception as e:
            # 如果计算失败，返回原始df，后续的检查会因为缺少列而自然失败。
            print(f"警告: 指标计算失败. {e}")
            pass
            
        return df

    def calculate_signals(self, event, held_symbols: list, positions: dict):
        """
        现在只对 `symbol_list` 中 **不包含** 在 `held_symbols` 里的股票进行检查。
        """
        if event.type != 'MARKET':
            return

        for s in self.symbol_list:
            if s not in held_symbols:
                df_raw = self.data_handler.get_latest_bars(s, N=252)
                if df_raw.empty or len(df_raw) < self.long_term_ma_period + 20:
                    continue
                
                df_daily = self._calculate_indicators(df_raw)
                
                setup_type, setup_data = self._find_high_quality_setup_backtest(s, df_daily)
                if setup_type == PraetorianSetupType.NONE:
                    continue
                
                is_triggered, trigger_info = self._find_precision_trigger_backtest(s, setup_type, setup_data, df_daily)
                if not is_triggered:
                    continue

                # --- 在发送信号前，计算止损价 ---
                # --- 【核心修改】在这里调用“火控计算机” ---
                stop_loss_price = self.calculate_atr_stop_loss(df_daily)
                
                # 如果无法计算出科学的止损价，则放弃这次交易，这是专业风控的体现
                if stop_loss_price is None:
                    continue

                current_timestamp = df_daily.index[-1]
                print(
                    f"[{current_timestamp.strftime('%Y-%m-%d')}] ★★★ 买入信号 ★★★\n"
                    f"  - 股票: {s}\n"
                    f"  - 日线背景: {setup_type.value}\n"
                    f"  - 作战命令: {trigger_info}\n"
                    f"  - 科学止损价: {stop_loss_price:.2f}"
                    )
                signal = SignalEvent(s, current_timestamp, 'LONG', strategy_name=self.name,stop_loss_price=stop_loss_price)
                events.put(signal)

    def calculate_signals_v1(self, event):
        """
        【修改】主逻辑现在会检查股票是否已持仓，只对未持仓的股票检查买入信号。
        """
        if event.type != 'MARKET':
            return

        # --- 【新增】从 Portfolio 获取当前持仓 ---
        # 杠精注释：在真实系统中，这个信息应该由 Backtest 引擎传入，
        # 但为了简化，我们这里假设可以直接访问（这是一个小小的架构妥协）。
        # 在我们的最终版 backtest.py 中，这个信息将通过 event 或参数传递。
        # 此处我们依赖于 Portfolio 的状态，这在我们的架构中是允许的。
        
        # 实际上，Portfolio的持仓状态是独立的，策略不应该直接知道。
        # 策略只负责产生信号。Portfolio会根据信号和持仓状态决定是否下单。
        # 所以我们不需要 `held_symbols`，只需要避免对同一支股票重复发送买入信号即可。
        
        for s in self.symbol_list:
            df_daily = self.data_handler.get_latest_bars(s, N=252)
            if df_daily.empty or len(df_daily) < self.long_term_ma_period + 20:
                continue

            setup_type, setup_data = self._find_high_quality_setup_backtest(s, df_daily)
            if setup_type == PraetorianSetupType.NONE:
                continue
            
            is_triggered, trigger_info = self._find_precision_trigger_backtest(s, setup_type, setup_data, df_daily)
            if not is_triggered:
                continue

            current_timestamp = df_daily.index[-1]
            print(
                f"[{current_timestamp.strftime('%Y-%m-%d')}] ★★★ 买入信号 ★★★\n"
                f"  - 股票: {s}\n"
                f"  - 日线背景: {setup_type.value}\n"
                f"  - 作战命令: {trigger_info}"
            )
            # 【关键】发送的是 'LONG' 信号
            signal = SignalEvent(s, current_timestamp, 'LONG', strategy_name=self.name)
            events.put(signal)

    def _find_high_quality_setup_backtest(self, symbol: str, df_daily: pd.DataFrame) -> Tuple[PraetorianSetupType, Optional[Dict]]:
        # 1. 检查“缺口突击”Setup
        is_gap_setup, gap_data = self._check_gap_commando_setup_backtest(df_daily)
        if is_gap_setup:
            return PraetorianSetupType.GAP_COMMANDO, gap_data

        # 2. 检查“大势龙头·回调”Setup
        is_pullback_setup, pullback_data = self._check_apex_predator_setup_backtest(df_daily)
        if is_pullback_setup:
            return PraetorianSetupType.APEX_PREDATOR_PULLBACK, pullback_data

        return PraetorianSetupType.NONE, None

    def _check_apex_predator_setup_backtest(self, df_history: pd.DataFrame) -> Tuple[bool, Optional[Dict]]:
        # --- 【代码移除】不再需要 df = df_history.copy() 和重复的指标计算 ---
        # --- 因为所有指标都已提前在主DataFrame上计算完毕 ---
        if len(df_history) < 2: return False, None
        yesterday = df_history.iloc[-2]
        
        # 健壮性检查：确保我们需要的列都存在
        required_cols = ['close', 'ema_long', 'ma10', 'ma20', 'ma30', 'avg_volume_20', 'volume']
        if yesterday.isnull().any() or not all(col in yesterday.index for col in required_cols):
            return False, None
        
        if not (yesterday['close'] > yesterday['ema_long']): return False, None
        max_ma = max(yesterday['ma10'], yesterday['ma20'], yesterday['ma30'])
        min_ma = min(yesterday['ma10'], yesterday['ma20'], yesterday['ma30'])
        if (max_ma - min_ma) / min_ma > self.ma_distance_threshold: return False, None
        
        # 【逻辑优化】确保切片不会越界
        start_index = -(self.lookback_period + 2)
        if abs(start_index) > len(df_history): return False, None # 如果数据不够长
        
        volume_quantile_threshold = df_history['volume'].iloc[start_index:-2].quantile(self.volume_spike_quantile)
        if not (df_history['volume'].iloc[start_index:-2] > volume_quantile_threshold).any(): return False, None
        
        if yesterday['volume'] > yesterday['avg_volume_20'] * self.volume_shrink_ratio: return False, None
        
        is_vcp_confirmed = False
        vcp_start_index = -(self.vcp_lookback_period + 2)
        if abs(vcp_start_index) <= len(df_history):
            consolidation_df = df_history.iloc[vcp_start_index:-2]
            if not consolidation_df.empty:
                platform_high = consolidation_df['high'].max()
                platform_low = consolidation_df['low'].min()
                if platform_low > 0:
                    platform_width_pct = (platform_high - platform_low) / platform_low
                    if platform_width_pct <= self.vcp_max_width_pct:
                        is_vcp_confirmed = True
       
        setup_data = {
            "key_support_level": yesterday['ma10'],
            "is_vcp_setup": is_vcp_confirmed,
            "yesterday_high": yesterday['high'],
            "yesterday_atr": yesterday.get('atr', 0) # 使用.get增加健壮性
        }
        return True, setup_data

    def _check_gap_commando_setup_backtest(self, df_daily: pd.DataFrame) -> Tuple[bool, Optional[Dict]]:
        if len(df_daily) < self.lookback_period + 2: return False, None
        
        today = df_daily.iloc[-1]
        yesterday = df_daily.iloc[-2]
        
        gap_pct = (today['open'] / yesterday['close']) - 1.0
        if gap_pct < self.gap_min_pct: return False, None
        
        launchpad_df = df_daily.iloc[-(self.lookback_period + 2):-2]
        if launchpad_df.empty: return False, None
        max_price_in_pad = launchpad_df['high'].max()
        if today['open'] < max_price_in_pad: return False, None
        
        return True, {"breakout_level_daily": max_price_in_pad}
        
    def _find_precision_trigger_backtest(self, symbol: str, setup_type: PraetorianSetupType, setup_data: Dict, df_daily: pd.DataFrame) -> Tuple[bool, str]:
        today = df_daily.iloc[-1]

        if setup_type == PraetorianSetupType.APEX_PREDATOR_PULLBACK:
            yesterday_high = setup_data['yesterday_high']
            yesterday_atr = setup_data['yesterday_atr']
            
            breakout_level = yesterday_high + self.atr_multiplier_breakout * yesterday_atr
            if today['high'] > breakout_level and today['close'] > breakout_level:
                # --- 【关键修正】从df_daily.iloc[-2]获取avg_volume_20 ---
                avg_vol_yesterday = df_daily.iloc[-2].get('avg_volume_20')
                if avg_vol_yesterday and today['volume'] > avg_vol_yesterday * 1.5:
                     return True, f"模拟闪电战: 日线级别放量突破ATR增强位({breakout_level:.2f})"
            
            is_strong_bullish_candle = (today['close'] > today['open']) and \
                                       ((today['high'] - today['close']) / (today['high'] - today['low'] + 1e-9) < 0.3)
            
            past_10_days = df_daily.iloc[-11:-1]
            down_day_volumes = past_10_days[past_10_days['close'] < past_10_days['open']]['volume']
            max_down_volume = down_day_volumes.max() if not down_day_volumes.empty else 0
            
            if is_strong_bullish_candle and today['volume'] > max_down_volume:
                return True, "模拟阵地战: 出现日线级别的口袋支点信号"

        elif setup_type == PraetorianSetupType.GAP_COMMANDO:
            if today['close'] > today['open']:
                return True, "模拟缺口突击: 缺口后收出阳线，承接有力"

        return False, "当日K线形态未满足任何扳机条件"

# ==============================================================================
# === 【卖出策略】趋势跟踪止损 (回测专用版) ===
# ==============================================================================
class TrendFollowerSellStrategyForBacktest(BacktestStrategy):
    """
    【V2.0 升级版】
    - 引入了 ATR 容忍度，使其逻辑更接近实盘，避免被轻易洗出。
    """
    def __init__(self, data_handler, symbol_list, **kwargs):
        super().__init__(data_handler, symbol_list, **kwargs)
        self.ma_period = kwargs.get('ma_period', 20)
        # --- 从参数接收 ATR 配置 ---
        self.atr_period = kwargs.get('atr_period', 14)
        self.atr_tolerance_multiplier = kwargs.get('atr_tolerance_multiplier', 0.5)
        # --- 从参数接收蜜月期天数 ---
        self.grace_period_days = kwargs.get('grace_period_days', 2)
        
    @property
    def name(self):
        return f"趋势跟踪止损-ATR容忍 ({self.ma_period}日线-回测版)"

    def calculate_signals(self, event, held_symbols: list, positions: dict):
        if event.type != 'MARKET': return
            
        for s in held_symbols:
            # --- 【核心修改】在这里获取持仓对象 ---
            position = positions.get(s)
            if not position:
                continue # 如果因为某些原因找不到，就跳过
            
            # --- 【一级战地加固】---
            # 先获取数据，再进行任何操作。并且只获取一次。
            required_bars = max(self.ma_period, self.atr_period) + 20
            df_daily = self.data_handler.get_latest_bars(s, N=required_bars)
            
            # 核心防御：如果连一根K线都没有，那就什么也别做了
            if df_daily.empty:
                continue
            # --- 加固结束 ---

            # current_timestamp = df_daily.index[-1]
            # if self._is_in_grace_period(position, s, current_timestamp):
            #     continue # 处于蜜月期，跳过该股票的卖出检查
            # --- 新增结束 ---

            df_daily['ma'] = ta.sma(df_daily['close'], length=self.ma_period)
            df_daily['atr'] = ta.atr(df_daily['high'], df_daily['low'], df_daily['close'], length=self.atr_period)
            
            today = df_daily.iloc[-1]
            yesterday = df_daily.iloc[-2] # 我们需要昨天的ATR来做今天的决策

            if pd.isna(today['ma']) or pd.isna(yesterday['atr']):
                continue

            # --- 【核心逻辑升级】 ---
            # 1. 战略判断：今天的收盘价是否低于均线？
            is_below_ma = today['close'] < today['ma']
            
            # 2. 战术确认：是否也跌破了ATR容忍带？
            tolerance_threshold = today['ma'] - (yesterday['atr'] * self.atr_tolerance_multiplier)
            is_below_tolerance = today['close'] < tolerance_threshold

            # 只有当两个条件都满足时，才发出卖出信号
            if is_below_ma and is_below_tolerance:
            # --- 升级结束 ---
                current_timestamp = df_daily.index[-1]
                print(
                    f"[{current_timestamp.strftime('%Y-%m-%d')}] ◆◆◆ 卖出信号 ◆◆◆\n"
                    f"  - 股票: {s}\n"
                    f"  - 原因: {self.name} - 价格({today['close']:.2f})跌破ATR容忍带({tolerance_threshold:.2f})。"
                )
                signal = SignalEvent(s, current_timestamp, 'SHORT', strategy_name=self.name)
                events.put(signal)
    
    # --- 【新增】将你提供的代码移植进来，并进行关键修改 ---
    def _is_in_grace_period(self, position, symbol: str, current_timestamp: datetime) -> bool:
        """
        [回测版战术豁免模块]
        - 【关键修改】使用回测引擎传入的 current_timestamp 替代 datetime.now()
        """
        try:
            # 杠精注释：在我们的新架构下，position.entry_timestamp 就是一个datetime对象
            entry_timestamp = position.entry_timestamp
            
            # --- 【核心修正】用当前回测时间来计算持仓天数 ---
            holding_duration = current_timestamp - entry_timestamp
            
            if holding_duration.days < self.grace_period_days:
                # 为了避免日志刷屏，可以只在第一次豁免时打印
                # print(f"[{current_timestamp.strftime('%Y-%m-%d')}] [{self.name}] for {symbol}: [蜜月期豁免]")
                return True

        except Exception as e:
            # print(f"[{self.name}][{symbol}] 检查蜜月期时出错: {e}")
            return False

        return False