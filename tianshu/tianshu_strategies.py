# tianshu/tianshu_strategies.py
import pandas as pd
import pandas_ta as ta
from .event import SignalEvent, events
from your_original_project.strategies.base_strategy import StrategyType, PraetorianSetupType # 假设可以这样引用

# 这是一个改造后的策略基类，用于回测框架
class BacktestStrategy:
    """
    所有用于“天枢”回测系统的策略都应继承此类。
    它规定了策略与回测引擎的交互方式。
    """
    def __init__(self, data_handler, **kwargs):
        """
        【核心改造点】:
        构造函数不再接收 quote_ctx 和 config，而是接收 data_handler。
        所有数据都必须从 data_handler 获取。
        """
        self.data_handler = data_handler
        self.symbol_list = data_handler.symbol_list
        self.bought = {s: 'OUT' for s in self.symbol_list}

    @property
    def name(self):
        raise NotImplementedError

    def calculate_signals(self, event):
        """
        这是策略的主逻辑入口，由回测引擎在每个时间点调用。
        """
        raise NotImplementedError

# --- 范例：改造你最复杂的 PraetorianStrategy ---
class PraetorianStrategyForBacktest(BacktestStrategy):
    """
    这是你的 PraetorianStrategy 的回测专用版本。
    注意看，所有 self.quote_ctx 的调用都被替换了。
    """
    def __init__(self, data_handler, **kwargs):
        super().__init__(data_handler, **kwargs)
        # --- 所有的参数都可以从kwargs传入，保持了灵活性 ---
        self.long_term_ma_period = kwargs.get('long_term_ma_period', 60)
        self.key_support_ma_period = kwargs.get('key_support_ma_period', 10)
        # ... (此处省略你策略中的所有其他参数初始化)
        self.launchpad_lookback = 20
        self.atr_period = 14
        self.vcp_atr_ratio = 0.75
        self.max_volatility_pct = 0.15
        self.gap_min_pct = 0.04
        # ... 等等

    @property
    def name(self):
        return "PraetorianStrategy (Backtest Version)"

    def calculate_signals(self, event):
        """
        【核心改造点】: 主逻辑现在在这里，由 MarketEvent 驱动。
        """
        if event.type != 'MARKET':
            return

        for s in self.symbol_list:
            if self.bought[s] == 'OUT': # 简单处理，避免重复买入
                # 【核心改造点】: 所有 self._find_high_quality_setup 的调用，
                # 现在都直接在这里实现，并且数据源是 self.data_handler。
                
                # --- 模拟 _find_high_quality_setup ---
                df_daily = self.data_handler.get_latest_bars(s, N=200) # 获取足够的回看数据
                if df_daily.empty or len(df_daily) < self.long_term_ma_period:
                    continue

                # --- 模拟 _check_gap_commando_setup ---
                is_gap_setup, _ = self._check_gap_commando_setup_backtest(s, df_daily)
                if is_gap_setup and self.bought[s] == 'OUT':
                    print(f"[{df_daily.index[-1].strftime('%Y-%m-%d')}] {s} 触发买入信号: 缺口突击")
                    signal = SignalEvent(s, df_daily.index[-1], 'LONG', strategy_name=self.name)
                    events.put(signal)
                    self.bought[s] = 'LONG'

    def _check_gap_commando_setup_backtest(self, symbol, df_daily):
        """
        这是一个策略内部的辅助方法，但是是回测版本的。
        它不进行任何API调用，所有数据都来自传入的df_daily。
        """
        try:
            if len(df_daily) < self.launchpad_lookback + 2: return False, None
            
            # 【改造点】: 不再需要 get_realtime_quote，因为我们是在历史中
            # 我们用最新一根K线的数据来模拟“今天”
            today = df_daily.iloc[-1]
            yesterday = df_daily.iloc[-2]
            
            launchpad_df = df_daily.iloc[-(self.launchpad_lookback + 2):-2]
            if launchpad_df.empty: return False, None
            
            # --- VCP检查 (逻辑不变) ---
            launchpad_df['atr'] = ta.atr(launchpad_df['high'], launchpad_df['low'], launchpad_df['close'], length=self.atr_period)
            # ... (省略VCP和形态紧凑度的完整代码，因为逻辑和你原来的一样)
            
            # --- 缺口检查 (逻辑改变) ---
            today_open = today['open']
            gap_pct = (today_open / yesterday['close']) - 1.0
            if gap_pct < self.gap_min_pct: return False, {}
            
            max_price_in_pad = launchpad_df['high'].max()
            if today_open < max_price_in_pad: return False, {}

            return True, {} # 假设Setup满足
        except Exception:
            return False, None