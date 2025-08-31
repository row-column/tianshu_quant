import os, sys
project_path = os.path.dirname(os.path.abspath(__file__))
sys.path.append(project_path)
from tianshu.backtest import Backtest
from tianshu.data import HistoricDataHandler
from tianshu.execution import SimulatedExecutionHandler
from tianshu.portfolio import Portfolio
from tianshu.tianshu_strategies import (
    PraetorianStrategyForBacktest,
    MacdReversalStrategyForBacktest,
    MacdReversalStrategyProForBacktest,
    MomentumContinuationStrategyForBacktest,
    PredatorAmbushStrategyForBacktest,
    TrendFollowerSellStrategyForBacktest,
    MacdReversalSellStrategyForBacktest,
    FixedStopLossStrategyForBacktest,
    ApexPredatorExitStrategyForBacktest,
    NextDaySellStrategyProForBacktest,
    IntradayHighStallATRForBacktest
)
import warnings
warnings.filterwarnings("ignore")
# --- 回测配置 ---
# 确保这些股票的数据文件存在于 data/ 目录中
# symbol_list = ['0005.HK', '9988.HK',"00981.HK","01810.HK","07226.HK"]
symbol_list = [
    '00002.HK', '01347.HK', '07200.HK', 'JNJ.US',
    '00005.HK', '01357.HK', '07226.HK', 'JPM.US',
    '00016.HK', '01398.HK', '09868.HK', 'LLYX.US',
    '0005.HK', '01810.HK', '09988.HK', 'MSFU.US',
    '00165.HK', '02359.HK', '09992.HK', 'PLTU.US',
    '00268.HK', '02800.HK', '9988.HK', 'SOXL.US',
    '00388.HK', '02899.HK', 'AAPU.US', 'TSLL.US',
    '00981.HK', '03750.HK', 'AMDL.US', 'WMT.US',
    '01024.HK', '06060.HK', 'AMZU.US'
]
# symbol_list = ['0005.HK', '9988.HK',"00981.HK","01810.HK","07226.HK"]

initial_capital = 100000.0

# --- 【核心修改】创建策略的“图纸和原材料”清单 ---
praetorian_params = {
    'long_term_ma_period': 60,
    'key_support_ma_period': 10,
    'ma_distance_threshold': 0.08,
    'volume_spike_quantile': 0.90,
    'volume_shrink_ratio': 0.8,
    'atr_period': 14,
    'lookback_period': 20,
    'vcp_lookback_period': 15,
    'vcp_max_width_pct': 0.12,
    'gap_min_pct': 0.04,
    'atr_multiplier_breakout': 0.25,
}

sell_strategy_params = { 'ma_period': 20 }

# 这是一个配置列表，每个元素都是一个字典，描述了如何构建一个策略
strategies_to_run = [
    {
        'class': PraetorianStrategyForBacktest,
        'params': praetorian_params
    },
    {
        'class': MacdReversalStrategyForBacktest,
        # 'params': {'k_period_minutes':60}
    },
    # {
    #     'class': MacdReversalStrategyProForBacktest,
    # },
    # {
    #     'class': MomentumContinuationStrategyForBacktest,
    # },
    {
        'class': PredatorAmbushStrategyForBacktest,
    },
    {
        'class': MacdReversalSellStrategyForBacktest,
    },
    {
        'class': FixedStopLossStrategyForBacktest,
    },
    {
        'class': ApexPredatorExitStrategyForBacktest,
    },
    # {
    #     'class': NextDaySellStrategyProForBacktest,
    # },

    # {
    #     'class': IntradayHighStallATRForBacktest,
    #     'params':  {
    #         'upper_shadow_ratio': 1.5,
    #         'retrace_atr_multiplier': 1.8,
    #     }
    # },
    # {
    #     'class': TrendFollowerSellStrategyForBacktest,
    #     'params': sell_strategy_params
    # }
]

# --- 启动回测 ---
if __name__ == "__main__":
    backtest = Backtest(
        symbol_list=symbol_list,
        initial_capital=initial_capital,
        data_handler_cls=HistoricDataHandler,
        execution_handler_cls=SimulatedExecutionHandler,
        portfolio_cls=Portfolio,
        # 【核心修改】将“配置清单”交给回测引擎工厂
        strategy_config_list=strategies_to_run
    )
    backtest.simulate_trading()