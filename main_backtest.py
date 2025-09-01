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
    NarrativeWBottomStrategyForBacktest,
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

SYMBOLS_TO_DOWNLOAD_HK = [
    '00002.HK', '01347.HK', '07200.HK', '00005.HK', '01357.HK', '07226.HK',
    '00016.HK', '01398.HK', '09868.HK', '0005.HK', '01810.HK', '09988.HK',
    '00165.HK', '02359.HK', '09992.HK', '00268.HK', '02800.HK', '9988.HK',
    '00388.HK', '02899.HK', '00981.HK', '03750.HK', '01024.HK', '06060.HK',
    '6690.HK', '2809.HK', '6618.HK', '2357.HK', '2382.HK', '968.HK',
    '1816.HK', '2015.HK', '1211.HK', '1276.HK', '883.HK', '6160.HK',
    '9880.HK', '1772.HK', '1797.HK', '2228.HK', '3690.HK', '9660.HK',
    '9618.HK', '81299.HK', '2845.HK', '2806.HK', '2828.HK'
]

SYMBOLS_TO_DOWNLOAD_US = [
    'JNJ.US', 'JPM.US', 'LLYX.US', 'MSFU.US', 'PLTU.US', 'SOXL.US', 'AAPU.US',
    'TSLL.US', 'AMDL.US', 'WMT.US', 'AMZU.US', 'SNOU.US', 'SNOW.US',
    'RERE.US', 'AFRM.US', 'UMAC.US', 'HIMS.US', 'NFLU.US', 'RXRX.US',
    'NNE.US', 'CRMG.US', 'KPDD.US', 'PDD.US', 'TEM.US', 'LULU.US', 'IWM.US',
    'OPEN.US', 'PHM.US', 'LEN.US', 'DHI.US', 'APPX.US', 'FUTU.US', 'BULL.US',
    'KO.US', 'TMDX.US', 'LFMD.US', 'DXYZ.US', 'SPCE.US', 'LUNR.US',
    'SOLZ.US', 'ETHA.US', 'OUST.US', 'AEVA.US', 'HSAI.US', 'PONY.US',
    'RIVN.US', 'ACHR.US', 'SEZL.US', 'SHLS.US', 'NXT.US', 'ARRY.US', 'RUN.US',
    'FSLR.US', 'EVGO.US', 'EOSE.US', 'MVST.US', 'LEU.US', 'SMR.US',
    'OKTA.US', 'CRWL.US', 'CRWD.US', 'RBRK.US', 'SOUN.US', 'NBIS.US',
    'QMCO.US', 'IONQ.US', 'QBTS.US', 'QUBT.US', 'RGTI.US', 'UNHG.US',
    'UNH.US', 'BLSH.US', 'CVX.US', 'XOM.US', 'BMNR.US', 'TLRY.US', 'VALN.US',
    'SE.US', 'TEMT.US', 'TME.US', 'ASTS.US', 'MP.US', 'RKLB.US', 'CRCL.US',
    'GE.US', 'RTX.US', 'CRWV.US', 'MVLL.US', 'CONL.US', 'TQQQ.US', 'ASMG.US',
    'ALAB.US', 'RDTL.US', 'SPXL.US', 'DIG.US', 'ERX.US', 'XLV.US', 'SMH.US',
    'SPYU.US', 'UDOW.US', 'TNA.US', 'UPRO.US', 'AVGX.US'
]
symbol_list = SYMBOLS_TO_DOWNLOAD_HK + SYMBOLS_TO_DOWNLOAD_US
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
        'params': {'k_period_minutes':60}
    },
    # {
    #     'class': NarrativeWBottomStrategyForBacktest,
    #     'params': {
    #         'k_period_minutes':240,
    #         'ookback_period':150,
    #         'capitulation_vol_ratio':2.0,
    #         'volume_contraction_ratio':0.5,
    #         'higher_low_tolerance':1.005,
    #         'breakout_vol_ratio':1.8
    #         }
    # },
    # {
    #     'class': NarrativeWBottomStrategyForBacktest,
    # },
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