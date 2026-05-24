import os, sys
project_path = os.path.dirname(os.path.abspath(__file__))
sys.path.append(project_path)
from tianshu.backtest import Backtest
from tianshu.data import HistoricDataHandler
from tianshu.execution import SimulatedExecutionHandler
from tianshu.portfolio import Portfolio
from tianshu.tianshu_strategies import (
    PraetorianStrategyForBacktest,
    ConservativeMA20BreakoutBuyStrategyForBacktest,
    ConservativeExitStrategyForBacktest,
    MacdStructuralReversalStrategyV2ForBacktest,
    MacdReversalStrategyForBacktest,
    MacdStructuralReversalStrategyForBacktest,
    MacdOverlordStrategyForBacktest,
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
from datetime import datetime
import warnings
warnings.filterwarnings("ignore")
# --- 回测配置 ---
# 确保这些股票的数据文件存在于 data/ 目录中

SYMBOLS_TO_DOWNLOAD_HK = ['03800.HK', '14651.HK', '17969.HK', '02809.HK', '06618.HK', 
                          '02015.HK', '01398.HK', '03750.HK', '01276.HK', '00883.HK', 
                          '06160.HK', '09880.HK', '01772.HK', '02228.HK', '09660.HK', 
                          '01357.HK', '09618.HK', '81299.HK', '02845.HK', '02806.HK', 
                          '02828.HK','00700.HK', '09988.HK', '01810.HK', '07200.HK', 
                          '07226.HK', '02899.HK', '01024.HK', '01357.HK', '09626.HK', 
                          '02800.HK', '02269.HK', '09688.HK', '01299.HK', 
                          '09992.HK', '02252.HK', '02359.HK', '00939.HK',
                          '03988.HK', '09999.HK', '03968.HK', 
                          '02628.HK', '02388.HK', '09961.HK', '00941.HK'
                          ]

SYMBOLS_TO_DOWNLOAD_US = ['SNPS.US', 'ORCX.US', 'MNDY.US', 'JPM.US', 'BBAI.US', 
                          'FSLR.US', 'WMT.US', 'BABX.US', 'ICLR.US', 'FORD.US', 
                          'JNJ.US', 'JPMO.US', 'AVGX.US', 'FIG.US', 'HMY.US', 
                          'TSLT.US', 'SNOU.US', 'SNOW.US', 'AFRM.US', 'PLTU.US', 
                          'UMAC.US', 'HIMS.US', 'NFLU.US', 'RXRX.US', 'NNE.US', 
                          'CRMG.US', 'PDD.US', 'TEM.US', 'IWM.US', 'OPEN.US', 
                          'PHM.US', 'LEN.US', 'DHI.US', 'APPX.US', 'FUTU.US', 
                          'BULL.US', 'KO.US', 'TMDX.US', 'LFMD.US', 'DXYZ.US',
                          'LUNR.US', 'SOLZ.US', 'ETHA.US', 'OUST.US', 'AEVA.US', 
                          'HSAI.US', 'PONY.US', 'RIVN.US', 'ACHR.US', 'SEZL.US', 
                          'NXT.US', 'EVGO.US', 'EOSE.US', 'MVST.US', 'LEU.US', 
                          'SMR.US', 'OKTA.US', 'CRWL.US', 'SOUN.US', 'NBIS.US', 
                          'QMCO.US', 'IONQ.US', 'QBTS.US', 'QUBT.US', 'RGTI.US', 
                          'UNHG.US', 'UNH.US', 'XOM.US', 'TLRY.US', 'VALN.US', 
                          'SE.US', 'TEMT.US', 'TME.US', 'ASTS.US', 'MP.US', 
                          'RKLB.US', 'CRCL.US', 'GE.US', 'RTX.US', 'MVLL.US', 
                          'TQQQ.US', 'ASMG.US', 'ALAB.US', 'RDTL.US', 'SPXL.US', 
                          'SMH.US', 'SPYU.US', 'UDOW.US', 'TNA.US', 'UPRO.US',
                          'NVDA.US', 'NVDX.US', 'NVDL.US', 'AMDL.US', 'OKLO.US', 
                          'TSLA.US', 'TSLL.US', 'LLY.US', 'LLYX.US', 
                          'GGLL.US', 'SOXL.US', 'AAPL.US', 'AAPU.US', 
                          'META.US', 'AMZN.US', 'GOOGL.US', 'TSM.US', 'MSFT.US', 
                          'MSFU.US', 'METU.US', 'AMZU.US', 'ASML.US',
                          'ROBN.US', 'AVGO.US', 'WMT.US', 
                          'COST.US', 'SMCI.US', 'SMCX.US']

SYMBOLS_TO_DOWNLOAD_CN = ['688775.SH', '688981.SH', '002384.SZ', '300308.SZ', 
                          '688041.SH', '300476.SZ', '603019.SH', '601012.SH', 
                          '600536.SH', '000975.SZ', '300750.SZ', '300347.SZ', 
                          '600900.SH', '601939.SH', '300195.SZ', '603799.SH', 
                          '601288.SH', '300748.SZ', '002475.SZ', '601138.SH', 
                          '688668.SH', '600183.SH', '300548.SZ', '300570.SZ', 
                          '300394.SZ', '002195.SZ', '002837.SZ', '002241.SZ', 
                          '600549.SH', '516780.SH', '159770.SZ', '515070.SH', 
                          '159202.SZ', '516100.SH', '159381.SZ', '515010.SH', 
                          '515980.SH', '159869.SZ', '301076.SZ', '601677.SH', 
                          '002714.SZ', '002078.SZ', '688561.SH', '000534.SZ', 
                          '300735.SZ', '300475.SZ', '603496.SH', '301536.SZ', 
                          '600989.SH', '002039.SZ', '002812.SZ', '600728.SH', 
                          '300441.SZ', '002444.SZ', '002174.SZ', '600930.SH', 
                          '601668.SH', '000959.SZ', '300408.SZ', '300014.SZ', 
                          '600809.SH', '600585.SH', '300204.SZ', '603083.SH', 
                          '603127.SH', '002747.SZ', '600021.SH', '603583.SH', 
                          '601089.SH', '600309.SH', '688767.SH', '688689.SH', 
                          '688677.SH', '688548.SH', '688391.SH', '688322.SH', 
                          '688305.SH', '688279.SH', '688160.SH', '688123.SH', 
                          '688115.SH', '605488.SH', '605389.SH', '605333.SH', 
                          '605319.SH', '603662.SH', '603655.SH', '603535.SH', 
                          '603456.SH', '603439.SH', '603380.SH', '603350.SH', 
                          '603196.SH', '603122.SH', '603088.SH', '603038.SH', 
                          '600960.SH', '600939.SH', '600933.SH', '600880.SH', 
                          '600759.SH', '600758.SH', '600754.SH', '600629.SH', 
                          '600617.SH', '600615.SH', '600575.SH', '600540.SH', 
                          '600252.SH', '600248.SH', '600231.SH', '600221.SH', 
                          '600191.SH', '600188.SH', '301603.SZ', '301520.SZ', 
                          '301429.SZ', '301382.SZ', '301186.SZ', '301129.SZ', 
                          '301125.SZ', '301110.SZ', '301059.SZ', '301012.SZ', 
                          '300990.SZ', '300970.SZ', '300964.SZ', '300949.SZ', 
                          '300935.SZ', '300923.SZ', '300814.SZ', '300813.SZ', 
                          '300749.SZ', '300739.SZ', '300736.SZ', '300661.SZ', 
                          '300660.SZ', '300635.SZ', '300551.SZ', '300507.SZ', 
                          '300467.SZ', '300370.SZ', '300350.SZ', '300235.SZ', 
                          '300155.SZ', '300121.SZ', '300110.SZ', '300086.SZ', 
                          '300076.SZ', '300063.SZ', '300049.SZ', '300022.SZ', 
                          '300021.SZ', '300013.SZ', '003041.SZ', '003033.SZ', 
                          '003008.SZ', '002970.SZ', '002906.SZ', '002864.SZ', 
                          '002628.SZ', '002590.SZ', '002542.SZ', '002535.SZ', 
                          '002534.SZ', '002510.SZ', '002489.SZ', '002451.SZ', 
                          '002377.SZ', '002154.SZ', '002137.SZ', '002086.SZ', '002061.SZ', '002042.SZ', '002008.SZ', '002006.SZ', '001333.SZ', '001306.SZ', '000922.SZ', '000782.SZ', '000739.SZ', '000727.SZ', '000720.SZ', '000567.SZ', '000559.SZ', '000401.SZ', '000008.SZ', '300442.SZ', '601689.SH', '002472.SZ', '603728.SH', '603486.SH', '002371.SZ', '600276.SH', '603259.SH', '003816.SZ', '002555.SZ', '603893.SH', '000021.SZ', '002637.SZ', '002164.SZ', '601918.SH', '601101.SH', '002131.SZ', '603124.SH', '603119.SH', '600409.SH', '688258.SH']

# symbol_list = SYMBOLS_TO_DOWNLOAD_HK + SYMBOLS_TO_DOWNLOAD_US
symbol_list = list(dict.fromkeys(SYMBOLS_TO_DOWNLOAD_US))
symbol_list = list(dict.fromkeys(SYMBOLS_TO_DOWNLOAD_CN))
# symbol_list = SYMBOLS_TO_DOWNLOAD_CN
# symbol_list = ['01810.HK', '00981.HK',"02899.HK","1211.HK","RBRK.US",'TMDX.US','IONQ.US','RKLB.US']

initial_capital = 100000.0
BACKTEST_START_DATE = "2026-02-18"
BACKTEST_START_DATE = "2025-05-21"
BACKTEST_END_DATE = None
data_handler_kwargs = {
    'start_date': BACKTEST_START_DATE,
    'end_date': BACKTEST_END_DATE,
}

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
# 历史组合配置保留在这里，便于以后切回原有多策略组合。
strategies_to_run = [
    
    # 买入：MACD底背离，60分钟 / 240分钟 / 日线
    # {
    #     'class': MacdReversalStrategyForBacktest,
    #     'params': {'k_period_minutes': 180},
    #     'data_periods': ['180m', '1d'],
    # },
    # {
    #     'class': MacdReversalStrategyForBacktest,
    #     'params': {'k_period_minutes': 240},
    #     'data_periods': ['240m', '1d'],
    # },
    {
        'class': MacdReversalStrategyForBacktest,
        'params': {'k_period_minutes': 1440},
        'data_periods': ['1d'],
    },

    # 买入：MACD趋势反转Pro，主周期 60分钟 / 240分钟 / 日线，5分钟做最终企稳确认
    # {
    #     'class': MacdReversalStrategyProForBacktest,
    #     'params': {'k_period_minutes': 60},
    #     'data_periods': ['60m', '5m', '1d'],
    # },
    # {
    #     'class': MacdReversalStrategyProForBacktest,
    #     'params': {'k_period_minutes': 240},
    #     'data_periods': ['240m', '5m', '1d'],
    # },
    # {
    #     'class': MacdReversalStrategyProForBacktest,
    #     'params': {'k_period_minutes': 1440},
    #     'data_periods': ['1d', '5m'],
    # },

    # {
    #     'class': MacdReversalStrategyProForBacktest,
    #     'params': {
    #         'k_period_minutes': 240,
    #         'confirmation_period': '5m',
    #         'require_daily_filter': False,
    #         'allow_scout_without_confirmation': True,
    #     },
    #     'data_periods': ['240m', '5m', '1d'],
    # },
    # {
    #     'class': MacdReversalStrategyProForBacktest,
    #     'params': {
    #         'k_period_minutes': 1440,
    #         'confirmation_period': '5m',
    #         'require_daily_filter': False,
    #         'allow_scout_without_confirmation': True,
    #     },
    #     'data_periods': ['1d', '5m'],
    # },

    # 卖出：MACD顶背离，60分钟 / 240分钟 / 日线
    {
        'class': MacdReversalSellStrategyForBacktest,
        'params': {'k_period_minutes': 60},
        'data_periods': ['60m', '1d'],
    },
    {
        'class': MacdReversalSellStrategyForBacktest,
        'params': {'k_period_minutes': 240},
        'data_periods': ['240m', '1d'],
    },
    # {
    #     'class': MacdReversalSellStrategyForBacktest,
    #     'params': {'k_period_minutes': 1440},
    #     'data_periods': ['1d'],
    # },
    # 卖出：尖兵-斩首，60分钟 / 240分钟 / 日线
    # {
    #     'class': ApexPredatorExitStrategyForBacktest,
    #     'params': {'k_period_minutes': 60},
    #     'data_periods': ['60m', '1d'],
    # },
    # {
    #     'class': ApexPredatorExitStrategyForBacktest,
    #     'params': {'k_period_minutes': 240},
    #     'data_periods': ['240m', '1d'],
    # },
    # {
    #     'class': ApexPredatorExitStrategyForBacktest,
    #     'params': {'k_period_minutes': 1440},
    #     'data_periods': ['1d'],
    # },
    # 风控兜底
    {
        'class': FixedStopLossStrategyForBacktest,
    },
]

# 当前默认回测 tmp_data/trading_system_ft_by.py 迁移出的 FT 交易系统逻辑。
# 只加载日线数据，避免分钟策略把回测执行周期切到 60m/240m 后污染日线信号。
strategies_to_run = [
    {
        'class': ConservativeMA20BreakoutBuyStrategyForBacktest,
        'params': {
            'target_value_pct': 0.085,
            'stop_loss_ratio': 0.07,
        },
        'data_periods': ['1d'],
    },
    {
        'class': ConservativeExitStrategyForBacktest,
        'data_periods': ['1d'],
    },
    {
        'class': MacdReversalStrategyForBacktest,
        'params': {'k_period_minutes': 1440},
        'data_periods': ['1d'],
    },

    # 卖出：MACD顶背离，60分钟 / 240分钟 / 日线
    {
        'class': MacdReversalSellStrategyForBacktest,
        'params': {'k_period_minutes': 60},
        'data_periods': ['60m', '1d'],
    },
    {
        'class': MacdReversalSellStrategyForBacktest,
        'params': {'k_period_minutes': 240},
        'data_periods': ['240m', '1d'],
    },
]

# --- 启动回测 ---
if __name__ == "__main__":

    is_single_stock_test:bool = False
    # is_single_stock_test:bool = True

    if not is_single_stock_test:
        print("--- 开始对【整个投资组合】进行回测 ---")

        backtest = Backtest(
            symbol_list=symbol_list,
            initial_capital=initial_capital,
            data_handler_cls=HistoricDataHandler,
            execution_handler_cls=SimulatedExecutionHandler,
            portfolio_cls=Portfolio,
            # 【核心修改】将“配置清单”交给回测引擎工厂
            strategy_config_list=strategies_to_run,
            data_handler_kwargs=data_handler_kwargs,
        )
        backtest.simulate_trading(is_show=True)
        print("--- 组合回测结束 ---")
    else:
        # --- 【第二部分：新增功能】对每个股票进行独立回测，并写入文件 ---
        print("\n\n--- 开始【单股票独立】回测分析 ---")
        
        # --- 【核心修改】 ---
        # 1. 定义输出文件名和路径
        output_filename = os.path.join(project_path, "logs", f"single_stock_performance_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt")
        
        # 2. 提取目录路径
        output_dir = os.path.dirname(output_filename)

        # 3. 在写入前，确保目录存在。如果不存在，就创建它。
        #    os.makedirs(..., exist_ok=True) 是幂等的，这意味着即使目录已经存在，它也不会报错。
        #    这才是编写健壮代码的正确姿势！
        os.makedirs(output_dir, exist_ok=True)

        print(f"详细报告将写入文件: {output_filename}")

        with open(output_filename, 'w', encoding='utf-8') as f:
            # 杠精注释：这里我们遍历的是原始的、完整的 symbol_list
            for symbol in symbol_list:
                print(f"正在分析股票: {symbol}...")
                
                # 写入文件头
                f.write(f"股票代码：{symbol}\n")
                
                # 为单只股票创建一个全新的、独立的回测实例
                # 杠精注释：注意这里的 symbol_list=[symbol]，这确保了回测环境的纯净性
                backtest_single = Backtest(
                    symbol_list=[symbol],  # <-- 关键点在这里！
                    initial_capital=initial_capital,
                    data_handler_cls=HistoricDataHandler,
                    execution_handler_cls=SimulatedExecutionHandler,
                    portfolio_cls=Portfolio,
                    strategy_config_list=strategies_to_run,
                    data_handler_kwargs=data_handler_kwargs,
                )
                
                # 调用修改后的方法，把文件句柄传进去
                backtest_single.simulate_trading(output_file=f)
                
                # 在文件中增加一个分隔符，让报告更美观
                f.write("\n" + "="*80 + "\n\n")
                # --- 【杠精的最终奥义】 ---
                # 杠精注释：在完成单次循环的所有写入操作后，强制将文件缓冲区的内容写入磁盘。
                # 这确保了即使程序在下一次循环中意外中断，已完成的结果也不会丢失。
                # 这才是兼顾了性能与数据安全性的专业做法！
                f.flush()

        print(f"--- 单股票独立回测分析完成，请查看 {output_filename} ---")
