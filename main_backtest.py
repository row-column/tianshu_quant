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

SYMBOLS_TO_DOWNLOAD_HK = [
    '01347.HK', '07200.HK',  '01357.HK', '07226.HK',
    '01398.HK', '0005.HK', '01810.HK', '09988.HK',
    '02359.HK', '09992.HK', '00268.HK', '02800.HK',
    '00388.HK', '02899.HK', '00981.HK', '03750.HK', 
    '01024.HK', '06060.HK','6690.HK', '2809.HK', '6618.HK',
    '2015.HK', '1211.HK', '1276.HK', '883.HK', '6160.HK','1772.HK', 
    '1797.HK', '2228.HK', '9660.HK',
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
    'RIVN.US', 'ACHR.US', 'SEZL.US', 'NXT.US',
    'EVGO.US', 'EOSE.US', 'MVST.US', 'LEU.US', 'SMR.US',
    'OKTA.US', 'CRWL.US', 'RBRK.US', 'SOUN.US', 'NBIS.US',
    'QMCO.US', 'IONQ.US', 'QBTS.US', 'QUBT.US', 'RGTI.US', 'UNHG.US',
    'UNH.US', 'BLSH.US', 'XOM.US', 'BMNR.US', 'TLRY.US', 'VALN.US',
    'SE.US', 'TEMT.US', 'TME.US', 'ASTS.US', 'MP.US', 'RKLB.US', 'CRCL.US',
    'GE.US', 'RTX.US', 'CRWV.US', 'MVLL.US', 'CONL.US', 'TQQQ.US', 'ASMG.US',
    'ALAB.US', 'RDTL.US', 'SPXL.US','XLV.US', 'SMH.US',
    'UDOW.US', 'TNA.US', 'UPRO.US'
]
SYMBOLS_TO_DOWNLOAD_CN = [
    "688775.SH","688981.SH","002384.SZ","300308.SZ","688041.SH",
    "300476.SZ","603019.SH","601012.SH","600536.SH","000975.SZ",
    "300750.SZ","300347.SZ","600900.SH","601939.SH","300195.SZ",
    "603799.SH","601288.SH","300748.SZ","002475.SZ","601138.SH",
    "688668.SH","600183.SH","300548.SZ","300570.SZ","300394.SZ",
    "002195.SZ","002837.SZ","002241.SZ","600549.SH","516780.SH",
    "159770.SZ","515070.SH","159202.SZ","516100.SH","159381.SZ",
    "515010.SH","515980.SH","600410.SH","159869.SZ","600930.SH"
]
# symbol_list = SYMBOLS_TO_DOWNLOAD_HK + SYMBOLS_TO_DOWNLOAD_US
symbol_list = SYMBOLS_TO_DOWNLOAD_CN
# symbol_list = ['01810.HK', '00981.HK',"02899.HK","1211.HK","RBRK.US",'TMDX.US','IONQ.US','RKLB.US']

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
    # {
    #     'class': PredatorAmbushStrategyForBacktest,
    # },
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

    is_single_stock_test:bool = True

    if not is_single_stock_test:
        print("--- 开始对【整个投资组合】进行回测 ---")

        backtest = Backtest(
            symbol_list=symbol_list,
            initial_capital=initial_capital,
            data_handler_cls=HistoricDataHandler,
            execution_handler_cls=SimulatedExecutionHandler,
            portfolio_cls=Portfolio,
            # 【核心修改】将“配置清单”交给回测引擎工厂
            strategy_config_list=strategies_to_run
        )
        backtest.simulate_trading(is_show=True)
        print("--- 组合回测结束 ---")
    else:
        # --- 【第二部分：新增功能】对每个股票进行独立回测，并写入文件 ---
        print("\n\n--- 开始【单股票独立】回测分析 ---")
        
        # --- 【核心修改】 ---
        # 1. 定义输出文件名和路径
        output_filename = os.path.join(project_path, "logs", "single_stock_performance_report.txt")
        
        # 2. 提取目录路径
        output_dir = os.path.dirname(output_filename)

        # 3. 在写入前，确保目录存在。如果不存在，就创建它。
        #    杠精注释：os.makedirs(..., exist_ok=True) 是幂等的，这意味着即使目录已经存在，它也不会报错。
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
                    strategy_config_list=strategies_to_run
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