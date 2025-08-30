# main_backtest.py
from tianshu.backtest import Backtest
from tianshu.data import HistoricDataHandler
from tianshu.execution import SimulatedExecutionHandler
from tianshu.portfolio import Portfolio
# 【关键】从你的新策略文件中导入改造后的策略
from tianshu.tianshu_strategies import PraetorianStrategyForBacktest

# --- 回测配置 ---
symbol_list = ['AAPL.US', 'TSLA.US']
initial_capital = 100000.0

# 策略需要的参数
praetorian_params = {
    'long_term_ma_period': 60,
    'key_support_ma_period': 10,
    # ... 其他所有PraetorianStrategy需要的参数
}

# --- 启动回测 ---
if __name__ == "__main__":
    backtest = Backtest(
        symbol_list=symbol_list,
        initial_capital=initial_capital,
        data_handler_cls=HistoricDataHandler,
        execution_handler_cls=SimulatedExecutionHandler,
        portfolio_cls=Portfolio,
        strategy_cls=PraetorianStrategyForBacktest, # 使用你改造后的策略
        strategy_params=praetorian_params # 传入参数
    )
    backtest.simulate_trading()