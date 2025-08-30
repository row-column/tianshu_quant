# tianshu/backtest.py
from .event import events
from .performance import show_performance_stats

class Backtest:
    """封装了回测的所有设置和组件。"""
    def __init__(
        self, 
        symbol_list, 
        initial_capital,
        data_handler_cls, 
        execution_handler_cls, 
        portfolio_cls, 
        strategy_cls,
        strategy_params=None # 允许传入策略参数
    ):
        self.symbol_list = symbol_list
        self.initial_capital = initial_capital
        
        self.data_handler = data_handler_cls(self.symbol_list)
        self.portfolio = portfolio_cls(self.data_handler, self.initial_capital)
        self.broker = execution_handler_cls(self.data_handler)
        
        # 动态创建策略实例
        if strategy_params is None:
            strategy_params = {}
        self.strategy = strategy_cls(data_handler=self.data_handler, **strategy_params)

    def _run_backtest(self):
        print(f"开始回测策略: {self.strategy.name}...")
        while True:
            if self.data_handler.continue_backtest:
                self.data_handler.update_bars()
            else:
                break
            
            while True:
                try:
                    event = events.get(block=False)
                except Exception:
                    break
                else:
                    if event is not None:
                        if event.type == 'MARKET':
                            self.strategy.calculate_signals(event)
                            self.portfolio.update_timeindex(event)
                        elif event.type == 'SIGNAL':
                            self.portfolio.on_signal(event)
                        elif event.type == 'ORDER':
                            self.broker.execute_order(event)
                        elif event.type == 'FILL':
                            self.portfolio.on_fill(event)
        
        print("回测结束。")

    def simulate_trading(self):
        self._run_backtest()
        show_performance_stats(self.portfolio.equity_curve, self.initial_capital)