import os, sys
project_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_path)
from .event import events
from .performance import show_performance_stats

class Backtest:
    """
    回测主引擎。
    - 负责接收策略的“图纸”（类）和“原材料”（参数）。
    - 在内部完成策略的实例化，确保所有依赖被正确注入。
    """
    def __init__(
        self, 
        symbol_list, 
        initial_capital,
        data_handler_cls, 
        execution_handler_cls, 
        portfolio_cls, 
        # --- 【核心修改】现在接收一个“策略配置列表” ---
        strategy_config_list: list 
    ):
        self.symbol_list = symbol_list
        self.initial_capital = initial_capital
        
        # --- 【关键顺序】第一步：创建所有依赖项 ---
        self.data_handler = data_handler_cls(self.symbol_list)
        self.portfolio = portfolio_cls(self.data_handler, self.initial_capital)
        self.broker = execution_handler_cls(self.data_handler)
        
        # --- 【核心修改】第二步：扮演工厂，使用依赖项来实例化策略 ---
        self.strategy_list = []
        for config in strategy_config_list:
            strategy_class = config['class']
            strategy_params = config.get('params', {})
            
            # 杠精注释：在这里，我们将健康的 data_handler 和 symbol_list 注入到
            # 每一个策略的构造函数中。这才是灵魂注入的正确时刻！
            self.strategy_list.append(
                strategy_class(
                    data_handler=self.data_handler,
                    symbol_list=self.symbol_list,
                    **strategy_params
                )
            )

    def _run_backtest(self):
        print("开始回测...")
        for strategy in self.strategy_list:
            print(f"  - 已加载策略: {strategy.name}")
            
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
                            held_symbols = self.portfolio.get_held_symbols()
                            # 构建一个包含完整持仓对象的字典
                            positions_map = {s: self.portfolio.get_position(s) for s in held_symbols}
                            # 将 positions_map 传递给所有策略
                            for strategy in self.strategy_list:
                                strategy.calculate_signals(event, held_symbols, positions_map)
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