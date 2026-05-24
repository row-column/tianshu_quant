import os, sys
project_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_path)
from .event import events
from .performance import show_performance_stats
from .periods import DAY_PERIOD_KEY, normalize_period_key

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
        strategy_config_list: list,
        data_handler_kwargs: dict | None = None,
    ):
        self.symbol_list = list(dict.fromkeys(symbol_list))
        self.initial_capital = initial_capital
        self.strategy_config_list = strategy_config_list
        self.data_handler_kwargs = data_handler_kwargs or {}
        
        # --- 【关键顺序】第一步：创建所有依赖项 ---
        data_kwargs = self._build_data_handler_kwargs()
        self.data_handler = data_handler_cls(self.symbol_list, **data_kwargs)
        self.portfolio = portfolio_cls(self.data_handler, self.initial_capital)
        self.broker = execution_handler_cls(self.data_handler)
        
        # --- 【核心修改】第二步：扮演工厂，使用依赖项来实例化策略 ---
        self.strategy_list = []
        for config in self.strategy_config_list:
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

    def _build_data_handler_kwargs(self):
        data_kwargs = dict(self.data_handler_kwargs)
        configured_periods = data_kwargs.get('periods')
        if configured_periods is None:
            configured_periods = self._infer_required_periods()
        data_kwargs['periods'] = configured_periods
        return data_kwargs

    def _infer_required_periods(self):
        periods = {DAY_PERIOD_KEY}
        for config in self.strategy_config_list:
            strategy_class = config['class']
            params = config.get('params', {})
            for period in config.get('data_periods', []):
                periods.add(normalize_period_key(period))
            if 'k_period_minutes' in params:
                periods.add(normalize_period_key(params['k_period_minutes']))
            else:
                default_period = getattr(strategy_class, 'DEFAULT_PRIMARY_PERIOD_MINUTES', None)
                if default_period is not None:
                    periods.add(normalize_period_key(default_period))
            for period in getattr(strategy_class, 'DEFAULT_REQUIRED_PERIODS', []):
                periods.add(normalize_period_key(period))
            if 'period' in params:
                periods.add(normalize_period_key(params['period']))
            if 'confirmation_period' in params:
                periods.add(normalize_period_key(params['confirmation_period']))
        return sorted(periods)

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
                            if hasattr(self.broker, 'execute_pending_orders'):
                                self._handle_fills(self.broker.execute_pending_orders(event))
                            held_symbols = self.portfolio.get_held_symbols()
                            # 构建一个包含完整持仓对象的字典
                            positions_map = {s: self.portfolio.get_position(s) for s in held_symbols}
                            # 将 positions_map 传递给所有策略
                            for strategy in self.strategy_list:
                                primary_period = getattr(strategy, 'primary_period_key', DAY_PERIOD_KEY)
                                if hasattr(self.data_handler, 'has_any_new_bar') and not self.data_handler.has_any_new_bar(primary_period):
                                    continue
                                strategy.calculate_signals(event, held_symbols, positions_map)
                            self.portfolio.update_timeindex(event)
                        elif event.type == 'SIGNAL':
                            self.portfolio.on_signal(event)
                        elif event.type == 'ORDER':
                            self._handle_fills(self.broker.execute_order(event))
                        elif event.type == 'FILL':
                            self.portfolio.on_fill(event)
        
        print("回测结束。")

    def _handle_fills(self, fills):
        for fill in fills or []:
            self.portfolio.on_fill(fill)

    def simulate_trading(self,is_show:bool=False,output_file=None):
        self._run_backtest()
        show_performance_stats(self.portfolio.equity_curve, self.initial_capital,is_show=is_show,output_file=output_file)
