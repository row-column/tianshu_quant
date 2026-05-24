import os, sys
project_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_path)
from abc import ABC, abstractmethod
from .event import FillEvent, events
from .periods import normalize_period_key

class ExecutionHandler(ABC):
    @abstractmethod
    def execute_order(self, event):
        raise NotImplementedError

class SimulatedExecutionHandler(ExecutionHandler):
    """
    【增强版】模拟执行处理器。
    - 引入了简单的滑点和佣金模型，让回测更接近真实。
    """
    def __init__(self, data_handler, commission_rate=0.0005, slippage_pct=0.001):
        self.data_handler = data_handler
        self.commission_rate = commission_rate
        self.slippage_pct = slippage_pct
        self.pending_orders = []

    def execute_order(self, event):
        """
        接收 OrderEvent，生成 FillEvent。
        默认在订单创建后的下一根K线成交，避免当根信号当根成交。
        """
        if event.type == 'ORDER':
            self.pending_orders.append(event)
            return self.execute_pending_orders()
        return []

    def execute_pending_orders(self, event=None):
        current_time = getattr(event, 'datetime', None) or getattr(self.data_handler, 'current_time', None)
        fills = []
        remaining_orders = []
        for order in self.pending_orders:
            order_period = normalize_period_key(getattr(order, 'period', None))
            bar_data = self._get_execution_bar(order, order_period)
            if bar_data is None:
                remaining_orders.append(order)
                continue
            if current_time is not None and bar_data.name > current_time:
                remaining_orders.append(order)
                continue
            fills.append(self._build_fill_event(order, order_period, bar_data))
        self.pending_orders = remaining_orders
        return fills

    def _get_execution_bar(self, order, order_period):
        order_created_at = getattr(order, 'datetime', None)
        if hasattr(self.data_handler, 'get_next_bar_after'):
            return self.data_handler.get_next_bar_after(order.symbol, order_created_at, period=order_period)
        bars = self.data_handler.get_latest_bars(order.symbol, N=1, period=order_period)
        return None if bars.empty else bars.iloc[0]

    def _build_fill_event(self, order, order_period, bar_data):
        close_price = bar_data['close']

        # --- 模拟滑点 ---
        if order.direction == 'BUY':
            fill_price = close_price * (1 + self.slippage_pct)
        else: # SELL
            fill_price = close_price * (1 - self.slippage_pct)

        fill_cost = fill_price * order.quantity
        commission = fill_cost * self.commission_rate

        fill_event = FillEvent(
            datetime=bar_data.name, # .name 是索引的名称，即时间戳
            symbol=order.symbol,
            exchange='SIMULATED',
            quantity=order.quantity,
            direction=order.direction,
            fill_cost=fill_cost,
            # initial_price 和 avg_cost 对于首次买入，就是成交价
            initial_price=fill_price,
            avg_cost=fill_price,
            # 作为忠实的“传话筒”，把风险信息原封不动地抄送
            initial_risk=order.initial_risk,
            entry_strategy_name=getattr(order, 'entry_strategy_name', 'Unknown'),
            stop_loss_price=getattr(order, 'stop_loss_price', 0.0),
            commission=commission,
            strategy_params=getattr(order, 'strategy_params', {}),
            metadata=getattr(order, 'metadata', {}),
        )
        print(
            f"[{bar_data.name.strftime('%Y-%m-%d %H:%M')}] 模拟成交({order_period}): "
            f"{order.direction} {order.quantity}股 {order.symbol} @ {fill_price:.2f}"
        )
        return fill_event
