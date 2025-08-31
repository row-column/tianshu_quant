import os, sys
project_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_path)
from abc import ABC, abstractmethod
from .event import FillEvent, events

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

    def execute_order(self, event):
        """
        接收 OrderEvent，生成 FillEvent。
        成交价模拟：使用当前K线的收盘价，并加入滑点。
        """
        if event.type == 'ORDER':
            bars = self.data_handler.get_latest_bars(event.symbol, N=1)
            if not bars.empty:
                bar_data = bars.iloc[0]
                close_price = bar_data['close']
                
                # --- 模拟滑点 ---
                if event.direction == 'BUY':
                    fill_price = close_price * (1 + self.slippage_pct)
                else: # SELL
                    fill_price = close_price * (1 - self.slippage_pct)

                fill_cost = fill_price * event.quantity
                commission = fill_cost * self.commission_rate

                fill_event = FillEvent(
                    datetime=bar_data.name, # .name 是索引的名称，即时间戳
                    symbol=event.symbol,
                    exchange='SIMULATED',
                    quantity=event.quantity,
                    direction=event.direction,
                    fill_cost=fill_cost,
                    commission=commission
                )
                events.put(fill_event)
                print(
                    f"[{bar_data.name.strftime('%Y-%m-%d')}] 模拟成交: "
                    f"{event.direction} {event.quantity}股 {event.symbol} @ {fill_price:.2f}"
                )