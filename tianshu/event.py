import os, sys
project_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_path)
from queue import Queue

class Event:
    """事件对象的基类。"""
    pass

class MarketEvent(Event):
    """当数据处理器提供一个新的市场数据（如K线）时触发。"""
    def __init__(self, datetime=None):
        self.type = 'MARKET'
        self.datetime = datetime

class SignalEvent(Event):
    """当策略对象产生一个交易信号时触发。"""
    def __init__(
        self,
        symbol,
        datetime,
        signal_type,
        strength=1.0,
        strategy_name="",
        stop_loss_price=None,
        period=None,
        execution_period=None,
        target_value_pct=None,
        strategy_params=None,
        metadata=None,
    ):
        self.type = 'SIGNAL'
        self.symbol = symbol
        self.datetime = datetime
        self.signal_type = signal_type # 'LONG' 或 'SHORT'
        self.strength = strength
        self.strategy_name = strategy_name # 记录是哪个策略产生的信号
        self.stop_loss_price = stop_loss_price
        self.period = period # 信号产生周期
        self.execution_period = execution_period # 期望成交周期；为空时由组合模块使用默认成交周期
        self.target_value_pct = target_value_pct # 按目标市值比例建仓；为空时沿用风险预算建仓
        self.strategy_params = strategy_params or {}
        self.metadata = metadata or {}

class OrderEvent(Event):
    """当Portfolio对象希望下单时触发。"""
    def __init__(
        self,
        symbol,
        order_type,
        quantity,
        direction,
        datetime=None,
        initial_risk=0.0,
        entry_strategy_name="",
        stop_loss_price=0.0,
        period=None,
        strategy_params=None,
        metadata=None,
    ):
        self.type = 'ORDER'
        self.symbol = symbol
        self.order_type = order_type # 'MKT'
        self.quantity = quantity
        self.direction = direction # 'BUY' 或 'SELL'
        self.datetime = datetime
        self.initial_risk = initial_risk
        self.entry_strategy_name = entry_strategy_name
        self.stop_loss_price = stop_loss_price
        self.period = period
        self.strategy_params = strategy_params or {}
        self.metadata = metadata or {}

class FillEvent(Event):
    """封装订单成交的细节。"""
    def __init__(self, datetime, symbol, exchange, quantity, direction, fill_cost,initial_price,avg_cost,initial_risk,entry_strategy_name,stop_loss_price,commission=0.0,strategy_params=None,metadata=None):
        self.type = 'FILL'
        self.datetime = datetime
        self.symbol = symbol
        self.exchange = exchange
        self.quantity = quantity
        self.direction = direction
        self.fill_cost = fill_cost
        self.initial_price = initial_price
        self.avg_cost = avg_cost
        self.initial_risk = initial_risk
        self.entry_strategy_name = entry_strategy_name
        self.stop_loss_price = stop_loss_price
        self.commission = commission
        self.strategy_params = strategy_params or {}
        self.metadata = metadata or {}

events = Queue()
