import os, sys
project_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_path)
from queue import Queue

class Event:
    """事件对象的基类。"""
    pass

class MarketEvent(Event):
    """当数据处理器提供一个新的市场数据（如K线）时触发。"""
    def __init__(self):
        self.type = 'MARKET'

class SignalEvent(Event):
    """当策略对象产生一个交易信号时触发。"""
    def __init__(self, symbol, datetime, signal_type, strength=1.0, strategy_name=""):
        self.type = 'SIGNAL'
        self.symbol = symbol
        self.datetime = datetime
        self.signal_type = signal_type # 'LONG' 或 'SHORT'
        self.strength = strength
        self.strategy_name = strategy_name # 记录是哪个策略产生的信号

class OrderEvent(Event):
    """当Portfolio对象希望下单时触发。"""
    def __init__(self, symbol, order_type, quantity, direction):
        self.type = 'ORDER'
        self.symbol = symbol
        self.order_type = order_type # 'MKT'
        self.quantity = quantity
        self.direction = direction # 'BUY' 或 'SELL'

class FillEvent(Event):
    """封装订单成交的细节。"""
    def __init__(self, datetime, symbol, exchange, quantity, direction, fill_cost, commission=0.0):
        self.type = 'FILL'
        self.datetime = datetime
        self.symbol = symbol
        self.exchange = exchange
        self.quantity = quantity
        self.direction = direction
        self.fill_cost = fill_cost
        self.commission = commission

events = Queue()