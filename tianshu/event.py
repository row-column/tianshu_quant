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
    def __init__(self, symbol, datetime, signal_type, strength=1.0, strategy_name="", stop_loss_price=None):
        self.type = 'SIGNAL'
        self.symbol = symbol
        self.datetime = datetime
        self.signal_type = signal_type # 'LONG' 或 'SHORT'
        self.strength = strength
        self.strategy_name = strategy_name # 记录是哪个策略产生的信号
        self.stop_loss_price = stop_loss_price

class OrderEvent(Event):
    """当Portfolio对象希望下单时触发。"""
    def __init__(self, symbol, order_type, quantity, direction, initial_risk=0.0):
        self.type = 'ORDER'
        self.symbol = symbol
        self.order_type = order_type # 'MKT'
        self.quantity = quantity
        self.direction = direction # 'BUY' 或 'SELL'
        # --- 【核心新增】附带上本次交易的单位风险 ---
        self.initial_risk = initial_risk

class FillEvent(Event):
    """封装订单成交的细节。"""
    def __init__(self, datetime, symbol, exchange, quantity, direction, fill_cost,initial_price,avg_cost,initial_risk, commission=0.0):
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
        self.commission = commission

events = Queue()