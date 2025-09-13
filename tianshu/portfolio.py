import os, sys
project_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_path)
import pandas as pd
from dataclasses import dataclass, field
import numpy as np
from .event import FillEvent,OrderEvent, events
from datetime import datetime

# ---一个轻量级的持仓对象，用于回测 ---
@dataclass
class BacktestPosition:
    symbol: str
    quantity: int
    entry_timestamp: datetime
    initial_price: float
    avg_cost: float
    initial_risk_per_share: float # 每股初始风险 (1R)
    entry_strategy_name: str # 【关键】记录是由哪个策略买入的
    stop_loss_price: float
    

class Portfolio:
    """
    投资组合类。
    - 实现了卖出信号的处理逻辑。
    - 提供了查询当前持仓的方法。
    - 实时跟踪和记录市值曲线。
    """
    def __init__(self, data_handler, initial_capital=100000.0, risk_per_trade=0.02):
        self.data_handler = data_handler
        self.initial_capital = initial_capital
        self.risk_per_trade = risk_per_trade 
        self.symbol_list = self.data_handler.symbol_list
         # --- 【修改】current_holdings 现在存储 BacktestPosition 对象 ---
        self.current_holdings: dict[str, BacktestPosition] = {}
        # self.all_positions = self._construct_all_positions()
        # self.current_holdings = self._construct_current_holdings()
        self.equity_curve = self._construct_equity_curve()
        self.cash = initial_capital
        self.total = initial_capital
        self.stop_loss_ratio = 0.04
        self.strategy_names = set()
        self.buy_percentages = {
            "禁卫军策略(回测版)-闪电战": 1.0,
            "禁卫军策略(回测版)-阵地战": 1.0,
            "MACD趋势反转(60min-回测版)": 1.0,
            "叙事性W底反转策略 (240min, 回测版)": 1.0,
            "叙事性W底反转策略 (720min, 回测版)": 1.0,
        }

    def _construct_all_positions(self):
        d = {s: pd.Series(dtype='float64') for s in self.symbol_list}
        return pd.DataFrame(d)

    def _construct_current_holdings(self):
        d = {s: 0.0 for s in self.symbol_list}
        d['cash'] = self.initial_capital
        d['commission'] = 0.0
        d['total'] = self.initial_capital
        return d

    def _construct_equity_curve(self):
        curve = pd.DataFrame(columns=['cash', 'total', 'returns'])
        return curve

    def update_timeindex(self, event):
        latest_datetime = self.data_handler.all_indices[self.data_handler.bar_index - 1]
        
        market_value = 0
        # --- 【核心修正】直接遍历 self.current_holdings.items() ---
        # 这里的 pos 永远都保证是一个 BacktestPosition 对象
        for symbol, pos in self.current_holdings.items():
            latest_bar = self.data_handler.get_latest_bars(symbol, N=1)
            if not latest_bar.empty:
                market_value += pos.quantity * latest_bar.iloc[0]['close']
        
        self.total = self.cash + market_value
        
        new_row = {'cash': self.cash, 'total': self.total}
        self.equity_curve.loc[latest_datetime] = new_row
        
        if len(self.equity_curve) > 1:
            self.equity_curve['returns'] = self.equity_curve['total'].pct_change()


    def on_signal(self, event):
        """
        响应 SignalEvent，现在可以处理买入和卖出两种信号。
        """
        if event.type != 'SIGNAL': return
        # --- 买入信号处理 ---
        if event.type == 'SIGNAL':
            if event.signal_type == 'LONG' and event.symbol not in self.current_holdings:
                # --- 【核心修改】在这里计算 risk_per_share ---
                latest_bar = self.data_handler.get_latest_bars(event.symbol, N=1)
                if latest_bar.empty: return
                price = latest_bar.iloc[0]['close']
                
                risk_per_share = 0.0
                fixed_risk_per_share = price * self.stop_loss_ratio
                fixed_stop_loss_price = price - fixed_risk_per_share
                if event.stop_loss_price and price > event.stop_loss_price:
                    # event.stop_loss_price = min(event.stop_loss_price,fixed_stop_loss_price)
                    # event.stop_loss_price = max(event.stop_loss_price,fixed_stop_loss_price)
                    # event.stop_loss_price = fixed_stop_loss_price
                    risk_per_share = price - event.stop_loss_price
                else:
                    print('执行到安全网逻辑')
                    risk_per_share = price * self.stop_loss_ratio # 安全网

                if risk_per_share <= 0: return

                quantity = self._calculate_position_size(price, risk_per_share,strategy_name=event.strategy_name)
                # quantity = self._calculate_position_size(price, risk_per_share,strategy_name=event.strategy_name,is_buy_percentage=False)
                
                if quantity > 0:
                    # --- 【核心修改】将 risk_per_share 放入 OrderEvent ---
                    order = OrderEvent(event.symbol, 
                                       'MKT', 
                                       quantity,
                                       'BUY',
                                       initial_risk=risk_per_share,
                                       entry_strategy_name=event.strategy_name,
                                       stop_loss_price=getattr(event, 'stop_loss_price', 0.0)
                                       )
                    events.put(order)
                    # self.strategy_names.add(event.strategy_name)
                    # print(self.strategy_names)
            
            elif event.signal_type == 'SHORT' and event.symbol in self.current_holdings:
                quantity = self.current_holdings[event.symbol].quantity
                order = OrderEvent(event.symbol, 'MKT', quantity, 'SELL') # 卖出时不需要风险参数
                events.put(order)
    
    def _calculate_position_size(self, price: float, risk_per_share: float,strategy_name:str=None,is_buy_percentage:bool=True) -> int:
        """(逻辑简化，职责更清晰)"""
        if is_buy_percentage:
            trade_risk_amount = self.total * self.risk_per_trade * self.buy_percentages[strategy_name]
            print('购买比例:',self.buy_percentages[strategy_name])
        else:
            trade_risk_amount = self.total * self.risk_per_trade
        
        quantity = int(trade_risk_amount / risk_per_share)
        
        if price * quantity > self.cash:
            quantity = int(self.cash / price)
            
        # 简单的港股手数处理
        return (quantity // 100) * 100 if quantity >= 100 else 0
    
    def _calculate_position_size_v2(self, symbol: str, stop_loss_price: float = None) -> int:
        latest_bar = self.data_handler.get_latest_bars(symbol, N=1)
        if latest_bar.empty: return 0
        price = latest_bar.iloc[0]['close']
        if stop_loss_price and price > stop_loss_price:
            risk_per_share = price - stop_loss_price
        else:
            risk_per_share = price * 0.05
        if risk_per_share <= 0: return 0
        trade_risk_amount = self.total * self.risk_per_trade
        quantity = int(trade_risk_amount / risk_per_share)
        if price * quantity > self.cash:
            quantity = int(self.cash / price)
        # 简单的手数处理
        return (quantity // 100) * 100 if quantity >= 100 else 0
    
    def _calculate_position_size_v1(self, symbol):
        """(无变化) 一个简单的基于固定风险百分比的仓位计算。"""
        latest_bar = self.data_handler.get_latest_bars(symbol, N=1)
        if latest_bar.empty: return 0
        
        price = latest_bar.iloc[0]['close']
        stop_loss_pct = 0.05
        risk_per_share = price * stop_loss_pct
        if risk_per_share == 0: return 0

        trade_risk_amount = self.current_holdings['total'] * self.risk_per_trade
        quantity = int(trade_risk_amount / risk_per_share)
        
        if price * quantity > self.current_holdings['cash']:
            quantity = int(self.current_holdings['cash'] / price)
            
        return quantity if quantity > 0 else 0

    def on_fill(self, event: FillEvent):
        """精确创建或销毁 BacktestPosition 对象"""
        if event.type != 'FILL': return

        if event.direction == 'BUY':
            # --- 【核心修改】创建持仓对象 ---
            position = BacktestPosition(
                symbol=event.symbol,
                quantity=event.quantity,
                entry_timestamp=event.datetime, # 成交时间就是建仓时间
                initial_price=event.initial_price,
                avg_cost=event.avg_cost, # 首次买入，成交价就是均价
                initial_risk_per_share=event.initial_risk,
                entry_strategy_name=event.entry_strategy_name,
                stop_loss_price=getattr(event, 'stop_loss_price', 0.0)
            )
            self.current_holdings[event.symbol] = position
            self.cash -= (event.fill_cost + event.commission)
        else: # SELL
            if event.symbol in self.current_holdings:
                del self.current_holdings[event.symbol]
                self.cash += (event.fill_cost - event.commission)


    # --- 一个关键的辅助方法 ---
    def get_held_symbols(self) -> list:
        """返回当前持有仓位的股票列表。"""
        # 现在它直接返回字典的键，因为字典里只有持仓股
        return list(self.current_holdings.keys())
        # return [symbol for symbol, quantity in self.current_holdings.items() if isinstance(quantity, (int, float)) and quantity > 0]

    # --- 获取持仓详细信息的接口 ---
    def get_position(self, symbol: str) -> BacktestPosition | None:
        """返回指定股票的持仓对象。"""
        return self.current_holdings.get(symbol)