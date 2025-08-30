# tianshu/portfolio.py
import pandas as pd
import numpy as np
from .event import OrderEvent, events

class Portfolio:
    """
    【增强版】投资组合类。
    - 实现了简单的基于总资金百分比的风险仓位管理。
    - 实时跟踪和记录市值曲线。
    """
    def __init__(self, data_handler, initial_capital=100000.0, risk_per_trade=0.02):
        self.data_handler = data_handler
        self.initial_capital = initial_capital
        self.risk_per_trade = risk_per_trade # 每笔交易承担总资金2%的风险
        self.symbol_list = self.data_handler.symbol_list

        self.all_positions = self._construct_all_positions()
        self.current_holdings = self._construct_current_holdings()
        self.equity_curve = self._construct_equity_curve()

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
        """在每个市场事件更新时，更新投资组合的总价值，并记录市值曲线。"""
        latest_datetime = self.data_handler.all_indices[self.data_handler.bar_index - 1]
        
        # 更新市值
        total_value = self.current_holdings['cash']
        for s in self.symbol_list:
            latest_bar = self.data_handler.get_latest_bars(s, N=1)
            if not latest_bar.empty:
                market_value = self.current_holdings[s] * latest_bar.iloc[0]['close']
                total_value += market_value
        
        self.current_holdings['total'] = total_value
        
        # 记录市值曲线
        new_row = {
            'cash': self.current_holdings['cash'],
            'total': self.current_holdings['total']
        }
        self.equity_curve.loc[latest_datetime] = new_row
        
        # 计算回报率
        if len(self.equity_curve) > 1:
            self.equity_curve['returns'] = self.equity_curve['total'].pct_change()

    def on_signal(self, event):
        """
        响应 SignalEvent，执行仓位管理并生成 OrderEvent。
        """
        if event.type == 'SIGNAL':
            quantity = self._calculate_position_size(event.symbol)
            if quantity > 0:
                direction = 'BUY' if event.signal_type == 'LONG' else 'SELL'
                order_type = 'MKT'
                
                # 杠精注释：在真实系统中，卖出逻辑会更复杂，需要检查当前持仓
                if direction == 'SELL' and self.current_holdings[event.symbol] == 0:
                    return # 没有持仓，不能卖

                # 简化：卖出时清空所有持仓
                if direction == 'SELL':
                    quantity = self.current_holdings[event.symbol]

                order = OrderEvent(event.symbol, order_type, quantity, direction)
                events.put(order)

    def _calculate_position_size(self, symbol):
        """
        一个简单的基于固定风险百分比的仓位计算。
        假设止损位在当前价格下方5%。
        """
        latest_bar = self.data_handler.get_latest_bars(symbol, N=1)
        if latest_bar.empty:
            return 0
        
        price = latest_bar.iloc[0]['close']
        stop_loss_pct = 0.05 # 假设5%的止损
        risk_per_share = price * stop_loss_pct
        
        if risk_per_share == 0:
            return 0

        # 根据总资金计算单笔交易可承担的风险金额
        trade_risk_amount = self.current_holdings['total'] * self.risk_per_trade
        
        quantity = int(trade_risk_amount / risk_per_share)
        
        # 检查现金是否足够
        if price * quantity > self.current_holdings['cash']:
            quantity = int(self.current_holdings['cash'] / price)
            
        return quantity if quantity > 0 else 0

    def on_fill(self, event):
        """响应 FillEvent，更新持仓状态。"""
        if event.type == 'FILL':
            if event.direction == 'BUY':
                self.current_holdings[event.symbol] += event.quantity
                self.current_holdings['cash'] -= (event.fill_cost + event.commission)
            else: # SELL
                self.current_holdings[event.symbol] -= event.quantity
                self.current_holdings['cash'] += (event.fill_cost - event.commission)
            
            self.current_holdings['commission'] += event.commission