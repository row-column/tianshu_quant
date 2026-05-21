import os, sys
project_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_path)
import pandas as pd
from dataclasses import dataclass, field
import numpy as np
from .event import FillEvent,OrderEvent, events
from datetime import datetime
from .periods import DAY_PERIOD_KEY, normalize_period_key

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
    strategy_params: dict = field(default_factory=dict)
    

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
        # self.buy_percentages = {
        #     "禁卫军策略(回测版)-闪电战": 1.0,
        #     "禁卫军策略(回测版)-阵地战": 1.0,
        #     "MACD趋势反转(60min-回测版)": 1.0,
        #     "叙事性W底反转策略 (240min, 回测版)": 1.0,
        #     "叙事性W底反转策略 (720min, 回测版)": 1.0,
        # }
        self.buy_percentages = {
            # 禁卫军作为核心，闪电战（动能突破）分配最高，阵地战（趋势回调）次之。
            "禁卫军策略(回测版)-闪电战": 0.40,
            "禁卫军策略(回测版)-阵地战": 0.35,

            # MACD反转是逆势信号，严格控制仓位，作为卫星策略。
            "MACD趋势反转(60min-回测版)": 0.20,
            # "MACD趋势反转(60min-回测版)": 0.50,
            "MACD霸主策略 (日线抽象回测版)": 0.20,

            # W底是大周期反转，信号可靠，给予较高权重。720分钟周期更可靠，仓位也更重。
            "叙事性W底反转策略 (240min, 回测版)": 0.30,
            "叙事性W底反转策略 (720min, 回测版)": 0.40,
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
        latest_datetime = getattr(event, 'datetime', None)
        if latest_datetime is None:
            latest_datetime = self.data_handler.current_time
        if latest_datetime is None and self.data_handler.bar_index > 0:
            latest_datetime = self.data_handler.all_indices[self.data_handler.bar_index - 1]
        
        market_value = 0
        # --- 【核心修正】直接遍历 self.current_holdings.items() ---
        # 这里的 pos 永远都保证是一个 BacktestPosition 对象
        for symbol, pos in self.current_holdings.items():
            latest_bar = self.data_handler.get_latest_bars(symbol, N=1, period=self.data_handler.valuation_period_key)
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
                execution_period = self._resolve_execution_period(event)
                latest_bar = self.data_handler.get_latest_bars(event.symbol, N=1, period=execution_period)
                if latest_bar.empty:
                    self._log_order_rejection(
                        event,
                        reason=f"成交周期({execution_period})没有可用K线",
                        execution_period=execution_period,
                    )
                    return
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

                if risk_per_share <= 0:
                    self._log_order_rejection(
                        event,
                        reason="单股风险小于等于0",
                        execution_period=execution_period,
                        price=price,
                        risk_per_share=risk_per_share,
                    )
                    return

                target_value_pct = getattr(event, 'target_value_pct', None)
                if target_value_pct is not None:
                    quantity_info = self._calculate_value_position_size(
                        price,
                        symbol=event.symbol,
                        target_value_pct=target_value_pct,
                        return_details=True,
                    )
                else:
                    # quantity = self._calculate_position_size(price, risk_per_share,strategy_name=event.strategy_name)
                    quantity_info = self._calculate_position_size(
                        price,
                        risk_per_share,
                        symbol=event.symbol,
                        strategy_name=event.strategy_name,
                        signal_strength=getattr(event, 'strength', 1.0),
                        is_buy_percentage=False,
                        return_details=True,
                    )
                quantity = quantity_info['final_quantity']
                
                if quantity > 0:
                    # --- 【核心修改】将 risk_per_share 放入 OrderEvent ---
                    order = OrderEvent(event.symbol, 
                                       'MKT', 
                                       quantity,
                                       'BUY',
                                       initial_risk=risk_per_share,
                                       entry_strategy_name=event.strategy_name,
                                       stop_loss_price=getattr(event, 'stop_loss_price', 0.0),
                                       period=execution_period,
                                       strategy_params=getattr(event, 'strategy_params', {}),
                                       metadata=getattr(event, 'metadata', {})
                                       )
                    events.put(order)
                    # self.strategy_names.add(event.strategy_name)
                    # print(self.strategy_names)
                else:
                    self._log_order_rejection(
                        event,
                        reason=(
                            "按目标市值比例和现金/交易单位计算后数量为0"
                            if target_value_pct is not None
                            else "按风险预算和交易单位计算后数量为0"
                        ),
                        execution_period=execution_period,
                        price=price,
                        stop_loss_price=getattr(event, 'stop_loss_price', None),
                        risk_per_share=risk_per_share,
                        signal_strength=getattr(event, 'strength', 1.0),
                        planned_amount=quantity_info['planned_amount'],
                        raw_quantity=quantity_info['raw_quantity'],
                        final_quantity=quantity,
                        board_lot=quantity_info['board_lot'],
                    )
            
            elif event.signal_type == 'SHORT' and event.symbol in self.current_holdings:
                position = self.current_holdings[event.symbol]
                requested_ratio = getattr(event, 'strength', 1.0)
                try:
                    requested_ratio = float(requested_ratio)
                except (TypeError, ValueError):
                    requested_ratio = 1.0
                requested_ratio = min(max(requested_ratio, 0.0), 1.0)
                raw_quantity = int(position.quantity * requested_ratio)
                quantity, _ = self._round_to_board_lot(event.symbol, raw_quantity)
                if requested_ratio >= 0.999:
                    quantity = position.quantity
                if quantity <= 0:
                    self._log_order_rejection(
                        event,
                        reason="卖出比例按交易单位取整后数量为0",
                        requested_ratio=requested_ratio,
                        raw_quantity=raw_quantity,
                        held_quantity=position.quantity,
                    )
                    return
                execution_period = self._resolve_execution_period(event)
                order = OrderEvent(
                    event.symbol,
                    'MKT',
                    quantity,
                    'SELL',
                    period=execution_period,
                    strategy_params=getattr(event, 'strategy_params', {}),
                    metadata=getattr(event, 'metadata', {}),
                ) # 卖出时不需要风险参数
                events.put(order)
    
    def _resolve_execution_period(self, event) -> str:
        period = getattr(event, 'execution_period', None)
        if period is None:
            period = getattr(self.data_handler, 'valuation_period_key', DAY_PERIOD_KEY)
        return normalize_period_key(period)

    def _get_board_lot(self, symbol: str) -> int:
        if symbol.endswith('.US'):
            return 1
        if symbol.endswith('.HK') or symbol.endswith('.SH') or symbol.endswith('.SZ'):
            return 100
        return 1

    def _round_to_board_lot(self, symbol: str, quantity: int) -> tuple[int, int]:
        board_lot = self._get_board_lot(symbol)
        if quantity < board_lot:
            return 0, board_lot
        return (quantity // board_lot) * board_lot, board_lot

    def _log_order_rejection(self, event, reason: str, **details):
        print(
            f"[下单失败] {event.symbol} {event.signal_type} by {event.strategy_name}: {reason}"
        )
        for key, value in details.items():
            if value is None:
                continue
            if isinstance(value, float):
                print(f"  - {key}: {value:.4f}")
            else:
                print(f"  - {key}: {value}")

    def _calculate_position_size(
        self,
        price: float,
        risk_per_share: float,
        symbol: str = "",
        strategy_name: str = None,
        signal_strength: float = 1.0,
        is_buy_percentage: bool = True,
        return_details: bool = False,
    ):
        """(逻辑简化，职责更清晰)"""
        if is_buy_percentage:
            trade_risk_amount = self.total * self.risk_per_trade * self.buy_percentages[strategy_name]
            print('购买比例:',self.buy_percentages[strategy_name])
        else:
            try:
                strength = float(signal_strength)
            except (TypeError, ValueError):
                strength = 1.0
            trade_risk_amount = self.total * self.risk_per_trade * max(strength, 0.0)
        
        raw_quantity = int(trade_risk_amount / risk_per_share)
        quantity = raw_quantity
        
        if price * quantity > self.cash:
            quantity = int(self.cash / price)
            
        final_quantity, board_lot = self._round_to_board_lot(symbol, quantity)
        if return_details:
            return {
                'planned_amount': trade_risk_amount,
                'raw_quantity': raw_quantity,
                'cash_capped_quantity': quantity,
                'final_quantity': final_quantity,
                'board_lot': board_lot,
            }
        return final_quantity

    def _calculate_value_position_size(
        self,
        price: float,
        symbol: str,
        target_value_pct: float,
        return_details: bool = False,
    ):
        try:
            pct = float(target_value_pct)
        except (TypeError, ValueError):
            pct = 0.0
        pct = max(pct, 0.0)
        target_trade_value = self.total * pct
        raw_quantity = int(target_trade_value / price)
        quantity = raw_quantity
        if price * quantity > self.cash:
            quantity = int(self.cash / price)
        final_quantity, board_lot = self._round_to_board_lot(symbol, quantity)
        if return_details:
            return {
                'planned_amount': target_trade_value,
                'raw_quantity': raw_quantity,
                'cash_capped_quantity': quantity,
                'final_quantity': final_quantity,
                'board_lot': board_lot,
            }
        return final_quantity
    
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
                stop_loss_price=getattr(event, 'stop_loss_price', 0.0),
                strategy_params=getattr(event, 'strategy_params', {})
            )
            self.current_holdings[event.symbol] = position
            self.cash -= (event.fill_cost + event.commission)
        else: # SELL
            if event.symbol in self.current_holdings:
                position = self.current_holdings[event.symbol]
                if event.quantity >= position.quantity:
                    del self.current_holdings[event.symbol]
                else:
                    position.quantity -= event.quantity
                    stage_flag = getattr(event, 'metadata', {}).get('conservative_exit_stage_flag')
                    if stage_flag:
                        state = position.strategy_params.setdefault('conservative_exit_state', {})
                        state[stage_flag] = True
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
