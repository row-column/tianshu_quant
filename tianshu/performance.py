import os, sys
project_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_path)
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def create_sharpe_ratio(returns, periods=252):
    """计算夏普比率。"""
    if returns.std() == 0: return 0
    return np.sqrt(periods) * (returns.mean()) / returns.std()

def create_drawdowns(equity_curve):
    """计算最大回撤。"""
    hwm = equity_curve.cummax()
    drawdown = (hwm - equity_curve) / hwm
    return drawdown.max()

def show_performance_stats(equity_curve, initial_capital):
    """展示回测的性能统计和图表。"""
    print("\n--- 天枢Quant回测性能报告 ---")
    
    total_return = (equity_curve['total'].iloc[-1] / initial_capital - 1.0)
    print(f"初始资本: ${initial_capital:,.2f}")
    print(f"最终总资产: ${equity_curve['total'].iloc[-1]:,.2f}")
    print(f"总回报率: {total_return:.2%}")
    
    returns = equity_curve['returns'].dropna()
    sharpe = create_sharpe_ratio(returns)
    max_dd = create_drawdowns(equity_curve['total'])
    
    print(f"夏普比率 (年化): {sharpe:.2f}")
    print(f"最大回撤: {max_dd:.2%}")
    
    # 绘制净值曲线
    fig, ax = plt.subplots(figsize=(12, 8))
    ax.plot(equity_curve.index, equity_curve['total'].values)
    ax.set_title('Portfolio Equity Curve')
    ax.set_xlabel('Date')
    ax.set_ylabel('Total Value ($)')
    plt.grid(True)
    plt.show()