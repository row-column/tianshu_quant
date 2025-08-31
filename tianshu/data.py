import os, sys
project_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_path)
import os
import pandas as pd
from abc import ABC, abstractmethod
from .event import MarketEvent, events
from config.settings import DATA_PATH

class DataHandler(ABC):
    @abstractmethod
    def get_latest_bars(self, symbol, N=1):
        raise NotImplementedError
    @abstractmethod
    def update_bars(self):
        raise NotImplementedError

class HistoricDataHandler(DataHandler):
    """从本地Parquet文件读取数据，用于回测。"""
    def __init__(self, symbol_list):
        self.symbol_list = symbol_list
        self.symbol_data = {}
        self.latest_symbol_data = {}
        self.continue_backtest = True
        self.bar_index = 0
        self._open_and_load_data()

    def _open_and_load_data(self):
        combined_index = None
        for s in self.symbol_list:
            filepath = os.path.join(DATA_PATH, f"{s}.parquet")
            self.symbol_data[s] = pd.read_parquet(filepath)
            # 确保索引是DatetimeIndex
            if not isinstance(self.symbol_data[s].index, pd.DatetimeIndex):
                 self.symbol_data[s]['timestamp'] = pd.to_datetime(self.symbol_data[s]['timestamp'])
                 self.symbol_data[s].set_index('timestamp', inplace=True)

            if combined_index is None:
                combined_index = self.symbol_data[s].index
            else:
                # 合并所有出现过的日期，并去重
                combined_index = combined_index.union(self.symbol_data[s].index)
        
        # 对齐所有数据
        self.all_indices = combined_index.sort_values()
        # 【关键移除】不再 reindex 和 pad 数据
        # 初始化每个股票的最新数据为一个空的DataFrame
        for s in self.symbol_list:
            # self.symbol_data[s] = self.symbol_data[s].reindex(index=self.all_indices, method='pad').dropna()
            # self.latest_symbol_data[s] = []
            self.latest_symbol_data[s] = pd.DataFrame() # 初始化为空DataFrame

    def get_latest_bars(self, symbol, N=1):
        """
        返回最新的N条数据。如果当天停牌，返回的就是昨天的数据。
        """
        try:
            return self.latest_symbol_data[symbol].tail(N)
        except (KeyError, IndexError):
            return pd.DataFrame() # 返回空DataFrame

    def update_bars(self):
        """
        只推送当天真实存在的K线。如果某股票当天停牌，则其 latest_symbol_data 不会更新，
        get_latest_bars 返回的就是前一天的数据，这完美模拟了现实。
        """
        if self.bar_index < len(self.all_indices):
            current_time = self.all_indices[self.bar_index]
            for s in self.symbol_list:
                # 尝试获取当前时间点的数据
                if current_time in self.symbol_data[s].index:
                    # 【逻辑修正】只更新到当前时间点，而不是追加
                    self.latest_symbol_data[s] = self.symbol_data[s][:current_time]
                '''  
                try:
                    # 获取当前时间点的一行数据
                    bar = self.symbol_data[s].loc[current_time]
                    # 追加到最新数据列表
                    self.latest_symbol_data[s] = self.symbol_data[s][self.symbol_data[s].index <= current_time]
                except KeyError:
                    # 当天该股票可能停牌
                    pass
                '''
            events.put(MarketEvent())
            self.bar_index += 1
        else:
            self.continue_backtest = False