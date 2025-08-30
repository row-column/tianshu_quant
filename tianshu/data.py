# tianshu/data.py
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
                combined_index = combined_index.union(self.symbol_data[s].index)
        
        # 对齐所有数据
        self.all_indices = combined_index.sort_values()
        for s in self.symbol_list:
            self.symbol_data[s] = self.symbol_data[s].reindex(index=self.all_indices, method='pad').dropna()
            self.latest_symbol_data[s] = []

    def get_latest_bars(self, symbol, N=1):
        try:
            return self.latest_symbol_data[symbol].tail(N)
        except (KeyError, IndexError):
            return pd.DataFrame() # 返回空DataFrame

    def update_bars(self):
        if self.bar_index < len(self.all_indices):
            current_time = self.all_indices[self.bar_index]
            for s in self.symbol_list:
                try:
                    # 获取当前时间点的一行数据
                    bar = self.symbol_data[s].loc[current_time]
                    # 追加到最新数据列表
                    self.latest_symbol_data[s] = self.symbol_data[s][self.symbol_data[s].index <= current_time]
                except KeyError:
                    # 当天该股票可能停牌
                    pass
            
            events.put(MarketEvent())
            self.bar_index += 1
        else:
            self.continue_backtest = False