import os, sys
project_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_path)
import os
import pandas as pd
from abc import ABC, abstractmethod
from .event import MarketEvent, events
from .periods import DAY_PERIOD_KEY, normalize_period_key, period_key_to_minutes
from config.settings import DATA_PATH

class DataHandler(ABC):
    @abstractmethod
    def get_latest_bars(self, symbol, N=1, period=None):
        raise NotImplementedError
    @abstractmethod
    def update_bars(self):
        raise NotImplementedError

class HistoricDataHandler(DataHandler):
    """从本地Parquet文件读取数据，用于回测。"""
    def __init__(self, symbol_list, periods=None, data_path=None, start_date=None, end_date=None):
        self.symbol_list = symbol_list
        self.data_path = data_path or DATA_PATH
        self.start_timestamp = self._parse_bound_timestamp(start_date, is_end=False)
        self.end_timestamp = self._parse_bound_timestamp(end_date, is_end=True)
        self.period_keys = [normalize_period_key(p) for p in (periods or [DAY_PERIOD_KEY])]
        if DAY_PERIOD_KEY not in self.period_keys:
            self.period_keys.append(DAY_PERIOD_KEY)
        self.period_keys = sorted(set(self.period_keys), key=period_key_to_minutes)
        self.execution_period = self.period_keys[0]
        self.valuation_period_key = self.execution_period
        self.symbol_data = {}
        self.latest_symbol_data = {}
        self.last_bar_timestamps = {}
        self.last_updated_periods = set()
        self.last_updated_symbols_by_period = {}
        self.current_time = None
        self.continue_backtest = True
        self.bar_index = 0
        self._open_and_load_data()

    def _parse_bound_timestamp(self, value, is_end=False):
        if value is None:
            return None
        ts = pd.Timestamp(value)
        if ts.tzinfo is None:
            ts = ts.tz_localize('UTC')
        else:
            ts = ts.tz_convert('UTC')
        if is_end and ts == ts.normalize():
            ts = ts + pd.Timedelta(days=1) - pd.Timedelta(nanoseconds=1)
        return ts

    def _candidate_paths_for(self, symbol, period_key):
        period_dir_path = os.path.join(self.data_path, period_key, f"{symbol}.parquet")
        if period_key == DAY_PERIOD_KEY:
            return [
                period_dir_path,
                os.path.join(self.data_path, f"{symbol}.parquet"),
            ]
        return [
            period_dir_path,
            os.path.join(self.data_path, f"{symbol}_{period_key}.parquet"),
        ]

    def _data_path_for(self, symbol, period_key):
        for filepath in self._candidate_paths_for(symbol, period_key):
            if os.path.exists(filepath):
                return filepath
        searched = ", ".join(self._candidate_paths_for(symbol, period_key))
        raise FileNotFoundError(
            f"找不到 {symbol} 的 {period_key} K线数据。已查找: {searched}"
        )

    def _load_symbol_period_data(self, symbol, period_key):
        filepath = self._data_path_for(symbol, period_key)
        df = pd.read_parquet(filepath)
        if not isinstance(df.index, pd.DatetimeIndex):
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            df.set_index('timestamp', inplace=True)
        if df.index.tz is None:
            df.index = df.index.tz_localize('UTC')
        df.sort_index(inplace=True)
        return df

    def _open_and_load_data(self):
        combined_index = None
        for period_key in self.period_keys:
            self.symbol_data[period_key] = {}
            self.latest_symbol_data[period_key] = {}
            self.last_bar_timestamps[period_key] = {}
            self.last_updated_symbols_by_period[period_key] = set()
            for s in self.symbol_list:
                self.symbol_data[period_key][s] = self._load_symbol_period_data(s, period_key)

                if period_key == self.execution_period:
                    if combined_index is None:
                        combined_index = self.symbol_data[period_key][s].index
                    else:
                        # 合并所有出现过的时间点，并去重
                        combined_index = combined_index.union(self.symbol_data[period_key][s].index)

                self.latest_symbol_data[period_key][s] = pd.DataFrame()
                self.last_bar_timestamps[period_key][s] = None
        
        # 对齐所有数据
        self.all_indices = combined_index.sort_values()
        if self.start_timestamp is not None:
            self.all_indices = self.all_indices[self.all_indices >= self.start_timestamp]
        if self.end_timestamp is not None:
            self.all_indices = self.all_indices[self.all_indices <= self.end_timestamp]
        if self.all_indices.empty:
            raise ValueError(
                f"回测时间范围内没有可用K线: start={self.start_timestamp}, end={self.end_timestamp}"
            )

    def get_latest_bars(self, symbol, N=1, period=None):
        """
        返回指定周期最新的N条数据。如果当前时间该股票没有新K线，返回上一根已知K线。
        """
        period_key = normalize_period_key(period)
        try:
            return self.latest_symbol_data[period_key][symbol].tail(N)
        except (KeyError, IndexError):
            return pd.DataFrame() # 返回空DataFrame

    def get_latest_bar(self, symbol, period=None):
        latest = self.get_latest_bars(symbol, N=1, period=period)
        return None if latest.empty else latest.iloc[0]

    def _slice_to_current_time(self, data, period_key, current_time):
        if period_key == DAY_PERIOD_KEY and self.execution_period != DAY_PERIOD_KEY:
            current_day_start = current_time.normalize()
            return data.loc[data.index.normalize() < current_day_start]
        return data.loc[:current_time]

    def has_any_new_bar(self, period=None):
        period_key = normalize_period_key(period)
        return period_key in self.last_updated_periods

    def has_new_bar(self, symbol, period=None):
        period_key = normalize_period_key(period)
        return symbol in self.last_updated_symbols_by_period.get(period_key, set())

    def update_bars(self):
        """
        以执行周期推进时间，并把所有已加载周期都切到当前时间之前的数据。
        """
        if self.bar_index < len(self.all_indices):
            current_time = self.all_indices[self.bar_index]
            self.current_time = current_time
            self.last_updated_periods = set()
            self.last_updated_symbols_by_period = {period_key: set() for period_key in self.period_keys}
            for period_key in self.period_keys:
                for s in self.symbol_list:
                    data = self.symbol_data[period_key][s]
                    latest_data = self._slice_to_current_time(data, period_key, current_time)
                    self.latest_symbol_data[period_key][s] = latest_data

                    latest_timestamp = None if latest_data.empty else latest_data.index[-1]
                    if latest_timestamp is not None and latest_timestamp != self.last_bar_timestamps[period_key][s]:
                        self.last_bar_timestamps[period_key][s] = latest_timestamp
                        self.last_updated_periods.add(period_key)
                        self.last_updated_symbols_by_period[period_key].add(s)
            events.put(MarketEvent(datetime=current_time))
            self.bar_index += 1
        else:
            self.continue_backtest = False
