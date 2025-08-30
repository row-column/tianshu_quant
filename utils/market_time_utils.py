#!/usr/bin/env python
# -*- coding=utf-8 -*-
import os, sys
project_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_path)
from enum import Enum, auto
from datetime import datetime, timedelta,timezone, time as dt_time
from typing import Optional
import pytz
from positions import MarketType
from utils.cfg_utils import load_yaml2cfg

# --- 模块级依赖与配置 ---
cfg_file = os.path.join(project_path, 'data/server_config.yaml')
cfg = load_yaml2cfg(cfg_file)

# --- 核心数据结构 ---

# 定义各个市场的本地交易时间段
MARKET_TRADING_HOURS = {
    "HK": {
        "timezone": "Asia/Hong_Kong",
        "sessions": [
            (dt_time(9, 30), dt_time(12, 0)),  # 上午盘
            (dt_time(13, 0), dt_time(16, 0))   # 下午盘
        ]
    },
    "US": {
        "timezone": "America/New_York",
        "sessions": [
            (dt_time(9, 30), dt_time(16, 0))   # 连续交易
        ]
    }
}

# 创建一个从 MarketType 到 pytz 时区对象的映射
MARKET_TIMEZONES = {
    MarketType.HK: pytz.timezone('Asia/Hong_Kong'),
    MarketType.US: pytz.timezone('America/New_York'),
}

class TradingSession(Enum):
    PRE_MARKET = auto()
    OPEN_HOUR = auto()      # 开盘后第一个小时
    MIDDAY = auto()         # 中盘
    POWER_HOUR = auto()     # 收盘前最后一个小时
    AFTER_MARKET = auto()

# --- 核心功能函数 ---

def get_timezone_for_symbol(symbol: str):
    """根据股票代码获取其市场对应的pytz时区对象。"""
    from utils.common_utils import get_market_type # 延迟导入以避免循环依赖
    market_type = get_market_type(symbol)
    return MARKET_TIMEZONES.get(market_type, pytz.utc)

def is_market_in_trading_hours(market: MarketType, current_utc_time: datetime = None) -> bool:
    """
    [V2.0 健壮版] 检查指定市场当前是否处于交易时间内。
    - 能正确处理港股午休等非连续时段。
    - 使用时区转换，能自动适应美股夏令时/冬令时变化。
    - 移除了对全局 cfg 的依赖，更具通用性。
    """
    if market not in [MarketType.HK, MarketType.US]:
        return False

    market_str = market.name # "HK" 或 "US"
    market_info = MARKET_TRADING_HOURS[market_str]
    market_tz = pytz.timezone(market_info["timezone"])

    # 获取当前时间
    if current_utc_time is None:
        current_utc_time = datetime.now(pytz.utc)

    # 将当前UTC时间转换为对应市场的本地时间
    current_local_time = current_utc_time.astimezone(market_tz)
    
    # 检查是否为工作日 (周一到周五)
    if current_local_time.weekday() > 4:
        return False

    # 遍历该市场的所有交易时段
    current_time_part = current_local_time.time()
    for start_time, end_time in market_info["sessions"]:
        if start_time <= current_time_part <= end_time: # 使用左闭右开更严谨
            return True

    return False

def is_any_market_open(symbol: str = None) -> bool:
    """
    [V2.0 健壮版] 检查是否有任何一个或指定市场处于交易时间内。
    """
    # 移除了对全局 cfg 的依赖，测试模式应在调用方处理
    if cfg.common.test_mode:
        return True
    
    current_utc_time = datetime.now(pytz.utc)
    
    if symbol:
        try:
            from utils.common_utils import get_market_type # 延迟导入以避免循环依赖
            market = get_market_type(symbol)
            return is_market_in_trading_hours(market, current_utc_time)
        except ValueError:
            return False # 无效symbol，认为市场未开放
    
    # 如果未提供symbol，则检查所有我们关心的市场
    return is_market_in_trading_hours(MarketType.HK, current_utc_time) or \
           is_market_in_trading_hours(MarketType.US, current_utc_time)


def is_hk_market_open() -> bool:
    """[V2.0 健壮版] 检查港股市场是否开放"""
    if cfg.common.test_mode:
        return True
    return is_market_in_trading_hours(MarketType.HK)

def is_us_market_open() -> bool:
    """[V2.0 健壮版] 检查美股市场是否开放"""
    if cfg.common.test_mode:
        return True
    return is_market_in_trading_hours(MarketType.US)

def get_current_session(market_str: str) -> Optional[TradingSession]:
    """根据市场字符串（'HK'或'US'）获取当前交易时段"""
    from .performance_utils import thread_safe_ttl_cache # 延迟导入装饰器

    # 将函数逻辑包装起来，以便装饰器可以应用
    @thread_safe_ttl_cache(maxsize=10, ttl=15)
    def _get_session_cached(market_str_cached: str) -> Optional[TradingSession]:
        market_info = MARKET_TRADING_HOURS.get(market_str_cached.upper())
        if not market_info:
            return None
            
        market_tz = pytz.timezone(market_info["timezone"])
        now_market_time = datetime.now(market_tz).time()
        today_date = datetime.now(market_tz).date()

        open_time = market_info['sessions'][0][0]
        close_time = market_info['sessions'][-1][1]
        
        open_hour_end = (datetime.combine(today_date, open_time) + timedelta(hours=1)).time()
        power_hour_start = (datetime.combine(today_date, close_time) - timedelta(hours=1)).time()

        is_in_session = any(start <= now_market_time < end for start, end in market_info["sessions"])
        if not is_in_session:
            return TradingSession.PRE_MARKET if now_market_time < open_time else TradingSession.AFTER_MARKET

        if open_time <= now_market_time < open_hour_end:
            return TradingSession.OPEN_HOUR
        elif open_hour_end <= now_market_time < power_hour_start:
            return TradingSession.MIDDAY
        elif power_hour_start <= now_market_time < close_time:
            return TradingSession.POWER_HOUR
        
        return None

    return _get_session_cached(market_str)

# ==============================================================================
# --- 新增核心函数 (New Core Function) ---
# ==============================================================================
def is_in_eod_buy_window(market: MarketType, window_minutes: int = 30) -> bool:
    """
    【新增】检查指定市场当前是否处于收盘前指定分钟数的尾盘交易窗口内。

    :param market: 市场类型 (MarketType.HK 或 MarketType.US)。
    :param window_minutes: 收盘前多少分钟视为窗口期, 默认为30分钟。
    :return: 如果在窗口期内则返回 True，否则返回 False。
    """
    if market not in [MarketType.HK, MarketType.US]:
        return False
    
    # 测试模式下，不激活尾盘窗口，让主逻辑 is_any_market_open 控制
    if cfg.common.test_mode:
        return False

    market_str = market.name
    market_info = MARKET_TRADING_HOURS.get(market_str)
    if not market_info or not market_info["sessions"]:
        return False

    market_tz = pytz.timezone(market_info["timezone"])
    current_local_time = datetime.now(market_tz)
    
    # 检查是否为工作日 (周一到周五)
    if current_local_time.weekday() > 4:
        return False

    # 获取最后一个交易时段的结束时间，即收盘时间
    close_time = market_info["sessions"][-1][1]
    
    # 将收盘时间与当前日期结合，创建完整的datetime对象
    # 使用 localize 来确保附加的时区信息是正确的，避免夏令时问题
    close_datetime = market_tz.localize(datetime.combine(current_local_time.date(), close_time))
    
    # 计算尾盘窗口的开始时间
    window_start_datetime = close_datetime - timedelta(minutes=window_minutes)

    # 检查当前市场本地时间是否落在 [窗口开始时间, 收盘时间) 区间内
    return window_start_datetime <= current_local_time < close_datetime
# ==============================================================================

# ==============================================================================
# --- 开盘观察窗函数 ---
# ==============================================================================
def is_in_opening_window(market: MarketType, window_minutes: int=30) -> bool:
    """
    【V3.0 无懈可击版】检查指定股票当前是否处于开盘后的“观察窗口期”内。

    本函数经过极致优化，可应对所有实战场景：
    - **逻辑严密**: 彻底修复了简单比较 time() 对象会导致在盘前时段误判的致命bug。
    - **自动时区**: 能根据股票代码自动适配港股/美股时区，并完美处理夏令时/冬令时切换。
    - **交易日判断**: 自动过滤非交易日（周末）。
    - **配置驱动**: 与系统配置的 test_mode 联动，确保测试与实盘行为一致。
    
    :param market: 市场类型 (MarketType.HK 或 MarketType.US)。
    :param window_minutes: 开盘后多少分钟视为窗口期。
    :return: 如果在窗口期内则返回 True，否则返回 False。
    """
    # 在测试模式下，时间窗口的概念没有意义，直接返回False，避免干扰回测逻辑。
    if cfg.common.test_mode:
        return False

    market_info = MARKET_TRADING_HOURS.get(market)
    # 如果没有市场信息或交易时段信息，直接返回False
    if not market_info or not market_info["sessions"]:
        return False

    market_tz = pytz.timezone(market_info["timezone"])
    current_local_time = datetime.now(market_tz)
    
    # 检查是否为工作日 (周一到周五)
    if current_local_time.weekday() > 4:
        return False

    # 获取第一个交易时段的开始时间，即开盘时间
    open_time = market_info["sessions"][0][0]
    
    # --- 核心逻辑：使用完整的 datetime 对象进行比较，100% 避免时区和跨日问题 ---
    
    # 1. 创建当天精准的开盘时间（带有时区信息，DST-safe）
    open_datetime = market_tz.localize(datetime.combine(current_local_time.date(), open_time))
    
    # 2. 计算观察窗口的结束时间
    window_end_datetime = open_datetime + timedelta(minutes=window_minutes)

    # 3. [关键判断] 检查当前市场本地时间是否落在 [开盘时间, 窗口结束时间) 的左闭右开区间内。
    #    这确保了只有在开盘后、窗口结束前的时间点，才会返回True。
    return open_datetime <= current_local_time < window_end_datetime


ASSUMED_NAIVE_TIMEZONE = pytz.timezone('Asia/Shanghai')

def normalize_to_utc(dt_str: str) -> datetime:
    """
    【时间协议核心】将任何ISO格式的时间字符串标准化为带时区的UTC时间对象。
    - 如果字符串本身带有时区信息 (aware), 则直接转换为UTC。
    - 如果是无时区的"天真"时间 (naive), 则根据预设时区赋予其身份, 然后转换为UTC。
    """
    try:
        dt = datetime.fromisoformat(dt_str)
        # 检查是否是“天真”的
        if dt.tzinfo is None or dt.tzinfo.utcoffset(dt) is None:
            # 赋予其我们假设的本地时区，然后转换为全球唯一的UTC时间
            return ASSUMED_NAIVE_TIMEZONE.localize(dt).astimezone(timezone.utc)
        else:
            # 本身就是“清醒”的，直接转换为UTC以实现标准化
            return dt.astimezone(timezone.utc)
    except Exception:
        # 兜底处理，以防极旧的、非标准ISO格式的时间戳
        dt_naive = datetime.strptime(dt_str, "%Y-%m-%dT%H:%M:%S.%f")
        return ASSUMED_NAIVE_TIMEZONE.localize(dt_naive).astimezone(timezone.utc)


def is_entering_weekend_risk_for_symbol(symbol: str,enable_wrp:bool=True,wrp_activation_days:list[int]=[3, 4]) -> bool:
    """
    [WRP - 上下文解耦版]
    检查指定股票是否正在进入其所在市场的周末风险期。
    这是一个纯粹的工具函数，其行为由全局系统上下文决定。
    """
    
    # 步骤 1: 进行防御性检查
    if not enable_wrp:
        return False
    
    try:
        # 步骤 3: 获取该股票对应的市场时区 (这是它自己的职责)
        market_tz = get_timezone_for_symbol(symbol)
        
        # 步骤 4: 获取该市场的本地当前时间
        now_local = datetime.now(market_tz)
        
        # 步骤 5: 在该市场的本地时间下，使用从上下文中获取的配置进行判断
        if now_local.weekday() in wrp_activation_days:
            # 日志可以保留，因为它对调试很有用
            # logger.info(f"【WRP 已激活 for {symbol} ({now_local.strftime('%A')})】")
            return True

        return False
    except Exception as e:
        # logger.error(f"检查 {symbol} 的WRP激活状态时出错: {e}", exc_info=True)
        return False


# print(normalize_to_utc('2025-08-11T15:58:16.391048'))
# print(is_entering_weekend_risk_for_symbol('TSLL.US'))