#!/usr/bin/env python
# -*- coding=utf-8 -*-
import os, sys
project_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_path)
from typing import  Dict, Any,Tuple
import logging
from longport.openapi import QuoteContext,Market,Period,AdjustType, OpenApiException, TradeSessions,SecurityQuote
from typing import Optional,List
import pandas_ta as ta
import pandas  as pd
import math
from datetime import date, timedelta
import time
from enum import Enum
from utils.performance_utils import thread_safe_ttl_cache,APIRateLimiter
logger = logging.getLogger(__name__)

quote_api_limiter = APIRateLimiter(requests_per_second=10)

# ==============================================================================
# === 【天枢系统 · 数据宪章】不同K线周期的最优数据获取量 (count) 定义 ===
# ==============================================================================
# 原理:
#   - 日线 (Day): 核心是满足年线(MA250)级别的趋势判断，250是行业标准。
#   - 中长周期 (60/240min): 需要回看数周到数月的趋势，200-250根K线提供了足够的历史深度。
#   - 日内核心周期 (15/30min): 需要覆盖过去数个交易日以观察日内结构，300-400根K线是理想选择。
#   - 短周期/确认周期 (1/5min): 需要覆盖当天及前一天的完整交易时段，以进行精细的形态和量价分析。
# ==============================================================================
OPTIMAL_KLINE_COUNT_MAP = {
    # --- 宏观与波段交易周期 ---
    id(Period.Day): 252,      # 日线级别：目标是覆盖一整年的交易日(~250天)。这足以计算任何长周期指标，如MA200、年线等，是判断长期趋势的基石。

    id(Period.Min_240): 200,  # 4小时线：200根K线约等于100个交易日（美股一天约2根），即接近半年的交易数据。足以进行中期波段的结构分析。

    id(Period.Min_180): 200,  # 3小时线：逻辑同上，提供数月的中期趋势视图。

    id(Period.Min_60): 250,   # 1小时线：250根K线约等于35-40个交易日（美股一天6.5根），覆盖一个半月的完整走势。对于小时级别的MACD、RSI等指标，提供了非常充分的“预热”和回看数据。

    # --- 日内交易核心周期 ---
    id(Period.Min_30): 300,   # 30分钟线：300根K线约等于23个交易日（美股一天13根），覆盖一个月的日内详细走势。对于“W底”这类需要较长回看周期的形态策略，这是必须的。

    id(Period.Min_15): 350,   # 15分钟线：350根K线约等于13个交易日（美股一天26根），覆盖超过两周的全部日内波动，足以识别关键的日内支撑阻力位。

    id(Period.Min_10): 400,   # 10分钟线: 400根K线约等于10个交易日（美股一天39根），提供非常精细的两周走势图。

    # --- 精细入场与确认周期 ---
    id(Period.Min_5): 250,    # 5分钟线：250根K线约等于3个完整的交易日（美股一天78根）。这对于“5分钟确认信号”至关重要，它需要看到昨天和前天的成交量均值和价格形态。

    id(Period.Min_3): 400,    # 3分钟线：400根K线约等于3个完整交易日（美股一天130根），提供更高频的确认视角。

    id(Period.Min_1): 480,    # 1分钟线：400根K线刚好覆盖一个完整的交易日还多一点（美股一天390根）。这对于需要分析当日完整VWAP或寻找尾盘突破的策略来说，是最低要求。
}

@thread_safe_ttl_cache(maxsize=512, ttl=5) # 缓存5秒，对于日内策略足够灵敏
def get_realtime_quote(quote_ctx: QuoteContext, symbol: str) -> Optional[SecurityQuote]:
    """
    【核心工具函数】获取单个股票的完整实时报价对象，并使用线程安全的TTL缓存。
    
    返回的是长桥原始的 Quote 对象，包含了开盘、最高、最低、最新价等所有信息。

    Args:
        quote_ctx: QuoteContext 实例.
        symbol: 股票代码.

    Returns:
        Quote 对象或在获取失败时返回 None.
    """
    logger.debug(f"正在请求 {symbol} 的实时报价...")
    try:
        #【核心修复】在发起真正的API调用前，先通过速率限制器等待。
        quote_api_limiter.wait()
        # 即使只查一个，也使用批量接口，保持代码模式统一
        quotes = quote_ctx.quote([symbol])
        if quotes and len(quotes) > 0:
            # 返回完整的 Quote 对象
            return quotes[0]
        else:
            logger.warning(f"无法获取 {symbol} 的有效实时报价对象")
            return None
    except Exception as e:
        logger.error(f"获取实时报价对象失败 for {symbol}: {e}", exc_info=True)
        return None

# 同时，我们可以优化现有的 get_current_price 函数，让它复用这个新的核心函数
def get_current_price(quote_ctx: QuoteContext, symbol: str) -> Optional[float]:
    """
    【优化版】从指定的 QuoteContext 获取股票的当前价格。
    """
    quote = get_realtime_quote(quote_ctx, symbol)
    if quote and quote.last_done:
        return float(quote.last_done)
    return None

def get_current_volume(quote_ctx: QuoteContext, symbol: str) -> int:
    """
    获取当前K线周期内的累计成交量
    
    Args:
        symbol: 股票代码
        
    Returns:
        int: 当前成交量，获取失败返回None
    """
    quote = get_realtime_quote(quote_ctx, symbol)
    if quote and quote.last_done:
        return float(quote.volume)
    return None

@thread_safe_ttl_cache(maxsize=256, ttl=3600)
def get_stock_static_info(quote_ctx: QuoteContext, symbol: str) -> Optional[Dict[str, Any]]:
    """
    【统一工具函数】通过股票代码，从指定的 QuoteContext 获取股票的静态信息。

    基于长桥证券API: http://open.longportapp.cn/zh-CN/docs/quote/pull/static

    Args:
        quote_ctx: 用于发起API调用的 QuoteContext 对象。
        symbol: 股票代码 (例如: '700.HK', 'AAPL.US')。

    Returns:
        一个包含股票静态信息的字典，如果无法获取信息则返回 None。
        字典包含 'symbol', 'name_cn', 'lot_size', 'eps' 等字段。
    """
    try:
        # 使用批量接口，即使只查询一个，也保持代码模式统一
        securityStaticInfo = quote_ctx.static_info([symbol])
        
        # 检查响应是否有效且包含数据
        if securityStaticInfo and securityStaticInfo and len(securityStaticInfo) > 0:
            static_data = securityStaticInfo[0]
            # 将SDK返回的Pydantic对象转换为字典，方便下游使用
            return {
                "symbol": static_data.symbol,
                "name_cn": static_data.name_cn,
                "name_en": static_data.name_en,
                "name_hk": static_data.name_hk,
                "exchange": static_data.exchange,
                "currency": static_data.currency,
                "lot_size": static_data.lot_size,
                "total_shares": static_data.total_shares,
                "circulating_shares": static_data.circulating_shares,
                "hk_shares": static_data.hk_shares,
                "eps": static_data.eps,
                "eps_ttm": static_data.eps_ttm,
                "bps": static_data.bps,
                "dividend_yield": static_data.dividend_yield,
                "stock_derivatives": static_data.stock_derivatives,
                "board": static_data.board,
            }
        else:
            logger.warning(f"无法获取 {symbol} 的有效静态信息")
            return None
    except Exception as e:
        logger.error(f"获取股票静态信息失败 {symbol}: {e}")
        return None

@thread_safe_ttl_cache(maxsize=512, ttl=300) # 缓存10秒，对于日内策略足够灵敏
def get_market_temperature(quote_ctx: QuoteContext, market: Market) -> int:
    """获取指定市场的当前温度值"""
    try:
        resp = quote_ctx.market_temperature(market)
        if resp and resp.temperature is not None:
            logger.info(f"获取到 {market} 市场温度: {resp.temperature} ({resp.description})")
            return resp.temperature
        else:
            raise ValueError(f"API未返回有效的温度数据 for {market}")
    except Exception as e:
        logger.error(f"获取 {market} 市场温度失败: {e}。将使用默认中性值。")
        return 50 # 返回一个默认的中性值

# ==============================================================================
# 核心历史数据获取函数
# ==============================================================================

def _estimate_bars_per_day(period: Period) -> int:
    """[辅助函数] 根据周期估算一个交易日大致的K线数量"""
    if period == Period.Day: return 1
    if period == Period.Min_60: return 7
    if period == Period.Min_30: return 14
    if period == Period.Min_15: return 28
    if period == Period.Min_10: return 39
    if period == Period.Min_5: return 80
    if period == Period.Min_3: return 135
    if period == Period.Min_1: return 400
    return 1

def _process_raw_bars(raw_bars: list, target_count: int) -> pd.DataFrame:
    """[辅助函数] 用于处理原始K线列表并转换为格式正确的DataFrame"""
    if not raw_bars:
        return pd.DataFrame()
        
    data_list = [
        {'timestamp': bar.timestamp, 'open': bar.open, 'high': bar.high, 
         'low': bar.low, 'close': bar.close, 'volume': bar.volume, 
         'turnover': bar.turnover} for bar in raw_bars
    ]
    df = pd.DataFrame(data_list)
    
    numeric_cols = ['open', 'high', 'low', 'close', 'volume', 'turnover']
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    df.dropna(subset=numeric_cols, inplace=True)
    
    df.drop_duplicates(subset=['timestamp'], keep='first', inplace=True)
    df.sort_values('timestamp', ascending=False, inplace=True)
    
    final_df = df.head(target_count).sort_values('timestamp', ascending=True).reset_index(drop=True)
    
    # [优化] 使用更健壮的时间转换
    final_df['timestamp'] = pd.to_datetime(final_df['timestamp'], unit='s', utc=True).dt.tz_convert('Asia/Shanghai')

    logger.debug(f"成功获取并整理了 {len(final_df)} 条数据。")
    return final_df

@thread_safe_ttl_cache(maxsize=500, ttl=30)
def get_klines_data(quote_ctx: QuoteContext, symbol: str, count: int, period: Period, adjust_type: AdjustType
                    ,trade_sessions: Optional[TradeSessions] = TradeSessions.Intraday) -> Optional[pd.DataFrame]:
    """
    获取标的 K 线
    - 数据深度: `candlesticks` 接口支持最多1000条数据，足以满足所有策略需求。
    注意：本接口只能获取到最近 1000 根 K 线，如需获取较长的历史数据，请访问接口：获取标的历史 K 线。
    """
    logger.info(f"正在为 {symbol} 获取 {count} 条【全时段】历史K线 (周期: {period})...")
    try:
        # ★★★ 使用“历史数据”专用限流器 ★★★
        quote_api_limiter.wait() 
        #动态构建API请求参数
        api_kwargs = {
            'symbol': symbol,
            'period': period,
            'adjust_type': adjust_type,
            'count': count
        }
        # 只有当 trade_sessions 被显式传递时，才将其加入到请求参数中
        if trade_sessions:
            api_kwargs['trade_sessions'] = trade_sessions
        
        raw_bars = quote_ctx.candlesticks(**api_kwargs)

        if not raw_bars:
            logger.warning(f"[{symbol}] API调用(candlesticks)未返回任何K线数据。")
            return None
        
        # 数据处理函数 _process_and_convert_bars_to_df 无需改变，继续复用
        final_df = _process_and_convert_bars_to_df(raw_bars, symbol)
        
        if final_df is None:
            return None

        if len(final_df) < count:
             logger.warning(f"[{symbol}] 数据不足: 目标 {count} 条, 实际获取 {len(final_df)} 条。")
        else:
             logger.info(f"[{symbol}] 成功获取 {len(final_df)} 条数据。")
        
        return final_df
    except OpenApiException as e:
        logger.info(f"[{symbol}] 获取历史数据时发生API错误: {e}")
        return None
    except Exception as e:
        # 捕获并打印您遇到的那种参数错误
        logger.error(f"[{symbol}] 获取历史数据时发生未知严重错误: {e}", exc_info=True)
        return None

@thread_safe_ttl_cache(maxsize=500, ttl=15)
def get_history_klines_data(quote_ctx: QuoteContext, symbol: str, count: int, period: Period, adjust_type: AdjustType,
                             trade_sessions: Optional[TradeSessions] = None) -> Optional[pd.DataFrame]:
    """
    获取历史K线数据。
    - 核心逻辑: 回归到语义最清晰、功能最强大的 `history_candlesticks_by_offset` 接口。
    - 关键参数: 经测试，该接口同样支持 `trade_sessions`，确保了港股数据的完整性。
    - 最终方案: 这是满足所有需求的、最健壮的唯一选择。
    Returns:
        pd.DataFrame or None: 
        - 如果成功，返回一个按【时间升序】排列的DataFrame (最旧的数据在第一行)。
        - 如果失败，返回 None。
    """
    # 【核心修复】在发起真正的API调用前，先通过速率限制器等待。
    # ★★★ 使用“历史数据”专用限流器 ★★★
    quote_api_limiter.wait()
    logger.info(f"正在为 {symbol} 获取 {count} 条【全时段】历史K线 (by_offset)(周期: {period})...")
    try:
        #动态构建API请求参数
        api_kwargs = {
            'symbol': symbol,
            'period': period,
            'adjust_type': adjust_type,
            'forward': False,
            'count': count
        }
        # 只有当 trade_sessions 被显式传递时，才将其加入到请求参数中
        if trade_sessions:
            api_kwargs['trade_sessions'] = trade_sessions
        
        raw_bars = quote_ctx.history_candlesticks_by_offset(**api_kwargs)

        # 2. 【健壮性检查 1】在进行任何处理前，先检查API是否返回了空列表。
        if not raw_bars:
            logger.warning(f"[{symbol}] API调用(by_offset)未返回任何K线数据。股票可能上市时间不足或已退市。")
            return None
        
        # 3. 【核心数据处理】调用全新的、无BUG的辅助函数。
        #    - 内部使用 to_numeric 和 dropna 清洗坏数据。
        #    - 内部正确地将时间戳转换为UTC标准时间，不做任何画蛇添足的时区转换。
        #    - 内部做了排序和去重，保证数据质量。
        final_df = _process_and_convert_bars_to_df(raw_bars, symbol)
        
        if final_df is None:
            return None # 如果数据清洗后变为空，也直接返回

        # 4. 【健壮性检查 2】日志记录实际获取的数据量，便于排查问题。
        if len(final_df) < count:
             logger.warning(f"[{symbol}] 数据不足: 目标 {count} 条, 实际获取 {len(final_df)} 条。")
        else:
             logger.info(f"[{symbol}] 成功获取 {len(final_df)} 条数据。")
       
        return final_df

    # 5. 【健壮性检查 3】完整的异常捕获，确保程序不会因为单个股票的数据问题而崩溃。
    except OpenApiException as e:
        logger.info(f"[{symbol}] 获取历史数据时发生API错误: {e}")
        return None
    except Exception as e:
        logger.error(f"[{symbol}] 获取历史数据时发生未知严重错误: {e}", exc_info=False)
        return None


def _process_and_convert_bars_to_df(raw_bars: list, symbol: str) -> Optional[pd.DataFrame]:
    """[全新辅助函数] 处理原始K线列表，进行清洗并转换为格式正确的DataFrame。"""
    if not raw_bars:
        logger.warning(f"[{symbol}] 传入的原始K线数据为空。")
        return None
        
    data_list = [
        {'timestamp': bar.timestamp, 'open': bar.open, 'high': bar.high, 
         'low': bar.low, 'close': bar.close, 'volume': bar.volume} for bar in raw_bars
    ]
    df = pd.DataFrame(data_list)
    
    # 进行数据清洗
    numeric_cols = ['open', 'high', 'low', 'close', 'volume']
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    df.dropna(subset=numeric_cols, inplace=True)
    
    if df.empty:
        logger.warning(f"[{symbol}] 数据经过清洗后变为空。")
        return None

    # [关键修正] 保持UTC时间，不进行任何时区转换，确保全球市场通用
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='s', utc=True)
    df.drop_duplicates(subset=['timestamp'], keep='first', inplace=True)
    df.set_index('timestamp', inplace=True)
    df.sort_index(ascending=True, inplace=True)
    # df.sort_values('timestamp', ascending=True, inplace=True)
    # 按时间索引排序，确保数据是升序的
    df.sort_index(ascending=True, inplace=True)

    return df
    
@thread_safe_ttl_cache(maxsize=500, ttl=30)
def get_history_klines_data_stable(quote_ctx: QuoteContext, symbol: str, count: int, period: Period, adjust_type: AdjustType) -> Optional[pd.DataFrame]:
    """
    [最终通用版] 获取历史K线数据。
    - 使用手动缓存机制，将 Period 和 AdjustType 对象转换为字符串作为缓存键，以解决 unhashable type 错误。
    - 融合了健壮的分批获取逻辑。
    """
    logger.debug(f"请求历史数据 for {symbol} (目标: {count} 条, 周期: {period})...")

    API_OFFSET_LIMIT = 50
    all_bars = []
    collected_timestamps = set()

    try:
        # --- # 步骤1: 使用 by_offset 获取第一批最新的数据
        logger.debug(f"  - 步骤1: 请求最新的 {min(count, API_OFFSET_LIMIT)} 条K线 (by_offset)...")
        initial_bars = quote_ctx.history_candlesticks_by_offset(
            symbol=symbol, period=period, adjust_type=adjust_type,
            forward=False, count=min(count, API_OFFSET_LIMIT)
        )

        if not initial_bars:
            logger.warning(f"[{symbol}] 初始K线请求(by_offset)失败，无法获取任何数据。")
            # 【告警集成】 4. 初始数据获取失败，这是严重问题
            error_msg = f"P1级告警: [数据源] 初始K线请求(by_offset)对 {symbol} 失败，无法获取任何数据！策略引擎可能失效。"
            logger.warning(error_msg)
            # send_weixin_notice(error_msg,['tmnbHCO'])
            return None # 直接失败

        for bar in initial_bars:
            if bar.timestamp not in collected_timestamps:
                all_bars.append(bar)
                collected_timestamps.add(bar.timestamp)
        
        # 如果需要的K线不多，第一批就已满足
        if len(all_bars) >= count:
            logger.debug(f"[{symbol}] 初始请求已满足 {len(all_bars)}/{count} 条数据，无需分批。")
            df = _process_raw_bars(all_bars, count)
            return df

        # 步骤2: 循环使用 by_date 向前回溯获取剩余数据
        oldest_bar_timestamp = min(b.timestamp for b in all_bars)

        while len(all_bars) < count:
            remaining_bars = count - len(all_bars)
            bars_per_day = _estimate_bars_per_day(period)
            # [关键] 使用更安全的动态回溯天数计算，增加缓冲区
            days_to_look_back = max(5, math.ceil(remaining_bars / bars_per_day) + 3)
            
            end_date = (oldest_bar_timestamp - timedelta(seconds=1)).date()
            start_date = end_date - timedelta(days=days_to_look_back)
            
            logger.debug(f"  -> 回溯请求，仍需 {remaining_bars} 条，日期范围: {start_date} -> {end_date}...")

            historical_bars = quote_ctx.history_candlesticks_by_date(
                symbol=symbol
                ,period=period
                ,adjust_type=adjust_type
                ,start=start_date 
                ,end=end_date
            )

            # [关键] 严谨的循环终止条件
            if not historical_bars:
                logger.warning(f"[{symbol}] 在日期范围 {start_date} -> {end_date} 已无更多历史数据，获取终止。")
                break

            newly_fetched_count = 0
            for bar in reversed(historical_bars):
                if bar.timestamp not in collected_timestamps:
                    all_bars.append(bar)
                    collected_timestamps.add(bar.timestamp)
                    newly_fetched_count += 1
            
            if newly_fetched_count == 0:
                logger.info(f"[{symbol}] 在日期范围 {start_date} -> {end_date} 内的数据已全部获取过，获取终止。")
                break

            oldest_bar_timestamp = min(b.timestamp for b in all_bars)
            time.sleep(0.2)

        # --- 步骤3：数据后处理 ---
        if not all_bars:
            # 这个分支理论上在 initial_bars 检查后不会进入，但作为保险
            return None
        final_df = _process_raw_bars(all_bars, count)
        if len(final_df) < count:
             logger.warning(f"[{symbol}] 数据不足: 目标 {count} 条, 实际获取 {len(final_df)} 条。")
        else:
             logger.info(f"[{symbol}] 获取成功，共 {len(final_df)}/{count} 条数据。")
        return final_df

    except Exception as e:
        logger.error(f"[{symbol}] 在获取历史数据时发生严重错误: {e}", exc_info=False)
         # 【告警集成】 5. 任何未预料的严重错误
        error_msg = f"P1级告警: [数据源] 获取历史数据(get_history_data)时对 {symbol} 发生严重错误: {e}"
        logger.error(error_msg, exc_info=False)
        return None

# ==============================================================================
# [核心重构] 终极、健壮、带缓存的 get_history_data 函数
# ==============================================================================
def get_history_klines_data_by_date(quote_ctx: QuoteContext, symbol: str, count: int, period: Period, adjust_type: AdjustType) -> Optional[pd.DataFrame]:
    """
    获取历史K线数据
    - 核心逻辑: 完全基于 `history_candlesticks_by_date`，并严格使用 `date` 对象作为参数。
    - 缓存机制: 已恢复并优化。
    - 健壮性: 在函数内部处理所有异常，永不崩溃。
    """
    logger.debug(f"[{symbol}] 缓存未命中，启动数据获取...")
    try:
        # 3. 计算回溯周期
        days_to_fetch = math.ceil(count * 1.8) + 60
        
        # 4. [关键修复] 创建正确的 `date` 对象
        end_date_obj = date.today()
        start_date_obj = end_date_obj - timedelta(days=days_to_fetch)

        # 5. [核心] 使用正确的API接口和正确的参数类型调用
        symbol_klines = quote_ctx.history_candlesticks_by_date(
            symbol=symbol,
            period=period,
            start=start_date_obj,
            end=end_date_obj,
            adjust_type=adjust_type
        )
        
        # 6. 数据校验
        if not symbol_klines:
            logger.warning(f"[{symbol}] 在 {start_date_obj} 到 {end_date_obj} 周期内API未返回任何数据。")
            return None
        
        # 7. 数据处理
        data_list = [{'timestamp': bar.timestamp, 'open': bar.open, 'high': bar.high, 'low': bar.low, 'close': bar.close, 'volume': bar.volume} for bar in symbol_klines]
        df = pd.DataFrame(data_list)
        
        numeric_cols = ['open', 'high', 'low', 'close', 'volume']
        for col in numeric_cols:
            df[col] = pd.to_numeric(df[col], errors='coerce')
        df.dropna(subset=numeric_cols, inplace=True)
        
        if df.empty:
            return None
            
        df.drop_duplicates(subset=['timestamp'], keep='first', inplace=True)
        df.sort_values('timestamp', ascending=True, inplace=True)
        
        final_df = df.reset_index(drop=True)
        
        logger.debug(f"[{symbol}] 数据获取成功并已缓存，返回 {len(final_df)} 条数据。")
        return final_df

    except OpenApiException as e:
        logger.warning(f"[{symbol}] 获取历史数据时发生API错误，已跳过。错误: {e}")
        return None
    except Exception as e:
        logger.error(f"[{symbol}] 获取历史数据时发生未知严重错误: {e}", exc_info=False)
        return None

# ==============================================================================
# 市场状态分析工具函数
# ==============================================================================
# 定义市场状态
class MarketRegime(Enum):
    ABOVE_MA = "Uptrend"
    BELOW_MA = "Downtrend"
    IN_RANGE = "Range-bound" # 方向不明的区间震荡
    UNKNOWN = "Unknown"

def get_market_regime(df: pd.DataFrame, current_price: float, trend_period: int) -> Tuple[str, str]:
    """
    [通用工具函数] 判断市场状态，结合了缓冲区的思想。
    此函数为纯计算函数，不适合直接缓存，由其调用者(check_market_health)缓存结果。
    """
    if df is None or len(df) < trend_period:
        return MarketRegime.IN_RANGE, f"数据不足以计算{trend_period}日MA"
    
    try:
        ma_series = df['close'].rolling(window=trend_period).mean()
        latest_ma = ma_series.iloc[-1]
        
        if pd.isna(latest_ma):
            return MarketRegime.IN_RANGE, f"{trend_period}日MA计算失败(NaN)"
        
        latest_ma = float(latest_ma)
        threshold = latest_ma * 0.015
        upper_bound = latest_ma + threshold
        lower_bound = latest_ma - threshold

        if current_price > upper_bound:
            return MarketRegime.ABOVE_MA, f"价格 ({current_price:.2f}) > 均线上轨 ({upper_bound:.2f})"
        elif current_price < lower_bound:
            return MarketRegime.BELOW_MA, f"价格 ({current_price:.2f}) < 均线下轨 ({lower_bound:.2f})"
        else:
            return MarketRegime.IN_RANGE, f"价格 ({current_price:.2f}) 在均线缓冲区内 [{lower_bound:.2f}, {upper_bound:.2f}]"

    except Exception as e:
        logger.error(f"计算市场状态时出错: {e}", exc_info=True)
        return MarketRegime.IN_RANGE, "计算市场状态时异常"

@thread_safe_ttl_cache(maxsize=512, ttl=43200) # 缓存12小时 (43200秒)，确保一天内只请求一次
def get_yesterday_close_price(quote_ctx: QuoteContext, symbol: str) -> Optional[float]:
    """
    获取指定股票的昨日收盘价，并使用长效缓存。
    Args:
        quote_ctx: QuoteContext 实例。
        symbol: 股票代码。

    Returns:
        昨日收盘价 (float) 或在获取失败时返回 None。
    """
    logger.debug(f"缓存未命中，正在为 {symbol} 请求昨日收盘价...")
    try:
        quote = get_realtime_quote(quote_ctx, symbol)
        if quote and quote.prev_close:
            return float(quote.prev_close)
        return None
    except Exception as e:
        logger.error(f"获取 {symbol} 昨日收盘价时发生错误: {e}", exc_info=True)
        return None

@thread_safe_ttl_cache(maxsize=200, ttl=3600) # 缓存1小时，用于稳定的头寸计算
def get_historical_atr(quote_ctx: QuoteContext,symbol: str,atr_period:int = 14) -> float:
    """
    [战略版] 计算并返回基于历史日线的、截至昨日的静态ATR。
    专门用于交易前的头寸规模计算，确保风险基石的稳定性。
    """
    total_bars_needed = atr_period + 20 
    df_day = get_klines_data(quote_ctx, symbol, total_bars_needed, Period.Day, AdjustType.NoAdjust)
    
    if df_day is None or len(df_day) < atr_period + 1:
        logger.warning(f"[{symbol}] 获取的日线数据不足，无法计算Historical ATR。")
        return 0.0
    
    try:
        atr_series = df_day.ta.atr(length=atr_period, append=False)
        if atr_series is None or atr_series.isna().all():
            return 0.0
        
        # 返回倒数第二个值，即“昨天”的ATR值，确保信号不漂移
        yesterday_atr = atr_series.iloc[-2]
        if pd.isna(yesterday_atr): return 0.0

        logger.info(f"[{symbol}] Historical ATR (昨日) 计算成功: {yesterday_atr:.4f}")
        return float(yesterday_atr)
    except Exception as e:
        logger.error(f"[{symbol}] 计算Historical ATR时发生错误: {e}", exc_info=True)
        return 0.0

@thread_safe_ttl_cache(maxsize=200, ttl=600) # 缓存10分钟，用于灵敏的盘中决策
def get_dynamic_atr(quote_ctx: QuoteContext,symbol: str,atr_period:int = 14) -> float:
    """
    [战术版] 计算并返回融合了盘中实时行情的动态ATR。
    专门用于盘中风险管理，如移动止损、动态利润目标等。
    """
    # --- 步骤 1: 获取历史基准 (调用上面的函数或复用其逻辑) ---
    total_bars_needed = atr_period + 50
    df_day = get_klines_data(quote_ctx, symbol, total_bars_needed, Period.Day, AdjustType.NoAdjust)
    
    if df_day is None or len(df_day) < atr_period + 1:
        return 0.0 # 无法计算历史基准

    try:
        atr_series = df_day.ta.atr(length=atr_period, append=False)
        if atr_series is None or atr_series.isna().all(): return 0.0
        # 获取昨日的ATR值和昨日的收盘价
        yesterday_atr = atr_series.iloc[-2]
        yesterday_close = df_day['close'].iloc[-2]
        
        if pd.isna(yesterday_atr) or pd.isna(yesterday_close): return 0.0

        # --- 步骤 2: 融合今日实时TR ---
        quote = get_realtime_quote(quote_ctx, symbol) # 假设这个工具函数存在
        if not quote or not quote.high or not quote.low or not quote.last_done:
            logger.info(f"[{symbol}] 无法获取今日实时行情，Dynamic ATR 将安全回退至昨日ATR: {yesterday_atr:.4f}")
            return float(yesterday_atr)
        # 计算今日的True Range (TR)
        high_today, low_today = float(quote.high), float(quote.low)
        current_tr = max(high_today - low_today, abs(high_today - yesterday_close), abs(low_today - yesterday_close))

        # --- 步骤 3: 使用标准平滑公式，动态更新ATR ---
        # ATR = ((前一日ATR * (N-1)) + 当日TR) / N
        dynamic_atr = ((yesterday_atr * (atr_period - 1)) + current_tr) / atr_period
        
        logger.info(f"[{symbol}] Dynamic ATR 计算成功: {dynamic_atr:.4f}")
        return float(dynamic_atr)
    except Exception as e:
        logger.error(f"[{symbol}] 计算Dynamic ATR时发生错误: {e}", exc_info=True)
        return 0.0 # 发生任何异常，都安全回退到0
        