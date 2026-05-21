import os, sys
project_path = os.path.dirname(os.path.abspath(__file__))
sys.path.append(project_path)
from datetime import date, timedelta
import pandas as pd
from longport.openapi import Config, QuoteContext, Period, AdjustType, TradeSessions
from utils.longport_api_utils import get_history_klines_data_by_range
from config.settings import LONGPORT_APP_KEY, LONGPORT_APP_SECRET, LONGPORT_ACCESS_TOKEN, DATA_PATH
from tianshu.periods import DAY_PERIOD_KEY, normalize_period_key

# --- 配置区 ---
# 定义你要回测的股票池
# SYMBOLS_TO_DOWNLOAD = ['00700.HK', '09988.HK', '01810.HK', '07200.HK', '07226.HK', '03750.HK', '01347.HK', '00981.HK', '02899.HK', '01024.HK', '00165.HK', '09698.HK', '09699.HK', '01357.HK', '09868.HK', '02800.HK', '02269.HK', '09688.HK', '01299.HK', '09626.HK', '00268.HK', '09992.HK', '02252.HK', '02359.HK', '06060.HK', '00005.HK', '00939.HK', '00388.HK', '01398.HK', '02318.HK', '03988.HK', '09999.HK', '00883.HK', '09618.HK', '03968.HK', '02015.HK', '00857.HK', '02628.HK', '02388.HK', '09961.HK', '00002.HK', '00016.HK', '02020.HK', '00941.HK','NVDA.US', 'NVDX.US', 'NVDL.US', 'AMD.US', 'AMDL.US', 'OKLO.US', 'TSLA.US', 'TSLL.US', 'TSLT.US', 'LLY.US', 'LLYX.US', 'PLTU.US', 'GGLL.US', 'SOXL.US', 'AAPL.US', 'AAPU.US', 'META.US', 'AMZN.US', 'GOOGL.US', 'TSM.US', 'MSFT.US', 'MSFU.US', 'METU.US', 'AMZU.US', 'ASML.US', 'ASMG.US', 'ROBN.US', 'AVGO.US', 'JNJ.US', 'JPM.US', 'WMT.US', 'COST.US', 'SMH.US', 'SMCI.US', 'SMCX.US', 'AVGX.US']
SYMBOLS_TO_DOWNLOAD_HK = ['03800.HK', '14651.HK', '17969.HK', '02809.HK', '06618.HK',
                          '02015.HK', '01398.HK', '03750.HK', '01276.HK', '00883.HK',
                          '06160.HK', '09880.HK', '01772.HK', '02228.HK', '09660.HK',
                          '01357.HK', '09618.HK', '81299.HK', '02845.HK', '02806.HK',
                          '02828.HK','00700.HK', '09988.HK', '01810.HK', '07200.HK',
                          '07226.HK', '02899.HK', '01024.HK', '01357.HK', '09626.HK',
                          '02800.HK', '02269.HK', '09688.HK', '01299.HK',
                          '09992.HK', '02252.HK', '02359.HK', '00939.HK',
                          '03988.HK', '09999.HK', '03968.HK',
                          '02628.HK', '02388.HK', '09961.HK', '00941.HK'
                          ]

SYMBOLS_TO_DOWNLOAD_US = ['SNPS.US', 'ORCX.US', 'MNDY.US', 'JPM.US', 'BBAI.US',
                          'FSLR.US', 'WMT.US', 'BABX.US', 'ICLR.US', 'FORD.US',
                          'JNJ.US', 'JPMO.US', 'AVGX.US', 'FIG.US', 'HMY.US',
                          'TSLT.US', 'SNOU.US', 'SNOW.US', 'AFRM.US', 'PLTU.US',
                          'UMAC.US', 'HIMS.US', 'NFLU.US', 'RXRX.US', 'NNE.US',
                          'CRMG.US', 'PDD.US', 'TEM.US', 'IWM.US', 'OPEN.US',
                          'PHM.US', 'LEN.US', 'DHI.US', 'APPX.US', 'FUTU.US',
                          'BULL.US', 'KO.US', 'TMDX.US', 'LFMD.US', 'DXYZ.US',
                          'LUNR.US', 'SOLZ.US', 'ETHA.US', 'OUST.US', 'AEVA.US',
                          'HSAI.US', 'PONY.US', 'RIVN.US', 'ACHR.US', 'SEZL.US',
                          'NXT.US', 'EVGO.US', 'EOSE.US', 'MVST.US', 'LEU.US',
                          'SMR.US', 'OKTA.US', 'CRWL.US', 'SOUN.US', 'NBIS.US',
                          'QMCO.US', 'IONQ.US', 'QBTS.US', 'QUBT.US', 'RGTI.US',
                          'UNHG.US', 'UNH.US', 'XOM.US', 'TLRY.US', 'VALN.US',
                          'SE.US', 'TEMT.US', 'TME.US', 'ASTS.US', 'MP.US',
                          'RKLB.US', 'CRCL.US', 'GE.US', 'RTX.US', 'MVLL.US',
                          'TQQQ.US', 'ASMG.US', 'ALAB.US', 'RDTL.US', 'SPXL.US',
                          'SMH.US', 'SPYU.US', 'UDOW.US', 'TNA.US', 'UPRO.US',
                          'NVDA.US', 'NVDX.US', 'NVDL.US', 'AMDL.US', 'OKLO.US',
                          'TSLA.US', 'TSLL.US', 'LLY.US', 'LLYX.US',
                          'GGLL.US', 'SOXL.US', 'AAPL.US', 'AAPU.US',
                          'META.US', 'AMZN.US', 'GOOGL.US', 'TSM.US', 'MSFT.US',
                          'MSFU.US', 'METU.US', 'AMZU.US', 'ASML.US',
                          'ROBN.US', 'AVGO.US', 'WMT.US', 'MU.US',
                          'COST.US', 'SMCI.US', 'SMCX.US']

SYMBOLS_TO_DOWNLOAD_CN = ['688775.SH', '688981.SH', '002384.SZ', '300308.SZ',
                          '688041.SH', '300476.SZ', '603019.SH', '601012.SH',
                          '600536.SH', '000975.SZ', '300750.SZ', '300347.SZ',
                          '600900.SH', '601939.SH', '300195.SZ', '603799.SH',
                          '601288.SH', '300748.SZ', '002475.SZ', '601138.SH',
                          '688668.SH', '600183.SH', '300548.SZ', '300570.SZ',
                          '300394.SZ', '002195.SZ', '002837.SZ', '002241.SZ',
                          '600549.SH', '516780.SH', '159770.SZ', '515070.SH',
                          '159202.SZ', '516100.SH', '159381.SZ', '515010.SH',
                          '515980.SH', '159869.SZ', '301076.SZ', '601677.SH',
                          '002714.SZ', '002078.SZ', '688561.SH', '000534.SZ',
                          '300735.SZ', '300475.SZ', '603496.SH', '301536.SZ',
                          '600989.SH', '002039.SZ', '002812.SZ', '600728.SH',
                          '300441.SZ', '002444.SZ', '002174.SZ', '600930.SH',
                          '601668.SH', '000959.SZ', '300408.SZ', '300014.SZ',
                          '600809.SH', '600585.SH', '300204.SZ', '603083.SH',
                          '603127.SH', '002747.SZ', '600021.SH', '603583.SH',
                          '601089.SH', '600309.SH', '688767.SH', '688689.SH',
                          '688677.SH', '688548.SH', '688391.SH', '688322.SH',
                          '688305.SH', '688279.SH', '688160.SH', '688123.SH',
                          '688115.SH', '605488.SH', '605389.SH', '605333.SH',
                          '605319.SH', '603662.SH', '603655.SH', '603535.SH',
                          '603456.SH', '603439.SH', '603380.SH', '603350.SH',
                          '603196.SH', '603122.SH', '603088.SH', '603038.SH',
                          '600960.SH', '600939.SH', '600933.SH', '600880.SH',
                          '600759.SH', '600758.SH', '600754.SH', '600629.SH',
                          '600617.SH', '600615.SH', '600575.SH', '600540.SH',
                          '600252.SH', '600248.SH', '600231.SH', '600221.SH',
                          '600191.SH', '600188.SH', '301603.SZ', '301520.SZ',
                          '301429.SZ', '301382.SZ', '301186.SZ', '301129.SZ',
                          '301125.SZ', '301110.SZ', '301059.SZ', '301012.SZ',
                          '300990.SZ', '300970.SZ', '300964.SZ', '300949.SZ',
                          '300935.SZ', '300923.SZ', '300814.SZ', '300813.SZ',
                          '300749.SZ', '300739.SZ', '300736.SZ', '300661.SZ',
                          '300660.SZ', '300635.SZ', '300551.SZ', '300507.SZ',
                          '300467.SZ', '300370.SZ', '300350.SZ', '300235.SZ',
                          '300155.SZ', '300121.SZ', '300110.SZ', '300086.SZ',
                          '300076.SZ', '300063.SZ', '300049.SZ', '300022.SZ',
                          '300021.SZ', '300013.SZ', '003041.SZ', '003033.SZ',
                          '003008.SZ', '002970.SZ', '002906.SZ', '002864.SZ',
                          '002628.SZ', '002590.SZ', '002542.SZ', '002535.SZ',
                          '002534.SZ', '002510.SZ', '002489.SZ', '002451.SZ',
                          '002377.SZ', '002154.SZ', '002137.SZ', '002086.SZ',
                          '002061.SZ', '002042.SZ', '002008.SZ', '002006.SZ',
                          '001333.SZ', '001306.SZ', '000922.SZ', '000782.SZ',
                          '000739.SZ', '000727.SZ', '000720.SZ', '000567.SZ',
                          '000559.SZ', '000401.SZ', '000008.SZ', '300442.SZ',
                          '601689.SH', '002472.SZ', '603728.SH', '603486.SH',
                          '002371.SZ', '600276.SH', '603259.SH', '003816.SZ',
                          '002555.SZ', '603893.SH', '000021.SZ', '002637.SZ',
                          '002164.SZ', '601918.SH', '601101.SH', '002131.SZ',
                          '603124.SH', '603119.SH', '600409.SH', '688258.SH']

# SYMBOLS_TO_DOWNLOAD = SYMBOLS_TO_DOWNLOAD_HK + SYMBOLS_TO_DOWNLOAD_US + SYMBOLS_TO_DOWNLOAD_CN
SYMBOLS_TO_DOWNLOAD = SYMBOLS_TO_DOWNLOAD_US

# 目标数据跨度。3 表示下载最近3年，即从 end_date 往前推3个自然年。
YEARS_TO_DOWNLOAD = 2

# 增量更新时，最近若干天会重刷一次，避免当天分钟线未收全、API修正历史K线等问题。
RECENT_REFRESH_DAYS = 7

# 是否忽略本地文件并全量重下。日常使用保持 False。
FORCE_FULL_REFRESH = False

# 保存前是否只保留目标日期范围内的数据。
PRUNE_TO_TARGET_RANGE = True

# 可用写法：Period.Day、Period.Min_60、"1d"、"60m"、60、180、"5m"。
DOWNLOAD_PERIODS = [
    Period.Day,
    Period.Min_60,
    Period.Min_180,
    Period.Min_240,
    Period.Min_5,
]

# 高级配置：如果某个周期要覆盖默认复权/交易时段/分片天数，可在这里写 dict。
DOWNLOAD_JOBS = [{"period": period} for period in DOWNLOAD_PERIODS]

PERIOD_KEY_TO_LONGPORT_PERIOD = {
    DAY_PERIOD_KEY: Period.Day,
    "1m": Period.Min_1,
    "3m": Period.Min_3,
    "5m": Period.Min_5,
    "10m": Period.Min_10,
    "15m": Period.Min_15,
    "30m": Period.Min_30,
    "60m": Period.Min_60,
    "120m": Period.Min_120,
    "180m": Period.Min_180,
    "240m": Period.Min_240,
}


def _output_file_path(symbol, period):
    period_key = normalize_period_key(period)
    if period_key == DAY_PERIOD_KEY:
        return os.path.join(DATA_PATH, f"{symbol}.parquet")
    return os.path.join(DATA_PATH, period_key, f"{symbol}.parquet")


def _subtract_years(day: date, years: int) -> date:
    if years <= 0:
        raise ValueError("YEARS_TO_DOWNLOAD 必须是正整数")
    try:
        return day.replace(year=day.year - years)
    except ValueError:
        return day.replace(year=day.year - years, day=28)


def _resolve_period(period):
    if isinstance(period, Period):
        return period
    period_key = normalize_period_key(period)
    try:
        return PERIOD_KEY_TO_LONGPORT_PERIOD[period_key]
    except KeyError as exc:
        raise ValueError(f"不支持的下载周期: {period}") from exc


def _normalize_download_job(job):
    if isinstance(job, dict):
        period = _resolve_period(job["period"])
        period_key = normalize_period_key(period)
        return {
            "period": period,
            "period_key": period_key,
            "adjust_type": job.get(
                "adjust_type",
                AdjustType.ForwardAdjust if period_key == DAY_PERIOD_KEY else AdjustType.NoAdjust,
            ),
            "trade_sessions": job.get("trade_sessions", TradeSessions.Intraday),
            "chunk_days": job.get("chunk_days"),
        }

    period = _resolve_period(job)
    period_key = normalize_period_key(period)
    return {
        "period": period,
        "period_key": period_key,
        "adjust_type": AdjustType.ForwardAdjust if period_key == DAY_PERIOD_KEY else AdjustType.NoAdjust,
        "trade_sessions": TradeSessions.Intraday,
        "chunk_days": None,
    }


def _unique_symbols(symbols):
    return list(dict.fromkeys(symbols))


def _load_existing_data(file_path):
    df = pd.read_parquet(file_path)
    if not isinstance(df.index, pd.DatetimeIndex):
        if "timestamp" not in df.columns:
            raise ValueError("本地数据既没有 DatetimeIndex，也没有 timestamp 列")
        df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
        df.set_index("timestamp", inplace=True)
    elif df.index.tz is None:
        df.index = df.index.tz_localize("UTC")
    else:
        df.index = df.index.tz_convert("UTC")

    df = df[~df.index.duplicated(keep="last")]
    df.sort_index(inplace=True)
    return df


def _filter_to_date_range(df, start_date: date, end_date: date):
    if df is None or df.empty:
        return pd.DataFrame()
    date_mask = (df.index.date >= start_date) & (df.index.date <= end_date)
    return df.loc[date_mask].copy()


def _date_bounds(df):
    if df is None or df.empty:
        return None, None
    return df.index.min().date(), df.index.max().date()


def _merge_date_ranges(ranges):
    valid_ranges = sorted((start, end) for start, end in ranges if start <= end)
    if not valid_ranges:
        return []

    merged = [valid_ranges[0]]
    for start, end in valid_ranges[1:]:
        last_start, last_end = merged[-1]
        if start <= last_end + timedelta(days=1):
            merged[-1] = (last_start, max(last_end, end))
        else:
            merged.append((start, end))
    return merged


def _build_fetch_ranges(existing_df, start_date: date, end_date: date):
    if FORCE_FULL_REFRESH or existing_df is None or existing_df.empty:
        return [(start_date, end_date)]

    first_date, last_date = _date_bounds(existing_df)
    ranges = []
    if first_date and first_date > start_date:
        ranges.append((start_date, min(first_date - timedelta(days=1), end_date)))

    if last_date:
        refresh_days = max(RECENT_REFRESH_DAYS, 1)
        recent_start = max(start_date, min(last_date, end_date) - timedelta(days=refresh_days - 1))
        ranges.append((recent_start, end_date))

    return _merge_date_ranges(ranges)


def _combine_klines(frames):
    usable_frames = [df for df in frames if df is not None and not df.empty]
    if not usable_frames:
        return pd.DataFrame()

    combined = pd.concat(usable_frames, axis=0)
    combined = combined[~combined.index.duplicated(keep="last")]
    combined.sort_index(inplace=True)
    return combined


def _validate_klines(df, symbol, period_key, start_date: date, end_date: date):
    errors = []
    warnings = []
    required_columns = {"open", "high", "low", "close", "volume"}

    if df is None or df.empty:
        errors.append("数据为空")
        return False, errors, warnings

    if not isinstance(df.index, pd.DatetimeIndex):
        errors.append("索引不是 DatetimeIndex")
    if df.index.has_duplicates:
        errors.append("时间索引存在重复")
    if not df.index.is_monotonic_increasing:
        errors.append("时间索引不是升序")

    missing_columns = required_columns - set(df.columns)
    if missing_columns:
        errors.append(f"缺少字段: {sorted(missing_columns)}")

    first_date, last_date = _date_bounds(df)
    if first_date and first_date > start_date:
        warnings.append(f"最早数据为 {first_date}，晚于目标开始日期 {start_date}，可能是上市时间不足或数据源无更早数据")
    if last_date and (end_date - last_date).days > 7:
        warnings.append(f"最新数据为 {last_date}，距离目标结束日期 {end_date} 超过7天")

    if errors:
        return False, errors, warnings
    return True, errors, warnings


def _atomic_save_parquet(df, file_path):
    output_dir = os.path.dirname(file_path)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)

    temp_path = f"{file_path}.tmp"
    df.to_parquet(temp_path)
    os.replace(temp_path, file_path)

# --- 主逻辑 ---
def download_data():
    """
    连接长桥API，按目标日期范围下载指定股票池的历史K线数据，并保存到本地data文件夹。
    """
    print("--- 天枢Quant数据下载器 ---")
    end_date = date.today()
    start_date = _subtract_years(end_date, YEARS_TO_DOWNLOAD)
    jobs = [_normalize_download_job(job) for job in DOWNLOAD_JOBS]
    symbols = _unique_symbols(SYMBOLS_TO_DOWNLOAD)

    print(f"目标时间范围: {start_date} -> {end_date} ({YEARS_TO_DOWNLOAD}年)")
    print(f"下载周期: {', '.join(job['period_key'] for job in jobs)}")
    print(f"股票数量: {len(symbols)}")

    # 确保数据目录存在
    if not os.path.exists(DATA_PATH):
        os.makedirs(DATA_PATH)
        print(f"已创建数据目录: {DATA_PATH}")

    # 初始化长桥API上下文
    try:
        config = Config(app_key=LONGPORT_APP_KEY, app_secret=LONGPORT_APP_SECRET, access_token=LONGPORT_ACCESS_TOKEN)
        quote_ctx = QuoteContext(config)
        print("长桥API连接成功。")
    except Exception as e:
        print(f"错误：无法初始化长桥API上下文，请检查配置。 {e}")
        return

    success_count = 0
    failure_count = 0

    # 循环下载每只股票的数据
    for symbol in symbols:
        print(f"\n正在处理: {symbol}...")

        for job in jobs:
            period = job["period"]
            period_key = job["period_key"]
            adjust_type = job["adjust_type"]
            trade_sessions = job["trade_sessions"]
            chunk_days = job["chunk_days"]
            file_path = _output_file_path(symbol, period)

            try:
                existing_df = None
                existing_load_failed = False
                if os.path.exists(file_path) and not FORCE_FULL_REFRESH:
                    try:
                        existing_df = _load_existing_data(file_path)
                        if PRUNE_TO_TARGET_RANGE:
                            existing_df = _filter_to_date_range(existing_df, start_date, end_date)
                    except Exception as e:
                        existing_load_failed = True
                        print(f"⚠️ 读取本地 {period_key} 数据失败，将全量重下: {file_path}，原因: {e}")

                fetch_ranges = [(start_date, end_date)] if existing_load_failed else _build_fetch_ranges(existing_df, start_date, end_date)
                fetched_frames = []
                range_failed = False

                for range_start, range_end in fetch_ranges:
                    print(f"  -> 下载 {period_key}: {range_start} -> {range_end}")
                    range_df = get_history_klines_data_by_range(
                        quote_ctx=quote_ctx,
                        symbol=symbol,
                        period=period,
                        adjust_type=adjust_type,
                        start=range_start,
                        end=range_end,
                        trade_sessions=trade_sessions,
                        chunk_days=chunk_days,
                    )
                    if range_df is None:
                        range_failed = True
                        break
                    if not range_df.empty:
                        fetched_frames.append(range_df)

                if range_failed:
                    failure_count += 1
                    print(f"❌ {symbol} 的 {period_key} 数据下载失败，未覆盖本地文件。")
                    continue

                frames_to_merge = []
                if existing_df is not None and not FORCE_FULL_REFRESH:
                    frames_to_merge.append(existing_df)
                frames_to_merge.extend(fetched_frames)

                final_df = _combine_klines(frames_to_merge)
                if PRUNE_TO_TARGET_RANGE:
                    final_df = _filter_to_date_range(final_df, start_date, end_date)

                is_valid, errors, warnings = _validate_klines(final_df, symbol, period_key, start_date, end_date)
                if not is_valid:
                    failure_count += 1
                    print(f"❌ {symbol} 的 {period_key} 数据校验失败: {'; '.join(errors)}")
                    continue

                _atomic_save_parquet(final_df, file_path)
                first_date, last_date = _date_bounds(final_df)
                success_count += 1
                print(f"✅ 保存 {len(final_df)} 条 {period_key} 数据 ({first_date} -> {last_date}) 到: {file_path}")
                for warning in warnings:
                    print(f"⚠️ {symbol} {period_key}: {warning}")

            except Exception as e:
                failure_count += 1
                print(f"❌ 下载 {symbol} 的 {period_key} 数据时发生严重错误: {e}")

    print(f"\n--- 所有任务完成：成功 {success_count} 个 symbol/period，失败 {failure_count} 个 ---")

if __name__ == "__main__":
    # 在运行此脚本前，你需要准备好你的配置文件
    # 比如在 config/settings.py 中定义好你的API Key
    # 并且把你的 longport_api_utils.py 放到 utils 目录中
    download_data()
