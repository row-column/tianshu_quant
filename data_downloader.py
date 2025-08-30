import os
import time
from datetime import datetime, timedelta

from longport.openapi import Config, QuoteContext, Period, AdjustType
from utils.longport_api_utils import get_history_klines_data
from config.settings import LONGPORT_APP_KEY, LONGPORT_APP_SECRET, LONGPORT_ACCESS_TOKEN, DATA_PATH

# --- 配置区 ---
# 定义你要回测的股票池
SYMBOLS_TO_DOWNLOAD = [
    "AAPL.US", "TSLA.US", "NVDA.US", # 美股
    "0700.HK", "9988.HK", "0005.HK"  # 港股
]

# 定义要下载的数据周期和时间范围
KLINE_PERIOD = Period.Day  # 我们以日线为例
DAYS_TO_DOWNLOAD = 365 * 1 # 下载过去5年的数据

# --- 主逻辑 ---
def download_data():
    """
    连接长桥API，下载指定股票池的历史K线数据，并保存到本地data文件夹。
    """
    print("--- 天枢Quant数据下载器 ---")
    
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

    # 循环下载每只股票的数据
    for symbol in SYMBOLS_TO_DOWNLOAD:
        print(f"\n正在处理: {symbol}...")
        
        file_path = os.path.join(DATA_PATH, f"{symbol}.parquet")
        
        # 简单检查，如果文件已存在，可以跳过（可根据需求修改为强制更新）
        if os.path.exists(file_path):
            print(f"数据文件已存在，跳过: {file_path}")
            continue

        try:
            # 调用API获取历史数据
            # 注意：get_history_klines_data 需要返回一个按时间升序的DataFrame
            df = get_history_klines_data(
                quote_ctx=quote_ctx,
                symbol=symbol,
                count=DAYS_TO_DOWNLOAD, # get_history_klines_data 按天数获取
                period=KLINE_PERIOD,
                adjust_type=AdjustType.ForwardAdjust, # 通常使用前复权
            )

            if df is not None and not df.empty:
                # 将DataFrame保存为Parquet格式，高效且带类型
                # DataFrame的索引（时间戳）也会被保存
                df.to_parquet(file_path)
                print(f"✅ 成功下载并保存 {len(df)} 条数据到: {file_path}")
            else:
                print(f"❌ 未能获取到 {symbol} 的数据。")

            # API调用之间最好有短暂延时，避免过于频繁触达限制
            time.sleep(0.5)

        except Exception as e:
            print(f"❌ 下载 {symbol} 时发生严重错误: {e}")

    print("\n--- 所有任务完成 ---")

if __name__ == "__main__":
    # 在运行此脚本前，你需要准备好你的配置文件
    # 比如在 config/settings.py 中定义好你的API Key
    # 并且把你的 longport_api_utils.py 放到 utils 目录中
    download_data()