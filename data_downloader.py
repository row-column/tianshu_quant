import os, sys
project_path = os.path.dirname(os.path.abspath(__file__))
sys.path.append(project_path)
import time
from datetime import datetime, timedelta
from longport.openapi import Config, QuoteContext, Period, AdjustType
from utils.longport_api_utils import get_history_klines_data
from config.settings import LONGPORT_APP_KEY, LONGPORT_APP_SECRET, LONGPORT_ACCESS_TOKEN, DATA_PATH

# --- 配置区 ---
# 定义你要回测的股票池
# SYMBOLS_TO_DOWNLOAD = ['00700.HK', '09988.HK', '01810.HK', '07200.HK', '07226.HK', '03750.HK', '01347.HK', '00981.HK', '02899.HK', '01024.HK', '00165.HK', '09698.HK', '09699.HK', '01357.HK', '09868.HK', '02800.HK', '02269.HK', '09688.HK', '01299.HK', '09626.HK', '00268.HK', '09992.HK', '02252.HK', '02359.HK', '06060.HK', '00005.HK', '00939.HK', '00388.HK', '01398.HK', '02318.HK', '03988.HK', '09999.HK', '00883.HK', '09618.HK', '03968.HK', '02015.HK', '00857.HK', '02628.HK', '02388.HK', '09961.HK', '00002.HK', '00016.HK', '02020.HK', '00941.HK','NVDA.US', 'NVDX.US', 'NVDL.US', 'AMD.US', 'AMDL.US', 'OKLO.US', 'TSLA.US', 'TSLL.US', 'TSLT.US', 'LLY.US', 'LLYX.US', 'PLTU.US', 'GGLL.US', 'SOXL.US', 'AAPL.US', 'AAPU.US', 'META.US', 'AMZN.US', 'GOOGL.US', 'TSM.US', 'MSFT.US', 'MSFU.US', 'METU.US', 'AMZU.US', 'ASML.US', 'ASMG.US', 'ROBN.US', 'AVGO.US', 'JNJ.US', 'JPM.US', 'WMT.US', 'COST.US', 'SMH.US', 'SMCI.US', 'SMCX.US', 'AVGX.US']
SYMBOLS_TO_DOWNLOAD_HK = [
    '00002.HK', '01347.HK', '07200.HK', '00005.HK', '01357.HK', '07226.HK',
    '00016.HK', '01398.HK', '09868.HK', '0005.HK', '01810.HK', '09988.HK',
    '00165.HK', '02359.HK', '09992.HK', '00268.HK', '02800.HK', '9988.HK',
    '00388.HK', '02899.HK', '00981.HK', '03750.HK', '01024.HK', '06060.HK',
    '6690.HK', '2809.HK', '6618.HK', '2357.HK', '2382.HK', '968.HK',
    '1816.HK', '2015.HK', '1211.HK', '1276.HK', '883.HK', '6160.HK',
    '9880.HK', '1772.HK', '1797.HK', '2228.HK', '3690.HK', '9660.HK',
    '9618.HK', '81299.HK', '2845.HK', '2806.HK', '2828.HK'
]

SYMBOLS_TO_DOWNLOAD_US = [
    'JNJ.US', 'JPM.US', 'LLYX.US', 'MSFU.US', 'PLTU.US', 'SOXL.US', 'AAPU.US',
    'TSLL.US', 'AMDL.US', 'WMT.US', 'AMZU.US', 'SNOU.US', 'SNOW.US',
    'RERE.US', 'AFRM.US', 'UMAC.US', 'HIMS.US', 'NFLU.US', 'RXRX.US',
    'NNE.US', 'CRMG.US', 'KPDD.US', 'PDD.US', 'TEM.US', 'LULU.US', 'IWM.US',
    'OPEN.US', 'PHM.US', 'LEN.US', 'DHI.US', 'APPX.US', 'FUTU.US', 'BULL.US',
    'KO.US', 'TMDX.US', 'LFMD.US', 'DXYZ.US', 'SPCE.US', 'LUNR.US',
    'SOLZ.US', 'ETHA.US', 'OUST.US', 'AEVA.US', 'HSAI.US', 'PONY.US',
    'RIVN.US', 'ACHR.US', 'SEZL.US', 'SHLS.US', 'NXT.US', 'ARRY.US', 'RUN.US',
    'FSLR.US', 'EVGO.US', 'EOSE.US', 'MVST.US', 'LEU.US', 'SMR.US',
    'OKTA.US', 'CRWL.US', 'CRWD.US', 'RBRK.US', 'SOUN.US', 'NBIS.US',
    'QMCO.US', 'IONQ.US', 'QBTS.US', 'QUBT.US', 'RGTI.US', 'UNHG.US',
    'UNH.US', 'BLSH.US', 'CVX.US', 'XOM.US', 'BMNR.US', 'TLRY.US', 'VALN.US',
    'SE.US', 'TEMT.US', 'TME.US', 'ASTS.US', 'MP.US', 'RKLB.US', 'CRCL.US',
    'GE.US', 'RTX.US', 'CRWV.US', 'MVLL.US', 'CONL.US', 'TQQQ.US', 'ASMG.US',
    'ALAB.US', 'RDTL.US', 'SPXL.US', 'DIG.US', 'ERX.US', 'XLV.US', 'SMH.US',
    'SPYU.US', 'UDOW.US', 'TNA.US', 'UPRO.US', 'AVGX.US'
]

SYMBOLS_TO_DOWNLOAD_CN = [
    "688775.SH","688981.SH","002384.SZ","300308.SZ","688041.SH",
    "300476.SZ","603019.SH","601012.SH","600536.SH","000975.SZ",
    "300750.SZ","300347.SZ","600900.SH","601939.SH","300195.SZ",
    "603799.SH","601288.SH","300748.SZ","002475.SZ","601138.SH",
    "688668.SH","600183.SH","300548.SZ","300570.SZ","300394.SZ",
    "002195.SZ","002837.SZ","002241.SZ","600549.SH","516780.SH",
    "159770.SZ","515070.SH","159202.SZ","516100.SH","159381.SZ",
    "159869.SZ","301076.SZ","601677.SH","002714.SZ","002078.SZ","688561.SH",
    "515010.SH","515980.SH","600410.SH","159869.SZ","600930.SH"
]
# SYMBOLS_TO_DOWNLOAD = SYMBOLS_TO_DOWNLOAD_HK + SYMBOLS_TO_DOWNLOAD_US
SYMBOLS_TO_DOWNLOAD = SYMBOLS_TO_DOWNLOAD_CN
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