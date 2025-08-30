#!/usr/bin/env python
# -*- coding=utf-8 -*-
import os, sys
project_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_path)
import json
from typing import List, Dict, Any,Optional
from positions import MarketType
import logging
from utils.market_time_utils import is_hk_market_open, is_us_market_open
from playwright.sync_api import sync_playwright, TimeoutError as PlaywrightTimeoutError
import time
import random
from enum import Enum
from datetime import datetime, date
from utils.cfg_utils import load_yaml2cfg
cfg_file        =   os.path.join(project_path, 'data/server_config.yaml')
cfg             =   load_yaml2cfg(cfg_file)
logger = logging.getLogger(__name__) 

def normalize_symbol_code(code: any) -> str or None:
    """
    将各种格式的股票代码标准化为 '代码.市场' 的统一格式。

    该函数具有强大的兼容性，可以处理以下输入格式：
    1.  无市场后缀的纯代码 (例如 'TSLL', '01024')
    2.  已标准化的代码 (例如 'BABA.US', '00700.HK')
    3.  CaiQian/旧格式前缀代码 (例如 'US.AAPL', 'HK.9988')
    4.  包含多余空格或大小写不一致的代码 (例如 '  tsll  ', 'hk.700')

    处理逻辑:
    - 如果代码仅由英文字母组成，则假定为美股，添加 '.US'。
    - 如果代码仅由数字组成，则假定为港股，添加 '.HK'。
    - 其他格式将被智能转换或在无法识别时返回其原始清理后的形态。

    Args:
        code (any): 各种格式的股票代码字符串。

    Returns:
        str or None: 标准化后的股票代码 (例如 'TSLL.US')。
                     如果输入为空或无效，则返回 None。
    
    Examples:
        >>> normalize_symbol_code('TSLL')
        'TSLL.US'
        >>> normalize_symbol_code('01024')
        '01024.HK'
        >>> normalize_symbol_code('US.AAPL')
        'AAPL.US'
        >>> normalize_symbol_code('hk.9988')
        '9988.HK'
        >>> normalize_symbol_code('BABA.US')
        'BABA.US'
        >>> normalize_symbol_code('   700.hk   ')
        '700.HK'
        >>> normalize_symbol_code(None)
        None
    """
    if not isinstance(code, str) or not code.strip():
        return None

    # 步骤 1: 清理输入，统一转换为大写并去除首尾空格
    code = code.strip().upper()

    # 步骤 2: 检查是否为已包含市场信息的格式
    if '.' in code:
        # 如果是 '代码.市场' 后缀格式，直接返回
        if code.endswith('.HK') or code.endswith('.US'):
            return code
        # 如果是 '市场.代码' 前缀格式，进行转换
        elif code.startswith('HK.'):
            return f"{code[3:]}.HK"
        elif code.startswith('US.'):
            return f"{code[3:]}.US"
        else:
            # 对于不识别的格式 (如 'BRK.A')，保持原样并记录
            logging.info(f"代码 '{code}' 包含'.'但格式未知，将按原样返回。")
            return code

    # 步骤 3: 处理无市场信息的纯代码
    # 仅由英文字母构成 -> 美股
    if code.isalpha():
        return f"{code}.US"
    # 仅由数字构成 -> 港股
    elif code.isdigit():
        return f"{code}.HK"
    
    # 步骤 4: 对于无法识别的混合代码，保持原样并发出警告
    logging.warning(f"无法自动识别代码 '{code}' 的市场，请手动添加.HK或.US后缀。")
    return code

def to_futu_symbol(symbol: str) -> str:
        """将系统标准代码 (e.g., '700.HK') 转换为富途格式 (e.g., 'HK.00700')。"""
        if '.' not in symbol: return symbol
        parts = symbol.split('.')
        if len(parts) != 2:
            return symbol.upper()
        market, code = parts[0], parts[1]
        market = market.upper()
        if market == 'HK':
            return f"HK.{code.zfill(5)}"
        elif market == 'US':
            return f"US.{code.upper()}"
        return symbol

def from_futu_symbol(futu_symbol: str) -> str:
    """将富途格式 (e.g., 'HK.00700') 转换为系统标准代码 (e.g., '00700.HK')。"""
    if '.' not in futu_symbol: return futu_symbol
    parts = futu_symbol.split('.')
    if len(parts) != 2:
        return futu_symbol.upper()
    market, code = parts[0], parts[1]
    return f"{code}.{market}"


def get_symbol_codes_from_json(file_path: str) -> List[str]:
    """
    从指定的JSON文件中读取数据并返回所有的键（股票代码）。

    这个函数会处理文件不存在或JSON解析错误等常见问题。

    Args:
        file_path: JSON文件的路径。

    Returns:
        一个包含所有股票代码的字符串列表。

    Raises:
        FileNotFoundError: 如果指定的文件路径不存在。
        ValueError: 如果文件内容不是有效的JSON格式。
    """
    if not os.path.exists(file_path):
        print(f"错误：找不到文件 '{file_path}'")
        return []

    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data: Dict[str, Any] = json.load(f)
        # 使用列表推导式高效地获取所有key
        return list(data.keys())
    except json.JSONDecodeError:
        raise ValueError(f"错误：文件 '{file_path}' 的内容不是有效的JSON格式。")
    except Exception as e:
        # 捕获其他可能的异常
        print(f"读取文件 '{file_path}' 时发生未知错误: {e}")
        return []

def get_stock_list(is_all:bool=False):
    hk_file_path = os.path.join('data', 'hk_stock.json')
    us_file_path = os.path.join('data', 'us_stock.json')
    ext_hk_file_path = os.path.join('data', 'hk_alpha_stock.json')
    ext_us_file_path = os.path.join('data', 'us_alpha_stock.json')
    hk_stock_list = []
    us_stock_list = []
    try:
        if is_all:
            # 调用函数分别读取港股和美股的代码
            hk_stock_list = get_symbol_codes_from_json(hk_file_path)
            us_stock_list = get_symbol_codes_from_json(us_file_path)
            ext_hk_stock_list = get_symbol_codes_from_json(ext_hk_file_path)
            ext_us_stock_list = get_symbol_codes_from_json(ext_us_file_path)
            us_stock_list.extend(ext_us_stock_list)
            hk_stock_list.extend(ext_hk_stock_list)
        else:
            if is_hk_market_open():
                hk_stock_list = get_symbol_codes_from_json(hk_file_path)
                ext_hk_stock_list = get_symbol_codes_from_json(ext_hk_file_path)
                hk_stock_list.extend(ext_hk_stock_list)
            if is_us_market_open():
                us_stock_list = get_symbol_codes_from_json(us_file_path)
                ext_us_stock_list = get_symbol_codes_from_json(ext_us_file_path)
                us_stock_list.extend(ext_us_stock_list)
            
        print("--- 港股代码 (hk_list) ---")
        print(hk_stock_list)
        print(f"\n共计 {len(hk_stock_list)} 个港股代码。\n")


        print("--- 美股代码 (us_list) ---")
        print(us_stock_list)
        print(f"\n共计 {len(us_stock_list)} 个美股代码。")

        return hk_stock_list,us_stock_list

    except (FileNotFoundError, ValueError) as e:
        print(e)
        return [],[]

def get_market_type(symbol: str) -> MarketType:
    symbol_upper = symbol.upper()
    if symbol_upper.endswith('.HK'):
        return MarketType.HK
    elif symbol_upper.endswith('.US'): 
        return MarketType.US
    raise ValueError(f"无效的股票代码格式: '{symbol}'. 请使用 '代码.市场' 格式。")


# ==============================================================================
#  【新增】股票代码映射工具 (轻量级函数式实现)
# ==============================================================================

# 步骤 1: 定义模块级的全局变量，作为映射数据的缓存
_ETF_MAP: Dict[str, str] = {}
_MAP_FILE_LOADED = False # [新增] 加载标记，防止不必要的重复加载

def _load_etf_map():
    """
    【私有函数】从YAML文件加载ETF映射数据到全局字典 `_ETF_MAP`。
    """
    global _ETF_MAP, _MAP_FILE_LOADED # 声明要修改全局变量
    map_file_path = os.path.join(project_path, 'data', 'etf_map.yaml')
    
    try:
        _ETF_MAP = load_yaml2cfg(map_file_path)
        logger.info(f"成功从 {map_file_path} 加载了 {len(_ETF_MAP)} 条ETF映射规则。")
    except FileNotFoundError:
        logger.warning(f"ETF映射文件 {map_file_path} 未找到，ETF映射功能将不可用。")
        _ETF_MAP = {}
    except Exception as e:
        logger.error(f"加载ETF映射文件时出错: {e}", exc_info=True)
        _ETF_MAP = {}
    
    _MAP_FILE_LOADED = True

def get_analysis_target(symbol: str) -> str:
    """
    【公开函数】获取用于分析的目标代码。
    如果需要，会首次加载映射文件。
    """
    # 延迟加载（Lazy Loading）：只在第一次调用时才加载文件
    if not _MAP_FILE_LOADED:
        _load_etf_map()
        
    return _ETF_MAP.get(symbol, symbol)

def reload_etf_map():
    """
    【公开函数】强制重新加载ETF映射文件，用于热更新。
    """
    logger.info("正在强制重新加载ETF映射规则...")
    _load_etf_map()


def normalize_symbol(symbol: str) -> str:
    """
    将股票代码规范化为统一格式。

    此工具函数是系统稳定运行的关键，它确保来自任何来源（API、文件、手动输入）
    的股票代码在进入核心逻辑前都被处理成唯一的、可比较的标准格式。
    
    核心功能:
    1.  对港股代码（.HK），将其数字部分用前导零补足至5位。
    2.  对所有其他市场的代码，统一转换为大写。
    3.  能优雅地处理格式不规范或无效的输入。

    Args:
        symbol (str): 原始的股票代码字符串。

    Returns:
        str: 规范化后的股票代码。

    Examples:
        >>> normalize_symbol("9868.HK")
        '09868.HK'
        >>> normalize_symbol("700.hk")
        '00700.HK'
        >>> normalize_symbol("BABA.US")
        'BABA.US'
        >>> normalize_symbol("aapl.us")
        'AAPL.US'
        >>> normalize_symbol("INVALID") # 无效格式
        'INVALID'
        >>> normalize_symbol(None) # None 输入
        ''
    """
    # 1. [健壮性] 处理无效输入，防止程序崩溃
    if not isinstance(symbol, str):
        return "" # 对于None或其他非字符串类型，返回空字符串

    # 2. [核心逻辑] 分割代码与市场后缀
    parts = symbol.split('.')
    
    # 3. [健壮性] 如果格式不正确（例如没有'.'），则直接返回大写原值
    if len(parts) != 2:
        return symbol.upper()
        
    code, market = parts
    market = market.upper()

    # 4. [核心逻辑] 对港股进行特殊处理
    if market == 'HK':
        return f"{code.zfill(5)}.{market}"
    
    # 5. [修复Bug & 统一处理] 对所有其他市场，返回 "代码.市场" 的大写形式
    return f"{code.upper()}.{market}"

class EnhancedJSONEncoder(json.JSONEncoder):
    """
    一个增强的、生产级别的JSON编码器。
    它可以优雅地处理标准库json无法序列化的特殊对象，
    例如 Enum (枚举) 和 datetime/date (日期时间)。
    这是保证数据持久化健壮性的核心工具。
    """
    def default(self, obj):
        # 如果对象是Enum的实例，返回它的 .value 属性 (例如 "HK", "building")
        if isinstance(obj, Enum):
            return obj.value
        
        # 如果对象是datetime或date的实例，将其转换为ISO 8601标准格式的字符串
        if isinstance(obj, (datetime, date)):
            return obj.isoformat()
            
        # 对于所有其他类型，调用父类的默认方法，这是标准的最佳实践
        return super().default(obj)


def get_negative_news(stock_code: str):
    """
    通过Playwright驱动真实浏览器，在Google上搜索相关的利空新闻。
    这个方法远比纯HTTP请求强大，能够应对JS动态加载和基础的反爬虫机制。

    Args:
        stock_code (str): 美股或港股的股票代码 (例如: 'AAPL', '0700.HK').

    Returns:
        None: 直接打印分析结果。
    """
    if not stock_code:
        print("杠一下：你是不是觉得我的时间不值钱？连个股票代码都不给，让我怎么分析？")
        return

    print(f"请求已受理。正在启动工业级浏览器引擎，为 '{stock_code}' 进行深度舆情扫描...")
    print("-" * 80)

    # 核心利空关键词库 - 这套词库本身就价值百万
    negative_keywords = [
        "利空", "暴跌", "蒸发", "抛售", "减持", "调查", "诉讼",
        "罚款", "违规", "亏损", "裁员", "倒闭", "风险", "高管离职",
        "内部问题", "供应链中断", "评级下调", "前景黯淡", "泡沫"
    ]

    found_negative_news = []

    with sync_playwright() as p:
        # 启动Chromium浏览器，headless=True表示无头模式（不显示UI界面），在服务器上运行的标配
        # 你可以改成 headless=False 来亲眼看看自动化浏览器是如何工作的
        try:
            browser = p.chromium.launch(headless=True)
        except Exception as e:
            print(f"致命错误：浏览器启动失败。是不是没按我的要求运行 'playwright install'？ 错误信息: {e}")
            return
            
        page = browser.new_page()

        for keyword in negative_keywords:
            query = f"{stock_code} {keyword}"
            search_url = f"https://www.google.com/search?q={query}&tbm=nws" # tbm=nws 表示搜索新闻

            print(f"正在执行高级搜索: '{query}'")

            try:
                # 导航到目标URL，设置超时时间，并等待DOM加载完成
                page.goto(search_url, timeout=30000, wait_until='domcontentloaded')

                # --------------------------------------------------------------
                # Playwright的精髓：智能等待。等待搜索结果的容器出现。
                # 这个选择器需要根据实际情况调整，但比requests稳定得多。
                # 我们给它5秒钟的时间加载，加载不出来就当没有。
                # --------------------------------------------------------------
                results_container_selector = '#main' 
                page.wait_for_selector(results_container_selector, timeout=5000)
                
                # Google新闻搜索结果的CSS选择器，和之前一样，但现在是动态加载也不怕了
                search_results_selector = 'a.WlydOe' 
                
                results = page.locator(search_results_selector).all()

                if not results:
                    # 有时候Google会用不同的布局，我们尝试备用选择器
                    search_results_selector_alt = 'a[data-ved]' # 一个更通用的链接选择器
                    results = page.locator(search_results_selector_alt).filter(has=page.locator('div[role="heading"]')).all()


                for result in results:
                    try:
                        title_element = result.locator('div[role="heading"]')
                        title = title_element.inner_text() if title_element.count() > 0 else "标题不可用"
                        
                        link = result.get_attribute('href')

                        if keyword in title:
                            news_item = {
                                "title": title,
                                "link": link,
                                "keyword": keyword
                            }
                            if not any(item['title'] == title for item in found_negative_news):
                                found_negative_news.append(news_item)
                    except Exception:
                        # 即使单个结果解析失败，也不影响大局，继续下一个
                        continue
                
                # 作为一个更高级的“人”，我们的暂停时间也更智能
                time.sleep(random.uniform(2, 5))

            except PlaywrightTimeoutError:
                print(f"提示：查询 '{query}' 页面加载超时或未找到结果。正常现象，可能该关键词下无新闻。")
                continue
            except Exception as e:
                print(f"警告：执行查询 '{query}' 时发生未知错误: {e}")
                continue
        
        # 优雅地关闭浏览器
        browser.close()

    # --------------------------------------------------------------------------
    # 最终分析与报告输出 - 现在的结果，含金量更高
    # --------------------------------------------------------------------------
    print("-" * 80)
    if found_negative_news:
        print(f"深度扫描完成！对于 '{stock_code}', 发现以下高确定性潜在利空新闻：")
        print("\n=== 工业级扫描结果汇总 ===\n")
        for i, news in enumerate(found_negative_news, 1):
            print(f"{i}. 标题: {news['title']}")
            print(f"   触发关键词: {news['keyword']}")
            print(f"   来源链接: {news['link']}\n")
        print("\n=== 杠精最终裁决 ===\n")
        print(f"结论：舆情存在显著负面信号。我通过真实浏览器内核抓取到了 {len(found_negative_news)} 条高度相关的利空报道。")
        print("操作建议：这才是专业的手法。之前的`requests`？那是小孩子过家家的玩具。现在你拿到的信息，是排除了大部分干扰后的高价值情报。下一步，别偷懒，立刻对这些信息进行深度分析和交叉验证。如果你连这点主动性都没有，华尔街不欢迎你。")

    else:
        print(f"深度扫描完成！对于 '{stock_code}', 未在近期新闻中发现明显的、直接的利空关键词。")
        print("\n=== 杠精最终裁决 ===\n")
        print("结论：表层舆情干净。但这恰恰可能是最危险的信号。")
        print("操作建议：利空消息可能隐藏在财报的附注里，或者在某个需要付费才能进入的行业论坛里，甚至在交易对手的脑子里。记住，当所有人都觉得安全的时候，就是风险最大的时候。保持你的仓位灵活，永远给自己留后路。")
    print("-" * 80)
