#!/usr/bin/env python
# -*- coding=utf-8 -*-
import os, sys
project_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_path)
import logging
import smtplib
from email.mime.text import MIMEText
from typing import List, Optional
import requests
import time
from datetime import datetime
import json
from utils.cfg_utils import load_yaml2cfg
import threading
# --- 模块级依赖与配置 ---
logger = logging.getLogger(__name__)
cfg_file = os.path.join(project_path, 'data/server_config.yaml')
cfg = load_yaml2cfg(cfg_file)

# --- 核心功能函数 ---

def send_email(
    subject: str,
    content: str,
    receivers: List[str]=['rowcolumn@163.com'],
    smtp_host: str = 'smtp.qq.com',
    smtp_port: int = 587,
    sender_email:str = '532033774@qq.com',
    smtp_user:str = '532033774',
    smtp_password:str = 'ehxswbhyugxpbhaj',
    timeout: int = 20,
    is_html: bool = False  # <--- 【新增】一个参数，用于区分邮件类型
) -> bool:
    """
    发送纯文本邮件，具有高健壮性和安全性的实现。支持纯文本和HTML格式。

    该函数会优先使用传入的认证信息。如果未传入，则会尝试从环境变量中读取
    (MAIL_SENDER, MAIL_USER, MAIL_PASS)，这是推荐的最佳实践。

    Args:
        subject (str): 邮件主题。
        content (str): 邮件正文内容 (纯文本)。
        receivers (List[str]): 收件人邮箱地址列表。
        smtp_host (str): SMTP服务器地址。默认为 'smtp.qq.com'。
        smtp_port (int): SMTP服务器端口。默认为 587 (适用于TLS)。
        sender_email (Optional[str]): 发件人邮箱地址。如果为None，则从环境变量 'MAIL_SENDER' 获取。
        smtp_user (Optional[str]): SMTP登录用户名。如果为None，则从环境变量 'MAIL_USER' 获取。
        smtp_password (Optional[str]): SMTP登录密码或授权码。如果为None，则从环境变量 'MAIL_PASS' 获取。
        timeout (int): 连接服务器的超时时间（秒）。

    Returns:
        bool: 发送成功返回 True，失败返回 False。
    """
    # --- 1. 安全地获取认证信息 ---
    # 优先使用函数参数，其次是环境变量。这种方式避免了将敏感信息硬编码在代码中。
    final_sender = sender_email or os.environ.get('MAIL_SENDER')
    final_user = smtp_user or os.environ.get('MAIL_USER')
    final_password = smtp_password or os.environ.get('MAIL_PASS')

    # --- 2. 前置校验 ---
    if not all([final_sender, final_user, final_password]):
        logger.error(
            "邮件发送失败：发件人、用户名或密码未提供。"
            "请通过函数参数或环境变量 (MAIL_SENDER, MAIL_USER, MAIL_PASS) 进行配置。"
        )
        return False
        
    if not receivers:
        logger.warning("邮件发送取消：收件人列表为空。")
        return False

    # --- 3. 构建邮件消息体 ---
    mime_type = 'html' if is_html else 'plain'
    message = MIMEText(content, mime_type, 'utf-8')
    message['Subject'] = subject
    message['From'] = final_sender
    message['To'] = ", ".join(receivers)  # 邮件头中收件人格式

    # --- 4. 建立SMTP连接并发送 ---
    server = None # 初始化server变量
    try:
        # 步骤 4.1: 建立连接
        server = smtplib.SMTP(smtp_host, smtp_port, timeout=timeout)
        server.starttls()
        
        # 步骤 4.2: 登录
        server.login(final_user, final_password)
        
        # 步骤 4.3: 发送邮件
        server.sendmail(final_sender, receivers, message.as_string())
        
        logger.info(f"邮件 '{subject}' 已成功发送至: {', '.join(receivers)}")
        return True
    # --- 5. 精细化的异常处理 ---
    except smtplib.SMTPAuthenticationError:
        logger.error("邮件发送失败：SMTP认证失败。请检查用户名和密码（或授权码）是否正确。", exc_info=True)
    except smtplib.SMTPConnectError:
        logger.error(f"邮件发送失败：无法连接到SMTP服务器 {smtp_host}:{smtp_port}。", exc_info=True)
    except smtplib.SMTPServerDisconnected:
        logger.error("邮件发送失败：SMTP服务器意外断开连接。", exc_info=True)
    except TimeoutError:
        logger.error(f"邮件发送失败：连接SMTP服务器 {smtp_host}:{smtp_port} 超时。", exc_info=True)
    except Exception as e:
        logger.error(f"邮件发送时发生未知错误: {e}", exc_info=True)
        # 如果发生任何异常，返回False
        return False

    # --- 6. 最终的清理工作 ---
    finally:
        if server:
            try:
                # 无论如何，都尝试正常退出会话
                server.quit()
            except Exception as e:
                # 即使退出时也可能发生错误（例如连接已断），记录警告即可，不影响主流程判断
                logger.warning(f"关闭SMTP连接时发生错误: {e}", exc_info=True)
                

def send_trade_notification(action: str, symbol: str,symbol_name: str,quantity: int, price: float, reason: str,title_keyword:str='trader'):
        """格式化并以非阻塞方式发送交易通知邮件。"""
        action_map = {"BUY": "买入开仓", "ADD": "盈利加仓", "PARTIAL SELL": "部分卖出", "LIQUIDATE": "清仓卖出"}
        action_cn = action_map.get(action.upper(), action)
        from utils.common_utils import get_market_type
        market = get_market_type(symbol)
        from positions import MarketType
        currency = "港币" if market == MarketType.HK else "美元"
        msg_body = f"""
-- 交易提醒 --
操作:     {action_cn}
股票代码: {symbol}
股票名称: {symbol_name}
数量:     {quantity} 股
价格:     {price:.3f} {currency}
总金额:   {quantity * price:,.2f} {currency}
原因:     {reason}
时间戳:   {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
        """
        try:
            # threading.Thread(target=send_email, args=(f'{title_keyword}-{symbol}', msg_body.strip()), daemon=True).start()
            send_email(subject=f'{title_keyword}-{symbol}',content=msg_body.strip())
            logger.info(f"已派发'{action_cn}'的邮件通知 -> {symbol}")
        except Exception as e:
            logger.error(f"派发邮件通知线程失败 -> {symbol}: {e}")

def send_strategy_notification(symbol: str,symbol_name: str, price: float, reason: str,title_keyword:str='新机会'):
        """格式化并以非阻塞方式发送交易通知邮件。"""
        from utils.common_utils import get_market_type
        market = get_market_type(symbol)
        from positions import MarketType
        currency = "港币" if market == MarketType.HK else "美元"
        msg_body = f"""
-- ✅ 发现买入信号 --
股票代码:  {symbol}
股票名称:  {symbol_name}
股票价格:  {price:.3f} {currency}
触发策略:  {reason}
时间戳:   {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
        """
        try:
            # threading.Thread(target=send_email, args=(f'{title_keyword}-{symbol}', msg_body.strip()), daemon=True).start()
            send_email(subject=f'{title_keyword}-{symbol}',content=msg_body.strip())
            logger.info(f"已派发发现买入信号的邮件通知 -> {symbol}")
        except Exception as e:
            logger.error(f"派发邮件通知线程失败 -> {symbol}: {e}")

def send_weixin_notice(
    message: str,
    user_ids: Optional[List[str]] = None,
    request_url: str = "https://miaotixing.com/trigger",
    timeout: int = 15
) -> bool:
    """通过"喵提醒"服务发送微信通知。"""
    target_users = user_ids or cfg.common.get('weixin_user_ids', ['tmnbHCO'])
    
    if not target_users:
        logger.warning("微信通知取消：接收用户ID列表为空或未配置。")
        return False
        
    if not isinstance(target_users, list):
        logger.error(f"微信通知配置错误：'weixin_user_ids' 应为列表。")
        return False

    headers = {"Content-Type": "application/json;charset=utf8", 'User-Agent': 'Python-Trading-System-Notifier/2.0'}
    
    success_count = 0
    for user_id in target_users:
        if not user_id: continue
        
        payload = {'id': user_id, 'text': message, 'ts': str(int(time.time())), 'type': 'json'}
        
        try:
            response = requests.post(url=request_url, params=payload, headers=headers, timeout=timeout)
            response.raise_for_status()
            
            result = response.json()
            if result.get('code') == 0:
                logger.info(f"微信通知已成功发送至用户ID: {user_id}。")
                success_count += 1
            else:
                logger.error(f"发送微信通知至 {user_id} 失败 (API): {result.get('msg', '未知错误')}")
        except requests.exceptions.RequestException as e:
            logger.error(f"发送微信通知至 {user_id} 失败 (网络): {e}")
        except Exception as e:
            logger.error(f"发送微信通知至 {user_id} 时发生未知错误: {e}", exc_info=True)
            
    return success_count > 0


if __name__ == "__main__":
    # send_weixin_notice('teset')
    # send_trade_notification('LIQUIDATE', 'TSLL.US',1,1,1,'内收盘前(10min)清仓')
    send_strategy_notification('TSLL.US',1,1,'内收盘前(10min)清仓')
    # send_email(subject='test',content='test')
    pass