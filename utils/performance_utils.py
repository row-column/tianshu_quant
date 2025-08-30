#!/usr/bin/env python
# -*- coding=utf-8 -*-
import os, sys
project_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_path)
import time
import threading
import logging
from functools import wraps
from cachetools import TTLCache
from collections import deque
logger = logging.getLogger(__name__)

class APIRateLimiter:
    """一个简单的、线程安全的API速率限制器"""
    def __init__(self, requests_per_second: int):
        self.requests_per_second = requests_per_second
        self.interval = 1.0 / requests_per_second
        self.last_request_time = 0
        self.lock = threading.Lock()

    def wait(self):
        with self.lock:
            now = time.time()
            elapsed = now - self.last_request_time
            if elapsed < self.interval:
                time.sleep(self.interval - elapsed)
            self.last_request_time = time.time()


class APIRateLimiter_v2:
    """
    【世界第一最终定稿 · 唯一正确版】
    基于“滑动时间窗口+请求队列”算法，是工业级限流器的标准实现。
    它简单、高效、100%线程安全，且逻辑无可辩驳。
    """
    def __init__(self, requests_per_second: int, requests_per_minute: int):
        if requests_per_second <= 0 or requests_per_minute <= 0:
            raise ValueError("速率必须是正数")
        
        # 秒级控制：两次请求的最小时间间隔
        self.interval_sec = 1.0 / requests_per_second
        
        # 分钟级控制：一分钟内允许的最大请求数
        self.limit_min = requests_per_minute
        
        # 请求时间戳队列 (使用双端队列，两头增删都是O(1)复杂度)
        self.requests = deque()
        
        self.lock = threading.Lock()

    def wait(self):
        with self.lock:
            while True:
                now = time.monotonic()

                # 1. 清理过期记录：从队列左侧移除所有超过1分钟窗口的旧时间戳
                while self.requests and now - self.requests[0] > 60:
                    self.requests.popleft()

                # 2. 检查分钟配额
                if len(self.requests) < self.limit_min:
                    # 分钟配额充足，再检查秒级配额
                    last_req_time = self.requests[-1] if self.requests else 0
                    if now - last_req_time >= self.interval_sec:
                        # 所有条件都满足，记录当前请求时间戳，跳出循环，允许通行
                        self.requests.append(now)
                        break
                
                # --- 如果任意条件不满足，计算需要休眠多久 ---
                
                # a. 因秒级限制需要等待的时间
                sleep_for_sec = self.interval_sec - (now - (self.requests[-1] if self.requests else 0))
                
                # b. 因分钟限制需要等待的时间 (等待队列中最早的那个请求过期)
                # 只有在分钟配额满时，这个等待才有意义
                sleep_for_min = float('inf')
                if len(self.requests) >= self.limit_min:
                    sleep_for_min = self.requests[0] + 60.1 - now # +0.1秒作为缓冲
                
                # 取两者中需要等待的、更短的那个时间。
                # 比如，可能秒级限制已经满足，但分钟超了，那就等分钟；
                # 或者分钟没超，但秒级太快了，那就等秒级。
                sleep_time = max(0, min(sleep_for_sec, sleep_for_min))
                
                # [关键] 先释放锁，再去休眠，避免阻塞其他无关线程
                self.lock.release()
                time.sleep(sleep_time if sleep_time > 0 else 0.01) # 最小休眠0.01秒，防止CPU空转
                self.lock.acquire()
                # 休眠结束后，循环会回到开头，重新检查所有条件

#==============================================================================
#  多线程机制的TTL缓存方法
# ==============================================================================
def thread_safe_ttl_cache(maxsize: int = 128, ttl: int = 15):
    """
    一个线程安全的、按条目过期的缓存装饰器。
    结合了 cachetools.TTLCache 和 threading.Lock 的优点。
    """
    # 为每个使用此装饰器的函数创建一个独立的缓存和锁
    cache = TTLCache(maxsize=maxsize, ttl=ttl)
    lock = threading.Lock()

    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            # 使用新的辅助函数来创建一个100%可哈希的键
            key = make_hashable_key(args, kwargs)

            # 双重检查锁定模式：先在锁外检查，提高性能
            cached_result = cache.get(key)
            if cached_result is not None:
                logger.debug(f"缓存命中 for {func.__name__} with key: {key}")
                return cached_result
            
            with lock:
                # 在锁内再次检查，防止在等待锁时其他线程已填充缓存
                cached_result = cache.get(key)
                if cached_result is not None:
                    logger.info(f"缓存命中 (in lock) for {func.__name__} with key: {key}")
                    return cached_result
                
                # 执行函数并缓存结果
                logger.info(f"缓存未命中，执行函数 {func.__name__} with key: {key}")
                result = func(*args, **kwargs)
                cache[key] = result
                return result
        return wrapper
    return decorator

def make_hashable_key(args, kwargs):
    """
    [辅助函数] 将传入的参数转换成一个完全可哈希的缓存键。
    它会尝试哈希每个参数，如果失败（例如遇到列表或自定义对象），
    则会将其转换为字符串。
    """
    key_parts = []
    # 1. 处理位置参数
    for arg in args:
        try:
            hash(arg)
            key_parts.append(arg)
        except TypeError:
            # 如果参数不可哈希，则使用其字符串表示形式
            key_parts.append(str(arg))

    # 2. 处理关键字参数
    if kwargs:
        # 对关键字参数按键名排序，确保 f(a=1, b=2) 和 f(b=2, a=1) 的缓存键相同
        sorted_kwargs = sorted(kwargs.items())
        for k, v in sorted_kwargs:
            try:
                hash(v)
                key_parts.append((k, v))
            except TypeError:
                key_parts.append((k, str(v)))
    
    return tuple(key_parts)