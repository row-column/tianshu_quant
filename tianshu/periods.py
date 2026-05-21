from __future__ import annotations

from typing import Any


DAY_PERIOD_KEY = "1d"

MINUTES_TO_PERIOD_KEY = {
    1: "1m",
    3: "3m",
    5: "5m",
    10: "10m",
    15: "15m",
    30: "30m",
    60: "60m",
    120: "120m",
    180: "180m",
    240: "240m",
    720: DAY_PERIOD_KEY,
    1440: DAY_PERIOD_KEY,
}

PERIOD_KEY_TO_MINUTES = {
    "1m": 1,
    "3m": 3,
    "5m": 5,
    "10m": 10,
    "15m": 15,
    "30m": 30,
    "60m": 60,
    "120m": 120,
    "180m": 180,
    "240m": 240,
    DAY_PERIOD_KEY: 1440,
}


def normalize_period_key(period: Any = None) -> str:
    if period is None:
        return DAY_PERIOD_KEY

    if isinstance(period, int):
        try:
            return MINUTES_TO_PERIOD_KEY[period]
        except KeyError as exc:
            raise ValueError(f"不支持的K线周期分钟数: {period}") from exc

    if isinstance(period, str):
        normalized = period.strip().lower()
        aliases = {
            "day": DAY_PERIOD_KEY,
            "daily": DAY_PERIOD_KEY,
            "d": DAY_PERIOD_KEY,
            "1day": DAY_PERIOD_KEY,
            "1440m": DAY_PERIOD_KEY,
            "720m": DAY_PERIOD_KEY,
        }
        normalized = aliases.get(normalized, normalized)
        if normalized in PERIOD_KEY_TO_MINUTES:
            return normalized
        if normalized.endswith("min"):
            minute_text = normalized[:-3]
            if minute_text.isdigit():
                return normalize_period_key(int(minute_text))
        if normalized.endswith("m") and normalized[:-1].isdigit():
            return normalize_period_key(int(normalized[:-1]))
        raise ValueError(f"不支持的K线周期: {period}")

    # 兼容 longport Period 这类枚举/对象。不同SDK版本字符串实现不完全一致，
    # 所以这里采用宽松匹配，只在边界层使用。
    period_name = getattr(period, "name", "") or str(period)
    period_text = period_name.lower()
    if "day" in period_text:
        return DAY_PERIOD_KEY
    for minutes in sorted(MINUTES_TO_PERIOD_KEY, reverse=True):
        if minutes in (720, 1440):
            continue
        if f"min_{minutes}" in period_text or f"min{minutes}" in period_text:
            return MINUTES_TO_PERIOD_KEY[minutes]

    raise ValueError(f"不支持的K线周期对象: {period}")


def period_key_to_minutes(period: Any = None) -> int:
    return PERIOD_KEY_TO_MINUTES[normalize_period_key(period)]


def period_key_for_minutes(minutes: int) -> str:
    return normalize_period_key(minutes)
