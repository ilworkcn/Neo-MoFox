"""LLM 统计模块

提供独立的、基于 SQLite 的 LLM 请求消耗统计，不依赖公共数据库。

使用方式:
    # 初始化（在 kernel 启动时调用一次）
    from src.kernel.llm.stats import init_llm_stats, get_llm_stats_collector
    await init_llm_stats(db_path="data/llm_stats/llm_stats.db")

    # 记录一次请求
    collector = get_llm_stats_collector()
    await collector.record(record)

    # 查询统计
    summary = await collector.get_summary()
    by_model = await collector.get_by_model()
    by_request = await collector.get_by_request_name()
    cache_rate = await collector.get_cache_hit_rate()
"""

from .collector import LLMStatsCollector, LLMRequestRecord
from .config import LLMStatsConfig
from .database import (
    close_llm_stats_db,
    get_llm_stats_collector,
    init_llm_stats,
)

__all__ = [
    "LLMStatsCollector",
    "LLMRequestRecord",
    "LLMStatsConfig",
    "init_llm_stats",
    "get_llm_stats_collector",
    "close_llm_stats_db",
]
