#!/usr/bin/env python3
"""一键清理 Booku Memory 插件 knowledge 层数据。

该脚本会执行以下操作：
1. 从插件配置加载 metadata_db_path 与 vector_db_path；
2. 批量硬删除 metadata 数据库中 bucket=knowledge, folder_id=default 的记录；
3. 删除向量库集合 booku_memory__knowledge__default；
4. 清理系统提醒中的“专业知识引导语”。

使用方式：
    uv run python scripts/clear_booku_knowledge.py
"""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path

# 允许脚本在仓库根目录外触发时仍可导入项目模块
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from plugins.booku_memory.config import BookuMemoryConfig
from plugins.booku_memory.service.metadata_repository import BookuMemoryMetadataRepository
from src.core.prompt import get_system_reminder_store
from src.kernel.logger import get_logger
from src.kernel.vector_db import get_vector_db_service

logger = get_logger("scripts.clear_booku_knowledge")

_TARGET_BUCKET = "knowledge"
_TARGET_FOLDER = "default"
_TARGET_COLLECTION = "booku_memory__knowledge__default"
_TARGET_REMINDER_BUCKET = "actor"
_TARGET_REMINDER_NAME = "专业知识引导语"
_DELETE_BATCH_SIZE = 500


def _load_booku_memory_config() -> BookuMemoryConfig:
    """加载 booku_memory 插件配置。"""
    return BookuMemoryConfig.load_for_plugin("booku_memory", auto_generate=True)


async def _clear_knowledge_metadata(config: BookuMemoryConfig) -> int:
    """分批硬删除 knowledge 层元数据记录。"""
    repo = BookuMemoryMetadataRepository(db_path=config.storage.metadata_db_path)
    deleted_total = 0

    await repo.initialize()
    try:
        while True:
            records = await repo.list_records_by_bucket(
                bucket=_TARGET_BUCKET,
                folder_id=_TARGET_FOLDER,
                limit=_DELETE_BATCH_SIZE,
                include_deleted=True,
            )
            if not records:
                break

            ids = [item.memory_id for item in records]
            deleted_total += await repo.hard_delete_records(ids)
            logger.info(f"已删除 metadata 记录批次: {len(ids)}")
    finally:
        await repo.close()

    return deleted_total


async def _clear_knowledge_vectors(config: BookuMemoryConfig) -> int:
    """删除 knowledge 层向量集合并返回删除前条数。"""
    vector_db = get_vector_db_service(config.storage.vector_db_path)

    before_count = await vector_db.count(_TARGET_COLLECTION)
    await vector_db.delete_collection(_TARGET_COLLECTION)

    return before_count


def _clear_knowledge_reminder() -> None:
    """清理知识库相关系统提醒。"""
    store = get_system_reminder_store()
    store.delete(_TARGET_REMINDER_BUCKET, _TARGET_REMINDER_NAME)


async def _run() -> int:
    """执行清理流程并返回进程退出码。"""
    config = _load_booku_memory_config()
    logger.info("开始清理 booku_memory knowledge 层数据")
    logger.info(f"metadata_db_path: {config.storage.metadata_db_path}")
    logger.info(f"vector_db_path: {config.storage.vector_db_path}")

    deleted_metadata = await _clear_knowledge_metadata(config)
    deleted_vectors = await _clear_knowledge_vectors(config)
    _clear_knowledge_reminder()

    logger.info("清理完成")
    logger.info(f"metadata 删除条数: {deleted_metadata}")
    logger.info(f"vector 删除条数(删除前计数): {deleted_vectors}")
    logger.info("专业知识引导语已清理")
    return 0


def main() -> None:
    """脚本入口。"""
    raise SystemExit(asyncio.run(_run()))


if __name__ == "__main__":
    main()
