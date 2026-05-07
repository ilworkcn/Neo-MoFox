"""Booku Memory 向量元数据清洗测试。"""

from __future__ import annotations

from plugins.booku_memory.service.booku_memory_service import BookuMemoryService


def test_sanitize_vector_metadata_drops_none_and_complex_values() -> None:
    """写入 Chroma 的 metadata 应只保留合法标量值。"""

    raw = {
        "title": "记忆标题",
        "bucket": "memory",
        "folder_id": "default",
        "person_id": None,
        "score": 0.5,
        "enabled": True,
        "tags": ["a", "b"],
        "extra": {"x": 1},
    }

    cleaned = BookuMemoryService._sanitize_vector_metadata(raw)

    assert cleaned == {
        "title": "记忆标题",
        "bucket": "memory",
        "folder_id": "default",
        "score": 0.5,
        "enabled": True,
    }