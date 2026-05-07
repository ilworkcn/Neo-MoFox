"""Booku Memory Service 状态统计测试。"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

import pytest

from plugins.booku_memory.config import BookuMemoryConfig
from plugins.booku_memory.service.booku_memory_service import BookuMemoryService


@dataclass
class _DummyPlugin:
    """最小插件桩。"""

    config: Any


class _FakeRepo:
    """用于状态统计测试的仓储桩。"""

    async def list_distinct_folder_ids(self) -> list[str]:
        """返回两个已有 folder。"""

        return ["folder-a", "folder-b"]

    async def get_recent_records(
        self,
        *,
        limit: int = 10,
        folder_id: str | None = None,
        include_archived: bool = True,
    ) -> list[Any]:
        """返回空 recent 列表。"""

        del limit, folder_id, include_archived
        return []

    async def get_bucket_counts(self, folder_id: str | None = None) -> dict[str, int]:
        """返回全局 bucket 统计。"""

        assert folder_id is None
        return {"memory": 6, "knowledge": 3}


class _UnexpectedUpsert:
    """若被调用说明创建校验未提前拦截。"""

    async def __call__(self, **kwargs: Any) -> dict[str, Any]:
        raise AssertionError(f"upsert_memory 不应被调用: {kwargs}")


class _FakeVectorDB:
    """用于状态统计测试的向量库桩。"""

    def __init__(self) -> None:
        """初始化集合计数。"""

        self._counts = {
            "booku_memory__memory__folder-a": 2,
            "booku_memory__memory__folder-b": 4,
            "booku_memory__knowledge": 3,
        }

    async def count(self, collection_name: str) -> int:
        """返回指定集合的条数。"""

        return int(self._counts.get(collection_name, 0))


@pytest.mark.asyncio
async def test_get_status_without_folder_uses_global_counts(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """未指定 folder 时应返回全局库存概览，而不是 default folder。"""

    cfg = BookuMemoryConfig()
    cfg.storage.metadata_db_path = str(tmp_path / "memory.db")
    cfg.storage.vector_db_path = str(tmp_path / "vector_store")

    vector_db = _FakeVectorDB()
    monkeypatch.setattr(
        "plugins.booku_memory.service.booku_memory_service.get_vector_db_service",
        lambda _path: vector_db,
    )

    service = BookuMemoryService(plugin=cast(Any, _DummyPlugin(config=cfg)))

    async def _fake_get_repo() -> _FakeRepo:
        return _FakeRepo()

    monkeypatch.setattr(service, "_get_repo", _fake_get_repo)

    result = await service.get_status()

    assert result["folder_id"] == "all"
    assert result["counts"]["metadata"] == {"memory": 6, "knowledge": 3}
    assert result["counts"]["vector"] == {"memory": 6, "knowledge": 3}
    assert result["recent"] == []
    assert result["folder_memory_ids"] == []


@pytest.mark.asyncio
async def test_search_memory_entries_falls_back_to_keyword_records_when_vector_misses(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """query 检索无向量命中时，应回退到 metadata keyword 检索。"""

    cfg = BookuMemoryConfig()
    cfg.storage.metadata_db_path = str(tmp_path / "memory.db")
    cfg.storage.vector_db_path = str(tmp_path / "vector_store")

    service = BookuMemoryService(plugin=cast(Any, _DummyPlugin(config=cfg)))

    record = SimpleNamespace(
        memory_id="person-1",
        title="满月月",
        folder_id="default",
        bucket="memory",
        source="manual",
        memory_type="person",
        status="active",
        person_id="qq:10001",
        relation_memory_ids=[],
        relation_aliases=[],
        event_start_at=0.0,
        event_end_at=0.0,
        related_people=[],
        knowledge_type="",
        address_or_coord="",
        place_type="",
        asset_type="",
        disposition_status="",
        procedure_type="",
        novelty_energy=0.0,
        created_at=0.0,
        updated_at=0.0,
        last_activated_at=0.0,
        activation_count=0,
        is_deleted=False,
        deleted_at=0.0,
        tags=[],
        core_tags=["满月月"],
        diffusion_tags=[],
        opposing_tags=[],
    )

    class _SearchRepo:
        async def search_records(self, **kwargs: Any) -> list[Any]:
            assert kwargs["keyword"] == "满月月"
            assert kwargs["memory_type"] == "person"
            return [record]

        async def get_records_map(self, memory_ids: list[str]) -> dict[str, Any]:
            return {memory_id: record for memory_id in memory_ids if memory_id == record.memory_id}

    async def _fake_get_repo() -> _SearchRepo:
        return _SearchRepo()

    async def _fake_retrieve_memories(**kwargs: Any) -> dict[str, Any]:
        assert kwargs["query_text"] == "满月月"
        return {"results": []}

    monkeypatch.setattr(service, "_get_repo", _fake_get_repo)
    monkeypatch.setattr(service, "retrieve_memories", _fake_retrieve_memories)

    result = await service.search_memory_entries(
        query_text="满月月",
        memory_type="person",
        top_n=5,
    )

    assert result["total"] == 1
    assert result["items"][0]["id"] == "person-1"
    assert result["items"][0]["title"] == "满月月"
    assert result["items"][0]["metadata"]["memory_type"] == "person"


@pytest.mark.asyncio
async def test_search_memory_entries_supports_tag_triplet_without_query(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """仅提供完整标签三元组时，也应走语义检索路径。"""

    cfg = BookuMemoryConfig()
    cfg.storage.metadata_db_path = str(tmp_path / "memory.db")
    cfg.storage.vector_db_path = str(tmp_path / "vector_store")

    service = BookuMemoryService(plugin=cast(Any, _DummyPlugin(config=cfg)))

    class _SearchRepo:
        async def search_records_by_tag_triplet(self, **kwargs: Any) -> list[Any]:
            assert kwargs["core_tags"] == ["复盘"]
            assert kwargs["diffusion_tags"] == ["项目"]
            assert kwargs["opposing_tags"] == ["跑题"]
            return []

        async def get_records_map(self, memory_ids: list[str]) -> dict[str, Any]:
            return {}

    async def _fake_get_repo() -> _SearchRepo:
        return _SearchRepo()

    async def _fake_retrieve_memories(**kwargs: Any) -> dict[str, Any]:
        assert kwargs["query_text"] == "复盘 项目 跑题"
        assert kwargs["core_tags"] == ["复盘"]
        assert kwargs["diffusion_tags"] == ["项目"]
        assert kwargs["opposing_tags"] == ["跑题"]
        return {
            "results": [
                {
                    "id": "mem-1",
                    "title": "项目复盘",
                    "metadata": {"memory_type": "event", "status": "active"},
                }
            ]
        }

    monkeypatch.setattr(service, "_get_repo", _fake_get_repo)
    monkeypatch.setattr(service, "retrieve_memories", _fake_retrieve_memories)

    result = await service.search_memory_entries(
        top_n=5,
        core_tags=["复盘"],
        diffusion_tags=["项目"],
        opposing_tags=["跑题"],
    )

    assert result["total"] == 1
    assert result["items"][0]["id"] == "mem-1"
    assert result["items"][0]["title"] == "项目复盘"


@pytest.mark.asyncio
async def test_search_memory_entries_falls_back_to_tag_triplet_records_when_vector_misses(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """tag 检索在向量 miss 时，应回退到元数据库的三元标签召回。"""

    cfg = BookuMemoryConfig()
    cfg.storage.metadata_db_path = str(tmp_path / "memory.db")
    cfg.storage.vector_db_path = str(tmp_path / "vector_store")

    service = BookuMemoryService(plugin=cast(Any, _DummyPlugin(config=cfg)))

    record = SimpleNamespace(
        memory_id="mem-tag-1",
        title="标签命中记忆",
        folder_id="default",
        bucket="memory",
        source="manual",
        memory_type="event",
        status="active",
        person_id=None,
        relation_memory_ids=[],
        relation_aliases=[],
        event_start_at=0.0,
        event_end_at=0.0,
        related_people=[],
        knowledge_type="",
        address_or_coord="",
        place_type="",
        asset_type="",
        disposition_status="",
        procedure_type="",
        novelty_energy=0.0,
        created_at=0.0,
        updated_at=0.0,
        last_activated_at=0.0,
        activation_count=0,
        is_deleted=False,
        deleted_at=0.0,
        tags=[],
        core_tags=["复盘"],
        diffusion_tags=["项目"],
        opposing_tags=["跑题"],
    )

    class _SearchRepo:
        async def search_records_by_tag_triplet(self, **kwargs: Any) -> list[Any]:
            assert kwargs["core_tags"] == ["复盘"]
            assert kwargs["diffusion_tags"] == ["项目"]
            assert kwargs["opposing_tags"] == ["跑题"]
            return [record]

        async def get_records_map(self, memory_ids: list[str]) -> dict[str, Any]:
            return {memory_id: record for memory_id in memory_ids if memory_id == record.memory_id}

    async def _fake_get_repo() -> _SearchRepo:
        return _SearchRepo()

    async def _fake_retrieve_memories(**kwargs: Any) -> dict[str, Any]:
        assert kwargs["core_tags"] == ["复盘"]
        return {"results": []}

    monkeypatch.setattr(service, "_get_repo", _fake_get_repo)
    monkeypatch.setattr(service, "retrieve_memories", _fake_retrieve_memories)

    result = await service.search_memory_entries(
        top_n=5,
        core_tags=["复盘"],
        diffusion_tags=["项目"],
        opposing_tags=["跑题"],
    )

    assert result["total"] == 1
    assert result["items"][0]["id"] == "mem-tag-1"
    assert result["items"][0]["title"] == "标签命中记忆"


@pytest.mark.asyncio
async def test_create_memory_requires_complete_tag_triplet(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """create_memory 在 service 层也必须拒绝不完整三元组。"""

    cfg = BookuMemoryConfig()
    cfg.storage.metadata_db_path = str(tmp_path / "memory.db")
    cfg.storage.vector_db_path = str(tmp_path / "vector_store")

    service = BookuMemoryService(plugin=cast(Any, _DummyPlugin(config=cfg)))
    monkeypatch.setattr(service, "upsert_memory", _UnexpectedUpsert())

    with pytest.raises(ValueError, match="创建记忆必须同时提供完整且非空"):
        await service.create_memory(
            title="新记忆",
            content="正文",
            core_tags=["核心"],
            diffusion_tags=["扩散"],
            opposing_tags=[],
        )