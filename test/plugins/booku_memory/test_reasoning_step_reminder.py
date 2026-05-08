"""Tests for booku_memory memory_command behavior.

此文件沿用原文件名，当前覆盖 memory_command 的关键语义：
- help 命令的本地帮助返回
- search/read/create/update/delete 的分发
- && 串联执行与失败短路
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, cast

import pytest

from plugins.booku_memory.agent.tools import BookuMemoryCommandTool


@dataclass
class _DummyPlugin:
    """最小插件桩对象。"""

    config: Any = None


class _FakeService:
    """用于替换真实服务，记录调用并返回固定结果。"""

    def __init__(self) -> None:
        self.calls: list[tuple[str, dict[str, Any]]] = []

    async def search_memory_entries(self, **kwargs: Any) -> dict[str, Any]:
        self.calls.append(("search", kwargs))
        return {"action": "search_memory_entries", "total": 1, "items": [{"id": "m1", "title": "t1", "metadata": {}}]}

    async def read_full_content(self, *, memory_ids: list[str]) -> dict[str, Any]:
        self.calls.append(("read", {"memory_ids": memory_ids}))
        return {"action": "read_full_content", "requested": len(memory_ids), "total": len(memory_ids), "items": []}

    async def create_memory(self, **kwargs: Any) -> dict[str, Any]:
        self.calls.append(("create", kwargs))
        return {"action": "create_memory", "mode": "created", "total": 1, "items": [{"id": "m2"}]}

    async def update_memory_by_id(self, **kwargs: Any) -> dict[str, Any]:
        self.calls.append(("update", kwargs))
        return {"action": "update_memory_by_id", "updated": 1, "items": [{"id": kwargs.get("memory_id", "")}]}

    async def delete_memories(self, *, memory_ids: list[str], hard: bool = False) -> dict[str, Any]:
        self.calls.append(("delete", {"memory_ids": memory_ids, "hard": hard}))
        return {"action": "delete_memories", "mode": "hard" if hard else "soft", "deleted": len(memory_ids)}


@pytest.mark.asyncio
async def test_memory_command_help_returns_local_manual(monkeypatch: pytest.MonkeyPatch) -> None:
    """help 命令应直接返回本地命令手册，不依赖服务可用性。"""

    def _fail_if_called(_plugin: Any) -> Any:
        raise AssertionError("help 不应访问底层 service")

    monkeypatch.setattr("plugins.booku_memory.agent.tools._service", _fail_if_called)

    tool = BookuMemoryCommandTool(plugin=cast(Any, _DummyPlugin()))
    ok, payload = await tool.execute(command="help")

    assert ok is True
    assert isinstance(payload, dict)
    assert payload.get("ok") is True
    first = payload.get("results", [])[0]
    assert first.get("result", {}).get("action") == "help"
    assert "search" in str(first.get("result", {}).get("content", ""))
    assert "create" in str(first.get("result", {}).get("content", ""))


@pytest.mark.asyncio
async def test_memory_command_dispatch_and_chain(monkeypatch: pytest.MonkeyPatch) -> None:
    """memory_command 应支持多命令串联并保持顺序执行。"""

    fake_service = _FakeService()
    monkeypatch.setattr("plugins.booku_memory.agent.tools._service", lambda _plugin: fake_service)

    tool = BookuMemoryCommandTool(plugin=cast(Any, _DummyPlugin()))
    ok, payload = await tool.execute(
        command=(
            "search -type person -person_id qq:10001 -topn 3 "
            "-core_tags 同学 -diffusion_tags 校园 -opposing_tags 陌生人 "
            "&& read -ids m1,m2 "
            "&& delete -id m2 -hard true"
        )
    )

    assert ok is True
    assert isinstance(payload, dict)
    assert payload.get("ok") is True
    assert payload.get("executed") == 3

    assert [name for name, _ in fake_service.calls] == ["search", "read", "delete"]
    assert fake_service.calls[0][1]["memory_type"] == "person"
    assert fake_service.calls[0][1]["person_id"] == "qq:10001"
    assert fake_service.calls[0][1]["core_tags"] == ["同学"]
    assert fake_service.calls[0][1]["diffusion_tags"] == ["校园"]
    assert fake_service.calls[0][1]["opposing_tags"] == ["陌生人"]


@pytest.mark.asyncio
async def test_memory_command_folder_option_is_ignored(monkeypatch: pytest.MonkeyPatch) -> None:
    """folder_id/folder 参数应被工具层忽略，不影响下游调用。"""

    fake_service = _FakeService()
    monkeypatch.setattr("plugins.booku_memory.agent.tools._service", lambda _plugin: fake_service)

    tool = BookuMemoryCommandTool(plugin=cast(Any, _DummyPlugin()))
    ok, payload = await tool.execute(
        command=(
            "search -query 复盘 -core_tags 复盘 -diffusion_tags 项目 -opposing_tags 闲聊 -folder_id events "
            "&& create -type event -title 年会 -content 内容 -folder archive "
            "-core_tags 年会 -diffusion_tags 公司 -opposing_tags 缺席"
        )
    )

    assert ok is True
    assert isinstance(payload, dict)
    assert payload.get("ok") is True
    assert [name for name, _ in fake_service.calls] == ["search", "create"]
    assert "folder_id" not in fake_service.calls[0][1]
    assert "folder_id" not in fake_service.calls[1][1]


@pytest.mark.asyncio
async def test_memory_command_requires_person_id_for_person_create(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """人物记忆创建必须携带 person_id。"""

    fake_service = _FakeService()
    monkeypatch.setattr("plugins.booku_memory.agent.tools._service", lambda _plugin: fake_service)

    tool = BookuMemoryCommandTool(plugin=cast(Any, _DummyPlugin()))
    ok, payload = await tool.execute(
        command=(
            "create -type person -title 张三 -content 测试人物 "
            "-core_tags 同学 -diffusion_tags 校园 -opposing_tags 陌生人"
        )
    )

    assert ok is False
    assert isinstance(payload, dict)
    assert payload.get("ok") is False
    first = payload.get("results", [])[0]
    assert "person_id" in str(first.get("error", ""))


@pytest.mark.asyncio
async def test_memory_command_search_requires_complete_tag_triplet(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """search 命令必须提供完整且非空的标签三元组。"""

    fake_service = _FakeService()
    monkeypatch.setattr("plugins.booku_memory.agent.tools._service", lambda _plugin: fake_service)

    tool = BookuMemoryCommandTool(plugin=cast(Any, _DummyPlugin()))
    ok, payload = await tool.execute(command="search -query 复盘 -core_tags 复盘")

    assert ok is False
    assert isinstance(payload, dict)
    assert payload.get("ok") is False
    first = payload.get("results", [])[0]
    assert "禁止只传一组或两组" in str(first.get("error", ""))
    assert fake_service.calls == []


@pytest.mark.asyncio
async def test_memory_command_search_without_tags_is_allowed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """search 未提供 tag 时应兼容为普通无筛选检索。"""

    fake_service = _FakeService()
    monkeypatch.setattr("plugins.booku_memory.agent.tools._service", lambda _plugin: fake_service)

    tool = BookuMemoryCommandTool(plugin=cast(Any, _DummyPlugin()))
    ok, payload = await tool.execute(command="search -topn 50")

    assert ok is True
    assert isinstance(payload, dict)
    assert payload.get("ok") is True
    assert [name for name, _ in fake_service.calls] == ["search"]
    assert fake_service.calls[0][1]["top_n"] == 50
    assert fake_service.calls[0][1]["core_tags"] == []
    assert fake_service.calls[0][1]["diffusion_tags"] == []
    assert fake_service.calls[0][1]["opposing_tags"] == []


@pytest.mark.asyncio
async def test_memory_command_create_requires_complete_tag_triplet(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """create 命令必须提供完整且非空的标签三元组。"""

    fake_service = _FakeService()
    monkeypatch.setattr("plugins.booku_memory.agent.tools._service", lambda _plugin: fake_service)

    tool = BookuMemoryCommandTool(plugin=cast(Any, _DummyPlugin()))
    ok, payload = await tool.execute(
        command="create -type event -title 年会 -content 内容 -core_tags 年会 -diffusion_tags 公司"
    )

    assert ok is False
    assert isinstance(payload, dict)
    assert payload.get("ok") is False
    first = payload.get("results", [])[0]
    assert "三组标签" in str(first.get("error", ""))
    assert fake_service.calls == []


@pytest.mark.asyncio
async def test_memory_command_update_rejects_partial_tag_triplet(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """update 命令只要传标签，就必须整组三元标签一起传。"""

    fake_service = _FakeService()
    monkeypatch.setattr("plugins.booku_memory.agent.tools._service", lambda _plugin: fake_service)

    tool = BookuMemoryCommandTool(plugin=cast(Any, _DummyPlugin()))
    ok, payload = await tool.execute(command="update -id mem-1 -core_tags 已归档")

    assert ok is False
    assert isinstance(payload, dict)
    assert payload.get("ok") is False
    first = payload.get("results", [])[0]
    assert "禁止只传一组或两组" in str(first.get("error", ""))
    assert fake_service.calls == []


@pytest.mark.asyncio
async def test_memory_command_stops_on_first_failed_segment(monkeypatch: pytest.MonkeyPatch) -> None:
    """命令链出现失败时应短路，不继续执行后续命令。"""

    fake_service = _FakeService()
    monkeypatch.setattr("plugins.booku_memory.agent.tools._service", lambda _plugin: fake_service)

    tool = BookuMemoryCommandTool(plugin=cast(Any, _DummyPlugin()))
    ok, payload = await tool.execute(
        command="read -ids m1 && unknown -x 1 && delete -id m1"
    )

    assert ok is False
    assert isinstance(payload, dict)
    assert payload.get("executed") == 2
    # 只执行了 read，第二段报错后 short-circuit
    assert [name for name, _ in fake_service.calls] == ["read"]
