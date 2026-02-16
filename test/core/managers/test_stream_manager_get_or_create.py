from __future__ import annotations

import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest


@pytest.mark.asyncio
async def test_get_or_create_stream_concurrent_calls_create_once(monkeypatch) -> None:
    """同一 stream_id 并发获取时应只创建一次流实例。"""
    from src.core.managers.stream_manager import StreamManager

    manager = StreamManager()
    stream_id = "stream-concurrent-001"
    fake_stream = SimpleNamespace(stream_id=stream_id, context=SimpleNamespace())

    manager._streams_crud.get_by = AsyncMock(return_value=None)
    manager._create_new_stream = AsyncMock(return_value=fake_stream)  # type: ignore[method-assign]

    first, second = await asyncio.gather(
        manager.get_or_create_stream(stream_id=stream_id, platform="qq"),
        manager.get_or_create_stream(stream_id=stream_id, platform="qq"),
    )

    assert first is fake_stream
    assert second is fake_stream
    assert manager._create_new_stream.await_count == 1
    assert manager._streams_crud.get_by.await_count == 1
    assert manager._create_new_stream.await_args.kwargs["stream_id"] == stream_id


@pytest.mark.asyncio
async def test_get_or_create_stream_returns_cached_instance_without_db(monkeypatch) -> None:
    """缓存中已有流时应直接返回，不触发查库/建流。"""
    from src.core.managers.stream_manager import StreamManager

    manager = StreamManager()
    stream_id = "stream-cached-001"
    cached_stream = SimpleNamespace(stream_id=stream_id, context=SimpleNamespace())
    manager._streams[stream_id] = cached_stream

    manager._streams_crud.get_by = AsyncMock(return_value=None)
    manager._create_new_stream = AsyncMock()  # type: ignore[method-assign]

    result = await manager.get_or_create_stream(stream_id=stream_id, platform="qq")

    assert result is cached_stream
    assert manager._streams_crud.get_by.await_count == 0
    assert manager._create_new_stream.await_count == 0


@pytest.mark.asyncio
async def test_create_new_stream_includes_bot_info(monkeypatch) -> None:
    """创建新流时应从适配器获取 bot 信息并保存到 ChatStream。"""
    from src.core.managers.stream_manager import StreamManager

    manager = StreamManager()
    manager._streams_crud.create = AsyncMock(return_value=None)

    # Mock user_query_helper
    helper = SimpleNamespace(generate_person_id=lambda platform, user_id: f"{platform}:{user_id}")
    monkeypatch.setattr(
        "src.core.utils.user_query_helper.get_user_query_helper",
        lambda: helper,
    )

    # Mock adapter manager get_bot_info_by_platform
    adapter_manager = SimpleNamespace(
        get_bot_info_by_platform=AsyncMock(
            return_value={"bot_id": "10001", "bot_nickname": "TestBot"}
        )
    )
    monkeypatch.setattr(
        "src.core.managers.adapter_manager.get_adapter_manager",
        lambda: adapter_manager,
    )

    stream = await manager._create_new_stream(
        platform="qq",
        user_id="u001",
        chat_type="private",
        stream_id="stream-new-001",
    )

    assert stream.bot_id == "10001"
    assert stream.bot_nickname == "TestBot"
    adapter_manager.get_bot_info_by_platform.assert_awaited_once_with("qq")
