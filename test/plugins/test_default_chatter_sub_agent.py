"""default_chatter.sub_agent 行为测试。"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock

import pytest

from plugins.default_chatter.config import DefaultChatterConfig
from plugins.default_chatter.plugin import (
    DefaultChatter,
    DefaultChatterPlugin,
    SendTextAction,
)
from src.core.models.message import Message
from src.core.models.stream import ChatStream


def _build_chatter() -> DefaultChatter:
    """构造默认聊天器实例。"""
    config = DefaultChatterConfig.from_dict({"plugin": {"enabled": True, "mode": "enhanced"}})
    plugin = DefaultChatterPlugin(config=config)
    return DefaultChatter(stream_id="test_stream", plugin=plugin)


def _build_chatter_with_config(plugin_overrides: dict[str, object]) -> DefaultChatter:
    """使用指定插件配置覆盖项构造默认聊天器实例。"""
    config = DefaultChatterConfig.from_dict(
        {"plugin": {"enabled": True, "mode": "enhanced", **plugin_overrides}}
    )
    plugin = DefaultChatterPlugin(config=config)
    return DefaultChatter(stream_id="test_stream", plugin=plugin)


@pytest.mark.asyncio
async def test_sub_agent_is_disabled_in_private_chat(monkeypatch: pytest.MonkeyPatch) -> None:
    """私聊场景应跳过 decide_should_respond。"""
    chatter = _build_chatter()
    stream = ChatStream(stream_id="s_private", platform="qq", chat_type="private")

    called = {"value": False}

    async def _fake_decide(**_kwargs: Any) -> dict[str, object]:
        called["value"] = True
        return {"reason": "should not be called", "should_respond": False}

    monkeypatch.setattr("plugins.default_chatter.plugin.decide_should_respond", _fake_decide)

    result = await chatter.sub_agent("hello", [], stream)

    assert result["should_respond"] is True
    assert "私聊场景" in result["reason"]
    assert called["value"] is False


@pytest.mark.asyncio
async def test_sub_agent_keeps_decision_flow_in_group_chat(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """群聊场景应继续走 decide_should_respond。"""
    chatter = _build_chatter()
    stream = ChatStream(stream_id="s_group", platform="qq", chat_type="group")

    captured: dict[str, Any] = {}

    async def _fake_decide(**kwargs: Any) -> dict[str, object]:
        captured.update(kwargs)
        return {"reason": "group decision", "should_respond": False}

    monkeypatch.setattr("plugins.default_chatter.plugin.decide_should_respond", _fake_decide)
    monkeypatch.setattr("plugins.default_chatter.plugin.random.random", lambda: 0.99)

    result = await chatter.sub_agent("group-msg", [], stream)

    assert result == {"reason": "group decision", "should_respond": False}
    assert captured["chatter"] is chatter
    assert captured["chat_stream"] is stream
    assert captured["unreads_text"] == "group-msg"
    assert captured["fallback_prompt"]
    assert "logger" in captured


@pytest.mark.asyncio
async def test_sub_agent_bypasses_llm_when_probability_hits(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """命中概率门时应直接响应，不再经过 decide_should_respond。"""
    chatter = _build_chatter()
    stream = ChatStream(
        stream_id="s_group",
        platform="qq",
        chat_type="group",
        bot_nickname="Neo",
    )
    setattr(stream.context, "_default_chatter_next_tick_bonus", 0.5)

    unread_msgs = [
        Message(content="Neo 你在吗", processed_plain_text="Neo 你在吗"),
        Message(content="小狐狸来看看", processed_plain_text="小狐狸来看看"),
    ]

    called = {"value": False}

    async def _fake_decide(**_kwargs: Any) -> dict[str, object]:
        called["value"] = True
        return {"reason": "should not be called", "should_respond": False}

    monkeypatch.setattr(
        "plugins.default_chatter.plugin.get_core_config",
        lambda: SimpleNamespace(
            personality=SimpleNamespace(
                nickname="Neo",
                alias_names=["小狐狸"],
            )
        ),
    )
    monkeypatch.setattr("plugins.default_chatter.plugin.decide_should_respond", _fake_decide)
    monkeypatch.setattr("plugins.default_chatter.plugin.random.random", lambda: 0.99)

    result = await chatter.sub_agent("group-msg", unread_msgs, stream)

    assert result["should_respond"] is True
    assert "概率直通响应" in result["reason"]
    assert called["value"] is False
    assert getattr(stream.context, "_default_chatter_next_tick_bonus", None) == 0.0


@pytest.mark.asyncio
async def test_send_text_marks_next_tick_bonus_after_success(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """send_text 成功后应为下一次 tick 写入概率加成。"""
    stream = ChatStream(stream_id="s_group", platform="qq", chat_type="group")
    action = SendTextAction(chat_stream=stream, plugin=DefaultChatterPlugin(config=DefaultChatterConfig()))

    monkeypatch.setattr(action, "_send_to_stream", AsyncMock(return_value=True))

    success, _detail = await action.execute(content="你好")

    assert success is True
    assert getattr(stream.context, "_default_chatter_next_tick_bonus", None) == 0.5


@pytest.mark.asyncio
async def test_sub_agent_skips_programmatic_controller_when_disabled(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """关闭程序化控制器后，群聊应始终回退到 decide_should_respond。"""
    chatter = _build_chatter_with_config({"enable_programmatic_controller": False})
    stream = ChatStream(
        stream_id="s_group",
        platform="qq",
        chat_type="group",
        bot_nickname="Neo",
    )
    setattr(stream.context, "_default_chatter_next_tick_bonus", 0.5)
    unread_msgs = [Message(content="Neo 你在吗", processed_plain_text="Neo 你在吗")]

    captured: dict[str, Any] = {}

    async def _fake_decide(**kwargs: Any) -> dict[str, object]:
        captured.update(kwargs)
        return {"reason": "llm only", "should_respond": False}

    monkeypatch.setattr(
        "plugins.default_chatter.plugin.get_core_config",
        lambda: SimpleNamespace(
            personality=SimpleNamespace(
                nickname="Neo",
                alias_names=["小狐狸"],
            )
        ),
    )
    monkeypatch.setattr("plugins.default_chatter.plugin.decide_should_respond", _fake_decide)
    monkeypatch.setattr("plugins.default_chatter.plugin.random.random", lambda: 0.0)

    result = await chatter.sub_agent("group-msg", unread_msgs, stream)

    assert result == {"reason": "llm only", "should_respond": False}
    assert captured["chatter"] is chatter
    assert getattr(stream.context, "_default_chatter_next_tick_bonus", None) == 0.5


@pytest.mark.asyncio
async def test_send_text_does_not_mark_bonus_when_controller_disabled(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """关闭程序化控制器后，send_text 不应写入下一 tick 加成。"""
    stream = ChatStream(stream_id="s_group", platform="qq", chat_type="group")
    plugin = DefaultChatterPlugin(
        config=DefaultChatterConfig.from_dict(
            {"plugin": {"enable_programmatic_controller": False}}
        )
    )
    action = SendTextAction(chat_stream=stream, plugin=plugin)

    monkeypatch.setattr(action, "_send_to_stream", AsyncMock(return_value=True))

    success, _detail = await action.execute(content="你好")

    assert success is True
    assert getattr(stream.context, "_default_chatter_next_tick_bonus", None) in (None, 0.0)
