"""Tests for LLMContextManager behavior."""

from __future__ import annotations

from typing import Any

from src.kernel.llm.context import LLMContextManager
from src.kernel.llm.payload import LLMPayload, Text, ToolCall, ToolResult
from src.kernel.llm.request import LLMRequest
from src.kernel.llm.roles import ROLE


class DummyTool:
    @classmethod
    def to_schema(cls) -> dict[str, Any]:
        return {"name": "dummy"}


def dummy_model() -> dict[str, Any]:
    return {
        "api_provider": "openai",
        "base_url": "https://api.openai.com/v1",
        "model_identifier": "gpt-4",
        "api_key": "sk-test",
        "client_type": "openai",
        "max_retry": 0,
        "timeout": 1,
        "retry_interval": 0,
        "price_in": 0.0,
        "price_out": 0.0,
        "temperature": 0.1,
        "max_tokens": 10,
        "extra_params": {},
    }


def test_context_manager_trims_full_groups() -> None:
    manager = LLMContextManager(max_payloads=5)
    payloads = [
        LLMPayload(ROLE.SYSTEM, Text("sys")),
        LLMPayload(ROLE.TOOL, DummyTool),
        LLMPayload(ROLE.USER, Text("q1")),
        LLMPayload(ROLE.ASSISTANT, Text("a1")),
        LLMPayload(ROLE.TOOL_RESULT, ToolResult({"ok": True})),
        LLMPayload(ROLE.USER, Text("q2")),
        LLMPayload(ROLE.ASSISTANT, Text("a2")),
    ]

    trimmed = manager.maybe_trim(payloads)

    assert len(trimmed) == 4
    assert trimmed[0].role == ROLE.SYSTEM
    assert trimmed[1].role == ROLE.TOOL
    assert trimmed[2].role == ROLE.USER
    assert trimmed[2].content[0].text == "q2"
    assert trimmed[3].role == ROLE.ASSISTANT


def test_context_manager_applies_hook() -> None:
    called = {"value": False}

    def hook(dropped_groups, remaining_payloads):
        called["value"] = True
        return [LLMPayload(ROLE.ASSISTANT, Text("summary"))]

    manager = LLMContextManager(max_payloads=4, compression_hook=hook)
    payloads = [
        LLMPayload(ROLE.SYSTEM, Text("sys")),
        LLMPayload(ROLE.USER, Text("q1")),
        LLMPayload(ROLE.ASSISTANT, Text("a1")),
        LLMPayload(ROLE.USER, Text("q2")),
        LLMPayload(ROLE.ASSISTANT, Text("a2")),
    ]

    trimmed = manager.maybe_trim(payloads)

    assert called["value"] is True
    assert len(trimmed) == 4
    assert trimmed[0].role == ROLE.SYSTEM
    assert trimmed[1].role == ROLE.ASSISTANT
    assert trimmed[1].content[0].text == "summary"
    assert trimmed[2].role == ROLE.USER
    assert trimmed[2].content[0].text == "q2"


def test_llm_request_uses_custom_context_manager() -> None:
    class CustomManager(LLMContextManager):
        def __init__(self) -> None:
            super().__init__(max_payloads=10)
            self.called = False

        def maybe_trim(self, payloads: list[LLMPayload]) -> list[LLMPayload]:
            self.called = True
            return payloads

    manager = CustomManager()
    request = LLMRequest([dummy_model()], context_manager=manager)
    request.add_payload(LLMPayload(ROLE.USER, Text("hello")))

    assert manager.called is True


def test_context_manager_trims_by_token_budget() -> None:
    manager = LLMContextManager(max_payloads=10)
    payloads = [
        LLMPayload(ROLE.USER, Text("q1")),
        LLMPayload(ROLE.ASSISTANT, Text("a1")),
        LLMPayload(ROLE.USER, Text("q2")),
        LLMPayload(ROLE.ASSISTANT, Text("a2")),
        LLMPayload(ROLE.USER, Text("q3")),
        LLMPayload(ROLE.ASSISTANT, Text("a3")),
    ]

    # 每条消息按 10 token 计，预算 25 时只能保留最后一组（2条消息）
    trimmed = manager.maybe_trim(
        payloads,
        max_token_budget=25,
        token_counter=lambda items: len(items) * 10,
    )

    assert len(trimmed) == 2
    assert trimmed[0].role == ROLE.USER
    assert trimmed[0].content[0].text == "q3"
    assert trimmed[1].role == ROLE.ASSISTANT


def test_context_manager_system_tool_equivalent_add_payload() -> None:
    manager = LLMContextManager(max_payloads=20)
    payloads: list[LLMPayload] = []

    payloads = manager.system(payloads, Text("sys"))
    payloads = manager.tool(payloads, DummyTool)

    assert len(payloads) == 2
    assert payloads[0].role == ROLE.SYSTEM
    assert payloads[0].content[0].text == "sys"
    assert payloads[1].role == ROLE.TOOL


def test_context_manager_reminder_creates_first_user() -> None:
    manager = LLMContextManager(max_payloads=20)
    payloads = [LLMPayload(ROLE.SYSTEM, Text("sys"))]

    payloads = manager.reminder(payloads, "你必须先输出结论")

    assert len(payloads) == 2
    assert payloads[0].role == ROLE.SYSTEM
    assert payloads[1].role == ROLE.USER
    assert payloads[1].content[0].text == "你必须先输出结论"


def test_context_manager_fills_missing_tool_result_placeholder() -> None:
    manager = LLMContextManager(max_payloads=20)
    payloads = [LLMPayload(ROLE.USER, Text("帮我调用工具"))]

    payloads = manager.add_payload(
        payloads,
        LLMPayload(
            ROLE.ASSISTANT,
            [
                Text("我将调用工具"),
                ToolCall(id="call_1", name="get_weather", args={"city": "上海"}),
            ],
        ),
    )

    assert len(payloads) == 4
    assert payloads[0].role == ROLE.USER
    assert payloads[1].role == ROLE.ASSISTANT
    assert payloads[2].role == ROLE.TOOL_RESULT
    result = next(part for part in payloads[2].content if isinstance(part, ToolResult))
    assert result.call_id == "call_1"
    assert result.value == ""
    assert payloads[3].role == ROLE.ASSISTANT
