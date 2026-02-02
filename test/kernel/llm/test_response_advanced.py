"""LLMResponse 高级功能测试。

测试覆盖：
1. stream_with_callback 方法
2. stream_with_buffer 方法
3. _ToolCallAccumulator 类
4. 各种边缘情况
"""

import pytest

from src.kernel.llm.model_client.base import StreamEvent
from src.kernel.llm.payload import LLMPayload, Text
from src.kernel.llm.request import LLMRequest
from src.kernel.llm.response import LLMResponse, _ToolCallAccumulator
from src.kernel.llm.roles import ROLE


def dummy_model_set():
    return [
        {
            "api_provider": "OpenAI",
            "base_url": "https://api.openai.com/v1",
            "model_identifier": "dummy",
            "api_key": "dummy-key",
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
    ]


async def _stream_text_events():
    """生成文本流事件。"""
    yield StreamEvent(text_delta="hel")
    yield StreamEvent(text_delta="lo")
    yield StreamEvent(text_delta=" world")


async def _stream_tool_events():
    """生成工具调用流事件。"""
    yield StreamEvent(tool_call_id="call_123", tool_name="calculator", tool_args_delta='{"a":')
    yield StreamEvent(tool_call_id="call_123", tool_args_delta=' 1, "b":')
    yield StreamEvent(tool_call_id="call_123", tool_args_delta=' 2}')


async def _stream_mixed_events():
    """生成混合流事件。"""
    yield StreamEvent(text_delta="Thinking")
    yield StreamEvent(tool_call_id="call_456", tool_name="search", tool_args_delta='{"query":')
    yield StreamEvent(text_delta="...")
    yield StreamEvent(tool_call_id="call_456", tool_args_delta=' "test"}')


@pytest.mark.asyncio
async def test_stream_with_callback():
    """测试流式响应 + 实时回调。"""
    chunks = []

    async def callback(chunk: str):
        chunks.append(chunk)

    req = LLMRequest(dummy_model_set(), request_name="test")
    resp = LLMResponse(
        _stream=_stream_text_events(),
        _upper=req,
        _auto_append_response=False,
        payloads=[LLMPayload(ROLE.USER, Text("hi"))],
        model_set=req.model_set,
    )

    result = await resp.stream_with_callback(callback)

    assert result == "hello world"
    assert chunks == ["hel", "lo", " world"]
    assert resp.message == "hello world"


@pytest.mark.asyncio
async def test_stream_with_callback_no_stream():
    """测试stream_with_callback处理非流式响应。"""
    chunks = []

    async def callback(chunk: str):
        chunks.append(chunk)

    req = LLMRequest(dummy_model_set(), request_name="test")
    resp = LLMResponse(
        _stream=None,
        _upper=req,
        _auto_append_response=False,
        payloads=[LLMPayload(ROLE.USER, Text("hi"))],
        model_set=req.model_set,
        message="static message",
    )

    result = await resp.stream_with_callback(callback)

    assert result == "static message"
    assert chunks == ["static message"]


@pytest.mark.asyncio
async def test_stream_with_callback_consumed_error():
    """测试stream_with_callback在已消费响应时抛出异常。"""
    async def callback(chunk: str):
        pass

    req = LLMRequest(dummy_model_set(), request_name="test")
    resp = LLMResponse(
        _stream=_stream_text_events(),
        _upper=req,
        _auto_append_response=False,
        payloads=[LLMPayload(ROLE.USER, Text("hi"))],
        model_set=req.model_set,
    )

    await resp.stream_with_callback(callback)

    with pytest.raises(Exception):  # LLMResponseConsumedError
        await resp.stream_with_callback(callback)


@pytest.mark.asyncio
async def test_stream_with_buffer():
    """测试带缓冲的流式响应。"""
    req = LLMRequest(dummy_model_set(), request_name="test")
    resp = LLMResponse(
        _stream=_stream_text_events(),
        _upper=req,
        _auto_append_response=False,
        payloads=[LLMPayload(ROLE.USER, Text("hi"))],
        model_set=req.model_set,
    )

    buffers = []
    async for chunk in resp.stream_with_buffer(buffer_size=5):
        buffers.append(chunk)

    # "hel" (3) + "lo" (2) = 5，应该先yield
    # " world" (6) 单独yield
    assert len(buffers) == 2
    assert buffers[0] == "hello"
    assert buffers[1] == " world"
    assert resp.message == "hello world"


@pytest.mark.asyncio
async def test_stream_with_buffer_no_stream():
    """测试stream_with_buffer处理非流式响应。"""
    req = LLMRequest(dummy_model_set(), request_name="test")
    resp = LLMResponse(
        _stream=None,
        _upper=req,
        _auto_append_response=False,
        payloads=[LLMPayload(ROLE.USER, Text("hi"))],
        model_set=req.model_set,
        message="static message",
    )

    buffers = []
    async for chunk in resp.stream_with_buffer(buffer_size=5):
        buffers.append(chunk)

    assert len(buffers) == 1
    assert buffers[0] == "static message"


@pytest.mark.asyncio
async def test_stream_with_buffer_consumed_error():
    """测试stream_with_buffer在已消费响应时抛出异常。"""
    req = LLMRequest(dummy_model_set(), request_name="test")
    resp = LLMResponse(
        _stream=_stream_text_events(),
        _upper=req,
        _auto_append_response=False,
        payloads=[LLMPayload(ROLE.USER, Text("hi"))],
        model_set=req.model_set,
    )

    async for _ in resp.stream_with_buffer(buffer_size=5):
        pass

    with pytest.raises(Exception):  # LLMResponseConsumedError
        async for _ in resp.stream_with_buffer(buffer_size=5):
            pass


@pytest.mark.asyncio
async def test_tool_call_accumulator_single_call():
    """测试工具调用累加器（单个工具调用）。"""
    acc = _ToolCallAccumulator()

    async def events():
        yield StreamEvent(tool_call_id="call_1", tool_name="calc", tool_args_delta='{"a":')
        yield StreamEvent(tool_call_id="call_1", tool_args_delta='1}')

    async for event in events():
        acc.apply(event)

    calls = acc.finalize()
    assert len(calls) == 1
    assert calls[0].id == "call_1"
    assert calls[0].name == "calc"
    assert calls[0].args == {"a": 1}


@pytest.mark.asyncio
async def test_tool_call_accumulator_multiple_calls():
    """测试工具调用累加器（多个工具调用）。"""
    acc = _ToolCallAccumulator()

    async def events():
        yield StreamEvent(tool_call_id="call_1", tool_name="tool1", tool_args_delta='{"x":1}')
        yield StreamEvent(tool_call_id="call_2", tool_name="tool2", tool_args_delta='{"y":2}')

    async for event in events():
        acc.apply(event)

    calls = acc.finalize()
    assert len(calls) == 2
    assert calls[0].id == "call_1"
    assert calls[0].name == "tool1"
    assert calls[1].id == "call_2"
    assert calls[1].name == "tool2"


@pytest.mark.asyncio
async def test_tool_call_accumulator_invalid_json():
    """测试工具调用累加器处理无效JSON。"""
    acc = _ToolCallAccumulator()

    async def events():
        yield StreamEvent(tool_call_id="call_1", tool_name="calc", tool_args_delta="not json")

    async for event in events():
        acc.apply(event)

    calls = acc.finalize()
    assert len(calls) == 1
    assert calls[0].id == "call_1"
    assert calls[0].args == "not json"  # 应该保持原始字符串


@pytest.mark.asyncio
async def test_tool_call_accumulator_empty_args():
    """测试工具调用累加器处理空参数。"""
    acc = _ToolCallAccumulator()

    async def events():
        yield StreamEvent(tool_call_id="call_1", tool_name="calc")

    async for event in events():
        acc.apply(event)

    calls = acc.finalize()
    assert len(calls) == 1
    assert calls[0].args == {}


@pytest.mark.asyncio
async def test_response_collects_tool_calls_from_stream():
    """测试响应从流中收集工具调用。"""
    req = LLMRequest(dummy_model_set(), request_name="test")
    resp = LLMResponse(
        _stream=_stream_tool_events(),
        _upper=req,
        _auto_append_response=False,
        payloads=[LLMPayload(ROLE.USER, Text("hi"))],
        model_set=req.model_set,
    )

    await resp

    assert len(resp.call_list) == 1
    assert resp.call_list[0].id == "call_123"
    assert resp.call_list[0].name == "calculator"
    assert resp.call_list[0].args == {"a": 1, "b": 2}


@pytest.mark.asyncio
async def test_response_collects_mixed_stream():
    """测试响应收集混合流（文本+工具调用）。"""
    req = LLMRequest(dummy_model_set(), request_name="test")
    resp = LLMResponse(
        _stream=_stream_mixed_events(),
        _upper=req,
        _auto_append_response=False,
        payloads=[LLMPayload(ROLE.USER, Text("hi"))],
        model_set=req.model_set,
    )

    chunks = []
    async for chunk in resp:
        chunks.append(chunk)

    assert chunks == ["Thinking", "..."]
    assert resp.message == "Thinking..."
    assert len(resp.call_list) == 1
    assert resp.call_list[0].name == "search"


@pytest.mark.asyncio
async def test_response_to_payload():
    """测试to_payload方法。"""
    req = LLMRequest(dummy_model_set(), request_name="test")
    resp = LLMResponse(
        _stream=None,
        _upper=req,
        _auto_append_response=False,
        payloads=[LLMPayload(ROLE.USER, Text("hi"))],
        model_set=req.model_set,
        message="test message",
    )

    payload = resp.to_payload()
    assert payload.role == ROLE.ASSISTANT
    assert payload.content == [Text("test message")]


@pytest.mark.asyncio
async def test_response_add_payload():
    """测试add_payload方法。"""
    req = LLMRequest(dummy_model_set(), request_name="test")
    resp = LLMResponse(
        _stream=None,
        _upper=req,
        _auto_append_response=False,
        payloads=[LLMPayload(ROLE.USER, Text("first"))],
        model_set=req.model_set,
    )

    new_payload = LLMPayload(ROLE.USER, Text("second"))
    resp.add_payload(new_payload)

    assert len(resp.payloads) == 2
    assert resp.payloads[1] == new_payload


@pytest.mark.asyncio
async def test_response_add_payload_with_position():
    """测试add_payload方法指定位置。"""
    req = LLMRequest(dummy_model_set(), request_name="test")
    resp = LLMResponse(
        _stream=None,
        _upper=req,
        _auto_append_response=False,
        payloads=[
            LLMPayload(ROLE.USER, Text("first")),
            LLMPayload(ROLE.USER, Text("third")),
        ],
        model_set=req.model_set,
    )

    new_payload = LLMPayload(ROLE.USER, Text("second"))
    resp.add_payload(new_payload, position=1)

    assert len(resp.payloads) == 3
    assert resp.payloads[1] == new_payload


@pytest.mark.asyncio
async def test_response_add_call_reflex():
    """测试add_call_reflex方法。"""
    req = LLMRequest(dummy_model_set(), request_name="test")
    resp = LLMResponse(
        _stream=None,
        _upper=req,
        _auto_append_response=False,
        payloads=[LLMPayload(ROLE.USER, Text("hi"))],
        model_set=req.model_set,
    )

    results = [
        LLMPayload(ROLE.TOOL_RESULT, Text("result1")),
        LLMPayload(ROLE.TOOL_RESULT, Text("result2")),
    ]

    resp.add_call_reflex(results)

    assert len(resp.payloads) == 3
    assert resp.payloads[1] == results[0]
    assert resp.payloads[2] == results[1]
