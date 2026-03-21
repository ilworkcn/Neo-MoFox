"""request_inspector 结构化渲染测试。"""

from src.kernel.llm.request_inspector import CapturedRequest, build_render_view


def test_build_render_view_renders_messages_tools_and_overview() -> None:
    """应将请求体转换为摘要、tools 与消息卡片。"""
    params = {
        "model": "demo-model",
        "stream": False,
        "temperature": 0.2,
        "messages": [
            {"role": "system", "content": "# System\nUse **markdown**"},
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "Hello\n- item"},
                    {"type": "image_url", "image_url": {"url": "data:image/png;base64,abc"}},
                ],
            },
            {
                "role": "assistant",
                "content": "Working on it",
                "tool_calls": [
                    {
                        "id": "call_1",
                        "type": "function",
                        "function": {
                            "name": "search_web",
                            "arguments": '{"query":"weather"}',
                        },
                    }
                ],
            },
            {
                "role": "tool",
                "tool_call_id": "call_1",
                "content": '{"temperature": 26}',
            },
        ],
        "tools": [
            {
                "type": "function",
                "function": {
                    "name": "search_web",
                    "description": "Search docs",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "query": {"type": "string", "description": "query text"},
                        },
                        "required": ["query"],
                    },
                },
            }
        ],
    }

    rendered = build_render_view(
        "chat.completions.create",
        "demo-model",
        params,
        {"api_provider": "OpenAI", "estimated_input_tokens": 128, "request_name": "demo_request"},
    )

    overview = {item["label"]: item["value"] for item in rendered["overview"]}
    assert overview["API"] == "chat.completions.create"
    assert overview["提供商"] == "OpenAI"
    assert overview["模型"] == "demo-model"
    assert overview["消息数"] == "4"
    assert overview["工具数"] == "1"
    assert overview["预估输入 Tokens"] == "128"
    assert overview["请求名称"] == "demo_request"
    assert overview["流式"] == "false"

    tool_card = rendered["tools"][0]
    assert tool_card["name"] == "search_web"
    assert tool_card["required"] == ["query"]
    assert tool_card["properties"][0]["type"] == "string"

    messages = rendered["messages"]
    assert messages[0]["blocks"][0]["type"] == "markdown"
    assert messages[1]["blocks"][1]["type"] == "media"
    assert messages[2]["blocks"][1]["type"] == "tool_call"
    assert messages[3]["blocks"][0]["type"] == "tool_result"
    assert messages[3]["meta"] == "tool_call_id: call_1"


def test_build_render_view_handles_unknown_message_shapes() -> None:
    """遇到未知消息结构时应回退为 unknown 块而不是崩溃。"""
    params = {
        "messages": [
            42,
            {"role": "assistant", "content": [{"type": "custom", "foo": "bar"}]},
        ]
    }

    rendered = build_render_view("chat.completions.create", "demo-model", params)

    assert rendered["messages"][0]["role"] == "unknown"
    assert rendered["messages"][0]["blocks"][0]["type"] == "unknown"
    assert rendered["messages"][1]["blocks"][0]["label"] == "未知 content 类型: custom"


def test_captured_request_to_full_includes_rendered_payload() -> None:
    """完整详情应同时返回 params 与 rendered 结构。"""
    record = CapturedRequest(
        id=1,
        ts=0.0,
        api_name="chat.completions.create",
        model="demo-model",
        params={"messages": [{"role": "user", "content": "hello"}], "tools": []},
        metadata={"api_provider": "OpenAI", "estimated_input_tokens": 12},
    )

    detail = record.to_full()

    assert detail["params"]["messages"][0]["content"] == "hello"
    assert detail["metadata"]["api_provider"] == "OpenAI"
    assert detail["estimated_input_tokens"] == 12
    assert detail["rendered"]["messages"][0]["blocks"][0]["text"] == "hello"