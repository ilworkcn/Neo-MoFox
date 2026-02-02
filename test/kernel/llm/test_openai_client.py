"""OpenAI客户端测试。

使用mock来模拟OpenAI SDK，避免依赖真实的API调用。
"""

import asyncio
from unittest.mock import AsyncMock, MagicMock, Mock, patch

import pytest

from src.kernel.llm import (
    Image,
    LLMPayload,
    ROLE,
    Text,
    Tool,
    ToolResult,
)


class TestIsDataUrl:
    """测试_is_data_url函数。"""

    def test_data_url_with_png(self):
        """测试PNG data URL。"""
        from src.kernel.llm.model_client.openai_client import _is_data_url

        assert _is_data_url("data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAUA")

    def test_data_url_with_jpeg(self):
        """测试JPEG data URL。"""
        from src.kernel.llm.model_client.openai_client import _is_data_url

        assert _is_data_url("data:image/jpeg;base64,/9j/4AAQSkZJRgABAQAAAQABAAD")

    def test_not_data_url(self):
        """测试非data URL。"""
        from src.kernel.llm.model_client.openai_client import _is_data_url

        assert not _is_data_url("https://example.com/image.png")
        assert not _is_data_url("/path/to/image.png")
        assert not _is_data_url("image.png")


class TestImageToDataUrl:
    """测试_image_to_data_url函数。"""

    def test_base64_format(self):
        """测试base64|格式转换。"""
        from src.kernel.llm.model_client.openai_client import _image_to_data_url

        b64_data = "iVBORw0KGgoAAAANSUhEUgAAAAUA"
        result = _image_to_data_url(f"base64|{b64_data}")

        assert result.startswith("data:image/png;base64,")
        assert b64_data in result

    def test_already_data_url(self):
        """测试已经是data URL格式。"""
        from src.kernel.llm.model_client.openai_client import _image_to_data_url

        url = "data:image/png;base64,iVBORw0KGgo"
        result = _image_to_data_url(url)

        assert result == url

    def test_file_not_found(self):
        """测试文件不存在。"""
        from src.kernel.llm.model_client.openai_client import _image_to_data_url

        with pytest.raises(FileNotFoundError, match="Image file not found"):
            _image_to_data_url("/nonexistent/file.png")

    @patch("src.kernel.llm.model_client.openai_client.Path")
    def test_valid_file_path(self, mock_path):
        """测试有效的文件路径。"""
        from src.kernel.llm.model_client.openai_client import _image_to_data_url

        # 创建mock文件
        mock_file = Mock()
        mock_file.exists.return_value = True
        mock_file.is_file.return_value = True
        mock_file.read_bytes.return_value = b"fake_image_data"
        mock_path.return_value = mock_file

        result = _image_to_data_url("/fake/path/image.png")

        assert result.startswith("data:image/png;base64,")
        # 验证base64编码
        assert "fake_image_data" not in result  # 应该被编码


class TestPayloadsToOpenAIMessages:
    """测试_payloads_to_openai_messages函数。"""

    def test_user_text_message(self):
        """测试用户文本消息。"""
        from src.kernel.llm.model_client.openai_client import _payloads_to_openai_messages

        payloads = [LLMPayload(ROLE.USER, Text("Hello"))]
        messages, tools = _payloads_to_openai_messages(payloads)

        assert len(messages) == 1
        assert messages[0]["role"] == "user"
        assert messages[0]["content"] == "Hello"
        assert len(tools) == 0

    def test_assistant_message(self):
        """测试助手消息。"""
        from src.kernel.llm.model_client.openai_client import _payloads_to_openai_messages

        payloads = [LLMPayload(ROLE.ASSISTANT, Text("Hi there"))]
        messages, tools = _payloads_to_openai_messages(payloads)

        assert len(messages) == 1
        assert messages[0]["role"] == "assistant"
        assert messages[0]["content"] == "Hi there"

    def test_multimodal_content(self):
        """测试多模态内容（文本+图片）。"""
        from src.kernel.llm.model_client.openai_client import _payloads_to_openai_messages

        payloads = [
            LLMPayload(
                ROLE.USER,
                [Text("What's in this image?"), Image("base64|abc123")],
            )
        ]
        messages, tools = _payloads_to_openai_messages(payloads)

        assert len(messages) == 1
        assert messages[0]["role"] == "user"
        content = messages[0]["content"]
        assert isinstance(content, list)
        assert len(content) == 2
        assert content[0]["type"] == "text"
        assert content[1]["type"] == "image_url"

    def test_tool_payload(self):
        """测试工具定义payload。"""
        from src.kernel.llm.model_client.openai_client import _payloads_to_openai_messages

        class MockTool:
            @classmethod
            def to_schema(cls):
                return {
                    "name": "test_tool",
                    "description": "A test tool",
                    "parameters": {"type": "object"},
                }

        payloads = [LLMPayload(ROLE.TOOL, [Tool(tool=MockTool)])]
        messages, tools = _payloads_to_openai_messages(payloads)

        # TOOL payload不应该进入messages
        assert len(messages) == 0
        assert len(tools) == 1
        assert tools[0]["type"] == "function"
        assert tools[0]["function"]["name"] == "test_tool"

    def test_tool_result_payload(self):
        """测试工具结果payload。"""
        from src.kernel.llm.model_client.openai_client import _payloads_to_openai_messages

        result = ToolResult(value="42", call_id="call_123", name="calculator")
        payloads = [LLMPayload(ROLE.TOOL_RESULT, [result])]

        messages, tools = _payloads_to_openai_messages(payloads)

        assert len(messages) == 1
        assert messages[0]["role"] == "tool"
        assert messages[0]["content"] == "42"
        assert messages[0]["tool_call_id"] == "call_123"

    def test_tool_result_with_text(self):
        """测试工具结果包含Text。"""
        from src.kernel.llm.model_client.openai_client import _payloads_to_openai_messages

        payloads = [
            LLMPayload(
                ROLE.TOOL_RESULT,
                [ToolResult(value="result", call_id="call_123"), Text("extra")],
            )
        ]

        messages, tools = _payloads_to_openai_messages(payloads)

        assert len(messages) == 1
        # 应该使用ToolResult的内容
        assert messages[0]["content"] == "result"

    def test_custom_object_with_to_text(self):
        """测试自定义对象带to_text方法。"""
        from src.kernel.llm.model_client.openai_client import _payloads_to_openai_messages

        class CustomResult:
            call_id = "custom_call"

            def to_text(self):
                return "custom result"

        payloads = [LLMPayload(ROLE.TOOL_RESULT, [CustomResult()])]

        messages, tools = _payloads_to_openai_messages(payloads)

        assert len(messages) == 1
        assert messages[0]["content"] == "custom result"
        assert messages[0]["tool_call_id"] == "custom_call"

    def test_empty_tool_result_content(self):
        """测试空的工具结果内容。"""
        from src.kernel.llm.model_client.openai_client import _payloads_to_openai_messages

        class BrokenObject:
            def to_text(self):
                raise Exception("Cannot convert")

        payloads = [LLMPayload(ROLE.TOOL_RESULT, [BrokenObject()])]

        messages, tools = _payloads_to_openai_messages(payloads)

        assert len(messages) == 1
        assert messages[0]["content"] == ""


class TestOpenAIChatClient:
    """测试OpenAIChatClient类。"""

    def test_init(self):
        """测试初始化。"""
        from src.kernel.llm.model_client.openai_client import OpenAIChatClient

        client = OpenAIChatClient()

        assert client._clients == {}
        assert hasattr(client, "_lock")

    def test_get_loop_key_with_running_loop(self):
        """测试获取事件循环key（有运行中的循环）。"""
        from src.kernel.llm.model_client.openai_client import OpenAIChatClient

        client = OpenAIChatClient()

        async def test():
            loop_key = client._get_loop_key()
            assert loop_key > 0
            return loop_key

        loop_key = asyncio.run(test())
        assert isinstance(loop_key, int)

    def test_get_loop_key_without_loop(self):
        """测试获取事件循环key（无运行中的循环）。"""
        from src.kernel.llm.model_client.openai_client import OpenAIChatClient

        client = OpenAIChatClient()

        # 在没有运行循环的情况下调用
        loop_key = client._get_loop_key()
        assert loop_key == 0

    @pytest.mark.asyncio
    async def test_create_non_streaming_response(self):
        """测试非流式响应。"""
        from src.kernel.llm.model_client.openai_client import OpenAIChatClient

        # 创建mock客户端
        mock_completion = MagicMock()
        mock_completion.choices = [MagicMock()]
        mock_completion.choices[0].message.content = "Hello, world!"
        mock_completion.choices[0].message.tool_calls = None

        mock_chat = AsyncMock()
        mock_chat.completions.create = AsyncMock(return_value=mock_completion)

        mock_openai_client = MagicMock()
        mock_openai_client.chat.completions.create = mock_chat.completions.create

        client = OpenAIChatClient()
        client._clients = {}  # 清空缓存
        client._get_client = MagicMock(return_value=mock_openai_client)

        payloads = [LLMPayload(ROLE.USER, Text("Hi"))]
        model_set = {
            "api_key": "test-key",
            "base_url": "https://api.test.com",
            "timeout": 30.0,
            "max_tokens": 100,
            "temperature": 0.7,
            "extra_params": {},
        }

        message, tool_calls, stream_iter = await client.create(
            model_name="gpt-4",
            payloads=payloads,
            tools=[],
            request_name="test",
            model_set=model_set,
            stream=False,
        )

        assert message == "Hello, world!"
        assert tool_calls == []
        assert stream_iter is None

    @pytest.mark.asyncio
    async def test_create_with_tool_calls(self):
        """测试包含工具调用的响应。"""
        from src.kernel.llm.model_client.openai_client import OpenAIChatClient

        # 创建mock工具调用
        mock_tc1 = MagicMock()
        mock_tc1.id = "call_123"
        mock_tc1.function.name = "calculator"
        mock_tc1.function.arguments = '{"a": 1, "b": 2}'

        mock_completion = MagicMock()
        mock_completion.choices = [MagicMock()]
        mock_completion.choices[0].message.content = None
        mock_completion.choices[0].message.tool_calls = [mock_tc1]

        mock_chat = AsyncMock()
        mock_chat.completions.create = AsyncMock(return_value=mock_completion)

        mock_openai_client = MagicMock()
        mock_openai_client.chat.completions.create = mock_chat.completions.create

        client = OpenAIChatClient()
        client._clients = {}
        client._get_client = MagicMock(return_value=mock_openai_client)

        payloads = [LLMPayload(ROLE.USER, Text("Calculate 1+2"))]
        model_set = {
            "api_key": "test-key",
            "base_url": None,
            "timeout": None,
            "max_tokens": None,
            "temperature": None,
            "extra_params": {},
        }

        message, tool_calls, stream_iter = await client.create(
            model_name="gpt-4",
            payloads=payloads,
            tools=[],
            request_name="test",
            model_set=model_set,
            stream=False,
        )

        assert message == ""
        assert len(tool_calls) == 1
        assert tool_calls[0]["id"] == "call_123"
        assert tool_calls[0]["name"] == "calculator"
        assert tool_calls[0]["args"] == {"a": 1, "b": 2}

    @pytest.mark.asyncio
    async def test_create_invalid_model_set_type(self):
        """测试无效的model_set类型。"""
        from src.kernel.llm.model_client.openai_client import OpenAIChatClient

        client = OpenAIChatClient()

        with pytest.raises(TypeError, match="OpenAIChatClient 期望 model_set 为单个模型配置 dict"):
            await client.create(
                model_name="gpt-4",
                payloads=[],
                tools=[],
                request_name="test",
                model_set="not_a_dict",  # type: ignore
                stream=False,
            )

    @pytest.mark.asyncio
    async def test_create_missing_api_key(self):
        """测试缺少api_key。"""
        from src.kernel.llm.model_client.openai_client import OpenAIChatClient

        client = OpenAIChatClient()
        model_set = {"api_key": ""}  # 空api_key

        with pytest.raises(ValueError, match="model.api_key 不能为空"):
            await client.create(
                model_name="gpt-4",
                payloads=[],
                tools=[],
                request_name="test",
                model_set=model_set,
                stream=False,
            )

    @pytest.mark.asyncio
    async def test_create_invalid_extra_params(self):
        """测试无效的extra_params。"""
        from src.kernel.llm.model_client.openai_client import OpenAIChatClient

        mock_completion = MagicMock()
        mock_completion.choices = [MagicMock()]
        mock_completion.choices[0].message.content = "OK"
        mock_completion.choices[0].message.tool_calls = None

        mock_chat = AsyncMock()
        mock_chat.completions.create = AsyncMock(return_value=mock_completion)

        mock_openai_client = MagicMock()
        mock_openai_client.chat.completions.create = mock_chat.completions.create

        client = OpenAIChatClient()
        client._clients = {}
        client._get_client = MagicMock(return_value=mock_openai_client)

        model_set = {
            "api_key": "test-key",
            "base_url": None,
            "timeout": None,
            "max_tokens": None,
            "temperature": None,
            "extra_params": "invalid",  # 不是dict
        }

        with pytest.raises(ValueError, match="model.extra_params 必须是 dict"):
            await client.create(
                model_name="gpt-4",
                payloads=[],
                tools=[],
                request_name="test",
                model_set=model_set,
                stream=False,
            )

    @pytest.mark.asyncio
    async def test_create_with_extra_params(self):
        """测试包含额外参数。"""
        from src.kernel.llm.model_client.openai_client import OpenAIChatClient

        mock_completion = MagicMock()
        mock_completion.choices = [MagicMock()]
        mock_completion.choices[0].message.content = "Response"
        mock_completion.choices[0].message.tool_calls = None

        mock_chat = AsyncMock()
        mock_chat.completions.create = AsyncMock(return_value=mock_completion)

        mock_openai_client = MagicMock()
        mock_openai_client.chat.completions.create = mock_chat.completions.create

        client = OpenAIChatClient()
        client._clients = {}
        client._get_client = MagicMock(return_value=mock_openai_client)

        payloads = [LLMPayload(ROLE.USER, Text("Hi"))]
        model_set = {
            "api_key": "test-key",
            "base_url": None,
            "timeout": None,
            "max_tokens": None,
            "temperature": None,
            "extra_params": {"top_p": 0.9, "presence_penalty": 0.1},
        }

        await client.create(
            model_name="gpt-4",
            payloads=payloads,
            tools=[],
            request_name="test",
            model_set=model_set,
            stream=False,
        )

        # 验证额外参数被传递
        call_kwargs = mock_chat.completions.create.call_args.kwargs
        assert call_kwargs["top_p"] == 0.9
        assert call_kwargs["presence_penalty"] == 0.1

