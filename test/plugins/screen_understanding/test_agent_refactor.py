"""screen_understanding agent refactor tests."""

from __future__ import annotations

import io
from types import SimpleNamespace
from typing import Any
from typing import cast
from unittest.mock import patch
from unittest.mock import AsyncMock

import pytest
import pybase64 as base64

from src.kernel.llm import LLMContextManager
from src.kernel.llm import LLMPayload
from src.kernel.llm import ROLE
from src.kernel.llm import Text
from src.kernel.llm import ToolCall
from src.kernel.llm import ToolResult

from plugins.screen_understanding.action_parser import ScreenControlAction
from plugins.screen_understanding.action_parser import parse_control_response
from plugins.screen_understanding.agent import ScreenControlAgent
from plugins.screen_understanding.agent import ScreenClickTool
from plugins.screen_understanding.agent.tools import ScreenCloseWindowTool
from plugins.screen_understanding.agent.tools import ScreenLaunchProgramTool
from plugins.screen_understanding.agent.tools import ScreenWaitTool
from plugins.screen_understanding.control_backends import build_control_backend_candidates
from plugins.screen_understanding.control_backends.wdotool_backend import WdotoolControlBackend
from plugins.screen_understanding.control_backends.xdotool_backend import XdotoolControlBackend
from plugins.screen_understanding.control_backends.ydotool_backend import YdotoolControlBackend
from plugins.screen_understanding.plugin import ScreenUnderstandingAdapter
from plugins.screen_understanding.plugin import ScreenUnderstandingPlugin
from plugins.screen_understanding.config import ScreenUnderstandingConfig


class _AdapterStub:
    """Minimal adapter stub for control-agent tests."""

    def __init__(self) -> None:
        self._analysis_model_set = object()
        self._control_model_set = object()
        self._model_set = self._analysis_model_set
        self.capture_current_frame = AsyncMock(
            return_value=SimpleNamespace(png_base64="ZmFrZQ==")
        )


class _FakeResponse:
    """Minimal follow-up response stub for tool-calling loops."""

    def __init__(self, call_batches: list[list[SimpleNamespace]]) -> None:
        self._call_batches = call_batches
        self._index = 0
        self.call_list = call_batches[0] if call_batches else []
        self.payloads: list[object] = []

    def __await__(self) -> object:
        async def _done() -> None:
            return None

        return _done().__await__()

    def add_payload(self, payload: object, position: object | None = None) -> "_FakeResponse":
        del position
        self.payloads.append(payload)
        return self

    async def send(self, stream: bool = False) -> "_FakeResponse":
        del stream
        self._index += 1
        self.call_list = self._call_batches[self._index] if self._index < len(self._call_batches) else []
        return self


class _FakeRequest:
    """Minimal request stub for the control agent's internal loop."""

    def __init__(self, response: _FakeResponse) -> None:
        self._response = response
        self.payloads: list[object] = []

    def add_payload(self, payload: object, position: object | None = None) -> "_FakeRequest":
        del position
        self.payloads.append(payload)
        return self

    async def send(self, stream: bool = False) -> _FakeResponse:
        del stream
        return self._response


@pytest.mark.asyncio
async def test_control_agent_runs_tool_chain_until_finish(monkeypatch: pytest.MonkeyPatch) -> None:
    """Agent 应通过私有 tools 执行动作，并在 finish tool 处结束。"""

    plugin = cast(Any, ScreenUnderstandingPlugin(config=ScreenUnderstandingConfig()))
    plugin.bind_adapter(_AdapterStub())
    monkeypatch.setattr(plugin, "describe_active_window", AsyncMock(return_value="app_id=code,title=VS Code"))

    agent = ScreenControlAgent(stream_id="stream-1", plugin=plugin)
    response = _FakeResponse(
        [
            [SimpleNamespace(id="call-1", name="tool-screen_click", args={"x": 10, "y": 20})],
            [SimpleNamespace(id="call-2", name="tool-screen_finish", args={"content": "设置页已经打开"})],
        ]
    )
    monkeypatch.setattr(
        agent,
        "create_llm_request",
        lambda **_kwargs: _FakeRequest(response),
    )
    monkeypatch.setattr(
        agent,
        "execute_local_usable",
        AsyncMock(
            return_value=(
                True,
                {
                    "kind": "action",
                    "action_type": "left_single",
                    "action_inputs": {"start_box": [10.0, 20.0, 10.0, 20.0]},
                    "execution_result": "已点击设置按钮",
                    "terminal": False,
                },
            )
        ),
    )

    success, result = await agent.execute(goal="打开设置", max_steps=3)

    assert success is True
    assert isinstance(result, dict)
    assert result["mode"] == "completed"
    assert result["goal"] == "打开设置"
    assert len(result["steps"]) == 1
    assert result["steps"][0]["action"] == "left_single"
    assert result["message"] == "设置页已经打开"


@pytest.mark.asyncio
async def test_control_agent_returns_stalled_when_repeating_same_action_without_progress(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """连续重复同一动作且界面无变化时，agent 应在工具链中提前停止。"""

    plugin = cast(Any, ScreenUnderstandingPlugin(config=ScreenUnderstandingConfig()))
    adapter = _AdapterStub()
    adapter.capture_current_frame = AsyncMock(
        side_effect=[
            SimpleNamespace(png_base64="ZmFrZQ=="),
            SimpleNamespace(png_base64="ZmFrZQ=="),
            SimpleNamespace(png_base64="ZmFrZQ=="),
            SimpleNamespace(png_base64="ZmFrZQ=="),
            SimpleNamespace(png_base64="ZmFrZQ=="),
            SimpleNamespace(png_base64="ZmFrZQ=="),
        ]
    )
    plugin.bind_adapter(adapter)
    monkeypatch.setattr(plugin, "describe_active_window", AsyncMock(return_value="app_id=code,title=VS Code"))

    agent = ScreenControlAgent(stream_id="stream-1", plugin=plugin)
    response = _FakeResponse(
        [
            [SimpleNamespace(id="call-1", name="tool-screen_click", args={"x": 10, "y": 20})],
            [SimpleNamespace(id="call-2", name="tool-screen_click", args={"x": 10, "y": 20})],
            [SimpleNamespace(id="call-3", name="tool-screen_click", args={"x": 10, "y": 20})],
        ]
    )
    monkeypatch.setattr(
        agent,
        "create_llm_request",
        lambda **_kwargs: _FakeRequest(response),
    )
    monkeypatch.setattr(
        agent,
        "execute_local_usable",
        AsyncMock(
            return_value=(
                True,
                {
                    "kind": "action",
                    "action_type": "left_single",
                    "action_inputs": {"start_box": [10.0, 20.0, 10.0, 20.0]},
                    "execution_result": "已点击按钮。",
                    "terminal": False,
                },
            )
        ),
    )

    success, result = await agent.execute(goal="点击按钮直到出现新窗口", max_steps=10)

    assert success is True
    assert isinstance(result, dict)
    assert result["mode"] == "stalled"
    assert len(result["steps"]) == 3


@pytest.mark.asyncio
async def test_screen_click_tool_converts_coordinates_to_action(monkeypatch: pytest.MonkeyPatch) -> None:
    """私有 click tool 应将坐标转换为统一的 action inputs 后交给插件执行。"""

    plugin = cast(Any, ScreenUnderstandingPlugin(config=ScreenUnderstandingConfig()))
    execute_mock = AsyncMock(return_value=(False, "已点击按钮"))
    monkeypatch.setattr(plugin, "_execute_control_action", execute_mock)

    tool = ScreenClickTool(plugin=plugin)
    success, result = await tool.execute(x=12, y=34)

    assert success is True
    assert isinstance(result, dict)
    assert result["action_type"] == "click"
    assert result["action_inputs"]["start_box"] == [12.0, 34.0, 12.0, 34.0]
    assert execute_mock.await_args is not None
    executed_action = execute_mock.await_args.args[0]
    assert executed_action.action_type == "click"
    assert executed_action.action_inputs["start_box"] == [12.0, 34.0, 12.0, 34.0]


@pytest.mark.asyncio
async def test_screen_wait_tool_passes_seconds_to_action(monkeypatch: pytest.MonkeyPatch) -> None:
    """wait tool 应把等待秒数透传给统一 action inputs。"""

    plugin = cast(Any, ScreenUnderstandingPlugin(config=ScreenUnderstandingConfig()))
    execute_mock = AsyncMock(return_value=(False, "等待 2.5 秒后继续观察。"))
    monkeypatch.setattr(plugin, "_execute_control_action", execute_mock)

    tool = ScreenWaitTool(plugin=plugin)
    success, result = await tool.execute(seconds=2.5)

    assert success is True
    assert isinstance(result, dict)
    assert result["action_type"] == "wait"
    assert result["action_inputs"] == {"seconds": 2.5}
    assert execute_mock.await_args is not None
    executed_action = execute_mock.await_args.args[0]
    assert executed_action.action_type == "wait"
    assert executed_action.action_inputs == {"seconds": 2.5}


@pytest.mark.asyncio
async def test_control_agent_uses_suspend_bridge_and_active_window_context(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """每轮工具调用后应以 _SUSPEND_ 承接，并在下一轮 user payload 带上活跃窗口信息。"""

    plugin = cast(Any, ScreenUnderstandingPlugin(config=ScreenUnderstandingConfig()))
    plugin.bind_adapter(_AdapterStub())
    monkeypatch.setattr(
        plugin,
        "describe_active_window",
        AsyncMock(
            side_effect=[
                "id=5, app_id=code, title=VS Code",
                "id=14, app_id=chrome, title=Chrome",
                "id=14, app_id=chrome, title=Chrome",
            ]
        ),
    )

    agent = ScreenControlAgent(stream_id="stream-1", plugin=plugin)
    response = _FakeResponse(
        [
            [SimpleNamespace(id="call-1", name="tool-screen_click", args={"x": 10, "y": 20})],
            [SimpleNamespace(id="call-2", name="tool-screen_finish", args={"content": "完成"})],
        ]
    )
    request = _FakeRequest(response)
    monkeypatch.setattr(agent, "create_llm_request", lambda **_kwargs: request)
    monkeypatch.setattr(
        agent,
        "execute_local_usable",
        AsyncMock(
            return_value=(
                True,
                {
                    "kind": "action",
                    "action_type": "click",
                    "action_inputs": {"start_box": [10.0, 20.0, 10.0, 20.0]},
                    "execution_result": "已点击设置按钮",
                    "terminal": False,
                },
            )
        ),
    )

    success, result = await agent.execute(goal="打开设置", max_steps=3)

    assert success is True
    assert isinstance(result, dict)
    assert result["mode"] == "completed"
    initial_system_payload = cast(LLMPayload, request.payloads[0])
    initial_system_texts = [
        content.text for content in initial_system_payload.content if isinstance(content, Text)
    ]
    assert any("你是一个负责桌面屏幕操控的 GUI Agent。" in text for text in initial_system_texts)

    initial_user_payload = cast(LLMPayload, request.payloads[1])
    initial_texts = [content.text for content in initial_user_payload.content if isinstance(content, Text)]
    assert any("当前活跃窗口信息" in text and "app_id=code" in text for text in initial_texts)
    assert not any("你是一个负责桌面屏幕操控的 GUI Agent。" in text for text in initial_texts)

    assistant_payloads = [
        cast(LLMPayload, payload) for payload in response.payloads if isinstance(payload, LLMPayload)
    ]
    assert any(
        any(isinstance(content, Text) and content.text == "_SUSPEND_" for content in payload.content)
        for payload in assistant_payloads
        if payload.role == ROLE.ASSISTANT
    )

    chained_user_payloads = [payload for payload in assistant_payloads if payload.role == ROLE.USER]
    assert len(chained_user_payloads) == 1
    chained_user_texts = [
        content.text for content in chained_user_payloads[0].content if isinstance(content, Text)
    ]
    assert any("当前任务目标" in text and "打开设置" in text for text in chained_user_texts)
    assert any("当前活跃窗口信息" in text and "app_id=chrome" in text for text in chained_user_texts)
    assert any("坐标参考网格和刻度" in text for text in chained_user_texts)
    assert any("不要重新开始任务" in text for text in chained_user_texts)
    assert any("必须始终围绕上面的当前任务目标推进" in text for text in chained_user_texts)
    assert not any("你是一个负责桌面屏幕操控的 GUI Agent。" in text for text in chained_user_texts)


@pytest.mark.asyncio
async def test_control_agent_uses_dedicated_control_model_set(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Control agent 应使用 control 专用模型，而不是常规监控模型。"""

    plugin = cast(Any, ScreenUnderstandingPlugin(config=ScreenUnderstandingConfig()))
    adapter = _AdapterStub()
    plugin.bind_adapter(adapter)
    monkeypatch.setattr(plugin, "describe_active_window", AsyncMock(return_value="app_id=code,title=VS Code"))

    agent = ScreenControlAgent(stream_id="stream-1", plugin=plugin)
    response = _FakeResponse(
        [[SimpleNamespace(id="call-1", name="tool-screen_finish", args={"content": "完成"})]]
    )
    captured_kwargs: dict[str, Any] = {}

    def _fake_create_llm_request(**kwargs: Any) -> _FakeRequest:
        captured_kwargs.update(kwargs)
        return _FakeRequest(response)

    monkeypatch.setattr(agent, "create_llm_request", _fake_create_llm_request)

    success, result = await agent.execute(goal="打开设置", max_steps=1)

    assert success is True
    assert isinstance(result, dict)
    assert captured_kwargs["model_set"] is adapter._control_model_set
    assert captured_kwargs["model_set"] is not adapter._analysis_model_set


def test_control_agent_annotates_screenshot_with_coordinate_grid() -> None:
    """Agent 发图前应为截图叠加坐标网格，帮助模型稳定定位。"""

    from PIL import Image as PILImage

    image = PILImage.new("RGB", (120, 120), color="white")
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    raw_base64 = base64.b64encode(buffer.getvalue()).decode("utf-8")

    annotated_base64 = ScreenControlAgent._annotate_image_with_coordinate_grid(raw_base64)

    assert annotated_base64 != raw_base64
    with PILImage.open(io.BytesIO(base64.b64decode(annotated_base64))) as annotated_image:
        assert annotated_image.size == (120, 120)


@pytest.mark.asyncio
async def test_screen_analysis_falls_back_to_legacy_model_set(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """周期截图分析在旧路径只设置 _model_set 时仍应可用。"""

    plugin = cast(Any, ScreenUnderstandingPlugin(config=ScreenUnderstandingConfig()))
    adapter = cast(Any, ScreenUnderstandingAdapter.__new__(ScreenUnderstandingAdapter))
    adapter.plugin = plugin
    adapter._analysis_model_set = None
    adapter._model_set = object()
    adapter._keyframes = [SimpleNamespace(frame_id="history", png_base64="aGlzdG9yeQ==")]

    frame = SimpleNamespace(frame_id="current", png_base64="Y3VycmVudA==")
    response = _FakeResponse([])
    response.message = "最新画面"
    captured_kwargs: dict[str, Any] = {}

    def _fake_create_llm_request(model_set: Any, request_name: str) -> _FakeRequest:
        captured_kwargs["model_set"] = model_set
        captured_kwargs["request_name"] = request_name
        return _FakeRequest(response)

    monkeypatch.setattr(
        "plugins.screen_understanding.plugin.create_llm_request",
        _fake_create_llm_request,
    )

    description = await adapter._describe_keyframe_buffer(frame)

    assert description == "最新画面"
    assert captured_kwargs["model_set"] is adapter._model_set
    assert captured_kwargs["request_name"] == plugin.config.analysis.request_name


@pytest.mark.asyncio
async def test_control_agent_executes_multiple_tool_calls_in_one_round(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """当模型同轮返回多个 tool call 时，agent 应按顺序执行整批调用。"""

    plugin = cast(Any, ScreenUnderstandingPlugin(config=ScreenUnderstandingConfig()))
    plugin.bind_adapter(_AdapterStub())
    monkeypatch.setattr(plugin, "describe_active_window", AsyncMock(return_value="app_id=code,title=VS Code"))

    agent = ScreenControlAgent(stream_id="stream-1", plugin=plugin)
    response = _FakeResponse(
        [
            [
                SimpleNamespace(id="call-1", name="tool-screen_click", args={"x": 10, "y": 20}),
                SimpleNamespace(id="call-2", name="tool-screen_hover", args={"x": 30, "y": 40}),
            ],
            [SimpleNamespace(id="call-3", name="tool-screen_finish", args={"content": "完成"})],
        ]
    )
    request = _FakeRequest(response)
    execute_mock = AsyncMock(
        side_effect=[
            (
                True,
                {
                    "kind": "action",
                    "action_type": "click",
                    "action_inputs": {"start_box": [10.0, 20.0, 10.0, 20.0]},
                    "execution_result": "已点击设置按钮",
                    "terminal": False,
                },
            ),
            (
                True,
                {
                    "kind": "action",
                    "action_type": "hover",
                    "action_inputs": {"start_box": [30.0, 40.0, 30.0, 40.0]},
                    "execution_result": "已移动到输入框",
                    "terminal": False,
                },
            ),
        ]
    )
    monkeypatch.setattr(agent, "create_llm_request", lambda **_kwargs: request)
    monkeypatch.setattr(agent, "execute_local_usable", execute_mock)
    sleep_mock = AsyncMock()
    monkeypatch.setattr(
        "plugins.screen_understanding.agent.control_agent.asyncio.sleep",
        sleep_mock,
    )

    success, result = await agent.execute(goal="打开设置", max_steps=3)

    assert success is True
    assert isinstance(result, dict)
    assert result["mode"] == "completed"
    assert execute_mock.await_count == 2
    sleep_mock.assert_awaited_once_with(1.0)
    assert len(result["steps"]) == 2
    assert result["steps"][0]["action"] == "click"
    assert result["steps"][1]["action"] == "hover"

    tool_result_payloads = [
        cast(LLMPayload, payload)
        for payload in response.payloads
        if isinstance(payload, LLMPayload) and payload.role == ROLE.TOOL_RESULT
    ]
    assert len(tool_result_payloads) == 2
    second_result = cast(ToolResult, tool_result_payloads[1].content[0])
    assert second_result.call_id == "call-2"
    assert second_result.value["action"] == "hover"


@pytest.mark.asyncio
async def test_screen_close_window_tool_delegates_to_plugin(monkeypatch: pytest.MonkeyPatch) -> None:
    """关闭窗口工具应委托给插件侧的本地窗口关闭能力。"""

    plugin = cast(Any, ScreenUnderstandingPlugin(config=ScreenUnderstandingConfig()))
    close_mock = AsyncMock(return_value="已关闭窗口：id=5, app_id=code, title=VS Code")
    monkeypatch.setattr(plugin, "close_target_window", close_mock)

    tool = ScreenCloseWindowTool(plugin=plugin)
    success, result = await tool.execute("VS Code")

    assert success is True
    assert result["action_type"] == "close_window"
    assert result["action_inputs"] == {"target": "VS Code"}
    assert "已关闭窗口" in result["execution_result"]


@pytest.mark.asyncio
async def test_screen_launch_program_tool_delegates_to_plugin(monkeypatch: pytest.MonkeyPatch) -> None:
    """启动程序工具应委托给插件侧的本地程序唤起能力。"""

    plugin = cast(Any, ScreenUnderstandingPlugin(config=ScreenUnderstandingConfig()))
    launch_mock = AsyncMock(return_value="已启动程序：firefox")
    monkeypatch.setattr(plugin, "launch_program", launch_mock)

    tool = ScreenLaunchProgramTool(plugin=plugin)
    success, result = await tool.execute("firefox")

    assert success is True
    assert result["action_type"] == "launch_program"
    assert result["action_inputs"] == {"command": "firefox"}
    assert result["execution_result"] == "已启动程序：firefox"


def test_context_manager_allows_assistant_bridge_between_tool_result_and_user() -> None:
    """tool_result 与下一轮 user 之间插入 assistant 承接后，上下文应合法。"""

    manager = LLMContextManager(max_payloads=20)
    payloads = [LLMPayload(ROLE.USER, Text("关闭窗口"))]

    payloads = manager.add_payload(
        payloads,
        LLMPayload(
            ROLE.ASSISTANT,
            [
                Text("开始执行"),
                ToolCall(id="call_1", name="tool-screen_click", args={"x": 10, "y": 20}),
            ],
        ),
    )
    payloads = manager.add_payload(
        payloads,
        LLMPayload(
            ROLE.TOOL_RESULT,
            ToolResult(value="已点击关闭按钮", call_id="call_1", name="tool-screen_click"),
        ),
    )
    payloads = manager.add_payload(payloads, LLMPayload(ROLE.ASSISTANT, Text("")))
    payloads = manager.add_payload(payloads, LLMPayload(ROLE.USER, Text("最新截图显示窗口仍在")))

    assert [payload.role for payload in payloads[-4:]] == [
        ROLE.USER,
        ROLE.ASSISTANT,
        ROLE.TOOL_RESULT,
        ROLE.USER,
    ] or [payload.role for payload in payloads[-5:]] == [
        ROLE.USER,
        ROLE.ASSISTANT,
        ROLE.TOOL_RESULT,
        ROLE.ASSISTANT,
        ROLE.USER,
    ]
    manager.validate_for_send(payloads)


@pytest.mark.asyncio
async def test_execute_control_action_uses_backend_strategy(monkeypatch: pytest.MonkeyPatch) -> None:
    """Plugin 控制动作执行应委托给解析出的 backend。"""

    plugin = cast(Any, ScreenUnderstandingPlugin(config=ScreenUnderstandingConfig()))

    class _BackendStub:
        backend_name = "stub"

        async def execute_action(self, action: ScreenControlAction) -> str:
            return f"stub:{action.action_type}"

    async def _fake_backend(*_args: object, **_kwargs: object) -> _BackendStub:
        return _BackendStub()

    monkeypatch.setattr(
        "plugins.screen_understanding.plugin.get_first_available_control_backend",
        _fake_backend,
    )

    terminal, result = await plugin._execute_control_action(
        ScreenControlAction(
            thought="点击一下",
            action_type="click",
            action_inputs={"start_box": [1, 2, 1, 2]},
            raw_text="Action: click(start_box='(1,2)')",
        )
    )

    assert terminal is False
    assert result == "stub:click"


@pytest.mark.asyncio
async def test_execute_control_action_wait_uses_requested_seconds(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Plugin wait 动作应按传入秒数暂停，而不是总是固定时长。"""

    plugin = cast(Any, ScreenUnderstandingPlugin(config=ScreenUnderstandingConfig()))
    sleep_mock = AsyncMock()
    monkeypatch.setattr("plugins.screen_understanding.plugin.asyncio.sleep", sleep_mock)

    terminal, result = await plugin._execute_control_action(
        ScreenControlAction(
            thought="等待页面加载",
            action_type="wait",
            action_inputs={"seconds": 2.5},
            raw_text="tool:wait",
        )
    )

    assert terminal is False
    assert result == "等待 2.5 秒后继续观察。"
    sleep_mock.assert_awaited_once_with(2.5)


def test_get_components_exposes_agent_instead_of_service_tool() -> None:
    """插件导出应包含 Adapter 和 Agent，而非旧 service/tool 组合。"""

    plugin = cast(Any, ScreenUnderstandingPlugin(config=ScreenUnderstandingConfig()))

    component_names = {component.__name__ for component in plugin.get_components()}

    assert component_names == {"ScreenUnderstandingAdapter", "ScreenControlAgent"}


def test_build_control_backend_candidates_respects_platform_aliases(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """backend 列表应把 auto/local_desktop 展开成平台相关候选。"""

    monkeypatch.setattr("plugins.screen_understanding.control_backends.sys.platform", "linux")

    candidates = build_control_backend_candidates(
        ["auto", "windows_uia", "xdotool", "xdotool"],
        command_timeout_seconds=1.0,
    )

    assert [candidate.backend_name for candidate in candidates] == ["wdotool", "ydotool", "xdotool", "windows_uia"]


def test_parse_control_response_supports_ui_tars_box_tokens() -> None:
    """parser 应兼容 UI-TARS 常见的 box token 坐标格式。"""

    action = parse_control_response(
        "Thought: 点击按钮\nAction: click(start_box='<|box_start|>(200,300)<|box_end|>')"
    )

    assert action.action_type == "click"
    assert action.action_inputs["start_box"] == [200.0, 300.0, 200.0, 300.0]


def test_xdotool_backend_extract_action_point_accepts_two_value_points() -> None:
    """xdotool backend 应兼容两元素 point 坐标，而不只接受四元素 box。"""

    backend = XdotoolControlBackend(command_timeout_seconds=1.0)

    point = backend._extract_action_point({"start_box": [200, 300]}, "start_box")

    assert point == (200, 300)


@pytest.mark.asyncio
async def test_wdotool_backend_keyboard_actions_do_not_require_start_box() -> None:
    """wdotool backend 的键盘动作不应错误依赖 start_box。"""

    backend = WdotoolControlBackend(command_timeout_seconds=1.0)

    with patch.object(backend, "_run_command", new=AsyncMock()) as run_command:
        result = await backend.execute_action(
            ScreenControlAction(
                thought="",
                action_type="hotkey",
                action_inputs={"key": "Ctrl+L"},
                raw_text="tool:hotkey",
            )
        )

    run_command.assert_awaited_once_with("wdotool", "key", "--clearmodifiers", "Ctrl+L")
    assert result == "已执行快捷键 Ctrl+L。"


@pytest.mark.asyncio
async def test_wdotool_backend_normalizes_enter_hotkey_alias() -> None:
    """wdotool backend 应将 Enter 这类常见别名转换成可识别的 keysym。"""

    backend = WdotoolControlBackend(command_timeout_seconds=1.0)

    with patch.object(backend, "_run_command", new=AsyncMock()) as run_command:
        result = await backend.execute_action(
            ScreenControlAction(
                thought="",
                action_type="hotkey",
                action_inputs={"key": "Enter"},
                raw_text="tool:hotkey",
            )
        )

    run_command.assert_awaited_once_with("wdotool", "key", "--clearmodifiers", "Return")
    assert result == "已执行快捷键 Enter。"


@pytest.mark.asyncio
async def test_wdotool_backend_type_uses_stdin_file_interface() -> None:
    """wdotool backend 文本输入应走 stdin/file 接口以减少掉字。"""

    backend = WdotoolControlBackend(command_timeout_seconds=1.0)

    with patch.object(backend, "_run_command_with_input", new=AsyncMock()) as run_command:
        result = await backend.execute_action(
            ScreenControlAction(
                thought="",
                action_type="type",
                action_inputs={"content": "github.com"},
                raw_text="tool:type",
            )
        )

    run_command.assert_awaited_once_with(
        "wdotool",
        "type",
        "--clearmodifiers",
        "--delay",
        str(backend._type_delay_ms),
        "--file",
        "-",
        input_text="github.com",
    )
    assert result == "已输入文本：github.com"


@pytest.mark.asyncio
async def test_xdotool_backend_keyboard_actions_do_not_require_start_box() -> None:
    """xdotool backend 的键盘动作不应错误依赖 start_box。"""

    backend = XdotoolControlBackend(command_timeout_seconds=1.0)

    with patch.object(backend, "_run_command", new=AsyncMock()) as run_command:
        result = await backend.execute_action(
            ScreenControlAction(
                thought="",
                action_type="type",
                action_inputs={"content": "github.com"},
                raw_text="tool:type",
            )
        )

    run_command.assert_awaited_once_with("xdotool", "type", "--delay", "1", "github.com")
    assert result == "已输入文本：github.com"


@pytest.mark.asyncio
async def test_ydotool_backend_keyboard_actions_do_not_require_start_box() -> None:
    """ydotool backend 的键盘动作不应错误依赖 start_box。"""

    backend = YdotoolControlBackend(command_timeout_seconds=1.0)

    with patch.object(backend, "_run_command", new=AsyncMock()) as run_command:
        result = await backend.execute_action(
            ScreenControlAction(
                thought="",
                action_type="hotkey",
                action_inputs={"key": "Enter"},
                raw_text="tool:hotkey",
            )
        )

    run_command.assert_awaited_once_with("ydotool", "key", "28:1", "28:0")
    assert result == "已执行快捷键 Enter。"


@pytest.mark.asyncio
async def test_ydotool_backend_click_waits_for_pointer_to_settle() -> None:
    """ydotool backend 点击前应等待鼠标落点稳定，避免被窗口管理器误判为拖拽。"""

    backend = YdotoolControlBackend(command_timeout_seconds=1.0)
    call_order: list[str] = []

    async def fake_move_absolute(x: int, y: int) -> None:
        assert (x, y) == (200, 300)
        call_order.append("move")

    async def fake_sleep(seconds: float) -> None:
        assert seconds == backend._pointer_settle_seconds
        call_order.append("sleep")

    async def fake_run_command(*command: str) -> None:
        assert command == (
            "ydotool",
            "click",
            "--next-delay",
            str(backend._pointer_click_event_delay_ms),
            "0xC0",
        )
        call_order.append("click")

    with patch.object(backend, "_move_absolute", new=fake_move_absolute), patch(
        "plugins.screen_understanding.control_backends.ydotool_backend.asyncio.sleep",
        new=fake_sleep,
    ), patch.object(backend, "_run_command", new=fake_run_command):
        result = await backend.execute_action(
            ScreenControlAction(
                thought="",
                action_type="click",
                action_inputs={"start_box": [200.0, 300.0, 200.0, 300.0]},
                raw_text="tool:click",
            )
        )

    assert call_order == ["move", "sleep", "click"]
    assert result == "已在 (200, 300) 执行左键单击。"


@pytest.mark.asyncio
async def test_xdotool_backend_is_available_requires_runtime_probe() -> None:
    """xdotool backend 应在真实探测失败时视为不可用。"""

    backend = XdotoolControlBackend(command_timeout_seconds=1.0)

    success_proc = AsyncMock()
    success_proc.returncode = 0
    success_proc.communicate = AsyncMock(return_value=(b"x:10 y:20 screen:0 window:1\n", b""))

    with patch("shutil.which", return_value="/usr/sbin/xdotool"), patch(
        "asyncio.create_subprocess_exec",
        return_value=success_proc,
    ):
        assert await backend.is_available() is True

    failing_proc = AsyncMock()
    failing_proc.returncode = 1
    failing_proc.communicate = AsyncMock(return_value=(b"", b"Error: Can't open display\n"))

    with patch("shutil.which", return_value="/usr/sbin/xdotool"), patch(
        "asyncio.create_subprocess_exec",
        return_value=failing_proc,
    ):
        assert await backend.is_available() is False


@pytest.mark.asyncio
async def test_wdotool_backend_is_available_requires_info_probe() -> None:
    """wdotool backend 应在 info 探测成功时视为可用。"""

    backend = WdotoolControlBackend(command_timeout_seconds=1.0)

    success_proc = AsyncMock()
    success_proc.returncode = 0
    success_proc.communicate = AsyncMock(return_value=(b"backend: kde\n", b""))

    with patch("shutil.which", return_value="/usr/sbin/wdotool"), patch(
        "asyncio.create_subprocess_exec",
        return_value=success_proc,
    ):
        assert await backend.is_available() is True

    failing_proc = AsyncMock()
    failing_proc.returncode = 1
    failing_proc.communicate = AsyncMock(return_value=(b"", b"backend unavailable\n"))

    with patch("shutil.which", return_value="/usr/sbin/wdotool"), patch(
        "asyncio.create_subprocess_exec",
        return_value=failing_proc,
    ):
        assert await backend.is_available() is False


@pytest.mark.asyncio
async def test_ydotool_backend_is_available_requires_daemon_connection() -> None:
    """ydotool backend 需要 client 可连接到 daemon socket 才视为可用。"""

    backend = YdotoolControlBackend(command_timeout_seconds=1.0)

    success_proc = AsyncMock()
    success_proc.returncode = 0
    success_proc.communicate = AsyncMock(return_value=(b"fd_daemon_socket: 3\n", b""))

    with patch("shutil.which", return_value="/usr/sbin/ydotool"), patch(
        "asyncio.create_subprocess_exec",
        return_value=success_proc,
    ):
        assert await backend.is_available() is True

    failing_proc = AsyncMock()
    failing_proc.returncode = 2
    failing_proc.communicate = AsyncMock(
        return_value=(
            b"failed to connect socket '/run/user/1000/.ydotool_socket': No such file or directory\n",
            b"",
        )
    )

    with patch("shutil.which", return_value="/usr/sbin/ydotool"), patch(
        "asyncio.create_subprocess_exec",
        return_value=failing_proc,
    ):
        assert await backend.is_available() is False