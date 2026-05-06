"""测试 MCPManager 的客户端工具接入行为。"""

from __future__ import annotations

import asyncio
from collections.abc import Generator
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, Mock

import pytest
from mcp.types import CallToolResult, TextContent, Tool

from src.core.components.registry import get_global_registry
from src.core.components.state_manager import get_global_state_manager
from src.core.components.types import ComponentState
from src.core.config.mcp_config import MCPConfig
from src.core.managers.tool_manager.mcp_manager import MCPManager
from src.kernel.concurrency import get_task_manager


@pytest.fixture(autouse=True)
def clear_component_state() -> Generator[None, None, None]:
    """隔离全局组件注册表和状态管理器。"""
    registry = get_global_registry()
    state_manager = get_global_state_manager()
    registry.clear()
    state_manager.clear()
    yield
    registry.clear()
    state_manager.clear()


def make_tool(name: str = "lookup") -> Tool:
    """创建用于测试的 MCP Tool 对象。"""
    return Tool(
        name=name,
        description="Lookup information",
        inputSchema={
            "type": "object",
            "properties": {"query": {"type": "string"}},
            "required": ["query"],
        },
    )


@pytest.mark.asyncio
async def test_discover_tools_registers_active_dynamic_tool() -> None:
    """发现 MCP 工具后应注册为可被 Chatter 收集的 ACTIVE Tool。"""
    manager = MCPManager()
    session = MagicMock()
    session.list_tools = AsyncMock(return_value=SimpleNamespace(tools=[make_tool()]))

    await manager._discover_tools("demo", session)

    signature = "mcp_provider:tool:mcp-demo-lookup"
    registry = get_global_registry()
    state_manager = get_global_state_manager()
    tool_cls = registry.get(signature)

    assert tool_cls is not None
    assert tool_cls.get_signature() == signature
    assert state_manager.get_state(signature) == ComponentState.ACTIVE
    assert tool_cls.to_schema()["function"]["name"] == "mcp-demo-lookup"


@pytest.mark.asyncio
async def test_discover_tools_normalizes_underscores_to_hyphens() -> None:
    """MCP 暴露给 LLM 的工具名应统一使用短横线。"""
    manager = MCPManager()
    session = MagicMock()
    session.list_tools = AsyncMock(
        return_value=SimpleNamespace(tools=[make_tool("list_allowed_directories")])
    )

    await manager._discover_tools("file_system", session)

    signature = "mcp_provider:tool:mcp-file-system-list-allowed-directories"
    tool_cls = get_global_registry().get(signature)

    assert tool_cls is not None
    assert tool_cls.to_schema()["function"]["name"] == "mcp-file-system-list-allowed-directories"


@pytest.mark.asyncio
async def test_dynamic_tool_executes_underlying_mcp_call() -> None:
    """动态 Tool 执行时应委托到底层 MCP session.call_tool。"""
    manager = MCPManager()
    session = MagicMock()
    session.list_tools = AsyncMock(return_value=SimpleNamespace(tools=[make_tool()]))
    session.call_tool = AsyncMock(
        return_value=CallToolResult(
            content=[TextContent(type="text", text="found")],
            isError=False,
        )
    )
    manager._sessions["demo"] = session

    await manager._discover_tools("demo", session)

    tool_cls = get_global_registry().get("mcp_provider:tool:mcp-demo-lookup")
    assert tool_cls is not None

    tool = tool_cls(MagicMock())
    ok, result = await tool.execute(query="paper")

    assert ok is True
    assert result == "found"
    session.call_tool.assert_awaited_once_with("lookup", {"query": "paper"})


@pytest.mark.asyncio
async def test_cleanup_unregisters_dynamic_tools_and_states() -> None:
    """清理 MCPManager 时应注销动态工具并移除组件状态。"""
    manager = MCPManager()
    session = MagicMock()
    session.list_tools = AsyncMock(return_value=SimpleNamespace(tools=[make_tool()]))

    await manager._discover_tools("demo", session)
    await manager.cleanup()

    signature = "mcp_provider:tool:mcp-demo-lookup"
    assert get_global_registry().get(signature) is None
    assert get_global_state_manager().get_state(signature) == ComponentState.UNLOADED
    assert manager._adapters == {}
    assert manager._tool_signatures == set()


@pytest.mark.asyncio
async def test_cleanup_ignores_transport_exception_group() -> None:
    """MCP 传输关闭异常不应导致 Bot 关闭失败。"""
    manager = MCPManager()
    manager._sessions["broken"] = MagicMock()
    manager._adapters["mcp-broken-lookup"] = MagicMock()
    manager._tool_signatures.add("mcp_provider:tool:mcp-broken-lookup")
    manager._exit_stack.aclose = AsyncMock(
        side_effect=ExceptionGroup("unhandled errors in a TaskGroup", [RuntimeError("broken remote")])
    )
    get_global_registry().register(Mock, "mcp_provider:tool:mcp-broken-lookup")
    get_global_state_manager().set_state(
        "mcp_provider:tool:mcp-broken-lookup",
        ComponentState.ACTIVE,
    )

    await manager.cleanup()

    assert manager._sessions == {}
    assert manager._adapters == {}
    assert manager._tool_signatures == set()
    assert get_global_registry().get("mcp_provider:tool:mcp-broken-lookup") is None
    assert (
        get_global_state_manager().get_state("mcp_provider:tool:mcp-broken-lookup")
        == ComponentState.UNLOADED
    )


@pytest.mark.asyncio
async def test_streamable_http_connect_ignores_session_cancel() -> None:
    """远端 HTTP MCP 初始化取消时不应打断 Bot 初始化。"""
    manager = MCPManager()
    manager._exit_stack.enter_async_context = AsyncMock(return_value=(MagicMock(), MagicMock()))
    manager._connect_session = AsyncMock(side_effect=asyncio.CancelledError("cancel scope"))

    ok = await manager.connect_streamable_http_server(
        name="remote",
        url="https://example.com/mcp",
        timeout=0,
    )

    assert ok is False


@pytest.mark.asyncio
async def test_initialize_isolates_cancelled_remote_connection(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """MCP remote 连接取消不应污染 initialize 调用方任务。"""
    manager = MCPManager()
    config = MCPConfig(
        mcp=MCPConfig.MCPSection(
            streamable_http_servers={"remote": "https://example.com/mcp"}
        )
    )
    monkeypatch.setattr("src.core.config.get_mcp_config", lambda: config)
    manager.connect_streamable_http_server_from_config = AsyncMock(
        side_effect=asyncio.CancelledError("cancel scope")
    )

    await manager.initialize()

    get_task_manager().cleanup_tasks()


@pytest.mark.asyncio
async def test_http_config_helpers_reject_missing_url() -> None:
    """HTTP 类 MCP 配置缺少 URL 时应返回失败而不是抛异常。"""
    manager = MCPManager()

    assert await manager.connect_sse_server_from_config("bad_sse", {}) is False
    assert await manager.connect_streamable_http_server_from_config("bad_http", {}) is False