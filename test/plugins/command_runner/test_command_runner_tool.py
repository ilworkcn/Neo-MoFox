"""command_runner 插件测试。"""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any, cast
from unittest.mock import AsyncMock, Mock, patch

import pytest

from plugins.command_runner.config import CommandRunnerConfig
from plugins.command_runner.plugin import CommandRunnerPlugin
from plugins.command_runner.tools import RunCommandTool


def _build_plugin() -> CommandRunnerPlugin:
    """创建测试用插件实例。"""

    return CommandRunnerPlugin(config=CommandRunnerConfig())


def test_get_components_respects_enabled_flag() -> None:
    """禁用插件时不应暴露工具组件。"""

    config = CommandRunnerConfig()
    config.plugin.enabled = False
    plugin = CommandRunnerPlugin(config=config)

    assert plugin.get_components() == []


@pytest.mark.asyncio
async def test_run_command_executes_non_risky_command() -> None:
    """普通命令应直接执行。"""

    plugin = cast(Any, _build_plugin())
    tool = RunCommandTool(plugin=cast(Any, plugin))

    process = AsyncMock()
    process.returncode = 0
    process.communicate = AsyncMock(return_value=(b"Neo-MoFox\n", b""))

    with patch("asyncio.create_subprocess_exec", return_value=process) as create_mock:
        success, result = await tool.execute("echo", ["Neo-MoFox"])
    payload = cast(dict[str, Any], result)

    assert success is True
    assert payload["exit_code"] == 0
    assert payload["stdout"] == "Neo-MoFox\n"
    create_mock.assert_awaited_once()


@pytest.mark.asyncio
async def test_risky_command_is_blocked_without_allowlist() -> None:
    """风险命令在未命中白名单时必须拒绝执行。"""

    plugin = cast(Any, _build_plugin())
    tool = RunCommandTool(plugin=cast(Any, plugin))

    success, result = await tool.execute("python", "-V")
    payload = cast(dict[str, Any], result)

    assert success is False
    assert payload["risky"] is True
    assert "白名单" in payload["reason"]


@pytest.mark.asyncio
async def test_risky_command_is_allowed_when_whitelisted() -> None:
    """风险命令命中白名单后应允许执行。"""

    plugin = cast(Any, _build_plugin())
    plugin.typed_config.policy.allow_executables = ["python"]
    tool = RunCommandTool(plugin=cast(Any, plugin))

    process = AsyncMock()
    process.returncode = 0
    process.communicate = AsyncMock(return_value=(b"Python 3.11.0\n", b""))

    with patch("asyncio.create_subprocess_exec", return_value=process):
        success, result = await tool.execute("python", ["-V"])
    payload = cast(dict[str, Any], result)

    assert success is True
    assert payload["risky"] is True
    assert any(rule.startswith("allow_executables:") for rule in payload["matched_rules"])


@pytest.mark.asyncio
async def test_blacklist_takes_precedence_over_whitelist() -> None:
    """命中黑名单时应优先拒绝，即使同时命中白名单。"""

    plugin = cast(Any, _build_plugin())
    plugin.typed_config.policy.allow_executables = ["python"]
    plugin.typed_config.policy.block_executables = ["python"]
    tool = RunCommandTool(plugin=cast(Any, plugin))

    success, result = await tool.execute("python", ["-V"])
    payload = cast(dict[str, Any], result)

    assert success is False
    assert "黑名单" in payload["reason"]


@pytest.mark.asyncio
async def test_working_directory_must_stay_inside_workspace() -> None:
    """工作目录越界时应拒绝执行。"""

    plugin = cast(Any, _build_plugin())
    tool = RunCommandTool(plugin=cast(Any, plugin))
    outside_dir = str(Path(plugin.workspace_root).resolve().parent)

    success, result = await tool.execute("echo", ["Neo-MoFox"], working_directory=outside_dir)

    assert success is False
    assert result == "工作目录必须位于项目根目录内"


@pytest.mark.asyncio
async def test_command_timeout_returns_failure_payload() -> None:
    """超时后应杀掉子进程并返回失败信息。"""

    plugin = cast(Any, _build_plugin())
    tool = RunCommandTool(plugin=cast(Any, plugin))

    process = AsyncMock()
    process.returncode = None
    process.communicate = AsyncMock(side_effect=[asyncio.TimeoutError(), (b"partial", b"timeout")])
    process.kill = Mock()

    with patch("asyncio.create_subprocess_exec", return_value=process):
        success, result = await tool.execute("echo", ["Neo-MoFox"], timeout_seconds=0.01)
    payload = cast(dict[str, Any], result)

    assert success is False
    assert "超时" in payload["reason"]
    process.kill.assert_called_once()