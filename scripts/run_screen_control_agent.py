#!/usr/bin/env python3
"""手动运行 screen_understanding 控制 Agent 的调试脚本。

该脚本会在当前仓库内直接装配 ScreenUnderstandingPlugin、
ScreenUnderstandingAdapter 与 ScreenControlAgent，随后按命令行参数
执行一次完整的截图驱动控制流程，方便本地调试 agent 工作流。

示例：
    uv run python scripts/run_screen_control_agent.py \
        --goal "打开 Firefox 并访问 GitHub" \
        --max-steps 6

    uv run python scripts/run_screen_control_agent.py \
        --goal "关闭当前窗口" \
        --capture-backend wf-recorder \
        --control-backend ydotool \
        --print-config
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from collections import deque
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any, Callable, Coroutine, cast

from mofox_wire import MessageEnvelope

# 允许脚本在仓库根目录外触发时仍可导入项目模块
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from plugins.screen_understanding.agent import ScreenControlAgent
from plugins.screen_understanding.config import ScreenUnderstandingConfig
from plugins.screen_understanding.plugin import ScreenUnderstandingAdapter
from plugins.screen_understanding.plugin import ScreenUnderstandingPlugin
from plugins.screen_understanding.plugin import _validate_config
from src.app.plugin_system.api.llm_api import get_model_set_by_name
from src.core.config.model_config import init_model_config
from src.kernel.logger import get_logger

logger = get_logger("scripts.run_screen_control_agent")


class _NullCoreSink:
    """最小 core sink，占位满足 adapter 构造要求。"""

    def __init__(self) -> None:
        self._outgoing_handlers: set[Callable[[MessageEnvelope], Coroutine[Any, Any, None]]] = set()

    def set_outgoing_handler(
        self,
        handler: Callable[[MessageEnvelope], Coroutine[Any, Any, None]] | None,
    ) -> None:
        """Register one outgoing handler; kept only for protocol compatibility."""

        if handler is not None:
            self._outgoing_handlers.add(handler)

    def remove_outgoing_handler(
        self,
        handler: Callable[[MessageEnvelope], Coroutine[Any, Any, None]],
    ) -> None:
        """Remove one previously registered outgoing handler."""

        self._outgoing_handlers.discard(handler)

    async def send(self, message: MessageEnvelope) -> None:
        """忽略 adapter 发送到 core 的消息。"""

        del message

    async def send_many(self, messages: list[MessageEnvelope]) -> None:
        """Ignore batches of incoming messages from the adapter."""

        del messages

    async def push_outgoing(self, envelope: MessageEnvelope) -> None:
        """Ignore outgoing messages because this debug script has no core loop."""

        del envelope

    async def close(self) -> None:
        """Release no-op sink resources."""

        self._outgoing_handlers.clear()


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for one agent debug run."""

    parser = argparse.ArgumentParser(
        description="手动运行 screen_understanding 的 ScreenControlAgent",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "示例:\n"
            "  uv run python scripts/run_screen_control_agent.py --goal \"打开设置\"\n"
            "  uv run python scripts/run_screen_control_agent.py --goal \"关闭当前窗口\" --max-steps 3\n"
            "  uv run python scripts/run_screen_control_agent.py --goal \"启动 Firefox\" --control-backend ydotool --capture-backend wf-recorder\n"
        ),
    )
    parser.add_argument("--goal", required=True, help="本次 agent 要完成的桌面任务目标")
    parser.add_argument("--max-steps", type=int, default=8, help="允许 agent 执行的最大轮数")
    parser.add_argument(
        "--stream-id",
        default="screen-control-debug",
        help="传给 ScreenControlAgent 的 stream_id",
    )
    parser.add_argument(
        "--session-id",
        default="manual-debug-session",
        help="写入 agent 结果中的调试 session_id",
    )
    parser.add_argument(
        "--model-name",
        default="",
        help="覆盖 config 中的 control.model_name",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=None,
        help="覆盖 config 中的 control.temperature",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=None,
        help="覆盖 config 中的 control.max_tokens",
    )
    parser.add_argument(
        "--capture-backend",
        action="append",
        default=None,
        help="覆盖截图后端优先级；可重复传入，例如 --capture-backend wf-recorder --capture-backend mss",
    )
    parser.add_argument(
        "--control-backend",
        action="append",
        default=None,
        help="覆盖控制后端优先级；可重复传入，例如 --control-backend ydotool --control-backend xdotool",
    )
    parser.add_argument("--display-index", type=int, default=None, help="覆盖截图显示器索引")
    parser.add_argument(
        "--backend-timeout",
        type=float,
        default=None,
        help="覆盖截图/控制命令超时时间（秒）",
    )
    parser.add_argument(
        "--portal-parent-window",
        default=None,
        help="覆盖 portal_parent_window，便于 Wayland portal 调试",
    )
    parser.add_argument(
        "--config-path",
        default="",
        help="从指定 TOML 路径加载 screen_understanding 配置；为空时使用插件默认配置路径",
    )
    parser.add_argument(
        "--model-config-path",
        default="",
        help="模型配置文件路径；为空时默认使用 config/model.toml",
    )
    parser.add_argument(
        "--print-config",
        action="store_true",
        help="执行前打印实际生效的 screen_understanding 配置摘要",
    )
    parser.add_argument(
        "--output",
        choices=["pretty", "json"],
        default="pretty",
        help="结果输出格式",
    )
    parser.add_argument(
        "--result-path",
        default="",
        help="将完整结果 JSON 写入指定文件路径",
    )
    return parser.parse_args()


def _load_config(args: argparse.Namespace) -> ScreenUnderstandingConfig:
    """Load and override screen_understanding config for this debug run."""

    if args.config_path:
        config = ScreenUnderstandingConfig.load(Path(args.config_path), auto_update=True)
    else:
        config = ScreenUnderstandingConfig.load_for_plugin("screen_understanding", auto_generate=True)

    if args.model_name:
        config.control.model_name = args.model_name.strip()
    if args.temperature is not None:
        config.control.temperature = args.temperature
    if args.max_tokens is not None:
        config.control.max_tokens = args.max_tokens
    if args.capture_backend:
        config.capture.backend_preference = [item.strip() for item in args.capture_backend if item.strip()]
    if args.control_backend:
        config.control.executor_backends = [item.strip() for item in args.control_backend if item.strip()]
    if args.display_index is not None:
        config.capture.display_index = args.display_index
    if args.backend_timeout is not None:
        config.capture.backend_timeout_seconds = args.backend_timeout
    if args.portal_parent_window is not None:
        config.capture.portal_parent_window = args.portal_parent_window

    _validate_config(config)
    return config


def _config_summary(config: ScreenUnderstandingConfig) -> dict[str, Any]:
    """Build a compact config summary suitable for CLI display."""

    return {
        "analysis": {
            "model_name": config.analysis.model_name,
            "temperature": config.analysis.temperature,
            "max_tokens": config.analysis.max_tokens,
        },
        "capture": {
            "display_index": config.capture.display_index,
            "backend_preference": list(config.capture.backend_preference),
            "backend_timeout_seconds": config.capture.backend_timeout_seconds,
            "portal_parent_window": config.capture.portal_parent_window,
        },
        "control": {
            "model_name": config.control.model_name,
            "temperature": config.control.temperature,
            "max_tokens": config.control.max_tokens,
            "executor_backends": list(config.control.executor_backends),
            "default_executor_backend": config.control.default_executor_backend,
        },
    }


def _resolve_model_config_path(args: argparse.Namespace) -> Path:
    """Resolve the model config path used by get_model_set_by_name."""

    if args.model_config_path:
        return Path(args.model_config_path).expanduser().resolve()
    return (PROJECT_ROOT / "config" / "model.toml").resolve()


def _serialize_result(value: Any) -> Any:
    """Convert result payloads into JSON-serializable objects."""

    if is_dataclass(value) and not isinstance(value, type):
        return {key: _serialize_result(item) for key, item in asdict(value).items()}
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, deque):
        return [_serialize_result(item) for item in value]
    if isinstance(value, dict):
        return {str(key): _serialize_result(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_serialize_result(item) for item in value]
    return value


def _print_result(result: dict[str, Any], output_mode: str) -> None:
    """Print agent result in the requested CLI format."""

    if output_mode == "json":
        print(json.dumps(_serialize_result(result), ensure_ascii=False, indent=2))
        return

    print("=== Screen Control Agent Result ===")
    print(f"success: {result.get('success')}")
    print(f"mode: {result.get('payload', {}).get('mode')}")
    print(f"goal: {result.get('payload', {}).get('goal')}")
    print(f"message: {result.get('payload', {}).get('message')}")
    print(f"steps: {len(result.get('payload', {}).get('steps', []))}")
    print(json.dumps(_serialize_result(result), ensure_ascii=False, indent=2))


async def _run(args: argparse.Namespace) -> int:
    """Assemble plugin pieces and execute one control-agent run."""

    config = _load_config(args)
    model_config_path = _resolve_model_config_path(args)
    init_model_config(str(model_config_path))

    if args.print_config:
        print(
            json.dumps(
                {
                    "screen_understanding": _config_summary(config),
                    "model_config_path": str(model_config_path),
                },
                ensure_ascii=False,
                indent=2,
            )
        )

    plugin = cast(Any, ScreenUnderstandingPlugin(config=config))
    adapter = ScreenUnderstandingAdapter(cast(Any, _NullCoreSink()), plugin=plugin)
    plugin.bind_adapter(adapter)
    adapter._analysis_model_set = get_model_set_by_name(
        config.analysis.model_name,
        temperature=config.analysis.temperature,
        max_tokens=config.analysis.max_tokens,
    )
    adapter._control_model_set = get_model_set_by_name(
        config.control.model_name,
        temperature=config.control.temperature,
        max_tokens=config.control.max_tokens,
    )
    adapter._model_set = adapter._analysis_model_set
    adapter._keyframes = deque(maxlen=config.analysis.keyframe_buffer_size)

    agent = ScreenControlAgent(stream_id=args.stream_id, plugin=plugin)
    logger.info(
        f"开始运行 ScreenControlAgent 调试脚本，goal={args.goal}, max_steps={args.max_steps}"
    )

    try:
        success, payload = await agent.execute(
            goal=args.goal,
            max_steps=args.max_steps,
            session_id=args.session_id,
        )
    finally:
        plugin.unbind_adapter(adapter)

    result = {
        "success": success,
        "payload": _serialize_result(payload),
        "config": _config_summary(config),
    }

    if args.result_path:
        result_path = Path(args.result_path)
        result_path.parent.mkdir(parents=True, exist_ok=True)
        result_path.write_text(
            json.dumps(result, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        logger.info(f"调试结果已写入 {result_path}")

    _print_result(result, args.output)
    return 0 if success else 1


def main() -> None:
    """Script entry point."""

    try:
        raise SystemExit(asyncio.run(_run(parse_args())))
    except KeyboardInterrupt:
        logger.warning("已手动中断 ScreenControlAgent 调试运行。")
        raise SystemExit(130)


if __name__ == "__main__":
    main()