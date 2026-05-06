"""SignalHandler 运行时信号处理测试。"""

from __future__ import annotations

import signal
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from src.app.runtime.signal_handler import SignalHandler


def test_register_and_restore_signal_handlers() -> None:
    """测试注册信号时会保存旧处理器，并可恢复。"""
    calls: list[tuple[int, object]] = []

    def fake_signal(signum: int, handler: object) -> signal.Handlers:
        calls.append((signum, handler))
        return signal.SIG_DFL

    bot = SimpleNamespace(logger=MagicMock(), _running=True)
    handler = SignalHandler(bot)  # type: ignore[arg-type]

    with patch("src.app.runtime.signal_handler.signal.signal", side_effect=fake_signal):
        handler.register_signals()
        assert handler._original_handlers[signal.SIGINT] == signal.SIG_DFL

        handler.restore_handlers()

    assert (signal.SIGINT, signal.SIG_DFL) in calls
