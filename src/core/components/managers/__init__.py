"""组件管理器层。

本包提供各类组件的管理器，负责组件的运行时编排和生命周期管理。
包括插件管理器、Action 管理器、Chatter 管理器、Command 管理器等。
"""

from src.core.components.managers.plugin_manager import get_plugin_manager, PluginManager
from src.core.components.managers.action_manager import get_action_manager, ActionManager
from src.core.components.managers.chatter_manager import get_chatter_manager, ChatterManager
from src.core.components.managers.command_manager import get_command_manager, CommandManager

__all__ = [
    # 主要管理器
    "get_plugin_manager",
    "PluginManager",
    "get_action_manager",
    "ActionManager",
    "get_chatter_manager",
    "ChatterManager",
    "get_command_manager",
    "CommandManager",
]
