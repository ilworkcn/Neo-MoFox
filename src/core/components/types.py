"""组件相关类型和枚举。

本模块定义了组件模块中使用的所有核心类型和枚举，包括聊天类型、组件类型
以及用于解析组件签名的实用函数。
"""

from enum import Enum
from typing import TypedDict


class ChatType(Enum):
    """聊天类型枚举。

    定义组件可以在其中活动的不同聊天上下文类型。
    """

    PRIVATE = "private"
    GROUP = "group"
    DISCUSS = "discuss"
    ALL = "all"


class ComponentType(Enum):
    """组件类型枚举。

    插件系统中所有可能的组件类型。
    """

    ACTION = "action"
    TOOL = "tool"
    ADAPTER = "adapter"
    CHATTER = "chatter"
    COMMAND = "command"
    COLLECTION = "collection"
    CONFIG = "config"
    EVENT_HANDLER = "event_handler"
    SERVICE = "service"
    ROUTER = "router"
    PLUGIN = "plugin"


class EventType(Enum):
    """事件类型枚举。

    定义事件处理器可以订阅的系统事件。
    """

    ON_START = "on_start"
    ON_STOP = "on_stop"
    ON_MESSAGE_RECEIVED = "on_message_received"
    ON_MESSAGE_SENT = "on_message_sent"
    ON_PLUGIN_LOADED = "on_plugin_loaded"
    ON_PLUGIN_UNLOADED = "on_plugin_unloaded"
    ON_COMPONENT_LOADED = "on_component_loaded"
    ON_COMPONENT_UNLOADED = "on_component_unloaded"
    ON_ERROR = "on_error"
    CUSTOM = "custom"  # 用于自定义事件


class ComponentState(Enum):
    """组件状态枚举。

    跟踪组件的生命周期状态。
    """

    UNLOADED = "unloaded"
    LOADED = "loaded"
    ACTIVE = "active"
    INACTIVE = "inactive"
    ERROR = "error"


class ComponentMeta(TypedDict, total=False):
    """组件元数据。

    组件的标准化元数据结构。
    """

    name: str
    version: str
    description: str
    author: str


class ComponentSignature(TypedDict):
    """组件签名类型字典。

    表示已解析的组件签名，格式为 'plugin_name:component_type:component_name'。
    """

    plugin_name: str
    component_type: ComponentType
    component_name: str


def parse_signature(signature: str) -> ComponentSignature:
    """解析组件签名字符串。

    解析格式为 'plugin_name:component_type:component_name' 的组件签名，
    并返回 ComponentSignature 类型字典。

    Args:
        signature: 组件签名字符串，例如 'my_plugin:action:send_message'

    Returns:
        ComponentSignature: 解析后的签名组件

    Raises:
        ValueError: 如果签名格式无效

    Examples:
        >>> parse_signature("my_plugin:action:send_message")
        {'plugin_name': 'my_plugin', 'component_type': ComponentType.ACTION, 'component_name': 'send_message'}

        >>> parse_signature("other_plugin:tool:calculator")
        {'plugin_name': 'other_plugin', 'component_type': ComponentType.TOOL, 'component_name': 'calculator'}
    """
    parts = signature.split(":")

    if len(parts) != 3:
        raise ValueError(
            f"无效的签名格式: '{signature}'。"
            f"期望格式为 'plugin_name:component_type:component_name'，但得到 {len(parts)} 个部分"
        )

    plugin_name, component_type_str, component_name = parts

    # 验证并转换组件类型
    try:
        component_type = ComponentType(component_type_str.lower())
    except ValueError:
        valid_types = [ct.value for ct in ComponentType]
        raise ValueError(
            f"未知的组件类型: '{component_type_str}'。"
            f"有效类型为: {', '.join(valid_types)}"
        )

    if not plugin_name:
        raise ValueError("插件名称不能为空")

    if not component_name:
        raise ValueError("组件名称不能为空")

    return ComponentSignature(
        plugin_name=plugin_name,
        component_type=component_type,
        component_name=component_name,
    )


def build_signature(
    plugin_name: str, component_type: ComponentType, component_name: str
) -> str:
    """构建组件签名字符串。

    从各个部分构建组件签名。

    Args:
        plugin_name: 插件名称
        component_type: 组件类型
        component_name: 组件名称

    Returns:
        str: 组件签名字符串

    Examples:
        >>> build_signature("my_plugin", ComponentType.ACTION, "send_message")
        'my_plugin:action:send_message'
    """
    return f"{plugin_name}:{component_type.value}:{component_name}"
