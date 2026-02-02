"""插件加载器和注册系统。

本模块提供插件注册装饰器和相关实用工具，用于管理插件发现和加载。
支持插件类用于注册自身的 @register_plugin 装饰器。
"""

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from src.core.components.base.plugin import BasePlugin

# 全局插件注册表
_plugin_registry: dict[str, type["BasePlugin"]] = {}


def register_plugin(cls: type["BasePlugin"]) -> type["BasePlugin"]:
    """注册插件类装饰器。

    此装饰器用于将插件类注册到全局插件注册表。
    每个插件必须定义 'plugin_name' 属性。

    Args:
        cls: 要注册的插件类

    Returns:
        注册后的类（本身不变）

    Raises:
        ValueError: 如果未定义 plugin_name 或插件已注册

    Examples:
        >>> @register_plugin
        ... class MyPlugin(BasePlugin):
        ...     plugin_name = "my_plugin"
        ...     plugin_description = "我的超棒插件"
        ...
        >>> # 插件现已注册，可以通过 get_plugin_class() 检索
    """
    # 检查是否定义了 plugin_name
    if not hasattr(cls, "plugin_name") or not cls.plugin_name:
        raise ValueError(
            f"插件类 '{cls.__name__}' 必须定义 'plugin_name' 属性"
        )

    plugin_name = cls.plugin_name

    # 检查重复注册
    if plugin_name in _plugin_registry:
        raise ValueError(
            f"插件 '{plugin_name}' 已被 "
            f"'{_plugin_registry[plugin_name].__name__}' 注册"
        )

    # 注册插件
    _plugin_registry[plugin_name] = cls

    return cls


def get_plugin_class(plugin_name: str) -> type["BasePlugin"] | None:
    """通过名称获取已注册的插件类。

    Args:
        plugin_name: 要检索的插件名称

    Returns:
        如果找到返回插件类，否则返回 None

    Examples:
        >>> plugin_cls = get_plugin_class("my_plugin")
        >>> if plugin_cls:
        ...     plugin_instance = plugin_cls(config)
    """
    return _plugin_registry.get(plugin_name)


def list_registered_plugins() -> list[str]:
    """列出所有已注册的插件名称。

    Returns:
        已注册的插件名称列表

    Examples:
        >>> plugins = list_registered_plugins()
        >>> ['my_plugin', 'other_plugin', 'awesome_plugin']
    """
    return list(_plugin_registry.keys())


def is_plugin_registered(plugin_name: str) -> bool:
    """检查插件是否已注册。

    Args:
        plugin_name: 要检查的插件名称

    Returns:
        如果插件已注册返回 True，否则返回 False

    Examples:
        >>> if is_plugin_registered("my_plugin"):
        ...     print("插件已加载")
    """
    return plugin_name in _plugin_registry


def unregister_plugin(plugin_name: str) -> bool:
    """注销插件。

    从注册表中移除插件。主要用于测试目的。

    Args:
        plugin_name: 要注销的插件名称

    Returns:
        如果插件已注销返回 True，如果未找到返回 False

    Examples:
        >>> unregister_plugin("my_plugin")
        True
    """
    if plugin_name in _plugin_registry:
        del _plugin_registry[plugin_name]
        return True
    return False


def clear_registry() -> None:
    """清除所有已注册的插件。

    从注册表中移除所有插件。主要用于测试目的。

    Examples:
        >>> clear_registry()
    """
    _plugin_registry.clear()


def get_registry_count() -> int:
    """获取已注册插件的数量。

    Returns:
        已注册插件的数量

    Examples:
        >>> count = get_registry_count()
        >>> 5
    """
    return len(_plugin_registry)


# 插件清单的数据类
from dataclasses import dataclass


@dataclass
class ComponentInclude:
    """组件包含声明。

    用于在 manifest.json 中声明插件包含的组件及其依赖项。

    Attributes:
        component_type: 组件类型（action, tool, chatter, command, collection, event_handler, adapter, service, router）
        component_name: 组件名称
        dependencies: 该组件依赖的其他组件签名列表
        enabled: 是否启用该组件（默认 True）
    """

    component_type: str
    component_name: str
    dependencies: list[str]  # 组件签名列表，如 ["other_plugin:tool:calculator"]
    enabled: bool = True


@dataclass
class PluginManifest:
    """插件清单数据。

    表示插件的 manifest.json 文件内容。

    Attributes:
        name: 唯一的插件名称/标识符
        version: 插件版本字符串
        description: 人类可读的描述
        author: 插件作者名称
        dependencies: 包含 'plugins' 和 'components' 列表的字典
        include: 插件包含的组件列表及组件级依赖
        entry_point: 相对于插件根目录的 Python 入口点文件
        min_core_version: 所需的最低核心版本
        _source_path: 内部：插件加载来源路径
    """

    name: str
    version: str
    description: str
    author: str
    dependencies: dict[str, list[str]]
    include: list[ComponentInclude]
    entry_point: str
    min_core_version: str
    _source_path: str  # 内部：清单加载来源路径


class PluginDependencyResolver:
    """使用拓扑排序的插件依赖解析器。

    分析插件依赖关系并确定正确的加载顺序以满足所有依赖。
    使用 Kahn 算法进行拓扑排序，使用 DFS 进行循环检测。

    Attributes:
        _plugins: 按名称索引的插件清单字典

    Examples:
        >>> resolver = PluginDependencyResolver()
        >>> resolver.add_plugin(manifest1)
        >>> resolver.add_plugin(manifest2)
        >>> load_order = resolver.resolve_load_order()
        >>> ['plugin1', 'plugin2']  # plugin2 依赖于 plugin1
    """

    def __init__(self) -> None:
        """初始化依赖解析器。"""
        self._plugins: dict[str, PluginManifest] = {}

    def add_plugin(self, manifest: PluginManifest) -> None:
        """将插件添加到依赖图。

        Args:
            manifest: 要添加的插件清单

        Examples:
            >>> resolver.add_plugin(plugin_manifest)
        """
        self._plugins[manifest.name] = manifest

    def resolve_load_order(self) -> list[str]:
        """使用拓扑排序解析插件加载顺序。

        基于插件的依赖关系使用 Kahn 算法确定正确的加载顺序。

        Returns:
            按依赖顺序排列的插件名称列表

        Raises:
            ValueError: 如果检测到循环依赖

        Examples:
            >>> order = resolver.resolve_load_order()
            >>> ['base_plugin', 'dependent_plugin', 'another_dependent']
        """
        # 构建依赖图
        in_degree: dict[str, int] = {name: 0 for name in self._plugins}
        graph: dict[str, set[str]] = {name: set() for name in self._plugins}

        for plugin_name, manifest in self._plugins.items():
            # 处理插件依赖
            for dep_ref in manifest.dependencies.get("plugins", []):
                dep_name = self._parse_plugin_ref(dep_ref)
                if dep_name in self._plugins:
                    graph[dep_name].add(plugin_name)
                    in_degree[plugin_name] += 1

        # Kahn 算法拓扑排序
        queue = [name for name, degree in in_degree.items() if degree == 0]
        load_order = []

        while queue:
            current = queue.pop(0)
            load_order.append(current)

            for dependent in graph[current]:
                in_degree[dependent] -= 1
                if in_degree[dependent] == 0:
                    queue.append(dependent)

        # 检查循环依赖
        if len(load_order) != len(self._plugins):
            remaining = [name for name in self._plugins if name not in load_order]
            raise ValueError(f"检测到循环依赖，涉及的插件: {remaining}")

        return load_order

    def check_circular_dependency(self) -> list[str] | None:
        """使用 DFS 检查循环依赖。

        对依赖图执行深度优先搜索以检测循环。

        Returns:
            如果找到循环则返回构成循环的插件名称列表，否则返回 None

        Examples:
            >>> cycle = resolver.check_circular_dependency()
            >>> if cycle:
            ...     print(f"检测到循环: {' -> '.join(cycle)}")
        """
        WHITE, GRAY, BLACK = 0, 1, 2
        color: dict[str, int] = {name: WHITE for name in self._plugins}
        cycle: list[str] = []

        def dfs(node: str, path: list[str]) -> bool:
            color[node] = GRAY
            path.append(node)

            manifest = self._plugins[node]
            for dep_ref in manifest.dependencies.get("plugins", []):
                dep_name = self._parse_plugin_ref(dep_ref)
                if dep_name not in self._plugins:
                    continue  # 外部依赖，跳过

                if color[dep_name] == GRAY:
                    # 找到循环
                    cycle_start = path.index(dep_name)
                    cycle.extend(path[cycle_start:])
                    cycle.append(dep_name)  # 回到起点
                    return True
                elif color[dep_name] == WHITE:
                    if dfs(dep_name, path):
                        return True

            path.pop()
            color[node] = BLACK
            return False

        for plugin_name in self._plugins:
            if color[plugin_name] == WHITE:
                if dfs(plugin_name, []):
                    return cycle

        return None

    def _parse_plugin_ref(self, ref: str) -> str:
        """解析插件引用字符串。

        从引用字符串中提取插件名称。
        未来版本可能支持版本约束。

        Args:
            ref: 插件引用字符串（例如 'plugin_name:>=1.0.0'）

        Returns:
            插件名称

        Examples:
            >>> resolver._parse_plugin_ref("my_plugin:>=1.0.0")
            'my_plugin'
            >>> resolver._parse_plugin_ref("other_plugin")
            'other_plugin'
        """
        return ref.split(":")[0]

    def clear(self) -> None:
        """清除解析器中的所有插件。

        Examples:
            >>> resolver.clear()
        """
        self._plugins.clear()
