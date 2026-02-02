"""服务组件基类。

本模块提供 BaseService 类，定义服务组件的基本行为。
Service 暴露特定功能供其他插件或组件调用。
可以实现 typing.Protocol 定义的接口标准（如 MemoryService, ConfigService 等）。
"""

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from src.core.components.base.plugin import BasePlugin


class BaseService:
    """服务组件基类。

    Service 暴露特定功能供其他插件或组件调用。
    提供插件间通信和功能复用的机制。

    服务可以实现 typing.Protocol 定义的接口标准，例如：
    - MemoryService: 内存管理服务
    - ConfigService: 配置管理服务
    - LogService: 日志服务

    外部调用者可以通过 Service Manager 获取 Service 组件的实例，
    然后直接调用服务方法，例如：

    >>> service = service_manager.get_service("my_memory")
    >>> await service.store("key", "value")

    Class Attributes:
        service_name: 服务名称
        service_description: 服务描述
        version: 服务版本

    Examples:
        >>> from src.core.models.protocols import MemoryService
        >>>
        >>> class MyMemoryService(BaseService):
        ...     # 实现 MemoryService 协议
        ...     service_name = "my_memory"
        ...     service_description = "我的内存服务"
        ...     version = "1.0.0"
        ...
        ...     async def store(self, key: str, value: Any) -> bool:
        ...         # 实现存储逻辑
        ...         return True
        ...
        ...     async def retrieve(self, key: str) -> Any | None:
        ...         # 实现检索逻辑
        ...         return None
    """

    # 服务元数据
    service_name: str = ""
    service_description: str = ""
    version: str = "1.0.0"

    # 组件级依赖（精确到组件签名）
    dependencies: list[str] = []  # 例如 ["other_plugin:service:database"]

    def __init__(self, plugin: "BasePlugin") -> None:
        """初始化服务组件。

        Args:
            plugin: 所属插件实例
        """
        self.plugin = plugin

    def get_info(self) -> dict[str, Any]:
        """获取服务信息。

        Returns:
            dict[str, Any]: 服务信息字典

        Examples:
            >>> info = service.get_info()
            >>> {
            ...     "name": "my_service",
            ...     "description": "我的服务",
            ...     "version": "1.0.0"
            ... }
        """
        return {
            "name": self.service_name,
            "description": self.service_description,
            "version": self.version,
        }
