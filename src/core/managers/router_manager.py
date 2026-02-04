"""Router 管理器。

本模块提供 Router 管理器，负责 Router 组件的注册、发现、挂载和生命周期管理。
Router 提供基于 FastAPI 的 HTTP 路由接口。
管理器维护 Router 组件的全局集合，并处理路由的动态挂载和卸载。
"""

from typing import TYPE_CHECKING, Any

from src.kernel.logger import get_logger

from src.core.components.registry import get_global_registry
from src.core.components.types import ComponentType
from src.core.transport.router.http_server import get_http_server

if TYPE_CHECKING:
    from src.core.components.base.router import BaseRouter
    from src.core.components.base.plugin import BasePlugin


logger = get_logger("router_manager")


class RouterManager:
    """Router 管理器。

    负责管理所有 Router 组件，提供查询、挂载和生命周期管理接口。
    自动将 Router 挂载到 HTTP 服务器，并处理启动和关闭流程。

    Attributes:
        _mounted_routers: 已挂载的 Router 实例字典

    Examples:
        >>> manager = RouterManager()
        >>> # 挂载插件的所有路由
        >>> await manager.mount_plugin_routers(plugin)
        >>> # 卸载插件的所有路由
        >>> await manager.unmount_plugin_routers("my_plugin")
    """

    def __init__(self) -> None:
        """初始化 Router 管理器。"""
        self._mounted_routers: dict[str, "BaseRouter"] = {}
        logger.info("Router 管理器初始化完成")

    def get_all_routers(self) -> dict[str, type["BaseRouter"]]:
        """获取所有已注册的 Router 组件。

        Returns:
            dict[str, type[BaseRouter]]: 将签名映射到 Router 类的字典

        Examples:
            >>> routers = manager.get_all_routers()
        """
        registry = get_global_registry()
        return registry.get_by_type(ComponentType.ROUTER)

    def get_routers_for_plugin(self, plugin_name: str) -> dict[str, type["BaseRouter"]]:
        """获取指定插件的所有 Router 组件。

        Args:
            plugin_name: 插件名称

        Returns:
            dict[str, type[BaseRouter]]: 将签名映射到 Router 类的字典

        Examples:
            >>> routers = manager.get_routers_for_plugin("my_plugin")
        """
        registry = get_global_registry()
        return registry.get_by_plugin_and_type(plugin_name, ComponentType.ROUTER)

    def get_router_class(self, signature: str) -> type["BaseRouter"] | None:
        """通过签名获取 Router 类。

        Args:
            signature: Router 组件签名

        Returns:
            type[BaseRouter] | None: Router 类，如果未找到则返回 None

        Examples:
            >>> router_cls = manager.get_router_class("my_plugin:router:api")
        """
        registry = get_global_registry()
        return registry.get(signature)

    def get_mounted_router(self, signature: str) -> "BaseRouter | None":
        """获取已挂载的 Router 实例。

        Args:
            signature: Router 组件签名

        Returns:
            BaseRouter | None: Router 实例，如果未挂载则返回 None

        Examples:
            >>> router = manager.get_mounted_router("my_plugin:router:api")
        """
        return self._mounted_routers.get(signature)

    def get_all_mounted_routers(self) -> dict[str, "BaseRouter"]:
        """获取所有已挂载的 Router 实例。

        Returns:
            dict[str, BaseRouter]: 将签名映射到 Router 实例的字典

        Examples:
            >>> routers = manager.get_all_mounted_routers()
        """
        return self._mounted_routers.copy()

    async def mount_router(
        self,
        signature: str,
        plugin: "BasePlugin",
    ) -> "BaseRouter":
        """挂载单个 Router。

        创建 Router 实例，挂载到 HTTP 服务器，并调用启动钩子。

        Args:
            signature: Router 组件签名
            plugin: 所属插件实例

        Returns:
            BaseRouter: 已挂载的 Router 实例

        Raises:
            ValueError: 如果 Router 类未找到
            RuntimeError: 如果 Router 已挂载

        Examples:
            >>> router = await manager.mount_router(
            ...     "my_plugin:router:api",
            ...     plugin
            ... )
        """
        # 检查是否已挂载
        if signature in self._mounted_routers:
            raise RuntimeError(f"Router 已挂载: {signature}")

        # 获取 Router 类
        router_cls = self.get_router_class(signature)
        if not router_cls:
            raise ValueError(f"Router 类未找到: {signature}")

        # 创建 Router 实例
        router_instance = router_cls(plugin=plugin)

        # 获取 HTTP 服务器
        http_server = get_http_server()

        # 挂载到 HTTP 服务器
        route_path = router_instance.get_route_path()
        http_server.app.mount(
            path=route_path,
            app=router_instance.get_app(),
            name=router_instance.router_name,
        )

        # 调用启动钩子
        await router_instance.startup()

        # 保存实例
        self._mounted_routers[signature] = router_instance

        logger.info(f"Router 已挂载: {signature} -> {route_path}")
        return router_instance

    async def unmount_router(self, signature: str) -> None:
        """卸载单个 Router。

        调用关闭钩子并从管理器中移除。

        Args:
            signature: Router 组件签名

        Examples:
            >>> await manager.unmount_router("my_plugin:router:api")
        """
        router_instance = self._mounted_routers.get(signature)
        if not router_instance:
            logger.warning(f"Router 未挂载，无法卸载: {signature}")
            return

        # 调用关闭钩子
        try:
            await router_instance.shutdown()
        except Exception as e:
            logger.error(f"Router 关闭钩子执行失败 ({signature}): {e}")

        # 移除实例
        self._mounted_routers.pop(signature, None)

        logger.info(f"Router 已卸载: {signature}")

    async def mount_plugin_routers(self, plugin: "BasePlugin") -> list["BaseRouter"]:
        """挂载插件的所有 Router 组件。

        Args:
            plugin: 插件实例

        Returns:
            list[BaseRouter]: 已挂载的 Router 实例列表

        Examples:
            >>> routers = await manager.mount_plugin_routers(plugin)
        """
        from src.core.components.types import build_signature
        
        plugin_name = plugin.plugin_name
        routers = self.get_routers_for_plugin(plugin_name)

        mounted_routers = []
        for component_name, router_cls in routers.items():
            # 构建完整签名
            signature = build_signature(plugin_name, ComponentType.ROUTER, component_name)
            try:
                router = await self.mount_router(signature, plugin)
                mounted_routers.append(router)
            except Exception as e:
                logger.error(f"挂载 Router 失败 ({signature}): {e}")

        logger.info(f"插件 {plugin_name} 的 Router 挂载完成: {len(mounted_routers)}/{len(routers)}")
        return mounted_routers

    async def unmount_plugin_routers(self, plugin_name: str) -> None:
        """卸载插件的所有 Router 组件。

        Args:
            plugin_name: 插件名称

        Examples:
            >>> await manager.unmount_plugin_routers("my_plugin")
        """
        from src.core.components.types import build_signature
        
        routers = self.get_routers_for_plugin(plugin_name)

        for component_name in routers.keys():
            # 构建完整签名
            signature = build_signature(plugin_name, ComponentType.ROUTER, component_name)
            try:
                await self.unmount_router(signature)
            except Exception as e:
                logger.error(f"卸载 Router 失败 ({signature}): {e}")

        logger.info(f"插件 {plugin_name} 的 Router 卸载完成")

    async def mount_all_routers(self) -> None:
        """挂载所有已注册的 Router 组件。

        需要配合插件系统使用，从插件实例获取。

        Examples:
            >>> await manager.mount_all_routers()
        """
        # TODO: 从插件系统获取所有插件实例并挂载
        logger.warning("mount_all_routers 需要插件系统支持，当前未实现")

    async def unmount_all_routers(self) -> None:
        """卸载所有已挂载的 Router 组件。

        Examples:
            >>> await manager.unmount_all_routers()
        """
        signatures = list(self._mounted_routers.keys())

        for signature in signatures:
            try:
                await self.unmount_router(signature)
            except Exception as e:
                logger.error(f"卸载 Router 失败 ({signature}): {e}")

        logger.info("所有 Router 已卸载")

    def get_router_info(self, signature: str) -> dict[str, Any] | None:
        """获取 Router 信息。

        Args:
            signature: Router 组件签名

        Returns:
            dict[str, Any] | None: Router 信息，如果未找到则返回 None

        Examples:
            >>> info = manager.get_router_info("my_plugin:router:api")
            >>> {
            ...     "signature": "my_plugin:router:api",
            ...     "name": "api",
            ...     "description": "API Router",
            ...     "route_path": "/api/v1/myrouter",
            ...     "mounted": True
            ... }
        """
        router_cls = self.get_router_class(signature)
        if not router_cls:
            return None

        router_instance = self.get_mounted_router(signature)
        is_mounted = router_instance is not None

        return {
            "signature": signature,
            "name": router_cls.router_name,
            "description": router_cls.router_description,
            "route_path": router_instance.get_route_path() if router_instance else None,
            "mounted": is_mounted,
        }

    def get_all_router_info(self) -> list[dict[str, Any]]:
        """获取所有 Router 的信息列表。

        Returns:
            list[dict[str, Any]]: Router 信息列表

        Examples:
            >>> info_list = manager.get_all_router_info()
        """
        all_routers = self.get_all_routers()
        return [
            info
            for signature in all_routers.keys()
            if (info := self.get_router_info(signature)) is not None
        ]

    async def reload_router(self, signature: str, plugin: "BasePlugin") -> "BaseRouter":
        """重新加载 Router。

        先卸载再挂载，用于热重载。

        Args:
            signature: Router 组件签名
            plugin: 所属插件实例

        Returns:
            BaseRouter: 重新挂载的 Router 实例

        Examples:
            >>> router = await manager.reload_router("my_plugin:router:api", plugin)
        """
        # 卸载
        await self.unmount_router(signature)

        # 挂载
        return await self.mount_router(signature, plugin)


# 全局 Router 管理器实例
_global_router_manager: RouterManager | None = None


def get_router_manager() -> RouterManager:
    """获取全局 Router 管理器实例。

    Returns:
        RouterManager: 全局 Router 管理器单例

    Examples:
        >>> manager = get_router_manager()
        >>> await manager.mount_plugin_routers(plugin)
    """
    global _global_router_manager
    if _global_router_manager is None:
        _global_router_manager = RouterManager()
    return _global_router_manager
