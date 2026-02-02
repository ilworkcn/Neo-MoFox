"""服务协议模型。

本模块使用 typing.Protocol 定义常见服务的接口标准。
Protocol 是 Python 的结构子类型工具，用于定义接口规范。
"""

from typing import Any, Protocol, runtime_checkable


@runtime_checkable
class MemoryService(Protocol):
    """内存服务协议。

    定义内存管理服务的标准接口。
    实现此协议的类可以作为组件系统的 Memory 服务使用。
    """

    async def store(self, key: str, value: Any) -> bool:
        """存储键值对。

        Args:
            key: 键
            value: 值

        Returns:
            bool: 是否成功
        """
        ...

    async def retrieve(self, key: str) -> Any | None:
        """检索值。

        Args:
            key: 键

        Returns:
            Any | None: 存储的值，如果不存在返回 None
        """
        ...

    async def delete(self, key: str) -> bool:
        """删除键值对。

        Args:
            key: 键

        Returns:
            bool: 是否成功
        """
        ...

    async def clear(self) -> None:
        """清除所有存储。"""
        ...

