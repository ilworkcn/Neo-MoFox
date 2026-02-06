"""消息发送器。

负责将 Message 发送到正确的 Adapter。
参考 old/chat/message_receive/uni_message_sender.py 的设计。
"""

from typing import TYPE_CHECKING, Any

from mofox_wire import MessageEnvelope

from src.kernel.logger import get_logger

if TYPE_CHECKING:
    from src.core.models.message import Message

logger = get_logger("message_sender")


class MessageSender:
    """消息发送器。

    负责将 Message 发送到 Adapter。

    职责：
    1. 使用 MessageConverter 将 Message 转换为 MessageEnvelope
    2. 根据 platform 推断目标 Adapter
    3. 通过 AdapterManager 获取 Adapter 实例
    4. 调用 Adapter._send_platform_message() 发送消息
    5. 触发发送事件

    Attributes:
        _converter: 消息转换器
        _adapter_manager: 适配器管理器引用

    Examples:
        >>> sender = MessageSender()
        >>> success = await sender.send_message(message)
    """

    def __init__(self) -> None:
        """初始化消息发送器。"""
        from src.core.transport.message_receive.converter import MessageConverter

        self._converter = MessageConverter()
        self._adapter_manager: Any = None
        logger.info("MessageSender 初始化完成")

    def set_adapter_manager(self, adapter_manager: Any) -> None:
        """设置适配器管理器引用。

        Args:
            adapter_manager: 适配器管理器实例

        Examples:
            >>> sender.set_adapter_manager(get_adapter_manager())
        """
        self._adapter_manager = adapter_manager
        logger.debug("MessageSender 设置适配器管理器")

    async def send_message(
        self,
        message: "Message",
        adapter_signature: str | None = None,
    ) -> bool:
        """发送消息到 Adapter。

        Args:
            message: 待发送的消息
            adapter_signature: 目标适配器签名（None 表示自动推断）

        Returns:
            bool: 是否发送成功

        Raises:
            ValueError: 如果消息格式不正确或无法确定目标 Adapter

        Examples:
            >>> success = await sender.send_message(message)
            >>> success = await sender.send_message(message, "my_plugin:adapter:qq")
        """
        try:
            # 1. 转换为 MessageEnvelope
            envelope = await self._converter.message_to_envelope(message)

            # 2. 确定目标 Adapter
            if not adapter_signature:
                adapter_signature = self._infer_adapter_signature(message)

            if not adapter_signature:
                logger.error(
                    f"无法确定目标 Adapter: platform={message.platform}, "
                    f"message_id={message.message_id}"
                )
                return False

            # 3. 获取 Adapter 实例
            if not self._adapter_manager:
                from src.core.managers.adapter_manager import get_adapter_manager

                self._adapter_manager = get_adapter_manager()

            adapter = self._adapter_manager.get_adapter(adapter_signature)

            if not adapter:
                logger.error(
                    f"Adapter 未找到: {adapter_signature}, "
                    f"message_id={message.message_id}"
                )
                return False

            # 4. 发送
            await adapter._send_platform_message(envelope)

            logger.info(
                f"消息发送成功: {message.message_id} → {adapter_signature}"
            )

            # 5. 触发发送事件
            await self._emit_send_event(message, envelope, adapter_signature)

            return True

        except ValueError as e:
            logger.error(f"消息格式错误: {e}")
            return False
        except Exception as e:
            logger.error(
                f"发送消息失败: message_id={message.message_id}, error={e}",
                exc_info=True,
            )
            return False

    def _infer_adapter_signature(self, message: "Message") -> str | None:
        """推断目标 Adapter 签名。

        根据 message.platform 查找匹配的 Adapter。

        Args:
            message: 消息对象

        Returns:
            str | None: Adapter 签名，如果未找到则返回 None
        """
        try:
            from src.core.components.registry import get_global_registry
            from src.core.components.types import ComponentType

            registry = get_global_registry()
            adapters = registry.get_by_type(ComponentType.ADAPTER)

            # 查找匹配平台的 Adapter
            for sig, adapter_cls in adapters.items():
                if hasattr(adapter_cls, "platform") and adapter_cls.platform == message.platform:
                    logger.debug(
                        f"推断 Adapter 签名: {sig} (platform={message.platform})"
                    )
                    return sig

            logger.warning(
                f"未找到匹配的 Adapter: platform={message.platform}"
            )
            return None

        except Exception as e:
            logger.error(f"推断 Adapter 签名失败: {e}")
            return None

    async def _emit_send_event(
        self,
        message: "Message",
        envelope: MessageEnvelope,
        adapter_signature: str,
    ) -> None:
        """触发消息发送事件。

        Args:
            message: 消息对象
            envelope: 消息信封
            adapter_signature: 适配器签名
        """
        try:
            # 尝试从事件管理器获取
            from src.core.managers.event_manager import get_event_manager
            from src.core.components.types import EventType

            event_mgr = get_event_manager()
            await event_mgr.publish_event(
                EventType.ON_MESSAGE_SENT,
                {
                    "message": message,
                    "envelope": envelope,
                    "adapter_signature": adapter_signature,
                },
            )
        except Exception as e:
            logger.warning(f"触发发送事件失败: {e}")


# 全局单例
_global_message_sender: "MessageSender | None" = None


def get_message_sender() -> MessageSender:
    """获取全局 MessageSender 单例。

    Returns:
        MessageSender: 全局 MessageSender 单例

    Examples:
        >>> sender = get_message_sender()
    """
    global _global_message_sender
    if _global_message_sender is None:
        _global_message_sender = MessageSender()
    return _global_message_sender


def set_message_sender(sender: MessageSender) -> None:
    """设置全局 MessageSender 单例。

    Args:
        sender: MessageSender 实例

    Examples:
        >>> set_message_sender(MessageSender())
    """
    global _global_message_sender
    _global_message_sender = sender


def reset_message_sender() -> None:
    """重置全局 MessageSender。

    主要用于测试场景，确保测试之间不会相互影响。

    Examples:
        >>> reset_message_sender()
    """
    global _global_message_sender
    _global_message_sender = None


__all__ = [
    "MessageSender",
    "get_message_sender",
    "set_message_sender",
    "reset_message_sender",
]
