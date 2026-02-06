"""会话循环生成器与驱动器。

提供 ``conversation_loop`` 异步生成器和 ``run_chat_stream`` 驱动器：

- ``conversation_loop``: 按需检查未读消息并产出 ``ConversationTick``
- ``run_chat_stream``: 消费 Tick 事件并调度 Chatter 处理

参考 old/chat/message_manager/distribution_manager.py 中同名实现。
"""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator, Awaitable, Callable
from typing import TYPE_CHECKING, Any

from src.core.transport.distribution.tick import ConversationTick
from src.kernel.logger import get_logger, COLOR

if TYPE_CHECKING:
    from src.core.models.stream import StreamContext
    from src.core.transport.distribution import StreamLoopManager

logger = get_logger("conversation_loop", display="会话循环", color=COLOR.MAGENTA)


# ============================================================================
# 异步生成器 — 核心循环逻辑
# ============================================================================


async def conversation_loop(
    stream_id: str,
    get_context_func: Callable[[str], Awaitable["StreamContext | None"]],
    calculate_interval_func: Callable[[str, bool], Awaitable[float]],
    flush_cache_func: Callable[[str], Awaitable[list[Any]]],
    check_force_dispatch_func: Callable[["StreamContext", int], bool],
    is_running_func: Callable[[], bool],
) -> AsyncIterator[ConversationTick]:
    """会话循环生成器 — 按需产出 Tick 事件。

    替代无限循环任务，改为事件驱动的生成器模式。
    只有调用 ``__anext__()`` 时才会执行，完全由消费者控制节奏。

    Args:
        stream_id: 流 ID
        get_context_func: 获取 StreamContext 的异步函数
        calculate_interval_func: 计算等待间隔的异步函数 (stream_id, has_messages) -> seconds
        flush_cache_func: 刷新缓存消息的异步函数
        check_force_dispatch_func: 检查是否需要强制分发的函数
        is_running_func: 检查是否继续运行的函数

    Yields:
        ConversationTick: 会话事件
    """
    tick_count = 0
    last_interval: float | None = None

    while is_running_func():
        try:
            # 1. 获取流上下文
            context = await get_context_func(stream_id)
            if not context:
                logger.warning(f"[生成器] stream={stream_id[:8]}, 无法获取流上下文")
                await asyncio.sleep(10.0)
                continue

            # 2. 刷新缓存消息到未读列表
            await flush_cache_func(stream_id)

            # 3. 检查未读消息数量
            unread_messages = context.unread_messages
            unread_count = len(unread_messages) if unread_messages else 0

            # 4. 检查是否需要强制分发
            force_dispatch = check_force_dispatch_func(context, unread_count)

            # 5. 有消息时产出 Tick
            if unread_count > 0 or force_dispatch:
                tick_count += 1
                yield ConversationTick(
                    stream_id=stream_id,
                    force_dispatch=force_dispatch,
                    tick_count=tick_count,
                )

            # 6. 计算并等待下次检查间隔
            has_messages = unread_count > 0
            interval = await calculate_interval_func(stream_id, has_messages)

            if last_interval is None or abs(interval - last_interval) > 0.01:
                logger.debug(f"[生成器] stream={stream_id[:8]}, 等待间隔: {interval:.2f}s")
                last_interval = interval

            await asyncio.sleep(interval)

        except asyncio.CancelledError:
            logger.info(f"[生成器] stream={stream_id[:8]}, 被取消")
            break
        except Exception as e:
            logger.error(f"[生成器] stream={stream_id[:8]}, 出错: {e}")
            await asyncio.sleep(5.0)


# ============================================================================
# 聊天流驱动器
# ============================================================================


async def run_chat_stream(
    stream_id: str,
    manager: "StreamLoopManager",
) -> None:
    """聊天流驱动器 — 消费 Tick 事件并调用 Chatter。

    为每个聊天流创建独立的 ``conversation_loop`` 生成器，
    通过 ``async for`` 消费 Tick 并调度 ``_process_stream_messages``。

    Args:
        stream_id: 流 ID
        manager: StreamLoopManager 实例
    """
    task_id = id(asyncio.current_task())
    logger.debug(f"[驱动器] stream={stream_id[:8]}, 任务ID={task_id}, 启动")

    try:
        # 创建生成器
        tick_generator = conversation_loop(
            stream_id=stream_id,
            get_context_func=manager._get_stream_context,
            calculate_interval_func=manager._calculate_interval,
            flush_cache_func=manager._flush_cached_messages_to_unread,
            check_force_dispatch_func=manager._needs_force_dispatch,
            is_running_func=lambda: manager.is_running,
        )

        # 消费 Tick 事件
        async for tick in tick_generator:
            try:
                context = await manager._get_stream_context(stream_id)
                if not context:
                    continue

                # 并发保护：Chatter 正在处理时跳过
                if context.is_chatter_processing:
                    logger.debug(
                        f"[驱动器] stream={stream_id[:8]}, Chatter 正在处理，跳过 Tick#{tick.tick_count}"
                    )
                    continue

                if tick.force_dispatch:
                    logger.info(f"[驱动器] stream={stream_id[:8]}, Tick#{tick.tick_count}, 强制分发")
                else:
                    logger.debug(f"[驱动器] stream={stream_id[:8]}, Tick#{tick.tick_count}, 开始处理")

                # 处理消息
                success = await manager._process_stream_messages(stream_id, context)

                manager._stats["total_process_cycles"] += 1
                if success:
                    logger.debug(f"[驱动器] stream={stream_id[:8]}, Tick#{tick.tick_count}, 处理成功")
                else:
                    manager._stats["total_failures"] += 1
                    logger.debug(f"[驱动器] stream={stream_id[:8]}, Tick#{tick.tick_count}, 处理失败")

            except asyncio.CancelledError:
                raise
            except Exception as e:
                logger.error(f"[驱动器] stream={stream_id[:8]}, 处理 Tick 时出错: {e}")
                manager._stats["total_failures"] += 1

    except asyncio.CancelledError:
        logger.info(f"[驱动器] stream={stream_id[:8]}, 任务ID={task_id}, 被取消")
    finally:
        # 清理任务记录
        try:
            context = await manager._get_stream_context(stream_id)
            if context and context.stream_loop_task:
                context.stream_loop_task = None
                logger.debug(f"[驱动器] stream={stream_id[:8]}, 清理任务记录")
        except Exception as e:
            logger.debug(f"清理任务记录失败: {e}")
