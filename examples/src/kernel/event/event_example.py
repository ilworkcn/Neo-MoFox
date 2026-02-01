"""Event bus 模块使用示例（简化版）。

演示如何使用event bus进行事件的发布和订阅。
"""

import asyncio
import sys
from pathlib import Path

# 添加项目根目录到Python路径
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent.parent))

from src.kernel.event import event_bus, Event


async def main():
    """主函数，演示event bus的各种用法。"""

    # ========== 示例1：基本的事件订阅和发布 ==========
    print("=" * 50)
    print("示例1：基本的事件订阅和发布")
    print("=" * 50)

    async def on_user_login(event: Event):
        """用户登录事件处理器"""
        user_id = event.data.get("user_id")
        username = event.data.get("username")
        print(f"[INFO] 用户登录：{username} (ID: {user_id})")

    # 订阅事件
    event_bus.subscribe("user_login", on_user_login)

    # 发布事件
    await event_bus.publish(
        Event(
            name="user_login",
            data={"user_id": "12345", "username": "张三"},
            source="auth_system",
        )
    )

    # ========== 示例2：多个处理器订阅同一事件 ==========
    print("\n" + "=" * 50)
    print("示例2：多个处理器订阅同一事件")
    print("=" * 50)

    async def log_to_file(event: Event):
        """将日志写入文件的处理器"""
        print(f"[FILE] 收到事件：{event.name}")

    async def log_to_database(event: Event):
        """将日志写入数据库的处理器"""
        print(f"[DB] 收到事件：{event.name}")

    async def send_notification(event: Event):
        """发送通知的处理器"""
        print(f"[NOTIFY] 新事件：{event.name}")

    # 多个处理器订阅同一事件
    event_bus.subscribe("new_message", log_to_file)
    event_bus.subscribe("new_message", log_to_database)
    event_bus.subscribe("new_message", send_notification)

    # 发布事件，所有处理器都会被调用
    await event_bus.publish(
        Event(
            name="new_message",
            data={"content": "你好，世界！"},
        )
    )

    # ========== 示例3：取消订阅 ==========
    print("\n" + "=" * 50)
    print("示例3：取消订阅")
    print("=" * 50)

    async def temporary_handler(event: Event):
        """临时处理器"""
        print("[TEMP] 临时处理器被调用")

    # 使用返回的取消订阅函数
    unsubscribe = event_bus.subscribe("temp_event", temporary_handler)

    print("第一次发布：")
    await event_bus.publish(Event(name="temp_event"))

    # 取消订阅
    unsubscribe()
    print("\n取消订阅后再次发布：")
    await event_bus.publish(Event(name="temp_event"))

    # ========== 示例4：同步处理器 ==========
    print("\n" + "=" * 50)
    print("示例4：同步处理器（非async）")
    print("=" * 50)

    def sync_handler(event: Event):
        """同步处理器"""
        print(f"[SYNC] 同步处理器处理事件：{event.name}")

    event_bus.subscribe("sync_event", sync_handler)
    await event_bus.publish(Event(name="sync_event"))

    # ========== 示例5：事件数据传递 ==========
    print("\n" + "=" * 50)
    print("示例5：复杂事件数据")
    print("=" * 50)

    async def handle_order_created(event: Event):
        """处理订单创建事件"""
        order_data = event.data
        print("[ORDER] 新订单创建：")
        print(f"   订单号：{order_data['order_id']}")
        print(f"   金额：¥{order_data['amount']:.2f}")
        print(f"   商品：{order_data['product']}")
        print(f"   来源：{event.source}")

    event_bus.subscribe("order_created", handle_order_created)

    await event_bus.publish(
        Event(
            name="order_created",
            data={
                "order_id": "ORD-2024-001",
                "amount": 299.99,
                "product": "机械键盘",
            },
            source="shop_system",
        )
    )

    # ========== 示例6：混合使用同步和异步处理器 ==========
    print("\n" + "=" * 50)
    print("示例6：混合使用同步和异步处理器")
    print("=" * 50)

    def sync_counter(event: Event):
        print("  [SYNC] 同步计数器")

    async def async_counter(event: Event):
        await asyncio.sleep(0.01)  # 模拟异步操作
        print("  [ASYNC] 异步计数器")

    event_bus.subscribe("counter_event", sync_counter)
    event_bus.subscribe("counter_event", async_counter)

    print("计数器事件：")
    await event_bus.publish(Event(name="counter_event"))

    # ========== 示例7：查看事件总线状态 ==========
    print("\n" + "=" * 50)
    print("示例7：事件总线状态")
    print("=" * 50)

    print("[STATS] 事件总线统计：")
    print(f"   总线名称：{event_bus.name}")
    print(f"   已订阅事件数：{event_bus.event_count}")
    print(f"   处理器总数：{event_bus.handler_count}")
    print(f"   已订阅的事件：{', '.join(sorted(event_bus.subscribed_events))}")

    # ========== 示例8：使用全局event_bus ==========
    print("\n" + "=" * 50)
    print("示例8：全局event_bus使用")
    print("=" * 50)

    # 全局event_bus已经自动导入并初始化
    # 可以在任何模块中使用它
    async def global_handler(event: Event):
        print(f"[GLOBAL] 全局事件总线收到：{event.name}")

    event_bus.subscribe("global_test", global_handler)
    await event_bus.publish(Event(name="global_test"))

    print("\n[SUCCESS] 所有示例执行完成！")


if __name__ == "__main__":
    # 运行示例
    asyncio.run(main())
