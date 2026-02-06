"""Bot 主类

Neo-MoFox 框架的核心协调器，负责系统初始化、插件加载和生命周期管理。
"""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any

from .console_ui import ConsoleUIManager, UILevel
from .exceptions import BotInitializationError, BotRuntimeError, BotShutdownError
from .signal_handler import SignalHandler


class Bot:
    """Neo-MoFox Bot 主类

    管理完整的 Bot 生命周期，包括：
    - Kernel 层初始化
    - Core 层组件加载
    - 插件发现和加载
    - 运行时管理
    - 优雅关闭

    Attributes:
        bot_name: Bot 名称
        bot_version: Bot 版本
        config_path: 配置文件路径
        plugins_dir: 插件目录
        log_dir: 日志目录
        ui_level: UI 详细程度
    """

    bot_name: str = "Neo-MoFox"
    bot_version: str = "0.1.0"

    def __init__(
        self,
        config_path: str = "config/core.toml",
        plugins_dir: str = "plugins",
        log_dir: str = "logs",
        ui_level: UILevel = UILevel.STANDARD,
    ) -> None:
        """初始化 Bot

        Args:
            config_path: 配置文件路径
            plugins_dir: 插件目录
            log_dir: 日志目录
            ui_level: UI 详细程度
        """
        self.config_path = config_path
        self.plugins_dir = plugins_dir
        self.log_dir = log_dir

        # UI 管理器
        self.ui = ConsoleUIManager(level=ui_level)

        # 状态标志
        self._initialized = False
        self._running = False
        self._shutdown_requested = False

        # Kernel 层组件（延迟初始化）
        self.config: Any = None
        self.logger: Any = None
        self.event_bus: Any = None
        self.task_manager: Any = None
        self.watchdog: Any = None
        self.vector_db: Any = None
        self.scheduler: Any = None
        self.storage: Any = None

        # Core 层组件（延迟初始化）
        self.plugin_loader: Any = None
        self.plugin_manager: Any = None
        self.http_server: Any = None
        self.load_order: list[str] = []
        self.manifests: dict[str, Any] = {}
        self.load_results: dict[str, bool] = {}

        # 统计数据
        self._stats: dict[str, Any] = {
            "plugins_loaded": 0,
            "plugins_failed": 0,
            "components_by_type": {},
        }

    async def initialize(self) -> None:
        """完整初始化流程

        按顺序初始化：
        1. Kernel 层（8 步）
        2. 插件发现
        3. 插件加载

        Raises:
            BotInitializationError: 初始化失败
        """
        try:
            # 显示启动横幅
            self.ui.show_banner(self.bot_version, self.bot_name)

            # Phase 1: Kernel 初始化
            await self._initialize_kernel()

            # Phase 2: Core 组件初始化
            await self._initialize_core()

            # Phase 3: 插件发现
            await self._discover_plugins()

            # Phase 4: 插件加载
            await self._load_plugins()

            self._initialized = True

            # 显示成功消息
            loaded = len([r for r in self.load_results.values() if r])
            total = len(self.load_results)
            failed = total - loaded

            if failed > 0:
                self.ui.display_warning(
                    f"Bot 已初始化，加载了 {loaded}/{total} 个插件（{failed} 个失败）"
                )
            else:
                self.ui.display_success(
                    f"Bot 初始化成功，加载了 {total} 个插件"
                )

        except Exception as e:
            self.ui.display_error(f"Initialization failed: {e}", e)
            raise BotInitializationError(str(e), "unknown") from e

    async def _initialize_kernel(self) -> None:
        """初始化 Kernel 层（8 步）

        1. Config
        2. Logger
        3. Event Bus
        4. Database
        5. Concurrency
        6. VectorDB
        7. Scheduler
        8. Storage
        """
        self.ui.update_phase_status("初始化内核", "启动中...")

        # Step 1: Config
        from src.core.config import init_core_config

        self.config = init_core_config(self.config_path)
        self.ui.update_phase_status("配置", "已加载")

        # Step 2: Logger
        from src.kernel.logger import get_logger, initialize_logger_system

        initialize_logger_system(log_dir=self.log_dir, log_level=self.config.bot.log_level)
        self.logger = get_logger("bot")
        self.ui.update_phase_status("日志", "已初始化")

        # Step 3: Event Bus
        from src.kernel.event import get_event_bus

        self.event_bus = get_event_bus()
        self.ui.update_phase_status("事件总线", "已启动")

        # Step 4: Database
        from src.kernel.db import init_database_from_config

        db_cfg = self.config.database
        await init_database_from_config(
            database_type=db_cfg.database_type,
            sqlite_path=db_cfg.sqlite_path,
            postgresql_host=db_cfg.postgresql_host,
            postgresql_port=db_cfg.postgresql_port,
            postgresql_database=db_cfg.postgresql_database,
            postgresql_user=db_cfg.postgresql_user,
            postgresql_password=db_cfg.postgresql_password,
            postgresql_schema=db_cfg.postgresql_schema,
            postgresql_ssl_mode=db_cfg.postgresql_ssl_mode,
            postgresql_ssl_ca=db_cfg.postgresql_ssl_ca,
            postgresql_ssl_cert=db_cfg.postgresql_ssl_cert,
            postgresql_ssl_key=db_cfg.postgresql_ssl_key,
            connection_pool_size=db_cfg.connection_pool_size,
            connection_timeout=db_cfg.connection_timeout,
            echo=db_cfg.echo,
        )
        self._stats["db_connected"] = True
        self.ui.update_phase_status("数据库", "已连接")

        # Step 5: Concurrency
        from src.kernel.concurrency import get_task_manager, get_watchdog

        self.task_manager = get_task_manager()
        self.watchdog = get_watchdog()
        self.ui.update_phase_status("并发管理", "已初始化")

        # Step 6: VectorDB
        from src.kernel.vector_db import get_vector_db_service

        # 确保 data 目录存在
        Path("data/chroma_db").mkdir(parents=True, exist_ok=True)
        self.vector_db = get_vector_db_service("data/chroma_db")
        self.ui.update_phase_status("向量数据库", "已初始化")

        # Step 7: Scheduler
        from src.kernel.scheduler import get_unified_scheduler

        self.scheduler = get_unified_scheduler()
        self.ui.update_phase_status("调度器", "已初始化")

        # Step 8: Storage
        from src.kernel.storage import JSONStore

        # 确保 data 目录存在
        Path("data/json_storage").mkdir(parents=True, exist_ok=True)
        self.storage = JSONStore("data/json_storage")
        self.ui.update_phase_status("存储", "已初始化")

    async def _initialize_core(self) -> None:
        """初始化 Core 层组件

        包括插件管理器、Action 管理器、Chatter 管理器、Command 管理器等。
        """
        # Step 1: 初始化 MessageReceiver 和 SinkManager
        from src.core.transport import MessageReceiver, SinkManager
        from src.core.transport.sink import set_sink_manager
        
        self.message_receiver = MessageReceiver()
        self.sink_manager = SinkManager(self.message_receiver)
        set_sink_manager(self.sink_manager)
        self.ui.update_phase_status("消息接收器", "已初始化")
        
        # Step 2: 导入其他manager以初始化
        from src.core.managers import (
            initialize_adapter_manager,
            initialize_router_manager,
            initialize_event_manager     
        )

        initialize_adapter_manager()
        initialize_router_manager()
        initialize_event_manager()

        self.ui.update_phase_status("核心管理器", "已初始化")
        
        # Step 3: 启动 HTTP 服务器
        from src.core.transport.router.http_server import get_http_server
        
        host = "127.0.0.1"
        port = 8000
        
        self.http_server = get_http_server(host=host, port=port)
        await self.http_server.start()
        self.ui.update_phase_status("HTTP服务器", "已启动")

    async def _discover_plugins(self) -> None:
        """发现插件并解析依赖"""
        self.ui.update_phase_status("发现插件", "扫描中...")

        from src.core.components.loader import PluginLoader

        self.plugin_loader = PluginLoader()
        self.load_order, self.manifests = await self.plugin_loader.plan_plugins(
            self.plugins_dir
        )

        # 显示插件加载计划
        self.ui.display_plugin_plan(self.load_order, self.manifests)

    async def _load_plugins(self) -> None:
        """加载插件"""
        self.ui.update_phase_status("加载插件", "启动中...")

        from src.core.managers import get_plugin_manager

        self.plugin_manager = get_plugin_manager()

        for plugin_name in self.load_order:
            manifest = self.manifests[plugin_name]
            try:
                success = await self.plugin_manager.load_plugin_from_manifest(
                    manifest._source_path, manifest
                )
                self.load_results[plugin_name] = success
                self.ui.update_plugin_progress(plugin_name, success)

            except Exception as e:
                self.load_results[plugin_name] = False
                self.ui.update_plugin_progress(plugin_name, False)
                if self.logger:
                    self.logger.error(f"插件 '{plugin_name}' 加载失败: {e}")
                # 继续加载其他插件（容错策略）

        # 更新统计
        self._stats["plugins_loaded"] = len(
            [r for r in self.load_results.values() if r]
        )
        self._stats["plugins_failed"] = len(
            [r for r in self.load_results.values() if not r]
        )
        from src.kernel.event import get_event_bus
        from src.core.components.types import EventType

        await get_event_bus().publish(EventType.ON_ALL_PLUGIN_LOADED, {})

    async def run(self) -> None:
        """主运行循环

        Raises:
            BotRuntimeError: Bot 未初始化
        """
        if not self._initialized:
            raise BotRuntimeError("Bot 未初始化。请先调用 initialize()。")

        self._running = True

        # 启动调度器
        await self.scheduler.start()
        self._stats["scheduler_running"] = True
        
        # 启动实时仪表盘（如果 UI 级别为 VERBOSE）
        if self.ui.level == UILevel.VERBOSE:
            self.ui.start_live_dashboard()

        # 启动信号处理器
        signal_handler = SignalHandler(self)
        signal_handler.register_signals()

        # 创建交互式命令解析器
        from .command_parser import CommandParser

        command_parser = CommandParser(self)

        self.logger.info("Neo-MoFox Bot 启动成功")
        self.logger.info("输入 /help 查看可用命令")

        # 主循环
        try:
            while self._running:
                try:
                    # 等待命令输入（带超时，以便可以检查其他状态）
                    should_continue = await asyncio.wait_for(
                        command_parser.read_and_execute(), timeout=1.0
                    )

                    if not should_continue:
                        break

                    # 更新仪表盘统计
                    if self.ui.level == UILevel.VERBOSE:
                        await self._update_runtime_stats()

                except asyncio.TimeoutError:
                    # 超时是正常的，继续循环
                    if self.ui.level == UILevel.VERBOSE:
                        await self._update_runtime_stats()
                    continue
                except asyncio.CancelledError:
                    break
                except Exception as e:
                    self.logger.error(f"主循环错误: {e}", exc_info=e)

        finally:
            # 停止实时仪表盘
            if self.ui.level == UILevel.VERBOSE:
                self.ui.stop_live_dashboard()

            # 恢复信号处理器
            signal_handler.restore_handlers()

    async def _update_runtime_stats(self) -> None:
        """更新运行时统计数据（用于仪表盘）"""
        stats = {
            "plugins_loaded": self._stats["plugins_loaded"],
            "plugins_failed": self._stats["plugins_failed"],
            "components_by_type": self._stats["components_by_type"],
            "tasks_active": len(self.task_manager.get_all_tasks()),
            "tasks_completed": 0,  # TODO: 从 TaskManager 获取
            "db_connected": self._stats["db_connected"],
            "scheduler_running": self._stats["scheduler_running"],
        }

        self.ui.update_dashboard_stats(stats)

    async def reload_plugin(
        self, plugin_name: str | None = None
    ) -> dict[str, bool]:
        """重新加载插件

        Args:
            plugin_name: 插件名，None 表示重新加载所有插件

        Returns:
            加载结果字典 {plugin_name: success}
        """
        results = {}

        try:
            if plugin_name:
                # 单插件重载
                if plugin_name not in self.manifests:
                    self.ui.display_error(f"未知插件: {plugin_name}")
                    return {plugin_name: False}

                # 卸载
                await self.plugin_manager.unload_plugin(plugin_name)

                # 重新加载
                manifest = self.manifests[plugin_name]
                success = await self.plugin_manager.load_plugin_from_manifest(
                    manifest._source_path, manifest
                )
                results[plugin_name] = success

            else:
                # 全部重载
                await self._unload_all_plugins()
                results = await self._load_plugins()

        except Exception as e:
            self.logger.error(f"插件重载失败: {e}", exc_info=e)
            if plugin_name:
                results[plugin_name] = False

        return results

    async def _unload_all_plugins(self) -> None:
        """卸载所有插件"""
        if not self.plugin_manager:
            return

        # 按相反顺序卸载
        for plugin_name in reversed(self.load_order):
            try:
                await self.plugin_manager.unload_plugin(plugin_name)
                if self.logger:
                    self.logger.info(f"插件已卸载: {plugin_name}")
            except Exception as e:
                if self.logger:
                    self.logger.warning(
                        f"插件 '{plugin_name}' 卸载失败: {e}"
                    )

        # 清空统计
        self.load_results.clear()
        self._stats["plugins_loaded"] = 0
        self._stats["plugins_failed"] = 0

    async def shutdown(self, timeout: float = 30.0) -> None:
        """优雅关闭 Bot

        Args:
            timeout: 关闭超时时间（秒）

        Raises:
            BotShutdownError: 关闭失败
        """
        if self._shutdown_requested:
            return

        self._shutdown_requested = True
        self._running = False

        if self.logger:
            self.logger.info("正在关闭 Neo-MoFox Bot...")
        else:
            print("正在关闭 Neo-MoFox Bot...")

        try:
            # 1. 停止接受新工作
            if self.logger:
                self.logger.info("停止接受新任务...")

            # 2. 卸载插件
            await self._unload_all_plugins()

            # 3. 停止调度器
            await self.scheduler.stop()
            self._stats["scheduler_running"] = False

            # 4. 停止 HTTP 服务器
            if self.http_server and self.http_server.is_running():
                if self.logger:
                    self.logger.info("停止 HTTP 服务器...")
                await self.http_server.stop()

            # 5. 停止 WatchDog
            if self.watchdog:
                self.watchdog.stop()

            # 6. 停止任务管理器（取消所有活动任务）
            active_tasks = self.task_manager.get_active_tasks()
            for task_info in active_tasks:
                self.task_manager.cancel_task(task_info.task_id)

            # 等待所有非守护任务完成
            await self.task_manager.wait_all_tasks()

            # 清理已完成的任务
            self.task_manager.cleanup_tasks()

            # 7. 关闭数据库
            from src.kernel.db import close_engine

            await close_engine()
            self._stats["db_connected"] = False

            # 8. 关闭向量数据库
            from src.kernel.vector_db import close_all_vector_db_services

            await close_all_vector_db_services()

            # 9. 关闭日志系统（停止事件广播）
            from src.kernel.logger import shutdown_logger_system

            shutdown_logger_system()

            self.ui.display_success("关闭完成")

        except Exception as e:
            if self.logger:
                self.logger.error(f"关闭过程中出错: {e}", exc_info=e)
            else:
                print(f"关闭过程中出错: {e}")
            raise BotShutdownError(f"关闭失败: {e}") from e

    async def start(self) -> None:
        """完整的启动流程（初始化 + 运行 + 关闭）

        这是推荐的启动方式。
        """
        try:
            await self.initialize()
            await self.run()
        except KeyboardInterrupt:
            # 用户中断
            if self.logger:
                self.logger.info("用户中断")
            else:
                print("\n[用户中断]")
        except Exception as e:
            if self.logger:
                self.logger.error(f"Bot 错误: {e}", exc_info=e)
            else:
                print(f"\n[致命错误: {e}]")
            raise
        finally:
            await self.shutdown()


__all__ = ["Bot"]
