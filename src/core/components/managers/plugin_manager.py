"""插件管理器。

本模块提供插件管理器，负责插件的发现、加载、卸载和生命周期管理。
支持文件夹、ZIP 压缩包和 .MFP 格式的插件。
使用 loader.py 中的依赖解析器进行拓扑排序，确保依赖顺序正确加载。
"""

import asyncio
import importlib.util
import json
import sys
import tempfile
import zipfile
from pathlib import Path
from typing import TYPE_CHECKING, Any

from src.kernel.logger import get_logger
from src.kernel.concurrency import get_task_manager

from src.core.components.loader import (
    PluginManifest,
    PluginDependencyResolver,
    get_plugin_class,
)
from src.core.components.registry import get_global_registry
from src.core.components.state_manager import get_global_state_manager
from src.core.components.types import ComponentState, build_signature

if TYPE_CHECKING:
    from src.core.components.base.plugin import BasePlugin


logger = get_logger("plugin_manager")


class PluginManager:
    """插件管理器。

    负责插件的发现、加载、卸载和生命周期管理。
    支持文件夹、ZIP 压缩包和 .MFP 格式的插件。
    使用 manifest.json 解析插件元数据和依赖关系。
    使用拓扑排序确定插件加载顺序。

    Attributes:
        _loaded_plugins: 已加载的插件实例字典
        _manifests: 插件清单字典
        _resolver: 依赖解析器
        _plugin_paths: 插件路径字典

    Examples:
        >>> manager = PluginManager()
        >>> await manager.load_all_plugins("plugins")
        >>> plugin = manager.get_plugin("my_plugin")
        >>> await manager.unload_plugin("my_plugin")
    """

    def __init__(self) -> None:
        """初始化插件管理器。"""
        self._loaded_plugins: dict[str, "BasePlugin"] = {}
        self._manifests: dict[str, PluginManifest] = {}
        self._resolver = PluginDependencyResolver()
        self._plugin_paths: dict[str, str] = {}
        self._failed_plugins: dict[str, str] = {}

        logger.info("插件管理器初始化完成")

    async def discover_plugins(self, plugins_dir: str) -> list[str]:
        """发现插件目录下的所有插件。

        扫描指定目录，查找所有有效的插件（文件夹、ZIP 或 .MFP 文件）。
        有效的插件必须包含 manifest.json 文件。

        Args:
            plugins_dir: 插件根目录路径

        Returns:
            list[str]: 发现的插件路径列表

        Examples:
            >>> discovered = await manager.discover_plugins("plugins")
            >>> ['plugins/my_plugin', 'plugins/other_plugin.zip']
        """
        discovered = []
        plugins_path = Path(plugins_dir)

        if not plugins_path.exists():
            logger.warning(f"插件目录不存在: {plugins_dir}")
            return discovered

        # 扫描文件夹
        for item in plugins_path.iterdir():
            if item.is_dir() and not item.name.startswith(".") and not item.name.startswith("__"):
                manifest_path = item / "manifest.json"
                if manifest_path.exists():
                    discovered.append(str(item))
                    logger.debug(f"发现插件文件夹: {item}")

            # 扫描 .zip 和 .mfp 文件
            elif item.suffix in (".zip", ".mfp"):
                # 验证是否包含 manifest.json
                try:
                    with zipfile.ZipFile(item, 'r') as zf:
                        if "manifest.json" in zf.namelist():
                            discovered.append(str(item))
                            logger.debug(f"发现插件压缩包: {item}")
                except Exception as e:
                    logger.warning(f"无法读取压缩包 {item}: {e}")

        logger.info(f"在 {plugins_dir} 中发现 {len(discovered)} 个插件")
        return discovered

    async def load_plugin(self, plugin_path: str) -> bool:
        """加载单个插件。

        从指定路径加载插件。支持文件夹、ZIP 和 .MFP 格式。
        首先加载 manifest.json，然后解析依赖，最后加载插件模块。

        Args:
            plugin_path: 插件路径（文件夹/ZIP/.MFP）

        Returns:
            bool: 是否加载成功

        Examples:
            >>> success = await manager.load_plugin("plugins/my_plugin")
            >>> True
        """
        # 1. 解析 manifest.json
        manifest = await self._load_manifest(plugin_path)
        if not manifest:
            error_msg = f"无法加载 manifest.json"
            self._failed_plugins[manifest.name if manifest else plugin_path] = error_msg
            return False

        # 2. 检查是否已加载
        if manifest.name in self._loaded_plugins:
            logger.warning(f"插件 '{manifest.name}' 已经加载")
            return True

        # 3. 检查版本兼容性
        if not self._check_version_compatibility(manifest):
            error_msg = f"核心版本不兼容，需要 {manifest.min_core_version}"
            self._failed_plugins[manifest.name] = error_msg
            logger.error(f"插件 '{manifest.name}' 加载失败: {error_msg}")
            return False

        # 4. 加载插件模块
        if plugin_path.endswith((".zip", ".mfp")):
            # 从 ZIP/MFP 加载
            plugin_module = await self._load_from_archive(plugin_path, manifest)
        else:
            # 从文件夹加载
            plugin_module = await self._load_from_folder(plugin_path, manifest)

        if not plugin_module:
            error_msg = f"插件模块加载失败"
            self._failed_plugins[manifest.name] = error_msg
            return False

        # 5. 查找 @register_plugin 注册的插件类
        plugin_class = get_plugin_class(manifest.name)
        if not plugin_class:
            error_msg = f"插件类未注册（未使用 @register_plugin 装饰器）"
            self._failed_plugins[manifest.name] = error_msg
            logger.error(f"插件 '{manifest.name}' 加载失败: {error_msg}")
            return False

        # 6. 加载插件配置
        # TODO: 从 config/plugins/{plugin_name}/ 加载配置
        # config = await self._load_plugin_config(manifest.name, plugin_path)

        # 7. 实例化插件
        try:
            plugin_instance = plugin_class(config=None)  # type: ignore
        except Exception as e:
            error_msg = f"插件实例化失败: {e}"
            self._failed_plugins[manifest.name] = error_msg
            logger.error(f"插件 '{manifest.name}' 加载失败: {error_msg}")
            return False

        # 8. 注册组件到全局注册表
        await self._register_components(plugin_instance)

        # 9. 调用生命周期钩子
        try:
            await plugin_instance.on_plugin_loaded()
        except Exception as e:
            logger.error(f"调用插件 '{manifest.name}' 的 on_plugin_loaded 钩子时出错: {e}")

        # 10. 更新状态
        self._loaded_plugins[manifest.name] = plugin_instance
        self._manifests[manifest.name] = manifest

        state_manager = get_global_state_manager()
        await state_manager.set_state_async(
            build_signature(manifest.name, ComponentType.PLUGIN, manifest.name),
            ComponentState.ACTIVE
        )

        logger.info(f"✅ 插件加载成功: {manifest.name} v{manifest.version}")
        return True

    async def load_all_plugins(self, plugins_dir: str) -> dict[str, bool]:
        """加载所有插件。

        发现指定目录下的所有插件，解析依赖关系，按拓扑排序顺序加载。

        Args:
            plugins_dir: 插件目录

        Returns:
            dict[str, bool]: 插件名到加载结果的映射

        Raises:
            ValueError: 如果检测到循环依赖

        Examples:
            >>> results = await manager.load_all_plugins("plugins")
            >>> {'my_plugin': True, 'other_plugin': False}
        """
        logger.info(f"开始加载插件目录: {plugins_dir}")

        # 1. 发现所有插件
        discovered = await self.discover_plugins(plugins_dir)

        if not discovered:
            logger.warning(f"在 {plugins_dir} 中未发现任何插件")
            return {}

        # 2. 加载所有 manifest
        temp_resolver = PluginDependencyResolver()
        manifests_to_load: dict[str, PluginManifest] = {}

        for path in discovered:
            manifest = await self._load_manifest(path)
            if manifest:
                temp_resolver.add_plugin(manifest)
                manifests_to_load[manifest.name] = manifest
                self._plugin_paths[manifest.name] = path

        # 3. 检测循环依赖
        cycle = temp_resolver.check_circular_dependency()
        if cycle:
            cycle_str = " -> ".join(cycle)
            raise ValueError(f"检测到循环依赖: {cycle_str}")

        # 4. 拓扑排序决定加载顺序
        load_order = temp_resolver.resolve_load_order()
        logger.info(f"插件加载顺序: {' -> '.join(load_order)}")

        # 5. 按顺序加载
        results = {}
        for plugin_name in load_order:
            plugin_path = manifests_to_load[plugin_name]._source_path
            success = await self.load_plugin(plugin_path)
            results[plugin_name] = success

            # 更新解析器
            if success:
                self._resolver.add_plugin(manifests_to_load[plugin_name])

        # 6. 显示加载统计
        success_count = sum(1 for v in results.values() if v)
        fail_count = len(results) - success_count
        logger.info(f"插件加载完成: 成功 {success_count}, 失败 {fail_count}")

        if self._failed_plugins:
            logger.warning("加载失败的插件:")
            for name, error in self._failed_plugins.items():
                logger.warning(f"  - {name}: {error}")

        return results

    async def unload_plugin(self, plugin_name: str) -> bool:
        """卸载插件。

        卸载指定插件，调用生命周期钩子并清理资源。

        Args:
            plugin_name: 插件名称

        Returns:
            bool: 是否卸载成功

        Examples:
            >>> success = await manager.unload_plugin("my_plugin")
            >>> True
        """
        if plugin_name not in self._loaded_plugins:
            logger.warning(f"插件 '{plugin_name}' 未加载")
            return False

        try:
            plugin = self._loaded_plugins[plugin_name]

            # 调用卸载钩子
            try:
                await plugin.on_plugin_unloaded()
            except Exception as e:
                logger.error(f"调用插件 '{plugin_name}' 的 on_plugin_unloaded 钩子时出错: {e}")

            # 更新状态
            state_manager = get_global_state_manager()
            await state_manager.set_state_async(
                build_signature(plugin_name, ComponentType.PLUGIN, plugin_name),
                ComponentState.UNLOADED
            )

            # TODO: 从全局注册表中移除组件

            # 移除引用
            del self._loaded_plugins[plugin_name]
            if plugin_name in self._manifests:
                del self._manifests[plugin_name]
            if plugin_name in self._plugin_paths:
                del self._plugin_paths[plugin_name]

            logger.info(f"✅ 插件卸载成功: {plugin_name}")
            return True

        except Exception as e:
            logger.error(f"❌ 插件卸载失败: {plugin_name} - {e}")
            return False

    async def reload_plugin(self, plugin_name: str) -> bool:
        """重载插件。

        先卸载插件，然后重新加载。

        Args:
            plugin_name: 插件名称

        Returns:
            bool: 是否重载成功

        Examples:
            >>> success = await manager.reload_plugin("my_plugin")
            >>> True
        """
        if plugin_name not in self._loaded_plugins:
            logger.warning(f"插件 '{plugin_name}' 未加载，无法重载")
            return False

        plugin_path = self._plugin_paths.get(plugin_name)
        if not plugin_path:
            logger.error(f"未找到插件 '{plugin_name}' 的路径")
            return False

        # 卸载
        if not await self.unload_plugin(plugin_name):
            return False

        # 重新加载
        return await self.load_plugin(plugin_path)

    def get_plugin(self, plugin_name: str) -> "BasePlugin | None":
        """获取插件实例。

        Args:
            plugin_name: 插件名称

        Returns:
            BasePlugin | None: 插件实例，如果未找到则返回 None

        Examples:
            >>> plugin = manager.get_plugin("my_plugin")
        """
        return self._loaded_plugins.get(plugin_name)

    def get_all_plugins(self) -> dict[str, "BasePlugin"]:
        """获取所有已加载插件。

        Returns:
            dict[str, BasePlugin]: 插件名到插件实例的字典

        Examples:
            >>> plugins = manager.get_all_plugins()
        """
        return self._loaded_plugins.copy()

    def list_loaded_plugins(self) -> list[str]:
        """列出所有已加载的插件名称。

        Returns:
            list[str]: 已加载插件名称列表

        Examples:
            >>> names = manager.list_loaded_plugins()
            >>> ['my_plugin', 'other_plugin']
        """
        return list(self._loaded_plugins.keys())

    def get_manifest(self, plugin_name: str) -> PluginManifest | None:
        """获取插件清单。

        Args:
            plugin_name: 插件名称

        Returns:
            PluginManifest | None: 插件清单，如果未找到则返回 None

        Examples:
            >>> manifest = manager.get_manifest("my_plugin")
        """
        return self._manifests.get(plugin_name)

    def is_plugin_loaded(self, plugin_name: str) -> bool:
        """检查插件是否已加载。

        Args:
            plugin_name: 插件名称

        Returns:
            bool: 插件是否已加载

        Examples:
            >>> if manager.is_plugin_loaded("my_plugin"):
            ...     print("插件已加载")
        """
        return plugin_name in self._loaded_plugins

    # === 私有方法 ===

    async def _load_manifest(self, plugin_path: str) -> PluginManifest | None:
        """加载 manifest.json。

        Args:
            plugin_path: 插件路径

        Returns:
            PluginManifest | None: 插件清单，如果加载失败则返回 None
        """
        try:
            if plugin_path.endswith((".zip", ".mfp")):
                # 从压缩包读取
                with zipfile.ZipFile(plugin_path, 'r') as zf:
                    manifest_data = json.loads(zf.read("manifest.json").decode('utf-8'))
            else:
                # 从文件夹读取
                manifest_file = Path(plugin_path) / "manifest.json"
                with open(manifest_file, 'r', encoding='utf-8') as f:
                    manifest_data = json.load(f)

            # 验证必需字段
            required_fields = ["name", "version", "description", "author", "dependencies", "entry_point"]
            for field in required_fields:
                if field not in manifest_data:
                    logger.error(f"manifest.json 缺少必需字段: {field}")
                    return None

            manifest = PluginManifest(
                name=manifest_data["name"],
                version=manifest_data["version"],
                description=manifest_data.get("description", ""),
                author=manifest_data.get("author", ""),
                dependencies=manifest_data.get("dependencies", {"plugins": [], "components": []}),
                entry_point=manifest_data.get("entry_point", "plugin.py"),
                min_core_version=manifest_data.get("min_core_version", "3.0.0"),
                _source_path=plugin_path
            )

            return manifest

        except Exception as e:
            logger.error(f"加载 manifest.json 失败 ({plugin_path}): {e}")
            return None

    def _check_version_compatibility(self, manifest: PluginManifest) -> bool:
        """检查核心版本兼容性。

        Args:
            manifest: 插件清单

        Returns:
            bool: 是否版本兼容
        """
        # TODO: 实现版本比较逻辑
        # 目前简单检查，未来可以使用 packaging.version 进行语义化版本比较
        return True

    async def _load_from_archive(self, archive_path: str, manifest: PluginManifest) -> Any | None:
        """从 ZIP/MFP 加载插件模块。

        Args:
            archive_path: 压缩包路径
            manifest: 插件清单

        Returns:
            加载的模块对象，失败返回 None
        """
        try:
            with zipfile.ZipFile(archive_path, 'r') as zf:
                # 提取到临时目录
                with tempfile.TemporaryDirectory() as tmpdir:
                    zf.extractall(tmpdir)

                    # 添加到 sys.path
                    sys.path.insert(0, tmpdir)

                    try:
                        # 动态导入
                        entry_point = Path(tmpdir) / manifest.entry_point
                        if not entry_point.exists():
                            logger.error(f"入口点不存在: {manifest.entry_point}")
                            return None

                        spec = importlib.util.spec_from_file_location(
                            manifest.name,
                            str(entry_point)
                        )
                        if spec is None or spec.loader is None:
                            logger.error(f"无法创建模块规范: {entry_point}")
                            return None

                        module = importlib.util.module_from_spec(spec)
                        sys.modules[manifest.name] = module
                        spec.loader.exec_module(module)

                        return module
                    finally:
                        # 从 sys.path 移除
                        if tmpdir in sys.path:
                            sys.path.remove(tmpdir)

        except Exception as e:
            logger.error(f"从压缩包加载插件模块失败 ({archive_path}): {e}")
            return None

    async def _load_from_folder(self, folder_path: str, manifest: PluginManifest) -> Any | None:
        """从文件夹加载插件模块。

        Args:
            folder_path: 文件夹路径
            manifest: 插件清单

        Returns:
            加载的模块对象，失败返回 None
        """
        try:
            folder = Path(folder_path)

            # 添加到 sys.path
            sys.path.insert(0, str(folder))

            try:
                entry_point = folder / manifest.entry_point
                if not entry_point.exists():
                    logger.error(f"入口点不存在: {manifest.entry_point}")
                    return None

                spec = importlib.util.spec_from_file_location(
                    manifest.name,
                    str(entry_point)
                )
                if spec is None or spec.loader is None:
                    logger.error(f"无法创建模块规范: {entry_point}")
                    return None

                module = importlib.util.module_from_spec(spec)
                sys.modules[manifest.name] = module
                spec.loader.exec_module(module)

                return module
            finally:
                # 从 sys.path 移除
                if str(folder) in sys.path:
                    sys.path.remove(str(folder))

        except Exception as e:
            logger.error(f"从文件夹加载插件模块失败 ({folder_path}): {e}")
            return None

    async def _register_components(self, plugin_instance: "BasePlugin") -> None:
        """注册插件的所有组件到全局注册表。

        Args:
            plugin_instance: 插件实例
        """
        registry = get_global_registry()

        # 获取插件的所有组件
        components = plugin_instance.get_components()
        plugin_name = plugin_instance.plugin_name

        for component_cls in components:
            # TODO: 注册组件到全局注册表
            # 需要组件类提供签名信息
            pass


# 全局插件管理器实例
_global_plugin_manager: PluginManager | None = None


def get_plugin_manager() -> PluginManager:
    """获取全局插件管理器实例。

    Returns:
        PluginManager: 全局插件管理器单例

    Examples:
        >>> manager = get_plugin_manager()
        >>> await manager.load_all_plugins("plugins")
    """
    global _global_plugin_manager
    if _global_plugin_manager is None:
        _global_plugin_manager = PluginManager()
    return _global_plugin_manager
