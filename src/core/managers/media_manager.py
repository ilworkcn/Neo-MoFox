"""媒体管理器。

负责图片和表情包的识别、存储和管理。

功能：
- 使用 VLM 识别图片和表情包内容
- 缓存识别结果到数据库，避免重复识别
- 管理媒体文件的存储和检索
- 支持按哈希值去重，节省存储和计算资源

设计原则：
- 优先从缓存读取，减少 VLM 调用
- 使用哈希值标识图片，避免重复处理
- 异步处理，不阻塞主流程
- 异常友好，识别失败不影响消息流转
"""

from __future__ import annotations

import hashlib
import time
from typing import Any

from src.kernel.logger import get_logger

logger = get_logger("media_manager")

# 单例实例
_media_manager: "MediaManager | None" = None


class MediaManager:
    """媒体管理器。
    
    管理图片、表情包等媒体资源的识别、存储和检索。
    
    主要功能：
    1. VLM 识别：调用 VLM 模型识别图片/表情包内容
    2. 缓存管理：使用哈希值缓存识别结果
    3. 数据库存储：持久化媒体信息
    4. 去重优化：相同内容的图片只识别一次
    
    Examples:
        >>> manager = get_media_manager()
        >>> description = await manager.recognize_image(base64_data, "image")
        >>> await manager.save_media_info(...)
    """

    def __init__(self):
        """初始化媒体管理器。"""
        self._vlm_model_set = None
        self._initialize_vlm()
        self._register_prompts()

    def _initialize_vlm(self) -> None:
        """初始化 VLM 模型配置。"""
        try:
            from src.app.plugin_system.api.llm_api import get_model_set_by_task

            self._vlm_model_set = get_model_set_by_task("vlm")
            self._vlm_available = self._vlm_model_set is not None
            
            if self._vlm_available:
                logger.info("VLM 模型已加载，媒体识别功能可用")
            else:
                logger.info("未配置 VLM 模型，媒体识别功能不可用")
        except Exception as e:
            logger.error(f"初始化 VLM 模型失败: {e}")

    def _register_prompts(self) -> None:
        """注册媒体识别相关的提示词模板。"""
        try:
            from src.core.prompt import PromptTemplate, get_prompt_manager
            
            manager = get_prompt_manager()
            
            # 注册图片识别提示词
            image_prompt = PromptTemplate(
                name="media.image_recognition",
                template="请简要描述这张图片的内容，用一句话概括。"
            )
            manager.register_template(image_prompt)
            
            # 注册表情包识别提示词
            emoji_prompt = PromptTemplate(
                name="media.emoji_recognition",
                template="请简要描述这个表情包的内容和含义，用一句话概括。"
            )
            manager.register_template(emoji_prompt)
            
            logger.debug("媒体识别提示词模板已注册")
        except Exception as e:
            logger.warning(f"注册提示词模板失败: {e}")

    # ──────────────────────────────────────────
    # 公共 API：媒体识别
    # ──────────────────────────────────────────

    async def recognize_media(
        self, 
        base64_data: str, 
        media_type: str,
        use_cache: bool = True
    ) -> str | None:
        """识别媒体内容（图片或表情包）。
        
        Args:
            base64_data: base64 编码的媒体数据
            media_type: 媒体类型，"image" 或 "emoji"
            use_cache: 是否使用缓存（默认 True）
            
        Returns:
            媒体的文字描述，识别失败返回 None
        """
        try:
            # 计算哈希值
            media_hash = self._compute_hash(base64_data)
            
            # 尝试从缓存读取
            if use_cache:
                cached_description = await self._get_cached_description(
                    media_hash, 
                    media_type
                )
                if cached_description:
                    logger.debug(f"从缓存获取{media_type}描述: {media_hash[:8]}...")
                    return cached_description
            
            # VLM 识别
            description = await self._recognize_with_vlm(base64_data, media_type)
            
            if description:
                # 保存到缓存
                await self._save_description_cache(
                    media_hash,
                    media_type,
                    description
                )
                logger.info(f"成功识别{media_type}: {description[:50]}...")
            
            return description
            
        except Exception as e:
            logger.error(f"识别{media_type}失败: {e}", exc_info=True)
            return None

    async def recognize_batch(
        self,
        media_list: list[tuple[str, str]],
        use_cache: bool = True
    ) -> list[tuple[int, str | None]]:
        """批量识别多个媒体。
        
        Args:
            media_list: [(base64_data, media_type), ...] 列表
            use_cache: 是否使用缓存
            
        Returns:
            [(index, description), ...] 列表，description 为 None 表示识别失败
        """
        results = []
        for idx, (base64_data, media_type) in enumerate(media_list):
            description = await self.recognize_media(
                base64_data,
                media_type,
                use_cache=use_cache
            )
            results.append((idx, description))
        return results

    # ──────────────────────────────────────────
    # 公共 API：数据库操作
    # ──────────────────────────────────────────

    async def save_media_info(
        self,
        media_hash: str,
        media_type: str,
        file_path: str | None = None,
        description: str | None = None,
        vlm_processed: bool = False
    ) -> None:
        """保存媒体信息到数据库。
        
        Args:
            media_hash: 媒体哈希值（作为唯一标识）
            media_type: 媒体类型（image/emoji）
            file_path: 文件路径（可选）
            description: 描述文本（可选）
            vlm_processed: 是否已经过 VLM 处理
        """
        try:
            from src.kernel.db.core.session import get_db_session
            from src.core.models.sql_alchemy import Images

            async with get_db_session() as session:
                # 查找现有记录
                from sqlalchemy import select
                stmt = select(Images).where(Images.path == media_hash)
                result = await session.execute(stmt)
                existing = result.scalar_one_or_none()

                if existing:
                    # 更新现有记录
                    existing.count += 1
                    if description:
                        existing.description = description
                    if vlm_processed:
                        existing.vlm_processed = True
                    logger.debug(f"更新媒体记录: {media_hash[:8]}... count={existing.count}")
                else:
                    # 创建新记录
                    new_image = Images(
                        image_id=media_hash,
                        path=file_path or media_hash,  # 如果没有路径，用哈希值
                        type=media_type,
                        description=description,
                        timestamp=time.time(),
                        vlm_processed=vlm_processed,
                        count=1
                    )
                    session.add(new_image)
                    logger.debug(f"创建新媒体记录: {media_hash[:8]}...")

                await session.commit()

        except Exception as e:
            logger.error(f"保存媒体信息失败: {e}", exc_info=True)

    async def get_media_info(self, media_hash: str) -> dict[str, Any] | None:
        """根据哈希值获取媒体信息。
        
        Args:
            media_hash: 媒体哈希值
            
        Returns:
            媒体信息字典，不存在返回 None
        """
        try:
            from src.kernel.db.core.session import get_db_session
            from src.core.models.sql_alchemy import Images
            from sqlalchemy import select

            async with get_db_session() as session:
                stmt = select(Images).where(Images.path == media_hash)
                result = await session.execute(stmt)
                media = result.scalar_one_or_none()

                if media:
                    return {
                        "id": media.id,
                        "image_id": media.image_id,
                        "path": media.path,
                        "type": media.type,
                        "description": media.description,
                        "count": media.count,
                        "timestamp": media.timestamp,
                        "vlm_processed": media.vlm_processed
                    }
                return None

        except Exception as e:
            logger.error(f"查询媒体信息失败: {e}", exc_info=True)
            return None

    # ──────────────────────────────────────────
    # 内部方法
    # ──────────────────────────────────────────

    async def _recognize_with_vlm(
        self, 
        base64_data: str, 
        media_type: str
    ) -> str | None:
        """使用 VLM 识别单个媒体。
        
        Args:
            base64_data: base64 编码的媒体数据
            media_type: 媒体类型（image 或 emoji）
            
        Returns:
            识别结果文本，失败返回 None
        """
        try:
            from src.app.plugin_system.api.llm_api import create_llm_request
            from src.kernel.llm import LLMContextManager, LLMPayload, ROLE, Text, Image
            from src.core.prompt import get_prompt_manager
            
            # 检查 VLM 模型是否可用
            if not self._vlm_model_set:
                logger.debug("VLM 模型不可用")
                return None

            # 创建 VLM 请求
            context_manager = LLMContextManager(max_payloads=3)
            request = create_llm_request(
                self._vlm_model_set,
                "image_recognition",
                context_manager=context_manager,
            )

            # 从提示词管理器获取提示词模板
            prompt_manager = get_prompt_manager()
            if media_type == "emoji":
                template = prompt_manager.get_template("media.emoji_recognition")
            else:
                template = prompt_manager.get_template("media.image_recognition")
            
            # 构建提示词（模板不需要参数，直接build）
            if template:
                prompt = template.build()

            # 处理 base64 数据：提取纯净的 base64 内容
            clean_base64 = self._extract_clean_base64(base64_data)
            
            # 使用标准的 data URL 格式（大多数 VLM API 都支持）
            # 假设是 PNG 图片，如果需要可以根据实际情况调整
            image_value = f"data:image/png;base64,{clean_base64}"

            # 添加 payload 并发送请求
            request.add_payload(LLMPayload(ROLE.USER, [Text(prompt), Image(image_value)]))
            response = await request.send(stream=False)
            await response

            # 提取并处理描述
            description = response.message.strip() if response.message else ""
            
            # 限制长度
            if len(description) > 100:
                description = description[:97] + "..."

            return description if description else None

        except Exception as e:
            logger.error(f"VLM 识别失败: {e}", exc_info=True)
            return None

    async def _get_cached_description(
        self,
        media_hash: str,
        media_type: str
    ) -> str | None:
        """从数据库缓存获取描述。
        
        Args:
            media_hash: 媒体哈希值
            media_type: 媒体类型
            
        Returns:
            缓存的描述，不存在返回 None
        """
        try:
            from src.kernel.db.core.session import get_db_session
            from src.core.models.sql_alchemy import ImageDescriptions
            from sqlalchemy import select

            async with get_db_session() as session:
                stmt = select(ImageDescriptions).where(
                    ImageDescriptions.image_description_hash == media_hash,
                    ImageDescriptions.type == media_type
                )
                result = await session.execute(stmt)
                desc = result.scalar_one_or_none()

                return desc.description if desc else None

        except Exception as e:
            logger.debug(f"查询缓存失败: {e}")
            return None

    async def _save_description_cache(
        self,
        media_hash: str,
        media_type: str,
        description: str
    ) -> None:
        """保存描述到缓存。
        
        Args:
            media_hash: 媒体哈希值
            media_type: 媒体类型
            description: 描述文本
        """
        try:
            from src.kernel.db.core.session import get_db_session
            from src.core.models.sql_alchemy import ImageDescriptions
            from sqlalchemy import select

            async with get_db_session() as session:
                # 检查是否已存在
                stmt = select(ImageDescriptions).where(
                    ImageDescriptions.image_description_hash == media_hash,
                    ImageDescriptions.type == media_type
                )
                result = await session.execute(stmt)
                existing = result.scalar_one_or_none()

                if not existing:
                    # 创建新缓存记录
                    new_desc = ImageDescriptions(
                        image_description_hash=media_hash,
                        type=media_type,
                        description=description,
                        timestamp=time.time()
                    )
                    session.add(new_desc)
                    await session.commit()
                    logger.debug(f"保存描述缓存: {media_hash[:8]}...")

        except Exception as e:
            logger.error(f"保存描述缓存失败: {e}", exc_info=True)

    @staticmethod
    def _extract_clean_base64(data: str) -> str:
        """提取纯净的 base64 数据（移除前缀和多余字符）。
        
        Args:
            data: 可能包含前缀的 base64 字符串
            
        Returns:
            纯净的 base64 字符串
        """
        # 移除可能的 data URL 前缀
        if data.startswith("data:"):
            # 提取 base64 部分
            if "base64," in data:
                data = data.split("base64,", 1)[1]
        elif data.startswith("base64|"):
            data = data[7:]
        
        # 移除可能的换行符和空格
        data = data.replace("\n", "").replace("\r", "").replace(" ", "")
        
        return data
    
    @staticmethod
    def _compute_hash(data: str) -> str:
        """计算数据的 SHA256 哈希值。
        
        Args:
            data: 待哈希的数据（base64 字符串）
            
        Returns:
            十六进制哈希字符串
        """
        # 使用提取的纯净 base64 数据计算哈希
        clean_data = MediaManager._extract_clean_base64(data)
        return hashlib.sha256(clean_data.encode()).hexdigest()


# ──────────────────────────────────────────
# 单例访问
# ──────────────────────────────────────────


def get_media_manager() -> MediaManager:
    """获取媒体管理器单例。
    
    Returns:
        MediaManager 实例
    """
    global _media_manager
    if _media_manager is None:
        _media_manager = MediaManager()
    return _media_manager


def initialize_media_manager() -> MediaManager:
    """初始化媒体管理器（用于显式初始化）。
    
    Returns:
        MediaManager 实例
    """
    global _media_manager
    _media_manager = MediaManager()
    return _media_manager
