from enum import Enum
from src.kernel.llm import LLMContextManager, LLMRequest, ModelSet, ToolRegistry, LLMUsable, ToolExecutor, ToolCall, ToolResult
from src.core.config import get_model_config

class TaskType(Enum):
    UTILS = "utils"
    UTILS_SMALL = "utils_small"
    ACTOR = "actor"
    SUB_ACTOR = "sub_actor"
    VLM = "vlm"
    VOICE = "voice"
    VIDEO = "video"
    TOOL_USE = "tool_use"
    
def create_llm_request(
    model_set: ModelSet,
    request_name: str = "",
    context_manager: LLMContextManager | None = None,
) -> LLMRequest:
    """创建 LLMRequest 实例

    Args:
        model_set: 模型集
        request_name: 请求名称（可选）
        context_manager: 上下文管理器（可选）

    Returns:
        LLMRequest 实例
    """
    return LLMRequest(
        model_set=model_set,
        request_name=request_name,
        context_manager=context_manager,
    )

def get_model_set_by_task(name: str) -> ModelSet:
    """根据任务名称获取 ModelSet

    Args:
        name: 模型集名称

    Returns:
        ModelSet 实例
    """
    return get_model_config().get_task(name)

def create_tool_registry(tools: list[type[LLMUsable]] | None = None) -> ToolRegistry:
    """创建工具注册表实例"""
    registry = ToolRegistry()
    if tools:
        for tool in tools:
            registry.register(tool)
    return registry
