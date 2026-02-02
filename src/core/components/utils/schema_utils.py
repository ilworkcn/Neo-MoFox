"""Schema 生成工具函数。

本模块提供 LLM Tool Schema 生成的通用工具函数。
供 Action 和 Tool 组件共享使用，避免非法依赖。
"""

import inspect
from typing import Any, get_args, get_origin, Callable


# Python 类型到 JSON Schema 类型的映射
_TYPE_MAPPING: dict[type, str] = {
    int: "integer",
    float: "number",
    str: "string",
    bool: "boolean",
    list: "array",
    dict: "object",
}


def map_type_to_json(type_hint: Any) -> str:
    """将 Python 类型注解映射到 JSON Schema 类型。

    Args:
        type_hint: Python 类型注解

    Returns:
        str: JSON Schema 类型字符串

    Examples:
        >>> map_type_to_json(int)
        'integer'
        >>> map_type_to_json(list[int])
        'array'
    """
    # 处理 None 类型
    if type_hint is type(None):
        return "null"

    # 处理 Annotated 类型
    origin = get_origin(type_hint)
    if origin is not None:
        # 如果是 Annotated[T, ...]，提取 T
        if origin.__name__ == "Annotated":
            args = get_args(type_hint)
            if args:
                return map_type_to_json(args[0])
        # 如果是 Union[T, None] 或 Optional[T]
        elif origin.__name__ in ("Union", "Optional"):
            args = get_args(type_hint)
            if args:
                # 过滤掉 None
                non_none_args = [arg for arg in args if arg is not type(None)]
                if non_none_args:
                    return map_type_to_json(non_none_args[0])
        # 如果是 list[T] 或其他泛型
        else:
            # 检查是否是 list、dict 等容器类型
            for container_type in (list, dict, set, tuple):
                if origin is container_type:
                    return _TYPE_MAPPING.get(container_type, "object")

    # 处理字符串类型的类型提示（如 "int"）
    if isinstance(type_hint, str):
        type_hint = eval(type_hint, {}, {})

    # 直接类型映射
    return _TYPE_MAPPING.get(type_hint, "string")


def parse_function_signature(
    func: Callable,
    component_name: str,
    component_description: str,
) -> dict[str, Any]:
    """解析函数签名并生成 LLM Tool Schema。

    Args:
        func: 要解析的函数（通常是 execute 方法）
        component_name: 组件名称
        component_description: 组件描述

    Returns:
        dict[str, Any]: OpenAI Tool 格式的 schema

    Examples:
        >>> schema = parse_function_signature(
        ...     my_action.execute,
        ...     "send_message",
        ...     "发送消息到用户"
        ... )
    """
    sig = inspect.signature(func)
    parameters = {}

    # 遍历函数参数
    for param_name, param in sig.parameters.items():
        # 跳过 self
        if param_name == "self":
            continue

        param_info: dict[str, Any] = {
            "type": map_type_to_json(param.annotation),
            "description": f"{param_name} 参数",
        }

        # 处理默认值
        if param.default != inspect.Parameter.empty:
            param_info["default"] = param.default

        # 添加参数描述（可以从 docstring 解析）
        parameters[param_name] = param_info

    return {
        "type": "function",
        "function": {
            "name": component_name,
            "description": component_description,
            "parameters": {
                "type": "object",
                "properties": parameters,
                "required": [
                    name
                    for name, param in sig.parameters.items()
                    if name != "self" and param.default == inspect.Parameter.empty
                ],
            },
        },
    }


def extract_description_from_docstring(func: Callable) -> str:
    """从函数的 docstring 提取描述。

    Args:
        func: 函数对象

    Returns:
        str: 提取的描述，如果没有 docstring 则返回空字符串

    Examples:
        >>> def example_func():
        ...     \"\"\"这是一个示例函数。\"\"\"
        ...     pass
        >>> extract_description_from_docstring(example_func)
        '这是一个示例函数。'
    """
    if func.__doc__:
        return func.__doc__.strip().split("\n")[0]
    return ""
