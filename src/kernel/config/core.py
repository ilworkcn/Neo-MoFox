"""src.kernel.config.core

实现层：承载 config 模块的全部逻辑实现。

`src.kernel.config.__init__` 应保持轻量，只负责对外导出与文档。
"""

from __future__ import annotations

import inspect
from pathlib import Path
from typing import Any, Callable, TypeVar, Self

import tomllib
from pydantic import BaseModel, ConfigDict, Field, TypeAdapter


SectionT = TypeVar("SectionT", bound="SectionBase")

__all__ = [
    "ConfigBase",
    "SectionBase",
    "config_section",
    "Field",
]


def config_section(name: str) -> Callable[[type[SectionT]], type[SectionT]]:
    """配置节装饰器。

    重要：该装饰器使用泛型返回类型，确保 IDE/Pylance 能保留被装饰类的具体类型，
    避免把 `SectionB` 降级成 `SectionBase`，从而导致字段（如 `value_b`）无法被识别。
    """

    def decorator(cls: type[SectionT]) -> type[SectionT]:
        cls.__config_section_name__ = name  # type: ignore[attr-defined]
        return cls

    return decorator


class SectionBase(BaseModel):
    """配置节基类。

    配置节是一组相关的配置选项。它们会被 ConfigBase 自动收集并映射到 TOML 节。
    """

    model_config = ConfigDict(extra="forbid")


class ConfigBase(BaseModel):
    """配置基类（静态可见）。

    配置类本身是一个 Pydantic 模型，所有配置节都应作为字段显式声明。
    这能让 IDE/Pylance 在访问 `config.xxx.yyy` 时正确进行类型推断。
    """

    model_config = ConfigDict(extra="forbid")

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Self:
        """从字典加载配置。"""

        return cls.model_validate(data)

    @classmethod
    def load(
        cls,
        path: str | Path,
        *,
        auto_update: bool = False,
    ) -> Self:
        """从 TOML 文件加载配置。

        当 ``auto_update=True`` 时，会将配置文件内容与模型定义的“签名”进行比对：
        - 配置节/字段是否存在
        - 注释文档（section docstring + Field.description）
        - 字段类型（由类型注解推导）
        - 默认值（由 Field.default / default_factory 推导）

        若签名不一致，将自动回写 TOML 文件使其更新到模型定义的版本；同时尽可能保留
        文件中已有且能通过类型校验的用户值。
        """

        path = Path(path)
        original_text = path.read_text(encoding="utf-8")
        with path.open("rb") as f:
            raw = tomllib.load(f)

        if not auto_update:
            return cls.from_dict(raw)

        merged = _merge_with_model_defaults(cls, raw)
        new_text = _render_toml_with_signature(cls, merged)

        if _normalize_newlines(original_text) != _normalize_newlines(new_text):
            path.write_text(new_text, encoding="utf-8")

        return cls.from_dict(merged)

    @classmethod
    def default(cls) -> dict[str, Any]:
        """生成默认配置字典。"""

        return cls().model_dump()


def _normalize_newlines(text: str) -> str:
    return text.replace("\r\n", "\n").replace("\r", "\n")


def _type_repr(annotation: Any) -> str:
    try:
        return getattr(annotation, "__name__", None) or str(annotation)
    except Exception:
        return "unknown"


def _toml_escape_string(value: str) -> str:
    # TOML 基本字符串，最小转义
    escaped = value.replace("\\", "\\\\").replace('"', '\\"')
    return f'"{escaped}"'


def _toml_format_key(key: str) -> str:
    # 尽量使用裸键；否则退回到字符串键
    if key and all(ch.isalnum() or ch in {"_", "-"} for ch in key):
        return key
    return _toml_escape_string(key)


def _toml_format_value(value: Any) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, int):
        return str(value)
    if isinstance(value, float):
        return repr(value)
    if isinstance(value, str):
        return _toml_escape_string(value)
    if isinstance(value, list):
        return "[" + ", ".join(_toml_format_value(v) for v in value) + "]"
    if isinstance(value, dict):
        items: list[str] = []
        for k in sorted(value.keys(), key=lambda x: str(x)):
            if not isinstance(k, str):
                continue
            items.append(f"{_toml_format_key(k)} = {_toml_format_value(value[k])}")
        return "{ " + ", ".join(items) + " }"
    if value is None:
        # TOML 不支持 null；用空字符串占位
        return _toml_escape_string("")
    return _toml_escape_string(str(value))


def _get_section_name(section_model: type[SectionBase], fallback: str) -> str:
    name = getattr(section_model, "__config_section_name__", None)
    return str(name) if name else fallback


def _eval_default_factory(factory: Any) -> Any:
    """兼容 Pydantic v2 的 default_factory 形态（可能需要 validated_data）。"""

    try:
        return factory()
    except TypeError:
        return factory({})


def _iter_sections(config_model: type[ConfigBase]) -> list[tuple[str, type[SectionBase]]]:
    sections: list[tuple[str, type[SectionBase]]] = []
    for field_name, model_field in config_model.model_fields.items():
        annotation = model_field.annotation
        if isinstance(annotation, type) and issubclass(annotation, SectionBase):
            sections.append((_get_section_name(annotation, field_name), annotation))
    sections.sort(key=lambda x: x[0])
    return sections


def _merge_with_model_defaults(
    config_model: type[ConfigBase],
    raw: dict[str, Any],
) -> dict[str, Any]:
    """将 raw 与模型默认值合并，并移除模型未定义的节/字段。

    保留 raw 中能通过字段类型校验的值；不合法的值回退到默认值/占位值。
    """

    merged: dict[str, Any] = {}
    for section_name, section_model in _iter_sections(config_model):
        raw_section = raw.get(section_name)
        if not isinstance(raw_section, dict):
            raw_section = {}

        section_out: dict[str, Any] = {}
        for key, field in section_model.model_fields.items():
            annotation = field.annotation
            default_value = (
                field.default
                if field.default is not None and field.default is not ...
                else None
            )
            if field.default_factory is not None:
                try:
                    default_value = _eval_default_factory(field.default_factory)
                except Exception:
                    default_value = None

            if key in raw_section:
                candidate = raw_section[key]
                try:
                    section_out[key] = TypeAdapter(annotation).validate_python(candidate)
                    continue
                except Exception:
                    pass

            if default_value is not None:
                section_out[key] = default_value
            else:
                section_out[key] = _placeholder_for_type(annotation)

        merged[section_name] = section_out
    return merged


def _placeholder_for_type(annotation: Any) -> Any:
    origin = getattr(annotation, "__origin__", None)
    if annotation in (str,):
        return ""
    if annotation in (int,):
        return 0
    if annotation in (float,):
        return 0.0
    if annotation in (bool,):
        return False
    if origin is list or annotation is list:
        return []
    if origin is dict or annotation is dict:
        return {}
    return ""


def _render_toml_with_signature(
    config_model: type[ConfigBase],
    data: dict[str, Any],
) -> str:
    """按模型签名生成带注释的 TOML（确定性输出）。"""

    lines: list[str] = []
    sections = _iter_sections(config_model)

    for idx, (section_name, section_model) in enumerate(sections):
        if idx != 0:
            lines.append("")

        section_doc = inspect.getdoc(section_model) or ""
        if section_doc:
            for doc_line in section_doc.splitlines():
                lines.append(f"# {doc_line}")

        lines.append(f"[{section_name}]")

        section_data = data.get(section_name)
        if not isinstance(section_data, dict):
            section_data = {}

        for field_name, field in section_model.model_fields.items():
            description = field.description or ""
            if description:
                for doc_line in description.splitlines():
                    lines.append(f"# {doc_line}")

            annotation = field.annotation
            type_text = _type_repr(annotation)

            default_text = None
            if field.default_factory is not None:
                try:
                    default_text = _toml_format_value(
                        _eval_default_factory(field.default_factory)
                    )
                except Exception:
                    default_text = None
            elif field.default is not None and field.default is not ...:
                default_text = _toml_format_value(field.default)

            sig_parts = [f"type={type_text}"]
            if default_text is not None:
                sig_parts.append(f"default={default_text}")
            else:
                sig_parts.append("default=<required>")

            lines.append("# signature: " + ", ".join(sig_parts))

            value = section_data.get(field_name)
            lines.append(f"{field_name} = {_toml_format_value(value)}")
            lines.append("")

        while lines and lines[-1] == "":
            lines.pop()

    return "\n".join(lines).rstrip() + "\n"
