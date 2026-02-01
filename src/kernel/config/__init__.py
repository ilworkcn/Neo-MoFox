"""src.kernel.config

内核配置模块。

该模块提供基于 Pydantic 的类型安全配置文件系统，支持自动类型校验与 TOML 存储。

本模块采用“静态可见”的配置模型设计：配置类本身继承 :class:`pydantic.BaseModel`，
并在类体中显式声明各配置节字段（这样 IDE/Pylance 能正确推断类型）。

典型使用示例：
    ```python
    from src.kernel.config import ConfigBase, SectionBase, config_section, Field

    class MyConfig(ConfigBase):
        @config_section("general")
        class GeneralSection(SectionBase):
            enabled: bool = Field(default=True, description="启用功能")

        general: GeneralSection = Field(default_factory=GeneralSection)

    my_config = MyConfig.load("config/my_config.toml")
    print(my_config.general.enabled)
    ```
"""

from __future__ import annotations

from .core import ConfigBase, SectionBase, config_section, Field

__all__ = [
    "ConfigBase",
    "SectionBase",
    "config_section",
    "Field",
]
