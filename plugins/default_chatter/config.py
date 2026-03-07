"""DefaultChatter 插件配置定义。"""

from __future__ import annotations

from typing import ClassVar, Literal

from src.core.components.base.config import BaseConfig, Field, SectionBase, config_section


class DefaultChatterConfig(BaseConfig):
    """DefaultChatter 配置。"""

    config_name: ClassVar[str] = "config"
    config_description: ClassVar[str] = "DefaultChatter 配置"

    @config_section("plugin", title="插件设置", tag="plugin", order=0)
    class PluginSection(SectionBase):
        """插件基础配置。"""

        @config_section("theme_guide", title="场景引导", tag="text", order=10)
        class ThemeGuideSection(SectionBase):
            """不同聊天类型的人设/语气引导。"""

            private: str = Field(
                default="你当前正处于“私聊”环境中，你可以以更贴近一对一陪伴感的交流方式与用户互动，关注用户情绪并提供更直接、细腻的回应。",
                description="私聊场景的额外提示词",
                label="私聊场景提示",
                input_type="textarea",
                rows=3,
                tag="text",
                order=0
            )
            group: str = Field(
                default="你当前正处于“群聊”环境中，你需要注意多人对话上下文，注意对方有没有真的在和你对话，群聊通常同时有多个用户，贸然插入对话会十分破坏气氛。优先回应与当前话题强相关或明确提及你的内容，表达简洁自然。",
                description="群聊场景的额外提示词",
                label="群聊场景提示",
                input_type="textarea",
                rows=3,
                tag="text",
                order=1
            )

        enabled: bool = Field(
            default=True,
            description="是否启用 DefaultChatter",
            label="启用插件",
            tag="plugin",
            order=0
        )
        mode: Literal["enhanced", "classical"] = Field(
            default="enhanced",
            description="执行模式: enhanced/classical",
            label="执行模式",
            input_type="select",
            choices=["enhanced", "classical"],
            tag="performance",
            hint="enhanced 模式更智能但消耗更多资源",
            order=1
        )
        reinforce_negative_behaviors: bool = Field(
            default=True,
            description="是否在每轮 user 提示词的 extra 板块中再次强调负面行为约束",
            label="增强负面行为约束",
            tag="ai",
            hint="开启后会在每轮对话中强调禁止行为",
            order=2
        )
        theme_guide: ThemeGuideSection = Field(
            default_factory=ThemeGuideSection,
            description="按聊天类型区分的额外提示词",
            label="场景引导配置",
            order=3
        )

    plugin: PluginSection = Field(default_factory=PluginSection)
