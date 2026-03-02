"""LLM context management utilities."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

from .payload import LLMPayload
from .payload.content import Content, Text
from .payload.tooling import LLMUsable, ToolCall, ToolResult
from .roles import ROLE

CompressionHook = Callable[[list[list[LLMPayload]], list[LLMPayload]], list[LLMPayload]]
TokenCounter = Callable[[list[LLMPayload]], int]


@dataclass(slots=True)
class LLMContextManager:
    """上下文管理器。

    默认职责：
    1. 接管 payload 列表写入（add_payload/system/tool/reminder）；
    2. 在写入后执行结构校验与最小修复（工具调用链补齐、孤立消息清理）；
    3. 最后按 max_payloads/token_budget 执行裁剪。

    对于 reminder：固定注入到“首个 USER 消息的首段”；若尚无 USER，则立即创建空 USER 并写入 reminder。
    """

    max_payloads: int | None = None
    compression_hook: CompressionHook | None = None
    _reminders: list[Text] | None = None

    def add_payload(
        self,
        payloads: list[LLMPayload],
        payload: LLMPayload,
        position: int | None = None,
    ) -> list[LLMPayload]:
        """向上下文追加 payload，并进行规范化与裁剪。

        Args:
            payloads: 现有 payload 列表。
            payload: 待追加 payload。
            position: 可选插入位置。

        Returns:
            list[LLMPayload]: 规范化后的 payload 列表。
        """

        updated = list(payloads)

        if position is not None:
            updated.insert(int(position), payload)
        elif updated and updated[-1].role == payload.role:
            updated[-1].content.extend(payload.content)
        else:
            updated.append(payload)

        updated = self._normalize_payloads(updated)
        updated = self._apply_reminders(updated)
        return self.maybe_trim(updated)

    def system(
        self,
        payloads: list[LLMPayload],
        content: Content | LLMUsable | list[Content | LLMUsable],
        position: int | None = None,
    ) -> list[LLMPayload]:
        """追加 SYSTEM payload，语义等同于 add_payload。"""

        return self.add_payload(
            payloads,
            LLMPayload(ROLE.SYSTEM, content),
            position=position,
        )

    def tool(
        self,
        payloads: list[LLMPayload],
        content: Content | LLMUsable | list[Content | LLMUsable],
        position: int | None = None,
    ) -> list[LLMPayload]:
        """追加 TOOL payload，语义等同于 add_payload。"""

        return self.add_payload(
            payloads,
            LLMPayload(ROLE.TOOL, content),
            position=position,
        )

    def reminder(
        self,
        payloads: list[LLMPayload],
        content: str | Text | list[str | Text],
    ) -> list[LLMPayload]:
        """登记 reminder 并注入到首个 USER 消息首段。

        reminder 作为仅次于 system 的固有提示词，不单独作为 role 出现。
        """

        items = content if isinstance(content, list) else [content]
        if self._reminders is None:
            self._reminders = []

        for item in items:
            text_part = item if isinstance(item, Text) else Text(str(item))
            self._reminders.append(text_part)

        updated = self._normalize_payloads(list(payloads))
        updated = self._apply_reminders(updated)
        return self.maybe_trim(updated)

    def _normalize_payloads(self, payloads: list[LLMPayload]) -> list[LLMPayload]:
        """默认结构校验与修复。

        规则：
        - 固定消息（SYSTEM/TOOL）保持原顺序；
        - 对话消息仅保留合法链路：USER -> ASSISTANT，或 USER -> ASSISTANT(tool_calls) -> TOOL_RESULT* -> ASSISTANT；
        - 对于 ASSISTANT tool_calls 缺失 TOOL_RESULT 的情况，自动补最小空结果占位；
        - 孤立 ASSISTANT/TOOL_RESULT 直接丢弃。
        """

        pinned = [p for p in payloads if p.role in {ROLE.SYSTEM, ROLE.TOOL}]
        convo = [p for p in payloads if p.role not in {ROLE.SYSTEM, ROLE.TOOL}]

        normalized: list[LLMPayload] = []
        idx = 0

        while idx < len(convo):
            payload = convo[idx]

            if payload.role == ROLE.USER:
                normalized.append(payload)
                idx += 1
                continue

            if payload.role == ROLE.ASSISTANT:
                if not normalized or normalized[-1].role not in {ROLE.USER, ROLE.TOOL_RESULT}:
                    idx += 1
                    continue

                repaired_assistant, expected_ids = self._repair_assistant_tool_calls(payload)
                normalized.append(repaired_assistant)
                idx += 1

                if not expected_ids:
                    continue

                seen_ids: set[str] = set()
                while idx < len(convo) and convo[idx].role == ROLE.TOOL_RESULT:
                    repaired_result, call_id = self._repair_tool_result_payload(convo[idx], expected_ids - seen_ids)
                    if repaired_result is not None:
                        normalized.append(repaired_result)
                    if call_id:
                        seen_ids.add(call_id)
                    idx += 1

                for missing_id in expected_ids - seen_ids:
                    normalized.append(
                        LLMPayload(
                            ROLE.TOOL_RESULT,
                            ToolResult(value="", call_id=missing_id),
                        )
                    )

                if idx >= len(convo) or convo[idx].role != ROLE.ASSISTANT:
                    normalized.append(LLMPayload(ROLE.ASSISTANT, Text("")))

                continue

            # TOOL_RESULT 孤立消息：直接清理
            idx += 1

        return pinned + normalized

    def _repair_assistant_tool_calls(
        self,
        payload: LLMPayload,
    ) -> tuple[LLMPayload, set[str]]:
        """修复 assistant tool call，补齐缺失 id。"""

        expected_ids: set[str] = set()
        repaired_content: list[Content | LLMUsable] = []
        tool_index = 0

        for part in payload.content:
            if not isinstance(part, ToolCall):
                repaired_content.append(part)
                continue

            call_id = part.id if part.id else f"auto_call_{tool_index}"
            tool_index += 1
            expected_ids.add(call_id)
            repaired_content.append(ToolCall(id=call_id, name=part.name, args=part.args))

        if not expected_ids:
            return payload, expected_ids

        return LLMPayload(ROLE.ASSISTANT, repaired_content), expected_ids

    def _repair_tool_result_payload(
        self,
        payload: LLMPayload,
        candidate_ids: set[str],
    ) -> tuple[LLMPayload | None, str | None]:
        """修复 TOOL_RESULT payload，必要时补齐 call_id。"""

        first_result: ToolResult | None = None
        for part in payload.content:
            if isinstance(part, ToolResult):
                first_result = part
                break

        if first_result is None:
            if len(candidate_ids) == 1:
                only_id = next(iter(candidate_ids))
                return LLMPayload(ROLE.TOOL_RESULT, ToolResult(value="", call_id=only_id)), only_id
            return None, None

        call_id = first_result.call_id
        if not call_id and len(candidate_ids) == 1:
            call_id = next(iter(candidate_ids))

        if call_id is None:
            return None, None
        if call_id not in candidate_ids and candidate_ids:
            return None, None

        repaired_parts: list[Content | LLMUsable] = []
        replaced = False
        for part in payload.content:
            if isinstance(part, ToolResult) and not replaced:
                repaired_parts.append(
                    ToolResult(
                        value=part.value,
                        call_id=call_id,
                        name=part.name,
                    )
                )
                replaced = True
            else:
                repaired_parts.append(part)

        return LLMPayload(ROLE.TOOL_RESULT, repaired_parts), call_id

    def _apply_reminders(self, payloads: list[LLMPayload]) -> list[LLMPayload]:
        """将 reminder 固定注入首个 USER 消息首段。"""

        if not self._reminders:
            return payloads

        reminder_parts: list[Content | LLMUsable] = [Text(item.text) for item in self._reminders]
        updated = list(payloads)

        user_index = next((idx for idx, p in enumerate(updated) if p.role == ROLE.USER), None)
        if user_index is None:
            insert_index = 0
            while insert_index < len(updated) and updated[insert_index].role in {ROLE.SYSTEM, ROLE.TOOL}:
                insert_index += 1
            updated.insert(insert_index, LLMPayload(ROLE.USER, reminder_parts))
            return updated

        first_user = updated[user_index]
        existing = first_user.content

        matched_prefix_len = 0
        while (
            matched_prefix_len < len(reminder_parts)
            and matched_prefix_len < len(existing)
            and self._is_same_text_part(existing[matched_prefix_len], reminder_parts[matched_prefix_len])
        ):
            matched_prefix_len += 1

        if matched_prefix_len == len(reminder_parts):
            return updated

        rebuilt = reminder_parts + existing[matched_prefix_len:]
        updated[user_index] = LLMPayload(ROLE.USER, rebuilt)
        return updated

    def _is_same_text_part(self, left: Content | LLMUsable, right: Content | LLMUsable) -> bool:
        """判断两个内容片段是否为同一文本片段。"""

        return isinstance(left, Text) and isinstance(right, Text) and left.text == right.text

    def maybe_trim(
        self,
        payloads: list[LLMPayload],
        *,
        max_token_budget: int | None = None,
        token_counter: TokenCounter | None = None,
    ) -> list[LLMPayload]:
        """
        根据 max_payloads 和 max_token_budget 对 payloads 进行裁剪。

        裁剪策略：
        1. 保留开头的系统/工具消息（pinned prefix）。
        2. 将剩余消息按用户/助手对话分组，整体裁剪掉较早的对话组。
        3. 如果提供了 compression_hook，则在裁剪掉一批对话组后，调用该 hook 生成压缩后的消息，并将其插入剩余消息的开头。
        4. 如果 max_token_budget 仍然超出，则继续裁剪剩余的对话组，直到满足预算。
        """

        trimmed = payloads

        # 首先根据 max_payloads 进行裁剪
        if (
            self.max_payloads is not None
            and self.max_payloads > 0
            and len(trimmed) > self.max_payloads
        ):
            trimmed = self._trim_by_payloads(trimmed, self.max_payloads)

        # 然后根据 max_token_budget 进行裁剪
        if (
            max_token_budget is not None
            and max_token_budget > 0
            and token_counter is not None
            and token_counter(trimmed) > max_token_budget
        ):
            trimmed = self._trim_by_tokens(trimmed, max_token_budget, token_counter)

        return trimmed

    def _trim_by_tokens(
        self,
        payloads: list[LLMPayload],
        token_budget: int,
        token_counter: TokenCounter,
    ) -> list[LLMPayload]:
        """
        根据 token_budget 对 payloads 进行裁剪
        """

        pinned, tail = self._split_pinned_prefix(payloads)
        groups = self._build_qa_groups(tail)
        if not groups:
            return payloads

        kept_groups = list(groups)
        dropped_groups: list[list[LLMPayload]] = []

        # 先尝试直接裁剪对话组，保留 pinned 和尽可能多的对话组，直到满足 token 预算
        while len(kept_groups) > 1:
            candidate = pinned + self._flatten_groups(kept_groups)
            if token_counter(candidate) <= token_budget:
                break
            dropped_groups.append(kept_groups.pop(0))

        remaining_payloads = self._flatten_groups(kept_groups)

        # 如果提供了 compression_hook，则在裁剪掉一批对话组后，调用该 hook 生成压缩后的消息，并将其插入剩余消息的开头
        hook_payloads = self._apply_compression_hook(dropped_groups, remaining_payloads)
        if hook_payloads:
            combined = pinned + hook_payloads + remaining_payloads
            while len(kept_groups) > 1 and token_counter(combined) > token_budget:
                kept_groups.pop(0)
                remaining_payloads = self._flatten_groups(kept_groups)
                combined = pinned + hook_payloads + remaining_payloads
            return combined

        return pinned + remaining_payloads

    def _trim_by_payloads(
        self, payloads: list[LLMPayload], max_payloads: int
    ) -> list[LLMPayload]:
        """
        根据 max_payloads 对 payloads 进行裁剪
        """
        pinned, tail = self._split_pinned_prefix(payloads)
        groups = self._build_qa_groups(tail)
        if not groups:
            return payloads

        kept_groups = list(groups)
        dropped_groups: list[list[LLMPayload]] = []

        # 先尝试直接裁剪对话组，保留 pinned 和尽可能多的对话组，直到满足 max_payloads 约束
        while (
            len(kept_groups) > 1
            and self._payload_len(pinned, kept_groups) > max_payloads
        ):
            dropped_groups.append(kept_groups.pop(0))

        remaining_payloads = self._flatten_groups(kept_groups)

        # 如果提供了 compression_hook，则在裁剪掉一批对话组后，调用该 hook 生成压缩后的消息，并将其插入剩余消息的开头
        hook_payloads = self._apply_compression_hook(dropped_groups, remaining_payloads)
        if hook_payloads:
            remaining_payloads = self._flatten_groups(kept_groups)

        # 最后检查一次总长度，如果仍然超出 max_payloads，则继续裁剪剩余的对话组，直到满足约束
        while len(kept_groups) > 1 and (
            len(pinned) + len(hook_payloads) + len(remaining_payloads) > max_payloads
        ):
            kept_groups.pop(0)
            remaining_payloads = self._flatten_groups(kept_groups)

        return pinned + hook_payloads + remaining_payloads

    def _split_pinned_prefix(
        self, payloads: list[LLMPayload]
    ) -> tuple[list[LLMPayload], list[LLMPayload]]:
        """将 payloads 拆分为 pinned 消息和对话消息两部分。

        pinned 消息定义为：所有 SYSTEM 和 TOOL 角色的消息，无论其出现在列表的任何位置，
        均视为固定部分，始终被保留，不参与裁剪。
        对话消息为剩余的 USER 和 ASSISTANT 消息，按原始顺序保留。
        """
        pinned_roles = {ROLE.SYSTEM, ROLE.TOOL}
        pinned = [p for p in payloads if p.role in pinned_roles]
        tail = [p for p in payloads if p.role not in pinned_roles]
        return pinned, tail

    def _build_qa_groups(self, payloads: list[LLMPayload]) -> list[list[LLMPayload]]:
        """将消息分组。一个组作为一个不可分割的最小裁剪单位。

        分组策略：
        1. 每一个 USER 角色开始一个新组。
        2. 后续的 ASSISTANT 和 TOOL_RESULT 消息紧跟在该 USER 组内。
        3. 如果在第一个 USER 之前有孤立的消息（如历史遗留），它们会各自独立成组。
        """
        groups: list[list[LLMPayload]] = []
        current: list[LLMPayload] = []

        for payload in payloads:
            # 遇到 USER 角色，开启新组
            if payload.role == ROLE.USER:
                if current:
                    groups.append(current)
                current = [payload]
            elif not current:
                # 处理第一个 USER 之前的孤立消息
                groups.append([payload])
            else:
                # 归入当前组（确保 user-assistant-tool_result 连带关系）
                current.append(payload)

        if current:
            groups.append(current)

        return groups

    def _apply_compression_hook(
        self,
        dropped_groups: list[list[LLMPayload]],
        remaining_payloads: list[LLMPayload],
    ) -> list[LLMPayload]:
        """调用压缩 hook 生成压缩后的消息"""
        if not self.compression_hook or not dropped_groups:
            return []
        return self.compression_hook(dropped_groups, remaining_payloads)

    def _flatten_groups(self, groups: list[list[LLMPayload]]) -> list[LLMPayload]:
        """将分组扁平化为单一列表"""
        return [payload for group in groups for payload in group]

    def _payload_len(
        self, pinned: list[LLMPayload], groups: list[list[LLMPayload]]
    ) -> int:
        """
        计算有效负载的总长度，包括 pinned 消息和对话组消息
        """
        return len(pinned) + sum(len(group) for group in groups)
