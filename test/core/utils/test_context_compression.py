"""Tests for context compression helpers."""

from __future__ import annotations

from src.core.utils.context_compression import _extract_summary_content


def test_extract_summary_content_discards_analysis_and_keeps_summary() -> None:
    """XML 提取成功时应只保留 summary 内容。"""
    raw_text = (
        "<analysis>ignored</analysis>"
        "<summary>保留这一段<nested>以及内部文本</nested></summary>"
    )

    assert _extract_summary_content(raw_text) == "保留这一段以及内部文本"


def test_extract_summary_content_falls_back_to_raw_text_on_parse_failure() -> None:
    """XML 提取失败时应直接回退原文。"""
    raw_text = "<summary>未闭合"

    assert _extract_summary_content(raw_text) == raw_text