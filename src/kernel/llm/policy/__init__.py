"""LLM policy 导出与默认策略工厂。"""

from __future__ import annotations

import threading
from collections.abc import Callable

from .base import ModelStep, Policy, PolicySession
from .load_balanced import LoadBalancedPolicy
from .round_robin import RoundRobinPolicy


def create_policy(policy_name: str) -> Policy:
    """根据策略名称创建 policy 实例。"""
    if policy_name == "round_robin":
        return RoundRobinPolicy()
    if policy_name == "load_balanced":
        return LoadBalancedPolicy()
    raise ValueError(f"Unsupported llm policy: {policy_name}")


def _builtin_default_policy_factory() -> Policy:
    """内建默认 policy 工厂。"""
    return LoadBalancedPolicy()


_default_policy_factory: Callable[[], Policy] = _builtin_default_policy_factory
_default_policy_lock = threading.Lock()
_default_policy_instance: Policy | None = None


def set_default_policy_factory(factory: Callable[[], Policy] | None) -> None:
    """设置默认 policy 工厂。

    上层可在初始化阶段通过依赖注入传入自定义工厂；传入 None 时恢复内建默认。
    """
    global _default_policy_factory, _default_policy_instance
    with _default_policy_lock:
        _default_policy_factory = factory or _builtin_default_policy_factory
        _default_policy_instance = None


def create_default_policy() -> Policy:
    """创建默认 policy。

    由上层通过依赖注入提供默认 policy 工厂；若未注入，则回退到内建默认
    load_balanced。同一默认工厂会复用一个 policy 实例，使轮询计数与负载均衡
    统计能够跨请求生效。调用 set_default_policy_factory 会重置该缓存。
    """
    global _default_policy_instance
    with _default_policy_lock:
        if _default_policy_instance is None:
            _default_policy_instance = _default_policy_factory()
        return _default_policy_instance

__all__ = [
    "Policy",
    "PolicySession",
    "ModelStep",
    "LoadBalancedPolicy",
    "RoundRobinPolicy",
    "create_policy",
    "create_default_policy",
    "set_default_policy_factory",
]
