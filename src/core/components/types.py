"""Component related types."""

from typing import TypedDict


class ComponentMeta(TypedDict, total=False):
    name: str
    version: str
    description: str
