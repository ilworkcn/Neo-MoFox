"""Action base for components."""

from typing import Protocol


class Action(Protocol):
    def run(self, *args, **kwargs):
        ...
