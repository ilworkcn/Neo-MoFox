"""Component registry for discovery and lookup."""

_registry = {}


def register(name: str, obj):
    _registry[name] = obj


def get(name: str):
    return _registry.get(name)
