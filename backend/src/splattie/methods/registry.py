"""Registry for available head generation methods."""

from __future__ import annotations

from typing import TYPE_CHECKING

from splattie.types import MethodInfo

if TYPE_CHECKING:
    from splattie.methods.base import HeadGenerationMethod


class MethodRegistry:
    """Singleton registry for head generation methods."""

    def __init__(self) -> None:
        self._methods: dict[str, type[HeadGenerationMethod]] = {}
        self._instances: dict[str, HeadGenerationMethod] = {}

    def register(self, method_cls: type[HeadGenerationMethod]) -> type[HeadGenerationMethod]:
        """Register a method class. Can be used as a decorator."""
        info = method_cls.__dict__.get("_info")
        if info is None:
            instance = method_cls()
            method_id = instance.info.id
        else:
            method_id = info.id
        self._methods[method_id] = method_cls
        return method_cls

    def get(self, method_id: str) -> HeadGenerationMethod:
        """Get or create a method instance."""
        if method_id not in self._instances:
            if method_id not in self._methods:
                msg = f"Unknown method: {method_id}. Available: {list(self._methods.keys())}"
                raise KeyError(msg)
            self._instances[method_id] = self._methods[method_id]()
        return self._instances[method_id]

    def list_available(self) -> list[MethodInfo]:
        """List all registered methods."""
        result = []
        for method_id in self._methods:
            instance = self.get(method_id)
            result.append(instance.info)
        return result

    @property
    def default_method_id(self) -> str | None:
        """Return the first registered method ID, or None."""
        return next(iter(self._methods), None)


registry = MethodRegistry()
