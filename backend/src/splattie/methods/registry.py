"""Registry for available asset generation methods."""

from __future__ import annotations

from typing import TYPE_CHECKING

from splattie.types import MethodInfo

if TYPE_CHECKING:
    from splattie.methods.base import AssetGenerationMethod


class MethodRegistry:
    """Singleton registry for asset generation methods."""

    def __init__(self) -> None:
        self._methods: dict[str, type[AssetGenerationMethod]] = {}
        self._instances: dict[str, AssetGenerationMethod] = {}

    def register(self, method_cls: type[AssetGenerationMethod]) -> type[AssetGenerationMethod]:
        """Register a method class. Can be used as a decorator."""
        info = method_cls.__dict__.get("_info")
        if info is None:
            instance = method_cls()
            method_id = instance.info.id
        else:
            method_id = info.id
        self._methods[method_id] = method_cls
        return method_cls

    def get(self, method_id: str) -> AssetGenerationMethod:
        """Get or create a method instance."""
        if method_id not in self._instances:
            if method_id not in self._methods:
                msg = f"Unknown method: {method_id}. Available: {list(self._methods.keys())}"
                raise KeyError(msg)
            self._instances[method_id] = self._methods[method_id]()
        return self._instances[method_id]

    def for_asset_type(self, asset_type: AssetType) -> AssetGenerationMethod:
        """Return the registered method for an asset category (head/body/object).

        Lets the API select by category so the method behind a category can change
        without altering the endpoint URL. Returns the first method registered for
        the type (one per type today: lam=head, lhm=body).
        """
        for method_id in self._methods:
            instance = self.get(method_id)
            if instance.info.asset_type == asset_type:
                return instance
        available = [(m.id, m.asset_type.value) for m in self.list_available()]
        msg = f"No method registered for asset type {asset_type.value!r}. Available: {available}"
        raise KeyError(msg)

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
