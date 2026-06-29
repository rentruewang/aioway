# Copyright (c) AIoWay Authors - All Rights Reserved

"The module that registeres the API, and expose in `aioway.api` module."

import copy
import types
from collections import abc as cabc
import dataclasses as dcls

__all__ = ["register_public_api", "public_api"]

type FuncOrClass = type | types.FunctionType

_REGISTRY: dict[str, FuncOrClass] = {}


def register_public_api[T: FuncOrClass](item: T) -> T:
    """
    Register the function or class as a public API.
    """

    if item.__name__ in _REGISTRY:
        raise KeyError(
            f"Name: {item.__name__} already registered. "
            f"Attempt to register: {item}, "
            f"found object with the same name: {_REGISTRY[item.__name__]}."
        )

    _REGISTRY[item.__name__] = item
    return item


@dcls.dataclass(frozen=True)
class _ReExportReg(cabc.Mapping[str, FuncOrClass]):
    "Re-export the items in the registry."

    module: str
    "The module name to rename."

    def __len__(self) -> int:
        return len(_REGISTRY)

    def __getitem__(self, name: str) -> FuncOrClass:
        item = _REGISTRY[name]
        item = copy.copy(item)
        item.__module__ = self.module
        return item

    def __iter__(self) -> cabc.Generator[str]:
        yield from _REGISTRY


def public_api(export: str | None = None) -> cabc.Mapping[str, FuncOrClass]:
    """
    Get the registered modules.

    Args:
        export: If supllied, set the `__module__` to the value given.

    Returns:
        The mapping of `__name__` to the publically registered modules.
    """

    if export is None:
        return _REGISTRY

    # Return a custom mapping s.t. we don't do extra computation everytime this is called.
    else:
        return _ReExportReg(export)
