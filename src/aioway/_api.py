# Copyright (c) AIoWay Authors - All Rights Reserved

"The module that registeres the API, and expose in `aioway.api` module."

import copy
import types
from collections import abc as cabc

__all__ = ["public", "registered"]

type FuncOrClass = type | types.FunctionType

_REGISTRY: dict[str, FuncOrClass] = {}


def public[T: FuncOrClass](item: T) -> T:
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


def registered(export: str | None = None) -> cabc.Mapping[str, FuncOrClass]:
    """
    Get the registered modules.

    Args:
        export: If supllied, set the `__module__` to the value given.

    Returns:
        The mapping of `__name__` to the publically registered modules.
    """

    if export is None:
        return _REGISTRY

    def re_export(item: FuncOrClass) -> FuncOrClass:
        item = copy.copy(item)
        item.__module__ = export
        return item

    return {key: re_export(val) for key, val in _REGISTRY.items()}
