# Copyright (c) AIoWay Authors - All Rights Reserved

"The module that registeres the API, and expose in `aioway.api` module."

import copy
import dataclasses as dcls
import typing
from collections import abc as cabc

__all__ = ["public_api", "public_items", "AiowayApi"]


_REGISTRY: dict[str, FuncOrClass] = {}


class FuncOrClass(typing.Protocol):
    __module__: str
    __name__: str
    __qualname__: str
    __doc__: str | None


@typing.runtime_checkable
class AiowayApi[**P = ..., T = typing.Any](FuncOrClass, typing.Protocol):
    "The aioway marker."

    __aioway_internal_ref__: FuncOrClass

    def __call__(self, *args: P.args, **kwargs: P.kwargs) -> T: ...


def public_api[T: FuncOrClass](item: T) -> T:
    """
    Register the function or class as a public API.
    """

    if isinstance(item, AiowayApi):
        raise ValueError(
            f"{item=} is already an API. Perhaps you're registering twice?"
        )

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

        original = _REGISTRY[name]
        item = copy.copy(original)

        # Mark it as `AiowayApi`.
        if typing.TYPE_CHECKING:
            assert isinstance(item, AiowayApi)

        item.__module__ = self.module
        item.__aioway_internal_ref__ = original
        return item

    def __iter__(self) -> cabc.Generator[str]:
        yield from _REGISTRY


def public_items(export: str | None = None) -> cabc.Mapping[str, FuncOrClass]:
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
