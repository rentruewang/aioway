# Copyright (c) AIoWay Authors - All Rights Reserved

"Extension of `pyt` from `torch`."

import abc
import dataclasses
import typing
from collections import abc as cabc

from .compat import GetAttrKey, KeyEntry, register_pytree_node

__all__ = [
    "PyTreeMixin",
    "PyTreeDictMixin",
    "PyTreeDataclassMixin",
    "PyTreeFlatten",
    "PyTreeFlattenWithKey",
]

type PyTreeFlatten = tuple[list, typing.Any]
type PyTreeFlattenWithKey = tuple[list[tuple[KeyEntry, typing.Any]], typing.Any]


class PyTreeDictKeyFunc(typing.Protocol):
    def __call__(self, key, /) -> typing.Any: ...


class PyTreeMixin(abc.ABC):
    """
    The mixin that enables the types to be compatible with `pytree` of `torch`.

    Using this is preferred because it's type safe,
    since implementing this should work with all pytree operations.
    """

    def __init_subclass__(cls) -> None:
        super().__init_subclass__()

        register_pytree_node(
            cls,
            flatten_fn=cls._tree_flatten,
            unflatten_fn=cls._tree_unflatten,
            flatten_with_keys_fn=cls._tree_flatten_with_keys,
        )

    @abc.abstractmethod
    def _tree_flatten(self) -> PyTreeFlatten:
        raise NotImplementedError

    @abc.abstractmethod
    def _tree_flatten_with_keys(self) -> PyTreeFlattenWithKey:
        "Flatten the given tree with list."
        raise NotImplementedError

    @classmethod
    @abc.abstractmethod
    def _tree_unflatten(cls, children: cabc.Iterable, context) -> typing.Self:
        "Rebuild from the children."
        raise NotImplementedError


class PyTreeDictMixin(PyTreeMixin, abc.ABC):
    """
    The mixin that decompose objects with `pytree`,
    hanlding all types that serializes to `dict`.
    """

    _PY_TREE_KEY: typing.ClassVar[PyTreeDictKeyFunc] = NotImplemented

    @typing.override
    @typing.final
    def _tree_flatten(self) -> PyTreeFlatten:
        d = self._tree_dict()
        return list(d.values()), (tuple(d), self._tree_context)

    @typing.override
    @typing.final
    def _tree_flatten_with_keys(self) -> PyTreeFlattenWithKey:
        d = self._tree_dict()
        return [(self._tree_key(k), v) for k, v in d.items()], (
            tuple(d),
            self._tree_context,
        )

    @property
    @abc.abstractmethod
    def _tree_context(self) -> typing.Any:
        "Everything else needed to rebuild."
        raise NotImplementedError

    @abc.abstractmethod
    def _tree_dict(self) -> dict[str, typing.Any]:
        "The dict to serailize. The values become flatten."
        raise NotImplementedError

    @typing.override
    @typing.final
    @classmethod
    def _tree_unflatten(cls, children, context) -> typing.Self:
        keys, ctx = context
        return cls._tree_from_dict(dict(zip(keys, children, strict=True)), ctx)

    @classmethod
    @abc.abstractmethod
    def _tree_from_dict(cls, mapping: dict[str, typing.Any], context) -> typing.Self:
        "Rebuild from the dict `_tree_dict` returned."
        raise NotImplementedError

    @classmethod
    def _tree_key(cls, key: str) -> KeyEntry:
        if cls._PY_TREE_KEY is NotImplemented:
            raise TypeError(
                f"`{cls.__qualname__}._PY_TREE_KEY` is not set. Class is abstract."
            )

        return cls._PY_TREE_KEY(key)


class PyTreeDataclassMixin(PyTreeDictMixin, abc.ABC):
    """
    Mixin for dataclasses. Since `cxx_pytree` cannot register data class.
    """

    _PY_TREE_KEY = GetAttrKey

    @typing.override
    def _tree_dict(self) -> dict[str, typing.Any]:
        if not dataclasses.is_dataclass(self):
            raise TypeError(
                f"`{type(self).__qualname__}` uses `PyTreeDataclassMixin` "
                "but is not a dataclass."
            )

        # If `init=False`, passing it to constructor crashes.
        return {
            field.name: getattr(self, field.name)
            for field in dataclasses.fields(self)
            if field.init
        }

    @property
    @typing.override
    def _tree_context(self) -> None:
        return None

    @classmethod
    @typing.override
    def _tree_from_dict(cls, mapping: dict[str, typing.Any], context) -> typing.Self:
        return cls(**mapping)
