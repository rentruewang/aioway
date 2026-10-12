# Copyright (c) AIoWay Authors - All Rights Reserved

"Extension of `pyt` from `torch`."

import abc
import typing
from collections import abc as cabc

from torch.utils import _pytree as pyt

__all__ = ["PyTree", "PyTreeDict", "PyTreeFlatten", "PyTreeFlattenWithKey"]

type PyTreeFlatten = tuple[list, typing.Any]
type PyTreeFlattenWithKey = tuple[list[tuple[pyt.KeyEntry, typing.Any]], typing.Any]


class PyTreeKeyFunc(typing.Protocol):
    def __call__(self, key: str, /) -> pyt.KeyEntry: ...


class PyTree(abc.ABC):
    """
    The mixin that enables the types to be compatible with `pytree` of `torch`.

    Using this is preferred because it's type safe,
    since implementing this should work with all pytree operations.
    """

    def __init_subclass__(cls) -> None:
        super().__init_subclass__()

        pyt.register_pytree_node(
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


class PyTreeDict(PyTree, abc.ABC):
    """
    The mixin that decompose objects with `pytree`,
    hanlding all types that serializes to `dict`.
    """

    def __init_subclass__(cls, pytree_key: PyTreeKeyFunc = NotImplemented) -> None:
        super().__init_subclass__()

        cls._PY_TREE_KEY = pytree_key

    @typing.override
    def _tree_flatten(self) -> PyTreeFlatten:
        return list(self._tree_dict().values()), self._tree_context

    @typing.override
    def _tree_flatten_with_keys(self) -> PyTreeFlattenWithKey:
        values = [(self._tree_key(k), v) for k, v in self._tree_dict().items()]
        return values, self._tree_context

    @property
    @abc.abstractmethod
    def _tree_context(self) -> typing.Any:
        "Everything else needed to rebuild."
        raise NotImplementedError

    @abc.abstractmethod
    def _tree_dict(self) -> dict[str, typing.Any]:
        "The dict to serailize. The values become flatten."
        raise NotImplementedError

    @classmethod
    def _tree_key(cls, key: str) -> pyt.KeyEntry:
        if cls._PY_TREE_KEY is NotImplemented:
            raise TypeError(
                "In class definition `pytree_key` is not provided. "
                "Class is abstract."
            )

        return cls._PY_TREE_KEY(key)
