# Copyright (c) AIoWay Authors - All Rights Reserved

"Extension of `pyt` from `torch`."

import abc
import dataclasses as dcls
import typing
from collections import abc as cabc

import torch
from torch.utils import _pytree as pyt

from aioway._utils import AnyDict, any_dict

__all__ = [
    "tree_leaves_typed",
    "tree_map_memo",
    "find_nested_tensors",
    "replace_tensors",
    "register_pytree_dcls",
    "PyTreeMixin",
]


def register_pytree_dcls[T: type](cls: T) -> T:
    """
    Decorator to register the dataclass into `pyt`.
    """

    if not isinstance(cls, type) or not dcls.is_dataclass(cls):
        raise TypeError("The decorator can only be used on dataclasses.")

    pyt.register_dataclass(cls)
    return cls


def tree_map_memo(
    obj, replace: cabc.Callable[..., object], memo: AnyDict | None = None
) -> typing.Any:
    """
    Decompose and replace. When this is called, `replace(obj)` is directly invoked.
    If it returns `NotImplemented`, then decomposing would continue.

    Args:
        obj: The object to maybe replace.
        types: The types to replace.
        replace: The replacer function.
        memo:
            Like `memo` in `__deepcopy__`,
            this is s.t. don't replace the same item with different ones.
    """

    memo = memo if memo is not None else any_dict()

    def replace_cached(item):
        if item not in memo:
            # If it is `NotImplemented`, no replacement is made.
            if (result := replace(item)) is NotImplemented:
                result = item

            memo[item] = result

        return memo[item]

    return pyt.tree_map(replace_cached, obj)


def tree_leaves_typed(obj, *types: type) -> list[typing.Any]:
    "Decompose the object based on the desired type."

    found = lambda item: isinstance(item, types)

    # Tree leaves without decomposing the target types.
    leaves = pyt.tree_leaves(obj, is_leaf=found)
    return [elem for elem in leaves if found(elem)]


def find_nested_tensors(obj: object) -> cabc.Iterator[torch.Tensor]:
    """
    Find and unpack tensors from containers.
    """

    yield from tree_leaves_typed(obj, torch.Tensor)


def replace_tensors(
    obj: object, replace: cabc.Callable[[torch.Tensor], object]
) -> typing.Any:
    """
    Replace tensors whenever encountered with the given function.

    This function has the `__torch_function__` disabled in the scope of the rendering,
    because it can mess with attribute access, which oftentimes means that
    this function fails also during debugging if `__torch_function__` is not disabled.
    Caused by `.device` / `.shape` / `.dtype` calls, which is used in `replace_tensors`.
    """

    from aioway.t import mode_off

    def maybe_replace(item):
        if not isinstance(item, torch.Tensor):
            return NotImplemented

        return replace(item)

    with mode_off():
        return tree_map_memo(obj, maybe_replace)


class PyTreeMixin(abc.ABC):
    """
    The mixin that enables the types to be compatible with `pytree` of `torch`.
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
    def _tree_children(self) -> dict[str, typing.Any]:
        "The named children pytree should descend into, in a fixed order."
        raise NotImplementedError

    @property
    @abc.abstractmethod
    def _tree_context(self) -> typing.Any:
        "Everything else needed to rebuild the instruction."
        raise NotImplementedError

    @classmethod
    @abc.abstractmethod
    def _tree_unflatten(cls, children: cabc.Iterable, context) -> typing.Self:
        "Rebuild from the children (in `_tree_children` order) and the context."
        raise NotImplementedError

    def _tree_flatten(self):
        return list(self._tree_children().values()), self._tree_context

    def _tree_flatten_with_keys(self):
        values = [(pyt.GetAttrKey(k), v) for k, v in self._tree_children().items()]
        return values, self._tree_context

    @abc.abstractmethod
    def _tree_children(self) -> dict[str, typing.Any]:
        "The named children pytree should descend into, in a fixed order."
        raise NotImplementedError

    @property
    @abc.abstractmethod
    def _tree_context(self) -> typing.Any:
        "Everything else needed to rebuild the instruction."
        raise NotImplementedError

    @classmethod
    @abc.abstractmethod
    def _tree_unflatten(cls, children: cabc.Iterable, context) -> typing.Self:
        "Rebuild from the children (in `_tree_children` order) and the context."
        raise NotImplementedError
