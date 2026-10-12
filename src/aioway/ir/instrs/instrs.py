# Copyright (c) AIoWay Authors - All Rights Reserved

"The instructions themselves."

import abc
import functools
import typing
from collections import abc as cabc

from torch.utils import _pytree as pyt

from aioway._utils import AnyDict, any_dict, find_common_base, is_list_of
from aioway.t import TList

__all__ = ["Instr", "InstrList"]


# The base instruction class ====


class Instr(abc.ABC):
    "The instruction base class."

    def __init_subclass__(cls, **kwargs) -> None:
        super().__init_subclass__(**kwargs)

        pyt.register_pytree_node(
            cls,
            flatten_fn=cls._tree_flatten,
            unflatten_fn=cls._tree_unflatten,
            flatten_with_keys_fn=cls._tree_flatten_with_keys,
        )

    def tree_map(self, func: cabc.Callable):
        """
        Call the `tree_map` function on the `Instr`.
        """

        return pyt.tree_map(func=func, tree=self)

    def tree_map_only(self, types: type | tuple[type, ...], func: cabc.Callable):
        """
        Call the `tree_map_only` function on the `Instr`.
        """

        return pyt.tree_map_only(types, func=func, tree=self)

    @functools.cached_property
    def inputs(self) -> TList:
        return self._inputs()

    @functools.cached_property
    def outputs(self) -> TList:
        return self._outputs()

    @abc.abstractmethod
    def _inputs(self) -> TList:
        "The list of tensors in the inputs."

        raise NotImplementedError

    @abc.abstractmethod
    def _outputs(self) -> TList:
        "The list of tensors in the outputs."

        raise NotImplementedError

    def _tree_flatten(self):
        return list(self._tree_children().values()), self._tree_context()

    def _tree_flatten_with_keys(self):
        values = [(pyt.GetAttrKey(k), v) for k, v in self._tree_children().items()]
        return values, self._tree_context()

    @abc.abstractmethod
    def _tree_children(self) -> dict[str, typing.Any]:
        "The named children pytree should descend into, in a fixed order."
        raise NotImplementedError

    @abc.abstractmethod
    def _tree_context(self) -> typing.Any:
        "Everything else needed to rebuild the instruction."
        raise NotImplementedError

    @classmethod
    @abc.abstractmethod
    def _tree_unflatten(cls, children: cabc.Iterable, context) -> typing.Self:
        "Rebuild from the children (in `_tree_children` order) and the context."
        raise NotImplementedError


# The instr list class ====


class InstrList[I: Instr = typing.Any]:
    "The list of `Instr`."

    def __init__(self, iterable: tuple[I, ...]) -> None:
        """
        Args:
            iterable: The iterable data.
        """

        self._seq = iterable

    def __len__(self) -> int:
        return len(self._seq)

    @typing.overload
    def __getitem__(self, idx: int) -> I: ...

    @typing.overload
    def __getitem__(self, idx: slice | list[int]) -> typing.Self: ...

    def __getitem__(self, idx):
        if isinstance(idx, int):
            return self._seq[idx]

        if isinstance(idx, slice):
            idx = list(range(len(self))[idx])

        if is_list_of(int)(idx):
            return type(self)(tuple(self._seq[i] for i in idx))

        raise IndexError(idx)

    def __iter__(self) -> cabc.Iterator[I]:
        return iter(self._seq)

    def index(self, instr: I, /) -> int:
        return self._instr_to_step[instr]

    @functools.cached_property
    def _instr_to_step(self) -> AnyDict[I, int]:
        "Mapping from thunks to their indices."

        return any_dict(Instr, *((thunk, idx) for idx, thunk in enumerate(self._seq)))

    @property
    def base(self) -> type[Instr] | None:
        """
        Get the common base type. If sequence is empty, return `None`.
        """

        # Pass in `self` works, but `tuple` for better performance.
        return find_common_base(self._seq)

    @classmethod
    def build(cls, iterable: cabc.Iterable[I] | typing.Self) -> typing.Self:
        if isinstance(iterable, InstrList):
            result: typing.Any = iterable
            return result

        return cls(tuple(iterable))
