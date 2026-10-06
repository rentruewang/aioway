# Copyright (c) AIoWay Authors - All Rights Reserved

"The instructions themselves."

import abc
import dataclasses as dcls
import functools
import typing
from collections import abc as cabc

from torch.utils import _pytree as pyt

from aioway.t import (
    TList,
)

__all__ = ["Instr"]


@typing.dataclass_transform(frozen_default=True, eq_default=False)
def instr_dcls[T](cls):
    """
    The dataclass transform wrapper for `Instr` classes.

    This also registeres the class to `pyt`.
    """

    cls = dcls.dataclass(frozen=True, eq=False, repr=False)(cls)
    pyt.register_dataclass(cls)
    return cls


# The base instruction class ====


@instr_dcls
class Instr(abc.ABC):
    "The instruction base class."

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
