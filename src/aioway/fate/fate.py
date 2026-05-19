# Copyright (c) AIoWay Authors - All Rights Reserved

"The module containing `Fate` interface, the implementation for fake aten operations."

import abc
import dataclasses as dcls
import typing

from torch import _ops

from aioway._keyed import Keyed
from aioway._types import dcls_no_repr
from aioway.decomps import find_nested_tensors
from aioway.fn import TorDisFn
from aioway.renders import render_fcall

__all__ = ["Fate", "find_fate", "all_fates"]


@dcls_no_repr
class Fate(Keyed[_ops.OpOverload], abc.ABC):
    """
    `Fate` stands for [f]ake [ate]n. Or [fa]ke [te]nsor. Or a tensor's [fate] (how it behaves).

    It overrides aten ops in fake mode and compute extra properties,
    such as storage costs and compute costs, as well as patching some operations with worst case.
    For example, boolean masking is data dependent, and is thus not supported by fake mode.
    """

    KEY: typing.ClassVar[_ops.OpOverload] = NotImplemented

    @typing.override
    def __repr__(self) -> str:
        return render_fcall("fate::" + self._name(), **dcls.asdict(self))

    def do(self) -> typing.Any:
        return self.KEY(**dcls.asdict(self))

    @abc.abstractmethod
    def ok(self) -> bool:
        """
        Whether or not the arguments are valid.
        """

        raise NotImplementedError

    @abc.abstractmethod
    def cost(self) -> int:
        """
        Return the cost of each operation.
        """

        raise NotImplementedError

    def inputs(self):
        yield from find_nested_tensors(self)


def find_fate(dispatch: TorDisFn) -> Fate:
    """
    Try finding a `Fate` operator with the thunk, and then wrap into `FateFn`.

    Returns `NotImplemented` if a candidate is not found.
    """

    for sub_type in Fate.find(dispatch.func):
        if not (fate := sub_type(*dispatch.args, **dispatch.kwargs)).ok():
            continue

        return fate
    else:
        return NotImplemented


@typing.no_type_check
def all_fates():
    """
    Get the registry for the fates.
    """
    return list(Fate.impls())
