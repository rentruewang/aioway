# Copyright (c) AIoWay Authors - All Rights Reserved

"The module containing `Fate` interface, the implementation for fake aten operations."

import abc
import re
import typing

from torch import _ops

from aioway._common import dcls_no_repr, find_nested_tensors
from aioway.op import Op

__all__ = ["Fate", "find_fate", "all_fates"]


@dcls_no_repr
class Fate(Op[_ops.OpOverload], abc.ABC):
    """
    `Fate` stands for [f]ake [ate]n. Or [fa]ke [te]nsor. Or a tensor's [fate] (how it behaves).

    It overrides aten ops in fake mode and compute extra properties,
    such as storage costs and compute costs, as well as patching some operations with worst case.
    For example, boolean masking is data dependent, and is thus not supported by fake mode.
    """

    KEY: typing.ClassVar[_ops.OpOverload] = NotImplemented

    @abc.abstractmethod
    def ok(self) -> bool:
        """
        Whether or not the arguments are valid.
        """

        raise NotImplementedError

    @typing.override
    @classmethod
    def name(cls) -> str:
        return "fate::" + _camel_to_snake(cls.__name__)

    @abc.abstractmethod
    def cost(self) -> int:
        """
        Return the cost of each operation.
        """

        raise NotImplementedError

    def inputs(self):
        yield from find_nested_tensors(self)


def find_fate(op: _ops.OpOverload, *args: typing.Any, **kwargs: typing.Any) -> Fate:
    """
    Try finding a `Fate` operator with the thunk, and then wrap into `FateFn`.

    Returns `NotImplemented` if a candidate is not found.
    """

    for sub_type in Fate.find(op):
        if not (fate := sub_type(*args, **kwargs)).ok():
            continue

        return fate

    return NotImplemented


@typing.no_type_check
def all_fates():
    """
    Get the registry for the fates.
    """
    return list(Fate.impls())


_CAMEL_CASE_REGEX = re.compile(r"(?<!^)(?=[A-Z])")


def _camel_to_snake(name: str) -> str:
    return re.sub(_CAMEL_CASE_REGEX, "_", name).lower()
