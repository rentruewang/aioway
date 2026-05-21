# Copyright (c) AIoWay Authors - All Rights Reserved

"The module containing `Fate` interface, the implementation for fake aten operations."

import abc
import dataclasses as dcls
import inspect
import typing
from collections import abc as cabc

from torch import _ops

from aioway.decomps import find_nested_tensors
from aioway.modes import TorDisFn
from aioway.renders import camel_to_snake, render_fcall

__all__ = ["Fate", "fate_dcls", "find_fate", "all_fates"]


@typing.dataclass_transform()
def fate_dcls(cls):
    "Decorator of dataclass for `Fate`."
    return dcls.dataclass(repr=False)(cls)


@fate_dcls
class Fate(abc.ABC):
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

    @classmethod
    def find(cls, key: _ops.OpOverload) -> cabc.Generator[type[typing.Self]]:
        """
        Recursively find the class tagged `key` in the subclass.

        This function iterates over the subclass by doing a DFS traversal.
        This has the benefit of being able to query new classes on the fly,
        while not maintaining a global dictionary.

        If this ends up being too slow, we'll change to a mapping-based method.

        Yields:
            All the subclasses with the `key` as key.
            Only concrete classes are considered.
        """

        for sub in cls.impls():
            if sub.KEY == key:
                yield sub

    @classmethod
    def impls(cls) -> cabc.Generator[type[typing.Self]]:
        """
        Walk the subclass tree, and get all the concrete subclasses that `Op` has.

        Yields:
            Subclasses if they are concrete (has `cls.is_concrete()` is `True`).
        """

        for sub in cls.__subclasses__():
            if sub.is_concrete():
                yield sub

            yield from sub.impls()

    @classmethod
    def _name(cls):
        return camel_to_snake(cls.__qualname__)

    @classmethod
    def is_concrete(cls) -> bool:
        """
        Check if the class can be initialized and found in registry.
        """

        # Concrete in class var and concrete in methods.
        return cls.KEY is not NotImplemented and not inspect.isabstract(cls)


def find_fate(dispatch: TorDisFn, /) -> Fate | None:
    """
    Try finding a `Fate` operator with the thunk, and then wrap into `FateFn`.

    Returns `None` if a candidate is not found.
    """

    for sub_type in Fate.find(dispatch.func):
        if not (fate := sub_type(*dispatch.args, **dispatch.kwargs)).ok():
            continue

        return fate
    else:
        return None


@typing.no_type_check
def all_fates():
    """
    Get the registry for the fates.
    """
    return list(Fate.impls())
