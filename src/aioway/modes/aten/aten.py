# Copyright (c) AIoWay Authors - All Rights Reserved

"The module containing `Aten` interface, the implementation for fake aten operations."

import abc
import dataclasses as dcls
import inspect
import typing
from collections import abc as cabc

from torch import _ops

from aioway._costs import Cost
from aioway._utils import camel_to_snake, find_nested_tensors, render_fcall

if typing.TYPE_CHECKING:
    from aioway.modes import TorchDispFn

__all__ = ["Aten", "aten_dcls", "find_aten", "all_aten_overrides"]


@typing.dataclass_transform(frozen_default=True)
def aten_dcls(cls, /):
    "Decorator of dataclass for `Aten`."
    return dcls.dataclass(repr=False, frozen=True)(cls)


@aten_dcls
class Aten(abc.ABC):
    """
    `Aten` are a bunch of `torch.ops.aten.*` overrides.

    It overrides aten ops in fake mode and compute extra properties,
    such as storage costs and compute costs, as well as patching some operations with worst case.
    For example, boolean masking is data dependent, and is thus not supported by fake mode.
    """

    KEY: typing.ClassVar[_ops.OpOverload] = NotImplemented
    """
    The `torch.ops.aten.*` operator that maps to the current `Aten`.
    Roughly 200 in total (we don't support that many yet).
    If `NotImplemented`, this class is considered abstract.
    """

    @typing.override
    def __repr__(self) -> str:
        return render_fcall("aten::" + self._name(), **dcls.asdict(self))

    @typing.final
    def __call__(self):
        result = self.forward()
        self.cost().commit()
        return result

    def forward(self):
        return self.KEY(**dcls.asdict(self))

    @abc.abstractmethod
    def ok(self) -> bool:
        """
        Whether or not the arguments are valid.
        """

        raise NotImplementedError

    @abc.abstractmethod
    def cost(self) -> Cost:
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


def find_aten(dispatch: TorchDispFn, /) -> Aten | None:
    """
    Try finding a `Aten` operator with the thunk, and then wrap into `AtenFn`.

    Returns `None` if a candidate is not found.
    """

    for sub_type in Aten.find(dispatch.func):
        if not (fate := sub_type(*dispatch.args, **dispatch.kwargs)).ok():
            continue

        return fate
    else:
        return None


@typing.no_type_check
def all_aten_overrides():
    """
    Get the registry for the `Aten` overrides.
    """

    return list(Aten.impls())
