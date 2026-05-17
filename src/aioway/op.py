# Copyright (c) AIoWay Authors - All Rights Reserved

"The module containing `Fate` interface, the implementation for fake aten operations."

import abc
import dataclasses as dcls
import inspect
import typing
from collections import abc as cabc

from aioway._common import dcls_no_repr, render_fcall
from aioway.tags import DimTag

__all__ = ["Op"]


@dcls_no_repr
class Op[K: cabc.Callable[..., object]](abc.ABC):
    """
    `Op` stands for [o]verridable [p]ass. Or [op]erator. It follows a pattern:
    it sits in a family of similar operations, and can be looked up by a key (and signature).

    An op is a custom (`aioway`) operation that may override the operator's behaviors at runtime,
    or at least provide some static, inspectable info on the current call.

    Right now, there are 2 `Op` kinds:
    1. `Fate` for ATen operations.
    2. `Might` for `nn.Module` init.
    """

    KEY: typing.ClassVar[K] = NotImplemented
    """
    The key that is set on the subclasses.

    When querying
    """

    def __init_subclass__(cls) -> None:
        # The key is not overwritten. This is considered an abstract class. Pass.
        if cls.KEY is NotImplemented:
            return

        # Pass abstract class naturally.
        if inspect.isabstract(cls):
            return

        if not callable(cls.KEY):
            raise TypeError(f"Non callable function key for {cls=}, {cls.KEY=}.")

    @typing.override
    def __repr__(self) -> str:
        return render_fcall(f"{self.name()}", **dcls.asdict(self))

    @typing.override
    def __hash__(self) -> int:
        return id(self)

    def do(self) -> typing.Any:
        """
        Generate the fake tensor.
        """

        return self.KEY(**dcls.asdict(self))

    @property
    def __aioway_dim_tag__(self) -> DimTag:
        "Also process `DimTag`s from the input."
        return NotImplemented

    @classmethod
    @abc.abstractmethod
    def name(cls) -> str:
        "The name of the class to be rendered."

        raise NotImplementedError

    @classmethod
    def find(cls, key: K) -> cabc.Generator[type[typing.Self]]:
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

        for sub in cls.__subclasses__():
            if sub.is_concrete() and sub.KEY == key:
                yield sub

            yield from sub.find(key)

    @classmethod
    def is_concrete(cls) -> bool:
        # Concrete in class var and concrete in methods.
        return cls.KEY is not NotImplemented and not inspect.isabstract(cls)
