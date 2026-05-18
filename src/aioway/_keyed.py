# Copyright (c) AIoWay Authors - All Rights Reserved

"The module containing `Fate` interface, the implementation for fake aten operations."

import abc
import inspect
import re
import typing
from collections import abc as cabc

__all__ = ["Keyed"]


_CAMEL_CASE_REGEX = re.compile(r"(?<!^)(?=[A-Z])")


class Keyed[K: cabc.Callable[..., object]](abc.ABC):
    """
    `Keyed` is a class where subclasses are indexed by their `KEY` attributes,
    providing easy way to find the subclass by walking the subclass tree.
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

        yield from _iter_ops(cls)

    @classmethod
    def _name(cls):
        return _camel_to_snake(cls.__qualname__)

    @classmethod
    def is_concrete(cls) -> bool:
        # Concrete in class var and concrete in methods.
        return cls.KEY is not NotImplemented and not inspect.isabstract(cls)


def _camel_to_snake(name: str) -> str:
    return re.sub(_CAMEL_CASE_REGEX, "_", name).lower()


def _iter_ops[T: Keyed](cls: type[T]) -> cabc.Generator[type[T]]:
    for sub in cls.__subclasses__():
        if sub.is_concrete():
            yield sub

        yield from _iter_ops(sub)
