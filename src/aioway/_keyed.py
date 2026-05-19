# Copyright (c) AIoWay Authors - All Rights Reserved

"The module containing `Fate` interface, the implementation for fake aten operations."

import abc
import inspect
import re
import typing
from collections import abc as cabc

from aioway._types import dcls_no_repr

__all__ = ["Keyed"]


_CAMEL_CASE_REGEX = re.compile(r"(?<!^)(?=[A-Z])")


@dcls_no_repr
class Keyed[K](abc.ABC):
    """
    `Keyed` is a class where subclasses are indexed by their `KEY_LIST` attributes,
    providing easy way to find the subclass by walking the subclass tree.
    """

    KEY_LIST: typing.ClassVar[tuple[K, ...]]
    key: K
    """
    During
    """

    def __init_subclass__(cls, key: K | list[K] | None = None) -> None:
        if key is None:
            cls.KEY_LIST = ()

        elif isinstance(key, list):
            cls.KEY_LIST = tuple(key)

        else:
            cls.KEY_LIST = (key,)

    @typing.final
    def __post_init__(self):
        if self.key not in self.KEY_LIST:
            raise KeyError(
                f"Key {self.key} is used to initialize {type(self)}, "
                f"but it's not inside {self.KEY_LIST}."
            )

        self._check_data()

    def _check_data(self) -> None:
        """
        Check the data stored in the dataclass. Normally this is `__post_init__`,
        but here `__post_init__` is used for key checking, so `__post_init__`
        would do the key checking and then call `._check_data()`.
        """

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
            if key in sub.KEY_LIST:
                yield sub

    @classmethod
    def impls(cls):
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
        return bool(cls.KEY_LIST) and not inspect.isabstract(cls)


def _camel_to_snake(name: str) -> str:
    return re.sub(_CAMEL_CASE_REGEX, "_", name).lower()


def _iter_ops[T](cls: type[Keyed[T]]) -> cabc.Generator[type[Keyed[T]]]:
    for sub in cls.__subclasses__():
        if sub.is_concrete():
            yield sub

        yield from _iter_ops(sub)
