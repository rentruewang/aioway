# Copyright (c) AIoWay Authors - All Rights Reserved

"Domain are a set of common 'type's we support."

from aioway.t import has_real
import abc
import typing

from collections import abc as cabc

__all__ = ["Domain"]


class Domain[T = typing.Any](abc.ABC):
    """
    The base type for `Domain`s.

    A `Domain` in `aioway` reprsents a batch of data,
    describing how it behaves and how we can interact with it.

    For instance, images, videos, texts, trees etc.
    """

    def bases(self) -> cabc.Iterable[Domain[T]]:
        """
        The base types that this base `Domain` is based on.

        For example, an `Image` is a `Grid`.
        """

        return ()

    @typing.final
    def check(self, obj: T, /) -> None:
        self.check_type(obj)

        if has_real(obj):
            self.check_data(obj)

    def check_type(self, obj: T, /) -> None:
        """
        Checks that can be run on fake values.
        """

    def check_data(self, obj: T) -> None:
        """
        Checks that should be run on real values.
        """
