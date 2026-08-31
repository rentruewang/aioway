# Copyright (c) AIoWay Authors - All Rights Reserved

import typing

from aioway.tspecs import TSpec

__all__ = ["Deductor", "DeductorLike", "DeductorCompat"]

type DeductorLike = Deductor | DeductorCompat
"""
Types compatible with `Deductor`.
"""


@typing.runtime_checkable
class Deductor(typing.Protocol):
    """
    `Deductor` converts from an input `TSpec` to another `TSpec`.

    It's the type of callables that consumes a torch object and outputs another one.
    """

    def __call__(self, tspec: TSpec, /) -> TSpec: ...


@typing.runtime_checkable
class DeductorCompat(typing.Protocol):
    """
    `DeductorCompat` can be converted to a `Deductor`.
    """

    def __deduct__(self) -> Deductor: ...
