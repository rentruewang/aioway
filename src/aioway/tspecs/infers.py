# Copyright (c) AIoWay Authors - All Rights Reserved

import dataclasses as dcls
import typing
from collections import abc as cabc

from .tspecs import TSpec

__all__ = [
    "TSpecInfer",
    "TSpecInferLike",
    "TSpecInferCompat",
    "MonoTSpecInfer",
    "PolyTSpecInfer",
]

type TSpecInferLike = TSpecInfer | TSpecInferCompat
"""
Types compatible with `TSpecInfer`.
"""


@typing.runtime_checkable
class TSpecInfer(typing.Protocol):
    """
    `TSpecInfer` converts from an input `TSpec` to another `TSpec`.

    It's the type of callables that consumes a torch object and outputs another one.
    """

    def __call__(self, tspec: TSpec, /) -> TSpec: ...


@typing.runtime_checkable
class TSpecInferCompat(typing.Protocol):
    """
    `TSpecInferCompat` can be converted to a `TSpecInfer`.
    """

    def __tspec_infer__(self) -> TSpecInfer: ...


@typing.final
@dcls.dataclass(frozen=True)
class MonoTSpecInfer(TSpecInfer):
    "The `TSpecInfer` that handles a single type."

    tspec_type: type[TSpec]
    function: cabc.Callable[[TSpec], TSpec]

    @typing.override
    def __call__(self, tspec: TSpec) -> TSpec:
        if not isinstance(tspec, self.tspec_type):
            raise TypeError(f"Only handles {self.tspec_type}, got {type(tspec)=}.")

        return self.function(tspec)


@typing.final
@dcls.dataclass(frozen=True)
class PolyTSpecInfer(TSpecInfer):
    "The `TSpecInfer` that handles multiple types."

    types_mapping: dict[type[TSpec], cabc.Callable[[TSpec], TSpec]]
    "The tspec type mapping."

    @typing.override
    def __call__(self, tspec: TSpec) -> TSpec:
        tspec_type = type(tspec)

        if (func := self.types_mapping.get(tspec_type)) is None:
            subset = list(self.types_mapping)
            raise TypeError(
                f"Does not know how to handle {type(tspec)=}. "
                f"Only handles a subset of {subset}."
            )

        return func(tspec)

    def convert(self, tspec: TSpec, /) -> TSpec:
        """
        Convert the input `tspec` to the output tspec.
        """

        raise NotImplementedError

    def accepts(self, tspec: type[TSpec], /) -> bool:
        "Whether or not this would accept tspec of specific type."

        raise NotImplementedError
