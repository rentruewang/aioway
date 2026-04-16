# Copyright (c) AIoWay Authors - All Rights Reserved

import abc
import dataclasses as dcls
import typing
from collections import abc as cabc

import torch
from torch import _ops, ops

from ..fn import Fn, Thunk

__all__ = ["register_preview"]

PREVIEW_CANDIDATES: dict[_ops.OpOverload, list[cabc.Callable[..., Preview]]] = {}


@dcls.dataclass(frozen=True)
class Preview(Fn, abc.ABC):
    """
    `Preview` is a preview for operations,
    allowing for multiple implementations for the same torch IR.
    """

    OP: typing.ClassVar[_ops.OpOverload]

    def __init_subclass__(cls) -> None:
        register_preview(cls.OP, cls)

    @abc.abstractmethod
    def ok(self) -> bool:
        """
        Whether or not the arguments are valid.
        """

        raise NotImplementedError

    @abc.abstractmethod
    def __call__(self) -> torch.Tensor:
        """
        Generate the fake tensor.
        """

        raise NotImplementedError

    @abc.abstractmethod
    def cost(self) -> int:
        """
        Return the cost of each operation.
        """

        raise NotImplementedError

    @property
    @typing.override
    def thunk(self) -> Thunk:
        return Thunk(self.OP, **dcls.asdict(self))


def find_preview(
    op: _ops.OpOverload, *args: typing.Any, **kwargs: typing.Any
) -> Preview:
    """
    Try finding a preview with the given `op` and its arguments.
    """

    if op not in PREVIEW_CANDIDATES:
        return NotImplemented

    for candidate in PREVIEW_CANDIDATES[op]:
        if not (preview := candidate(*args, **kwargs)).ok():
            continue

        return preview

    return NotImplemented


def register_preview(op: _ops.OpOverload, preview: type[Preview]):
    """
    Register a patching function that only runs under fake mode.

    The patch would be called. If the patching function returns `NotImplemented`,
    it will fall back to the default implementation (plain `func(*args, **kwargs)`).
    """

    if op not in PREVIEW_CANDIDATES:
        PREVIEW_CANDIDATES[op] = []

    PREVIEW_CANDIDATES[op].append(preview)


@dcls.dataclass(frozen=True)
class BooleanMasking(Preview):
    OP = ops.aten.index.Tensor

    self: torch.Tensor
    indices: list[torch.Tensor]

    def ok(self) -> bool:
        return len(self.indices) == 1 and self.indices[0].dtype == torch.bool

    @typing.override
    def __call__(this) -> torch.Tensor:
        return this.self

    @typing.override
    def cost(self) -> int:
        return self().numel()


@dcls.dataclass(frozen=True)
class IntSelect(Preview):
    OP = ops.aten.index.Tensor

    self: torch.Tensor
    indices: list[torch.Tensor]

    def ok(self):
        return len(self.indices) == 1 and self.indices[0].dtype == torch.int

    @typing.override
    def __call__(this):
        return this.self[this.indices]

    @typing.override
    def cost(self) -> int:
        return self().numel()
