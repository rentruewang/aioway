# Copyright (c) AIoWay Authors - All Rights Reserved

import abc
import dataclasses as dcls
import typing
from collections import abc as cabc

import torch
from torch import _ops, ops

from ..fn import TorchFn

__all__ = ["register_preview"]

PREVIEW_CANDIDATES: dict[_ops.OpOverload, list[cabc.Callable[..., Preview]]] = {}


class Preview(abc.ABC):
    """
    `Preview` is a preview for operations,
    allowing for multiple implementations for the same torch IR.
    """

    OP: typing.ClassVar[_ops.OpOverload]

    def __init_subclass__(cls) -> None:
        register_preview(cls.OP, cls)

    @typing.final
    def __call__(self):
        if not self.ok():
            return NotImplemented

        result = self.get()

        if not isinstance(result, torch.Tensor):
            raise TypeError("Should return a tensor.")

        return result

    @abc.abstractmethod
    def ok(self) -> bool:
        """
        Whether or not the arguments are valid.
        """

        raise NotImplementedError

    @abc.abstractmethod
    def get(self) -> torch.Tensor:
        """
        Generate the fake tensor.
        """

        raise NotImplementedError


def find_preview(
    op: _ops.OpOverload, *args: typing.Any, **kwargs: typing.Any
) -> Preview:
    """
    Try finding a preview with the given `op` and its arguments.
    """

    if op not in PREVIEW_CANDIDATES:
        return NotImplemented

    for candidate in PREVIEW_CANDIDATES[op]:
        if (preview := candidate(*args, **kwargs)) is NotImplemented:
            continue

        return preview

    return NotImplemented


class PreviewFn(TorchFn):
    def __init__(
        self,
        func: _ops.OpOverload,
        preview: Preview,
        types: tuple[type, ...],
        /,
        *args: typing.Any,
        **kwargs: typing.Any,
    ) -> None:
        super().__init__(func, types, *args, **kwargs)
        self._preview = preview

    @typing.override
    def do(self):
        if (result := self.preview()) is not NotImplemented:
            return result

        return self.func(*self.args, **self.kwargs)

    @property
    def preview(self):
        return self._preview


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

    tensor: torch.Tensor
    indices: list[torch.Tensor]

    def ok(self):
        return len(self.indices) == 1 and self.indices[0].dtype == torch.bool

    @typing.override
    def get(self) -> torch.Tensor:
        return self.tensor


@dcls.dataclass(frozen=True)
class IntSelect(Preview):
    OP = ops.aten.index.Tensor

    tensor: torch.Tensor
    indices: list[torch.Tensor]

    def ok(self):
        return len(self.indices) == 1 and self.indices[0].dtype == torch.int

    @typing.override
    def get(self):
        return self.tensor[self.indices]
