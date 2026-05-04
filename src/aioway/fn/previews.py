# Copyright (c) AIoWay Authors - All Rights Reserved

import abc
import dataclasses as dcls
import inspect
import re
import typing
from collections import abc as cabc

import numpy as np
import torch
from torch import _ops

from aioway._common import dcls_no_repr, render_fcall

from .fn import Fn
from .guards import TensorFilter, all_tensors

__all__ = ["find_preview", "PreviewFnFinder", "all_previews", "PreviewFn"]


_PREVIEW_CANDIDATES: dict[_ops.OpOverload, list[cabc.Callable[..., PreviewFn]]] = {}


class TensorFn(Fn, abc.ABC):
    @typing.override
    def __hash__(self) -> int:
        return id(self)

    @abc.abstractmethod
    @typing.override
    def do(self) -> torch.Tensor:
        raise NotImplementedError

    def parameters(self, select: TensorFilter = all_tensors, /):
        for tensor in self.tensors():
            if select(tensor):
                yield tensor

    @abc.abstractmethod
    def tensors(self) -> cabc.Iterator[torch.Tensor]:
        raise NotImplementedError


@dcls_no_repr
class TensorThunk(TensorFn):
    func: cabc.Callable[..., typing.Any]
    args: tuple[typing.Any, ...]
    kwargs: dict[str, typing.Any]

    @typing.override
    def __hash__(self) -> int:
        return id(self)

    @typing.override
    def do(self) -> torch.Tensor:
        return self.func(*self.args, **self.kwargs)

    @typing.override
    def tensors(self) -> cabc.Iterator[torch.Tensor]:
        yield from _discover_tensors(self.args)
        yield from _discover_tensors(self.kwargs)


def _discover_tensors(obj: object) -> cabc.Iterator[torch.Tensor]:
    if isinstance(obj, torch.Tensor):
        yield obj
        return

    if obj in [None, NotImplemented]:
        return

    if isinstance(obj, int | float | bool | str | np.ndarray):
        return

    if isinstance(obj, cabc.Sequence):
        for elem in obj:
            yield from _discover_tensors(elem)
        return

    if isinstance(obj, cabc.Mapping):
        for elem in obj.values():
            yield from _discover_tensors(elem)
        return

    raise TypeError(f"Unknown type: {type(obj)=}.")


@dcls_no_repr
class PreviewFn(TensorFn, abc.ABC):
    """
    `Preview` is a preview for operations,
    allowing for multiple implementations for the same torch IR.
    """

    IR: typing.ClassVar[_ops.OpOverload]
    """
    The torch IR that this `Preview` would be implementing.
    """

    def __init_subclass__(cls) -> None:
        cls.__register_preview()

    @typing.override
    def __repr__(self) -> str:
        return render_fcall(f"preview::{self.name()}", **dcls.asdict(self))

    @abc.abstractmethod
    def ok(self) -> bool:
        """
        Whether or not the arguments are valid.
        """

        raise NotImplementedError

    @abc.abstractmethod
    def do(self) -> torch.Tensor:
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

    @classmethod
    def __register_preview(cls):
        """
        Register a patching function that only runs under fake mode.

        The patch would be called. If the patching function returns `NotImplemented`,
        it will fall back to the default implementation (plain `func(*args, **kwargs)`).
        """

        # Abstract methods.
        if inspect.isabstract(cls):
            return

        # Abstract `ClassVar`.
        try:
            op = cls.IR
        except AttributeError:
            return

        # Mimick defaultdict behavior.
        # Using dict over defaultdict s.t. we don't need special handling in `repr`.
        if op not in _PREVIEW_CANDIDATES:
            _PREVIEW_CANDIDATES[op] = []

        _PREVIEW_CANDIDATES[op].append(cls)

    @classmethod
    def name(cls) -> str:
        return _camel_to_snake(cls.__name__)


def find_preview(
    op: _ops.OpOverload, *args: typing.Any, **kwargs: typing.Any
) -> PreviewFn:
    """
    Try finding a preview with the given `op` and its arguments.
    """

    if op not in _PREVIEW_CANDIDATES:
        return NotImplemented

    return PreviewFnFinder(op)(*args, **kwargs)


@dcls.dataclass(frozen=True)
class PreviewFnFinder:
    op: _ops.OpOverload

    def __repr__(self):
        name = type(self).__qualname__
        return f"{name}[{self.op.name()}]({self.candidates})"

    def __call__(self, *args, **kwargs):
        for candidate in self.candidates:
            if not (preview := candidate(*args, **kwargs)).ok():
                continue

            return preview

        return NotImplemented

    @property
    def candidates(self):
        return _PREVIEW_CANDIDATES[self.op]


def all_previews():
    return _PREVIEW_CANDIDATES


_CAMEL_CASE_REGEX = re.compile(r"(?<!^)(?=[A-Z])")


def _camel_to_snake(name: str) -> str:
    return re.sub(_CAMEL_CASE_REGEX, "_", name).lower()
