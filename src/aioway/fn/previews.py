# Copyright (c) AIoWay Authors - All Rights Reserved

import abc
import dataclasses as dcls
import inspect
import re
import typing
from collections import abc as cabc

import torch
from torch import _ops

from aioway._common import dcls_frozen_no_repr, render_fcall

from .fn import Fn
from .modes import HasParam, TDispatchFn

__all__ = ["find_preview", "all_previews", "PreviewFn", "Preview"]


_PREVIEW_CANDIDATES: dict[_ops.OpOverload, list[type[Preview]]] = {}


@dcls_frozen_no_repr
class Preview(HasParam, abc.ABC):
    IR: typing.ClassVar[_ops.OpOverload]
    """
    The torch IR that this `Preview` would be implementing.
    """

    def __init_subclass__(cls) -> None:
        cls.__register_preview()

    @typing.override
    def __repr__(self) -> str:
        return render_fcall(f"preview::{self.name()}", **dcls.asdict(self))

    def __hash__(self) -> int:
        return id(self)

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

    @classmethod
    def name(cls) -> str:
        return _camel_to_snake(cls.__name__)

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


@typing.final
@dcls.dataclass(frozen=True)
class PreviewFn(HasParam, Fn):
    """
    `PreviewFn` wraps a `Preview` object, which is split out so as to declutter subclasses for `Fn`.

    Each `Preview` is an implementation of an IR, and each IR can have multiple `Preview`s,
    each handling a subset of parameters (if `Preview.ok` is `False`, it's discarded.)
    """

    preview: Preview
    """
    The preview object that ends up being selected.
    """

    original: TDispatchFn
    "The original `TorchDispatchFn` from which the `Preview` is translated."

    @typing.override
    def do(self) -> torch.Tensor:
        return self.preview.do()

    @typing.override
    def tensors(self) -> cabc.Iterator[torch.Tensor]:
        yield from self.preview.tensors()

    @property
    def func(self):
        return self.original.func

    @property
    def types(self):
        return self.original.types

    @property
    def args(self):
        return self.original.args

    @property
    def kwargs(self):
        return self.original.kwargs


def find_preview(thunk: TDispatchFn) -> PreviewFn:
    """
    Try finding a preview with the given `op` and its arguments.
    """

    if thunk.func not in _PREVIEW_CANDIDATES:
        return NotImplemented

    for candidate in _PREVIEW_CANDIDATES[thunk.func]:
        if not (preview := candidate(*thunk.args, **thunk.kwargs)).ok():
            continue

        return PreviewFn(preview, original=thunk)

    return NotImplemented


def all_previews():
    return _PREVIEW_CANDIDATES


_CAMEL_CASE_REGEX = re.compile(r"(?<!^)(?=[A-Z])")


def _camel_to_snake(name: str) -> str:
    return re.sub(_CAMEL_CASE_REGEX, "_", name).lower()
