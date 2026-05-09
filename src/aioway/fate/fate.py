# Copyright (c) AIoWay Authors - All Rights Reserved

import abc
import dataclasses as dcls
import inspect
import re
import typing

import torch
from torch import _ops

from aioway._common import dcls_frozen_no_repr, render_fcall

__all__ = ["find_fate", "aten_to_fate", "Fate"]


_ATEN_TO_FATE_LIST: dict[_ops.OpOverload, list[type[Fate]]] = {}
"The registry to store `Fate` operators."


@dcls_frozen_no_repr
class Fate(abc.ABC):
    """
    `Fate` stands for [f]ake [ate]n. Or [fa]ke [te]nsor. Or a tensor's [fate] (how it behaves).

    It overrides aten ops in fake mode and compute extra properties,
    such as storage costs and compute costs, as well as patching some operations with worst case.
    For example, boolean masking is data dependent, and is thus not supported by fake mode.
    """

    IR: typing.ClassVar[_ops.OpOverload]
    """
    The torch IR that this `Fate` would be implementing.
    """

    def __init_subclass__(cls) -> None:
        cls.__register_preview()

    @typing.override
    def __repr__(self) -> str:
        return render_fcall(f"preview::{self.name()}", **dcls.asdict(self))

    @typing.override
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
        if op not in _ATEN_TO_FATE_LIST:
            _ATEN_TO_FATE_LIST[op] = []

        _ATEN_TO_FATE_LIST[op].append(cls)


def find_fate(op: _ops.OpOverload, *args: typing.Any, **kwargs: typing.Any) -> Fate:
    """
    Try finding a `Fate` operator with the thunk, and then wrap into `FateFn`.

    Returns `NotImplemented` if a candidate is not found.
    """

    if op not in _ATEN_TO_FATE_LIST:
        return NotImplemented

    for candidate in _ATEN_TO_FATE_LIST[op]:
        if not (preview := candidate(*args, **kwargs)).ok():
            continue

        return preview

    return NotImplemented


def aten_to_fate():
    """
    A mapping of `aten` ops to their `Fate` counterparts.
    """

    return _ATEN_TO_FATE_LIST


_CAMEL_CASE_REGEX = re.compile(r"(?<!^)(?=[A-Z])")


def _camel_to_snake(name: str) -> str:
    return re.sub(_CAMEL_CASE_REGEX, "_", name).lower()
