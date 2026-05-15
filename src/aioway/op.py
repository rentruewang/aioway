# Copyright (c) AIoWay Authors - All Rights Reserved

"The module containing `Fate` interface, the implementation for fake aten operations."

import abc
import dataclasses as dcls
import inspect
import typing

import torch
from torch import _ops

from aioway._common import dcls_frozen_no_repr, render_fcall

__all__ = ["find_fate", "aten_to_fate", "Fate"]


_ATEN_TO_FATE_LIST: dict[_ops.OpOverload, list[type[Fate]]] = {}
"The registry to store `Fate` operators."


@dcls_frozen_no_repr
class Op(abc.ABC):
    """
    `Op` stands for [o]verridable [p]ass. Or [op]erator. It follows a pattern:
    it sits in a family of similar operations, and can be looked up by a key (and signature).

    An op is a custom (`aioway`) operation that may override the operator's behaviors at runtime,
    or at least provide some static, inspectable info on the current call.

    Right now, there are 2 `Op` kinds:
    1. `Fate` for ATen operations.
    2. `Preview` for `nn.Module` init.
    """

    KEY: typing.ClassVar[typing.Any]
    """
    The torch IR that this `Fate` would be implementing.
    """

    def __init_subclass__(cls, key: typing.Any = None) -> None:
        if not inspect.isabstract(cls) and key is None:
            raise KeyError(f"Key is not specified for non abstract {cls=}.")

        cls.KEY = key

    @typing.override
    def __repr__(self) -> str:
        return render_fcall(f"fate::{self.name()}", **dcls.asdict(self))

    @typing.override
    def __hash__(self) -> int:
        return id(self)

    @abc.abstractmethod
    def ok(self) -> bool:
        """
        Whether or not the arguments are valid.
        """

        raise NotImplementedError

    def do(self) -> torch.Tensor:
        """
        Generate the fake tensor.
        """

        return self.KEY(**dcls.asdict(self))

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
            ir = cls.KEY
        except AttributeError:
            return

        # Mimick defaultdict behavior.
        # Using dict over defaultdict s.t. we don't need special handling in `repr`.
        if ir not in _ATEN_TO_FATE_LIST:
            _ATEN_TO_FATE_LIST[ir] = []

        _ATEN_TO_FATE_LIST[ir].append(cls)
