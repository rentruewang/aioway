# Copyright (c) AIoWay Authors - All Rights Reserved

"The preview module protocol definition."

import functools
import logging
import typing
from collections import abc as cabc

import torch
from torch import nn

from aioway._common import dcls_no_eq_no_repr
from aioway.ctx import enabled_fake_mode, fake_mode_func

from .de import defer, eager
from .tensors import TensorFn

__all__ = ["ModuleFn", "FakableModule"]

LOGGER = logging.getLogger(__name__)


class FakableModule[**P, M: nn.Module]:
    """
    `FakableModule` allows `nn.Module` to be used with fake mode.

    Example:
        `FakableModule(torch.nn.Linear, in_features=1, out_features=2)`.
        Note that the arguments following the first type argument will be passed into the `nn.Module` type,
        so they are exactly as those on the `torch` documentation.
    """

    def __init__(
        self, module: cabc.Callable[P, M], /, *args: P.args, **kwargs: P.kwargs
    ) -> None:
        self._module = module
        self._args = args
        self._kwargs = kwargs

    def __call__(self, tensor: torch.Tensor | TensorFn, /) -> torch.Tensor:
        tensor = eager(tensor)
        return self.module(tensor)

    @functools.cached_property
    @fake_mode_func
    def fake(self) -> M:
        return self._module(*self._args, **self._kwargs)

    @functools.cached_property
    def real(self) -> M:
        return self._module(*self._args, **self._kwargs)

    @property
    def module(self):
        return self.fake if enabled_fake_mode() else self.real

    def parameters(self) -> cabc.Generator[nn.Parameter]:
        yield from self.module.parameters()


@dcls_no_eq_no_repr
class ModuleFn[**P, M: nn.Module](TensorFn):
    """
    `ModuleFn` informs us how `nn.Module` would behave without initializing it.
    """

    tensor: TensorFn
    module: FakableModule[P, M]

    @typing.override
    def forward(self) -> torch.Tensor:
        tensor = self.tensor.do()
        module = self.module.module
        return module(tensor)

    @typing.override
    def _params_self(self) -> cabc.Generator[torch.Tensor]:
        yield from self.module.parameters()

    @classmethod
    def build(
        cls, tensor: torch.Tensor | TensorFn, module: FakableModule
    ) -> typing.Self:
        return cls(tensor=defer(tensor), module=module)
