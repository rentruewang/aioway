# Copyright (c) AIoWay Authors - All Rights Reserved

"The optimizer `Fn`."

import typing
from collections import abc as cabc

import torch
from torch import optim

from aioway._common import dcls_no_eq

from .fn import DoFn

__all__ = ["Optim"]


class OptimType(typing.Protocol):
    def __call__(
        self, params: cabc.Iterable[torch.Tensor], lr: float
    ) -> optim.Optimizer: ...


class Optim[O: optim.Optimizer]:
    """
    The wrapper type that generates optimizer related `Fn`, used to do the step operation.

    The init signature follows closly from `torch.optim.*` optimizers,
    having `lr` and `params` as input (to pass into `params`).
    """

    def __init__(
        self,
        optim_cls: OptimType,
        params: cabc.Iterable[torch.Tensor],
        lr: float,
    ) -> None:
        self._opt = optim_cls(params=params, lr=lr)

    def zero_grad(self):
        return OptimFn(self._opt, lambda opt: opt.zero_grad())

    def step(self):
        return OptimFn(self._opt, lambda opt: opt.step())


@dcls_no_eq
class OptimFn[O: optim.Optimizer](DoFn[None]):
    """
    The `Fn` that stores `torch.optim.Optimizer` methods.

    Since right now we only support `zero_grad()` and `step()`,
    the `function` signature is made to be `(Optimizer) -> None`.
    """

    optimizer: optim.Optimizer
    "The optimizer that is stored."

    function: cabc.Callable[[optim.Optimizer], None]
    "The function that will be called in `forward`."

    @typing.override
    def forward(self) -> None:
        return self.function(self.optimizer)
