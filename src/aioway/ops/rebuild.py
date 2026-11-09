# Copyright (c) AIoWay Authors - All Rights Reserved

"Rebuilds an `nn.Module`."

from __future__ import annotations

import copy
import dataclasses as dcls
import typing

import loguru as L
import torch
from torch import nn

from aioway.t import is_real_tensor

__all__ = [
    "ModuleRebuilder",
    "Init",
    "NoInit",
    "Constant",
    "Zeros",
    "Ones",
    "Uniform",
    "Normal",
    "TruncNormal",
    "XavierUniform",
    "XavierNormal",
    "KaimingUniform",
    "KaimingNormal",
    "Orthogonal",
]


# Init base ====


class Init(typing.Protocol):
    "Initializes a tensor in place."

    def __call__(self, tensor: torch.Tensor, /) -> None: ...


@dcls.dataclass(frozen=True)
class NoInit(Init):
    "Leave the tensor as allocated (uninitialized memory)."

    def __call__(self, tensor: torch.Tensor, /) -> None:
        pass


# Rebuilder ====


@dcls.dataclass(frozen=True)
class ModuleRebuilder:
    """
    Copy a module with fresh tensors from the current context, filled by `init`.

    Fresh means allocated under the active fake mode, default device, etc.
    The source module is never modified.
    Tensors shared between submodules (tied weights) stay shared in the copy.
    """

    init: Init = dcls.field(default_factory=NoInit)
    "The initialization strategy, applied in place to every parameter and buffer."

    def __call__(self, module: nn.Module, /) -> nn.Module:
        return copy.deepcopy(module, memo=self._tensor_memo(module))

    def _tensor_memo(self, module: nn.Module) -> dict[int, torch.Tensor]:
        """
        Get a `copy.deepcopy` memo for the tensors of the modules.
        """

        memo: dict[int, torch.Tensor] = {}

        for name, param in module.named_parameters():
            L.logger.trace("Rebuilding parameter {}", name)
            memo[id(param)] = nn.Parameter(
                self._new_like(param),
                requires_grad=param.requires_grad,
            )

        for name, buf in module.named_buffers():
            L.logger.trace("Rebuilding buffer {}", name)
            memo[id(buf)] = self._new_like(buf)

        return memo

    def _new_like(self, tensor: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            new = _empty_like(tensor)

            # Fake tensors have no data.
            if is_real_tensor(new):
                self.init(new)

            return new


# Constant inits ====


@dcls.dataclass(frozen=True)
class Constant(Init):
    "Fill with `val`."

    val: float

    def __call__(self, tensor: torch.Tensor, /) -> None:
        nn.init.constant_(tensor, self.val)


@dcls.dataclass(frozen=True)
class Zeros(Init):
    "Fill with 0."

    def __call__(self, tensor: torch.Tensor, /) -> None:
        nn.init.zeros_(tensor)


@dcls.dataclass(frozen=True)
class Ones(Init):
    "Fill with 1."

    def __call__(self, tensor: torch.Tensor, /) -> None:
        nn.init.ones_(tensor)


# Random inits ====


@dcls.dataclass(frozen=True)
class Uniform(Init):
    "Sample from U(a, b)."

    a: float = 0.0
    b: float = 1.0

    def __call__(self, tensor: torch.Tensor, /) -> None:
        nn.init.uniform_(tensor, a=self.a, b=self.b)


@dcls.dataclass(frozen=True)
class Normal(Init):
    "Sample from N(mean, std²)."

    mean: float = 0.0
    std: float = 1.0

    def __call__(self, tensor: torch.Tensor, /) -> None:
        nn.init.normal_(tensor, mean=self.mean, std=self.std)


@dcls.dataclass(frozen=True)
class TruncNormal(Init):
    "Sample from N(mean, std²), redrawing values outside [a, b]."

    mean: float = 0.0
    std: float = 1.0
    a: float = -2.0
    b: float = 2.0

    def __call__(self, tensor: torch.Tensor, /) -> None:
        nn.init.trunc_normal_(tensor, mean=self.mean, std=self.std, a=self.a, b=self.b)


# Fan-scaled inits ====


@dcls.dataclass(frozen=True)
class XavierUniform(Init):
    "Glorot uniform, scaled by fan_in + fan_out. Needs at least 2 dims."

    gain: float = 1.0

    def __call__(self, tensor: torch.Tensor, /) -> None:
        nn.init.xavier_uniform_(tensor, gain=self.gain)


@dcls.dataclass(frozen=True)
class XavierNormal(Init):
    "Glorot normal, scaled by fan_in + fan_out. Needs at least 2 dims."

    gain: float = 1.0

    def __call__(self, tensor: torch.Tensor, /) -> None:
        nn.init.xavier_normal_(tensor, gain=self.gain)


@dcls.dataclass(frozen=True)
class KaimingUniform(Init):
    "He uniform, scaled by fan `mode` and `nonlinearity`. Needs at least 2 dims."

    a: float = 0.0
    mode: typing.Literal["fan_in", "fan_out"] = "fan_in"
    nonlinearity: str = "leaky_relu"

    def __call__(self, tensor: torch.Tensor, /) -> None:
        nn.init.kaiming_uniform_(
            tensor, a=self.a, mode=self.mode, nonlinearity=self.nonlinearity
        )


@dcls.dataclass(frozen=True)
class KaimingNormal(Init):
    "He normal, scaled by fan `mode` and `nonlinearity`. Needs at least 2 dims."

    a: float = 0.0
    mode: typing.Literal["fan_in", "fan_out"] = "fan_in"
    nonlinearity: str = "leaky_relu"

    def __call__(self, tensor: torch.Tensor, /) -> None:
        nn.init.kaiming_normal_(
            tensor, a=self.a, mode=self.mode, nonlinearity=self.nonlinearity
        )


@dcls.dataclass(frozen=True)
class Orthogonal(Init):
    "(Semi-)orthogonal matrix, scaled by `gain`. Needs at least 2 dims."

    gain: float = 1.0

    def __call__(self, tensor: torch.Tensor, /) -> None:
        nn.init.orthogonal_(tensor, gain=self.gain)


# Helpers ====


def _empty_like(tensor: torch.Tensor) -> torch.Tensor:
    """
    Since `torch.empty_like` on a fake tensor would return a fake tensor,
    this should respect the current fake mode set in `aioway`.
    """

    return torch.empty(
        tensor.shape,
        device=tensor.device,
        dtype=tensor.dtype,
        layout=tensor.layout,
    )
