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
    "ConstantInit",
    "ZerosInit",
    "OnesInit",
    "UniformInit",
    "NormalInit",
    "TruncNormalInit",
    "XavierUniformInit",
    "XavierNormalInit",
    "KaimingUniformInit",
    "KaimingNormalInit",
    "OrthogonalInit",
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

    def __call__[T: nn.Module](self, module: T, /) -> T:
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
class ConstantInit(Init):
    "Wraps `torch.nn.init.constant_`."

    val: float
    "The fill value."

    def __call__(self, tensor: torch.Tensor, /) -> None:
        nn.init.constant_(tensor, self.val)


@dcls.dataclass(frozen=True)
class ZerosInit(Init):
    "Wraps `torch.nn.init.zeros_`."

    def __call__(self, tensor: torch.Tensor, /) -> None:
        nn.init.zeros_(tensor)


@dcls.dataclass(frozen=True)
class OnesInit(Init):
    "Wraps `torch.nn.init.ones_`."

    def __call__(self, tensor: torch.Tensor, /) -> None:
        nn.init.ones_(tensor)


# Random inits ====


@dcls.dataclass(frozen=True)
class UniformInit(Init):
    "Wraps `torch.nn.init.uniform_`."

    a: float = 0.0
    "Lower bound."

    b: float = 1.0
    "Upper bound."

    def __call__(self, tensor: torch.Tensor, /) -> None:
        nn.init.uniform_(tensor, a=self.a, b=self.b)


@dcls.dataclass(frozen=True)
class NormalInit(Init):
    "Wraps `torch.nn.init.normal_`."

    mean: float = 0.0
    "Mean of the normal distribution."

    std: float = 1.0
    "Standard deviation of the normal distribution."

    def __call__(self, tensor: torch.Tensor, /) -> None:
        nn.init.normal_(tensor, mean=self.mean, std=self.std)


@dcls.dataclass(frozen=True)
class TruncNormalInit(Init):
    "Wraps `torch.nn.init.trunc_normal_`."

    mean: float = 0.0
    "Mean of the normal distribution."

    std: float = 1.0
    "Standard deviation of the normal distribution."

    a: float = -2.0
    "Lower cutoff, values below are redrawn."

    b: float = 2.0
    "Upper cutoff, values above are redrawn."

    def __call__(self, tensor: torch.Tensor, /) -> None:
        nn.init.trunc_normal_(tensor, mean=self.mean, std=self.std, a=self.a, b=self.b)


# Fan-scaled inits ====
# These need tensors with at least 2 dims, so they fail on biases and norm layers.


@dcls.dataclass(frozen=True)
class XavierUniformInit(Init):
    "Wraps `torch.nn.init.xavier_uniform_`."

    gain: float = 1.0
    "Scaling factor."

    def __call__(self, tensor: torch.Tensor, /) -> None:
        nn.init.xavier_uniform_(tensor, gain=self.gain)


@dcls.dataclass(frozen=True)
class XavierNormalInit(Init):
    "Wraps `torch.nn.init.xavier_normal_`."

    gain: float = 1.0
    "Scaling factor."

    def __call__(self, tensor: torch.Tensor, /) -> None:
        nn.init.xavier_normal_(tensor, gain=self.gain)


@dcls.dataclass(frozen=True)
class KaimingUniformInit(Init):
    "Wraps `torch.nn.init.kaiming_uniform_`."

    a: float = 0.0
    "Negative slope of the rectifier after this layer (only for `leaky_relu`)."

    mode: typing.Literal["fan_in", "fan_out"] = "fan_in"
    "`fan_in` preserves the variance in the forward pass, `fan_out` in the backward."

    nonlinearity: str = "leaky_relu"
    "The activation after this layer, used to pick the gain."

    def __call__(self, tensor: torch.Tensor, /) -> None:
        nonlinearity: typing.Any = self.nonlinearity
        nn.init.kaiming_uniform_(
            tensor, a=self.a, mode=self.mode, nonlinearity=nonlinearity
        )


@dcls.dataclass(frozen=True)
class KaimingNormalInit(Init):
    "Wraps `torch.nn.init.kaiming_normal_`."

    a: float = 0.0
    "Negative slope of the rectifier after this layer (only for `leaky_relu`)."

    mode: typing.Literal["fan_in", "fan_out"] = "fan_in"
    "`fan_in` preserves the variance in the forward pass, `fan_out` in the backward."

    nonlinearity: str = "leaky_relu"
    "The activation after this layer, used to pick the gain."

    def __call__(self, tensor: torch.Tensor, /) -> None:
        nonlinearity: typing.Any = self.nonlinearity
        nn.init.kaiming_normal_(
            tensor, a=self.a, mode=self.mode, nonlinearity=nonlinearity
        )


@dcls.dataclass(frozen=True)
class OrthogonalInit(Init):
    "Wraps `torch.nn.init.orthogonal_`."

    gain: float = 1.0
    "Scaling factor."

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
