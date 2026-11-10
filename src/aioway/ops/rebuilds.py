# Copyright (c) AIoWay Authors - All Rights Reserved

"Rebuilds an `nn.Module`."

import abc
import contextlib as ctxl
import dataclasses as dcls
import typing

import loguru as L
import torch
from torch import nn

from aioway._utils import AnyDict, any_dict
from aioway.t import is_real_tensor

__all__ = [
    "ModuleRebuilder",
    "InitError",
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


@typing.dataclass_transform(frozen_default=True, kw_only_default=True)
def init_dcls(cls):
    return dcls.dataclass(frozen=True, kw_only=True)(cls)


class InitError(Exception):
    "An `Init` failed on one of the module's tensors."


@init_dcls
class Init(abc.ABC):
    "Initializes a tensor in place."

    allow_failure: bool = False

    @typing.final
    def __call__(self, tensor: torch.Tensor, /) -> None:
        try:
            self.init(tensor)
        except Exception as e:
            if self.allow_failure:
                return
            else:
                raise

    @abc.abstractmethod
    def init(self, tensor: torch.Tensor, /) -> None:
        raise NotImplementedError


@init_dcls
class NoInit(Init):
    "Leave the tensor as allocated (uninitialized memory)."

    def init(self, tensor: torch.Tensor, /) -> None:
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
        return self._replace_param_memo(module).replace(module)

    def _replace_param_memo(
        self, module: nn.Module
    ) -> AnyDict[torch.Tensor, torch.Tensor]:
        """
        Get the replacement memo for the tensors of the module.
        """

        memo: AnyDict[torch.Tensor, torch.Tensor] = any_dict()

        for name, param in module.named_parameters():
            L.logger.trace("Rebuilding parameter {}", name)
            memo[param] = nn.Parameter(
                self._new_like(name, param),
                requires_grad=param.requires_grad,
            )

        for name, buf in module.named_buffers():
            L.logger.trace("Rebuilding buffer {}", name)
            memo[buf] = self._new_like(name, buf)

        return memo

    def _new_like(self, name: str, tensor: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            new = _empty_like(tensor)

            # Fake tensors have no data.
            if not is_real_tensor(new):
                return new

            with self._init_guard(name, tensor):
                self.init(new)

            return new

    @ctxl.contextmanager
    def _init_guard(self, name: str, tensor: torch.Tensor):
        "Re-raise init failures as `InitError`, naming the tensor."

        try:
            yield
        except Exception as e:
            raise InitError(
                f"{self.init!r} failed on {name!r} "
                f"(shape={tuple(tensor.shape)}, dtype={tensor.dtype})"
            ) from e


# Constant inits ====


@init_dcls
class ConstantInit(Init):
    "Wraps `torch.nn.init.constant_`."

    val: float
    "The fill value."

    def init(self, tensor: torch.Tensor, /) -> None:
        nn.init.constant_(tensor, self.val)


@init_dcls
class ZerosInit(Init):
    "Wraps `torch.nn.init.zeros_`."

    def init(self, tensor: torch.Tensor, /) -> None:
        nn.init.zeros_(tensor)


@init_dcls
class OnesInit(Init):
    "Wraps `torch.nn.init.ones_`."

    def init(self, tensor: torch.Tensor, /) -> None:
        nn.init.ones_(tensor)


# Random inits ====


@init_dcls
class UniformInit(Init):
    "Wraps `torch.nn.init.uniform_`."

    a: float = 0.0
    "Lower bound."

    b: float = 1.0
    "Upper bound."

    def init(self, tensor: torch.Tensor, /) -> None:
        nn.init.uniform_(tensor, a=self.a, b=self.b)


@init_dcls
class NormalInit(Init):
    "Wraps `torch.nn.init.normal_`."

    mean: float = 0.0
    "Mean of the normal distribution."

    std: float = 1.0
    "Standard deviation of the normal distribution."

    def init(self, tensor: torch.Tensor, /) -> None:
        nn.init.normal_(tensor, mean=self.mean, std=self.std)


@init_dcls
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

    def init(self, tensor: torch.Tensor, /) -> None:
        nn.init.trunc_normal_(tensor, mean=self.mean, std=self.std, a=self.a, b=self.b)


# Fan-scaled inits (at least 2 dims, so no biases nor norms) ====


@init_dcls
class XavierUniformInit(Init):
    "Wraps `torch.nn.init.xavier_uniform_`."

    gain: float = 1.0
    "Scaling factor."

    def init(self, tensor: torch.Tensor, /) -> None:
        nn.init.xavier_uniform_(tensor, gain=self.gain)


@init_dcls
class XavierNormalInit(Init):
    "Wraps `torch.nn.init.xavier_normal_`."

    gain: float = 1.0
    "Scaling factor."

    def init(self, tensor: torch.Tensor, /) -> None:
        nn.init.xavier_normal_(tensor, gain=self.gain)


@init_dcls
class KaimingUniformInit(Init):
    "Wraps `torch.nn.init.kaiming_uniform_`."

    a: float = 0.0
    "Negative slope of the rectifier after this layer (only for `leaky_relu`)."

    mode: typing.Literal["fan_in", "fan_out"] = "fan_in"
    "`fan_in` preserves the variance in the forward pass, `fan_out` in the backward."

    nonlinearity: str = "leaky_relu"
    "The activation after this layer, used to pick the gain."

    def init(self, tensor: torch.Tensor, /) -> None:
        nonlinearity: typing.Any = self.nonlinearity
        nn.init.kaiming_uniform_(
            tensor, a=self.a, mode=self.mode, nonlinearity=nonlinearity
        )


@init_dcls
class KaimingNormalInit(Init):
    "Wraps `torch.nn.init.kaiming_normal_`."

    a: float = 0.0
    "Negative slope of the rectifier after this layer (only for `leaky_relu`)."

    mode: typing.Literal["fan_in", "fan_out"] = "fan_in"
    "`fan_in` preserves the variance in the forward pass, `fan_out` in the backward."

    nonlinearity: str = "leaky_relu"
    "The activation after this layer, used to pick the gain."

    def init(self, tensor: torch.Tensor, /) -> None:
        nonlinearity: typing.Any = self.nonlinearity
        nn.init.kaiming_normal_(
            tensor, a=self.a, mode=self.mode, nonlinearity=nonlinearity
        )


@init_dcls
class OrthogonalInit(Init):
    "Wraps `torch.nn.init.orthogonal_`."

    gain: float = 1.0
    "Scaling factor."

    def init(self, tensor: torch.Tensor, /) -> None:
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
