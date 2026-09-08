# Copyright (c) AIoWay Authors - All Rights Reserved

import dataclasses as dcls
import typing
from collections import abc as cabc

import tensordict as td
import torch
from torch import nn
from torchrl.data import tensor_specs as tspecs

from .signs import sign_reg

__all__ = ["ArgsTSpec", "NnArgs"]


class ArgsTSpec(tspecs.Composite):
    """
    The `Composite` subclass that represents an argument list.
    """


class Invoker(typing.Protocol):
    def __call__(self, module: nn.Module) -> typing.Any: ...


@dcls.dataclass(frozen=True)
class NnArgs:
    "The tensordict that marks something"

    args: typing.Any
    "The underlying tensordict or tensor."

    def __post_init__(self) -> None:
        _ = self.invoke

    @property
    def invoke(self) -> Invoker:
        "Invoke the module for you, with the unpacking taken care of."

        if isinstance(self.args, cabc.Mapping | td.TensorDict | td.TensorClass):
            return self._invoke_tdict

        if isinstance(self.args, torch.Tensor):
            return self._invoke_tensor

        raise TypeError(f"Unhandled type {type(self.args)=}.")

    def _invoke_tensor(self, module: nn.Module):
        return module(self.args)

    def _invoke_tdict(self, module: nn.Module) -> typing.Any:
        args, kwargs = self._apply_tdict(type(module))
        return module(*args, **kwargs)

    def _apply_tdict(
        self, nn_type: type[nn.Module]
    ) -> tuple[list[typing.Any], dict[str, typing.Any]]:
        "Apply `self` onto `nn.Module.forward` of the given `nn_type`."

        signature = sign_reg()[nn_type]

        args = []
        kwargs = {}

        for key, param in signature.params.items():
            if param.is_positional_only:
                args.append(self.args[key])

            else:
                kwargs[key] = self.args[key]

        return args, kwargs
