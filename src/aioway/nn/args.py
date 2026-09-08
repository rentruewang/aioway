# Copyright (c) AIoWay Authors - All Rights Reserved

import torch
import inspect
import typing
import tensordict as td
from torchrl.data import tensor_specs as tspecs
import dataclasses as dcls
from torch import nn
from .signs import sign_reg
from collections import abc as cabc

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

    arguments: typing.Any
    "The underlying tensordict."

    def __post_init__(self) -> None:
        _ = self.invoke

    @property
    def invoke(self) -> Invoker:
        "Invoke the module for you, with the unpacking taken care of."

        if isinstance(self.arguments, cabc.Mapping | td.TensorDict | td.TensorClass):
            return self._invoke_tdict

        if isinstance(self.arguments, torch.Tensor):
            return self._invoke_tensor

        raise TypeError(f"Unhandled type {type(self.argument)=}.")

    def _invoke_tensor(self, module: nn.Module):
        return module(self.arguments)

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
                args.append(self.arguments[key])

            else:
                kwargs[key] = self.arguments[key]

        return args, kwargs
