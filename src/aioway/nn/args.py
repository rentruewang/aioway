# Copyright (c) AIoWay Authors - All Rights Reserved

import inspect
import typing
import tensordict as td
from torchrl.data import tensor_specs as tspecs
import dataclasses as dcls
from torch import nn
from .signs import sign_reg

__all__ = ["ArgsTSpec"]


class ArgsTSpec(tspecs.Composite):
    """
    The `Composite` subclass that represents an argument list.
    """


@dcls.dataclass(frozen=True)
class Args:
    "The tensordict that marks something"

    tdict: td.TensorDict | td.TensorClass
    "The underlying tensordict."

    def invoke(self, module: nn.Module) -> typing.Any:
        "Invoke the module for you, with the unpacking taken care of."

        args, kwargs = self.apply(type(module))
        return module(*args, **kwargs)

    def apply(
        self, nn_type: type[nn.Module]
    ) -> tuple[list[typing.Any], dict[str, typing.Any]]:
        "Apply `self` onto `nn.Module.forward` of the given `nn_type`."

        signature = sign_reg()[nn_type]

        args = []
        kwargs = {}

        for key, param in signature.params.items():
            if param.is_positional_only:
                args.append(self.tdict[key])

            else:
                kwargs[key] = self.tdict[key]

        return args, kwargs
