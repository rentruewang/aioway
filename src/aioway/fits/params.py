# Copyright (c) AIoWay Authors - All Rights Reserved

import dataclasses as dcls
import typing

import torch
from torch import nn, optim

__all__ = [
    "ParamOptimInfo",
    "index_optim_params",
    "ParamModuleInfo",
    "index_module_params",
]

_OPT_PARAM = "params"


@dcls.dataclass(frozen=True, eq=False)
class ParamOptimInfo:
    "Get the parameter's information, from an optimizer."

    param: torch.Tensor
    "The parameter in question."

    group_idx: int
    "The index it stays in the optimizer."

    opt: optim.Optimizer
    "The optimizer it belongs to."

    @property
    def group(self) -> dict[str, typing.Any]:
        "The live param group dict. Edits here change the optimizer."
        return self.opt.param_groups[self.group_idx]

    @property
    def hyper_params(self) -> dict[str, typing.Any]:
        "Copy of the group's hyperparameters (everything except 'params')."
        return {k: v for k, v in self.group.items() if k != _OPT_PARAM}

    @property
    def state(self) -> dict[str, typing.Any]:
        "Per-parameter optimizer state."

        return self.opt.state[self.param]


def index_optim_params(opt: optim.Optimizer) -> dict[torch.Tensor, ParamOptimInfo]:
    "Map each parameter managed by `opt` to its ParamInfo."

    result: dict[torch.Tensor, ParamOptimInfo] = {}

    for index, group in enumerate(opt.param_groups):
        for param in group[_OPT_PARAM]:
            result[param] = ParamOptimInfo(param=param, group_idx=index, opt=opt)

    return result


@dcls.dataclass(frozen=True, eq=False)
class ParamModuleInfo:
    "Get the parameter's information, from a module."

    param: torch.Tensor
    "The parameter in question."

    path: str
    "Dotted path from the root module, e.g. 'encoder.layers.0.weight'."

    module: nn.Module
    "The root module it was found in."

    @property
    def owner(self) -> nn.Module:
        "The submodule that directly holds the parameter."
        prefix, _, _ = self.path.rpartition(".")
        return self.module.get_submodule(prefix)

    @property
    def attr(self) -> str:
        "Attribute name on the owner, e.g. 'weight'."

        return self.path.rpartition(".")[2]


def index_module_params(module: nn.Module) -> dict[torch.Tensor, ParamModuleInfo]:
    "Map each parameter of `module` to its ParamModuleInfo."

    result: dict[torch.Tensor, ParamModuleInfo] = {}

    for path, param in module.named_parameters():
        result[param] = ParamModuleInfo(param=param, path=path, module=module)

    return result
