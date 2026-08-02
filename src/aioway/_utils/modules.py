# Copyright (c) AIoWay Authors - All Rights Reserved

"A collection of module related utilities."

import torch
import torch.nn as nn
from torch import nn

__all__ = ["rebuild_module"]


def rebuild_module(module: nn.Module, /) -> nn.Module:
    """
    Rebuild the module recursively (initialize in the post order).

    Used when you want to re initialize the `nn.Module` in the current context.
    """

    for child in module.children():
        rebuild_module(child)

    _init_params(module)
    _init_buffers(module)

    return module


def _init_buffers(module: nn.Module):
    "Initialize non-learnable buffers in the current module."

    for name, buf in list(module.named_buffers(recurse=False)):
        setattr(module, name, torch.empty_like(buf))


def _init_params(module: nn.Module):
    "Initialize learnable parameters in the current module."

    for name, param in list(module.named_parameters(recurse=False)):
        new_param = nn.Parameter(
            torch.empty_like(param),
            requires_grad=param.requires_grad,
        )
        setattr(module, name, new_param)
