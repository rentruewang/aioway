# Copyright (c) AIoWay Authors - All Rights Reserved

"A collection of module related utilities."

import logging

import torch
import torch.nn as nn
from torch import nn

__all__ = ["rebuild_module"]

LOGGER = logging.getLogger(__name__)


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
        LOGGER.debug("Initalizing %s attribute", name)
        setattr(module, name, _empty_like(buf))


def _init_params(module: nn.Module):
    "Initialize learnable parameters in the current module."

    for name, param in list(module.named_parameters(recurse=False)):
        LOGGER.debug("Initalizing %s attribute", name)
        new_param = nn.Parameter(
            _empty_like(param),
            requires_grad=param.requires_grad,
        )
        setattr(module, name, new_param)


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
        requires_grad=tensor.requires_grad,
    )
