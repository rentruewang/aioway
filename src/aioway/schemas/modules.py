# Copyright (c) AIoWay Authors - All Rights Reserved

from torch import nn

from .utils import replace_tensors_with_attr

__all__ = ["hash_module_state_dict"]


def hash_module_state_dict(module: nn.Module) -> dict[str, int]:
    state_dict = module.state_dict()
    replaced = replace_tensors_with_attr(state_dict)
    return {key: hash(val) for key, val in replaced.items()}
