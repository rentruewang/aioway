# Copyright (c) AIoWay Authors - All Rights Reserved

from aioway.tensors import is_fake_mode_on
from aioway.nn import deduction_for
import typing

from torch import nn
import torch
from .rewrites import Rewriter

__all__ = ["NetMorphLinearDeeper"]


class NetMorphLinearDeeper(Rewriter):
    """
    Handless the sequential module with 2 `nn.Linear`, first and last layer.
    """

    @typing.override
    def handle(self, module: nn.Module) -> bool:
        if not isinstance(module, nn.Sequential):
            return False

        modules = list(module.children())

        # Handles the case where the first and last are linear, and only first and last.
        if sum(1 for mod in modules if isinstance(mod, nn.Linear)) != 2:
            return False

        first, last = modules[0], modules[-1]
        if not isinstance(first, nn.Linear) or not isinstance(last, nn.Linear):
            return False

        # The intermediate modules should not change shapes.
        if first.out_features != last.in_features:
            return False

        return True

    @typing.override
    def rewrite(self, module: nn.Sequential) -> nn.Sequential:
        modules = list(module.children())

        first, *middle, last = modules
        assert isinstance(first, nn.Linear)
        assert isinstance(last, nn.Linear)

        inserted = self._get_inserted(first)
        result = nn.Sequential(first, *middle, inserted, *middle, last)
        return result

    @torch.no_grad()
    def _get_inserted(self, first: nn.Linear) -> nn.Linear:
        features = first.out_features
        linear = nn.Linear(in_features=features, out_features=features)

        # In fake mode, these are automatically no-op.
        linear.weight.copy_(torch.eye(features))
        linear.bias.zero_()

        return linear
