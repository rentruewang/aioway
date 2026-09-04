# Copyright (c) AIoWay Authors - All Rights Reserved


import dataclasses as dcls
import typing

import torch
from torch import nn
from torchrl.data import tensor_specs as tspecs

from aioway.modes import fake_mode
from aioway.tspecs import sample_from_tspec

from .deductions import deduction_for


@dcls.dataclass(frozen=True)
class _SlidingDeduction:
    allowed_dims: tuple[int, ...]
    "The allowed dimensions."

    # Mark `module` as `typing.Any` to use with multiple types.
    def __call__(self, module: typing.Any, input: tspecs.Unbounded) -> tspecs.Unbounded:
        if input.ndim not in self.allowed_dims:
            return NotImplemented

        with fake_mode():
            module = module.module()
            output: torch.Tensor = module(sample_from_tspec(input))

        batch, *rest = output.shape
        assert batch == input.shape[0]

        return tspecs.Unbounded(torch.Size(rest))

    def register(self, *modules: type[nn.Module]) -> None:
        for module in modules:
            deduction_for(module).register(self)


_SlidingDeduction((1, 2)).register(nn.Conv1d, nn.AvgPool1d, nn.MaxPool1d)
_SlidingDeduction((3,)).register(nn.Conv2d, nn.AvgPool2d, nn.MaxPool2d)
_SlidingDeduction((4,)).register(nn.Conv3d, nn.AvgPool3d, nn.MaxPool3d)
