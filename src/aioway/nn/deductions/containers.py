# Copyright (c) AIoWay Authors - All Rights Reserved

from torch import nn

from aioway.tspecs import TSpec

from .deductions import deduction_for

__all__ = ["sequential_deduct"]


@deduction_for(nn.Sequential).register
def sequential_deduct(self: nn.Sequential, input: TSpec) -> TSpec:
    for sub in self.children():
        deduction = deduction_for(sub)

        if (output := deduction(sub, input)) is NotImplemented:
            return NotImplemented

        input = output
    return input
