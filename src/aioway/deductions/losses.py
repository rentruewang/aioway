# Copyright (c) AIoWay Authors - All Rights Reserved

import logging

from torch import nn
from torchrl.data import tensor_specs as tspecs

from aioway.tspecs import LossTSpec

from .deductions import deduction_for

__all__ = ["symmetric_loss_deduct"]

LOGGER = logging.getLogger(__name__)


_LOSS_TSPEC = LossTSpec()


@deduction_for(nn.L1Loss).register
@deduction_for(nn.SmoothL1Loss).register
@deduction_for(nn.MSELoss).register
def symmetric_loss_deduct(
    self, input: tspecs.Unbounded, target: tspecs.Unbounded
) -> LossTSpec:
    if input.shape != target.shape:
        return NotImplemented

    return _LOSS_TSPEC
