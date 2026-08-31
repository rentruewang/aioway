# Copyright (c) AIoWay Authors - All Rights Reserved

from torch import nn
from torchrl.data import tensor_specs as tspecs

from aioway.emits import sample_from_tspec


def linear_deduct(module: nn.Module, input: tspecs.Unbounded) -> tspecs.Unbounded:
    sample_from_tspec(input)
