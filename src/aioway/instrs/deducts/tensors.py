# Copyright (c) AIoWay Authors - All Rights Reserved

import dataclasses as dcls

import torch
from torchrl.data import tensor_specs as tspecs

from aioway.modes import fake_mode
from aioway.tspecs import TSpecLike

from ..instrs import Instr
from .deducts import Deductor

__all__ = ["identity_infer", "UnboundedInfer"]


def identity_infer[T: TSpecLike](tspec: T) -> T:
    return tspec


@dcls.dataclass(frozen=True)
class UnboundedInfer(Deductor):
    """
    Infer an unbounded output from an unbounded input.
    """

    instr: Instr

    def __call__(self, tspec: TSpecLike, /) -> tspecs.Unbounded:
        assert isinstance(tspec, tspecs.Unbounded)

        with fake_mode():
            module = self.instr.module()
            random = tspec.sample(torch.Size([1]))

        output = module(random)
        return tspecs.Unbounded(shape=output.shape)
