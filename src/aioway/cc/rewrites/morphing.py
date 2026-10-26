# Copyright (c) AIoWay Authors - All Rights Reserved

"`NetMorphLinearDeeper` on traced programs instead of `nn.Sequential`."

import dataclasses as dcls
import typing

import torch
from torch import nn

from aioway.ir import (
    ModuleCall,
    NnProc,
    OrderedIndex,
    Proc,
    Query,
    TypeSequential,
    track_module_thunks,
)
from aioway.t import fake_mode

from .rewrites import RewriteRule

__all__ = ["NetMorphLinearSeqDeeper"]


@dcls.dataclass(frozen=True)
class NetMorphLinearSeqDeeper(RewriteRule):
    """
    Deepens `Linear -> *middle -> Linear` in an `NnProc` into
    `Linear -> *middle -> identity Linear -> *middle -> Linear`.

    Like `NetMorphLinearDeeper`, but the part is found by querying the traced
    program, so the modules don't need to sit in an `nn.Sequential`.
    Run it in the fake mode the program was traced in.
    """

    middle: tuple[type[nn.Module], ...] = (nn.ReLU,)
    "Types of the modules between the two `nn.Linear`, in order."

    def map(self, proc: NnProc) -> NnProc:
        "A copy of `proc` with the first match deepened. Check `handle` first."

        result = proc.copy()
        queried = proc[self.query]
        result[self.query] = self._build_deeper_net(queried)
        return result

    def _build_deeper_net(self, sub: NnProc) -> NnProc:
        first, middle, last = _modules(sub)
        inserted = _identity_linear(first.out_features)

        (x,) = sub.inputs.values()
        assert sub.inputs.tensors.all_fake

        # By tracing additional 2 sets of `middles` and `inserted`,
        # the built program would be deeper.
        with track_module_thunks() as hist, fake_mode():
            for module in [first, *middle, inserted, *middle, last]:
                x = module(x)
                assert isinstance(x, torch.Tensor)

        return hist.procedure

    @property
    def query(self) -> _DeeperQuery:
        return _DeeperQuery(self)


@dcls.dataclass(frozen=True)
class _DeeperQuery(Query[ModuleCall]):
    "The custom query that only matches modules between `nn.Linear`s."

    deeper: NetMorphLinearSeqDeeper

    @typing.override
    def _find(self, proc: Proc[ModuleCall]) -> OrderedIndex:
        indices = self.query(proc)
        sub = proc[indices]

        first, *_, last = _modules(sub)

        # Only the last output may leave the part, and the middle keeps the shape.
        assert len(sub.outputs) == 1
        assert first.out_features == last.in_features

        return indices

    @property
    def query(self) -> TypeSequential[ModuleCall]:
        pattern = nn.Linear, *self.deeper.middle, nn.Linear
        return TypeSequential(pattern, key=lambda call: call.func)


class _SequentialDecomosed(typing.NamedTuple):
    first: nn.Linear
    middle: list[nn.Module]
    last: nn.Linear


def _modules(proc: Proc[ModuleCall]) -> _SequentialDecomosed:
    first, *middle, last = [call.func for call in proc]
    assert isinstance(first, nn.Linear)
    assert isinstance(last, nn.Linear)
    assert all(isinstance(m, nn.Module) for m in middle)
    return _SequentialDecomosed(first=first, last=last, middle=middle)


@torch.no_grad()
def _identity_linear(features: int) -> nn.Linear:
    linear = nn.Linear(features, features)

    # In fake mode, these are automatically no-op.
    linear.weight.copy_(torch.eye(features))
    linear.bias.zero_()

    return linear
