# Copyright (c) AIoWay Authors - All Rights Reserved

"`NetMorphLinearDeeper` on traced programs instead of `nn.Sequential`."

from aioway.t import fake_mode
import dataclasses as dcls

import torch
from IPython.core.guarded_eval import typing
from torch import nn

from aioway.ir import ModuleCall, NnProgram, TypeSequential, track_module_thunks

from .rewrites import Rewriter

__all__ = ["NetMorphLinearSeqDeeper"]


@dcls.dataclass(frozen=True)
class NetMorphLinearSeqDeeper(Rewriter):
    """
    Deepens `Linear -> *middle -> Linear` in an `NnProgram` into
    `Linear -> *middle -> identity Linear -> *middle -> Linear`.

    Like `NetMorphLinearDeeper`, but the part is found by querying the traced
    program, so the modules don't need to sit in an `nn.Sequential`.
    Run it in the fake mode the program was traced in.
    """

    middle: tuple[type[nn.Module], ...] = (nn.ReLU,)
    "Types of the modules between the two `nn.Linear`, in order."

    def __post_init__(self) -> None:
        if not self.middle:
            raise ValueError("Middle is empty.")

    @typing.override
    def handle(self, prog: NnProgram) -> bool:
        try:
            sub = prog[self.part]
        except LookupError:
            return False

        first, *_, last = _modules(sub)

        # Only the last output may leave the part, and the middle keeps the shape.
        return len(sub.outputs) == 1 and first.out_features == last.in_features

    def rewrite(self, prog: NnProgram) -> NnProgram:
        "A copy of `prog` with the first match deepened. Check `handle` first."

        result = prog.copy()
        result[self.part] = self._build_deeper_net(prog[self.part])
        return result

    def _build_deeper_net(self, sub: NnProgram) -> NnProgram:
        first, middle, last = _modules(sub)
        inserted = _identity_linear(first.out_features)

        (x,) = sub.inputs
        assert sub.inputs.all_fake

        # By tracing additional 2 sets of `middles` and `inserted`,
        # the built program would be deeper.
        with track_module_thunks() as hist, fake_mode():
            for module in [first, *middle, inserted, *middle, last]:
                x = module(x)
                assert isinstance(x, torch.Tensor)

        return hist.program

    @property
    def part(self) -> TypeSequential[ModuleCall]:
        pattern = nn.Linear, *self.middle, nn.Linear
        return TypeSequential(pattern, key=lambda call: call.func)


class _SequentialDecomosed(typing.NamedTuple):
    first: nn.Linear
    middle: list[nn.Module]
    last: nn.Linear


def _modules(prog: NnProgram) -> _SequentialDecomosed:
    first, *middle, last = [call.func for call in prog]
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
