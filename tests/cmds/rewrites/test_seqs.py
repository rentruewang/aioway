# Copyright (c) AIoWay Authors - All Rights Reserved

import inspect

import torch
from torch import nn
from torch import testing as tt

from aioway.cmds import NetMorphLinearOpDeeper, OpRewriter, unroll_sequential
from aioway.ops import ModuleOp, Op, OpFunc


def types(op: Op) -> list[type[nn.Module]]:
    return [type(o.net) for o in op.post_order() if isinstance(o, ModuleOp)]


def test_unroll_keeps_value():
    seq = nn.Sequential(nn.Linear(4, 8), nn.ReLU(), nn.Linear(8, 4))
    x = torch.randn(2, 4)

    tt.assert_close(unroll_sequential(seq)(x), seq(x))


def test_unroll_signature():
    func = unroll_sequential(nn.Sequential(nn.ReLU()))

    assert list(inspect.signature(func).parameters) == ["input"]


def test_unroll_flatten():
    seq = nn.Sequential(nn.Linear(4, 8), nn.Sequential(nn.ReLU(), nn.Linear(8, 4)))

    func = unroll_sequential(seq)

    assert types(func.output) == [nn.Linear, nn.ReLU, nn.Linear]


def test_unroll_then_deepen():
    seq = nn.Sequential(nn.Linear(4, 8), nn.ReLU(), nn.Linear(8, 4))
    x = torch.randn(2, 4)

    rewritten = OpRewriter(NetMorphLinearOpDeeper())(unroll_sequential(seq).output)
    result = OpFunc(rewritten)

    assert types(result.output) == [nn.Linear, nn.ReLU, nn.Linear, nn.ReLU, nn.Linear]
    tt.assert_close(result(x), seq(x))
