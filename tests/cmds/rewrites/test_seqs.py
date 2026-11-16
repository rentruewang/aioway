# Copyright (c) AIoWay Authors - All Rights Reserved

import torch
from torch import nn
from torch import testing as tt

from aioway.cmds import NetMorphLinearOpDeeper, OpRewriter, unroll_sequential
from aioway.ops import ModuleOp, Op, TensorOp


def types(op: Op) -> list[type[nn.Module]]:
    return [type(o.nn) for o in op.post_order() if isinstance(o, ModuleOp)]


def test_unroll_keeps_value():
    seq = nn.Sequential(nn.Linear(4, 8), nn.ReLU(), nn.Linear(8, 4))
    x = torch.randn(2, 4)

    tt.assert_close(unroll_sequential(seq, TensorOp(x))(), seq(x))


def test_unroll_nested():
    seq = nn.Sequential(nn.Linear(4, 8), nn.Sequential(nn.ReLU(), nn.Linear(8, 4)))

    op = unroll_sequential(seq, TensorOp(torch.randn(2, 4)))

    assert types(op) == [nn.Linear, nn.ReLU, nn.Linear]


def test_unroll_then_deepen():
    seq = nn.Sequential(nn.Linear(4, 8), nn.ReLU(), nn.Linear(8, 4))
    x = torch.randn(2, 4)

    result = OpRewriter(NetMorphLinearOpDeeper())(unroll_sequential(seq, TensorOp(x)))

    assert types(result) == [nn.Linear, nn.ReLU, nn.Linear, nn.ReLU, nn.Linear]
    tt.assert_close(result(), seq(x))
