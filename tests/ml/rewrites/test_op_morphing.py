# Copyright (c) AIoWay Authors - All Rights Reserved

import pytest
import torch
from torch import nn
from torch import testing as tt

from aioway.ml import NetMorphLinearOpDeeper, OpRewriter, OpTypeChain
from aioway.ops import CallOp, LayerOp, ModuleThunkOp, Op, SameShapeOp, TensorOp
from aioway.ops.thunks import ModuleOp


def x() -> TensorOp:
    return TensorOp(torch.randn(2, 4))


def block(input: Op, *middle: nn.Module) -> Op:
    "Linear, a shape-preserving middle, Linear."

    h = LayerOp(nn.Linear(4, 8), input)

    for module in middle or [nn.ReLU()]:
        h = SameShapeOp(module, h)

    return LayerOp(nn.Linear(8, 4), h)


def add(left: Op, right: Op) -> CallOp:
    return CallOp(func=torch.add, args=(left, right), kwargs={})


def types(op: Op) -> list[type[nn.Module]]:
    return [type(o.nn) for o in op.post_order() if isinstance(o, ModuleOp)]


def deeper(root: Op, *middle: type[nn.Module]) -> Op:
    rule = NetMorphLinearOpDeeper(middle=middle or (nn.ReLU,))
    return OpRewriter(rule)(root)


# Querying ====


def test_query_innermost_first():
    root = block(x())

    chain = OpTypeChain((nn.Linear, nn.ReLU, nn.Linear))(root)

    assert chain[-1] is root
    assert [type(op.module) for op in chain] == [nn.Linear, nn.ReLU, nn.Linear]


def test_query_no_match():
    with pytest.raises(LookupError):
        OpTypeChain((nn.Linear, nn.ReLU, nn.Linear))(block(x(), nn.Tanh()))


def test_query_needs_single_dependency():
    # `join` reads both branches, so no chain runs through it.
    left = LayerOp(nn.Linear(4, 4), x())
    join = ModuleThunkOp(func=nn.Bilinear(4, 4, 4), args=(left, x()), kwargs={})

    assert OpTypeChain((nn.Linear,))(join) == [left]

    with pytest.raises(LookupError):
        OpTypeChain((nn.Linear, nn.Bilinear))(join)


# Rewriting ====


def test_rewrite_deepens():
    result = deeper(block(x()))

    assert types(result) == [nn.Linear, nn.ReLU, nn.Linear, nn.ReLU, nn.Linear]


def test_rewrite_longer_middle():
    result = deeper(block(x(), nn.LayerNorm(8), nn.ReLU()), nn.LayerNorm, nn.ReLU)

    assert types(result) == [
        nn.Linear,
        nn.LayerNorm,
        nn.ReLU,
        nn.Linear,
        nn.LayerNorm,
        nn.ReLU,
        nn.Linear,
    ]


def test_rewrite_keeps_value():
    root = add(block(x()), x())

    tt.assert_close(deeper(root)(), root())


def test_rewrite_thunks():
    h = ModuleThunkOp(func=nn.Linear(4, 8), args=(x(),), kwargs={})
    h = ModuleThunkOp(func=nn.ReLU(), args=(h,), kwargs={})
    root = ModuleThunkOp(func=nn.Linear(8, 4), args=(h,), kwargs={})

    result = deeper(root)

    assert types(result) == [nn.Linear, nn.ReLU, nn.Linear, nn.ReLU, nn.Linear]
    tt.assert_close(result(), root())


def test_rewrite_keeps_original():
    root = block(x())
    before = types(root)

    deeper(root)

    assert types(root) == before


def test_rewrite_keeps_other_branch():
    root = add(block(x()), SameShapeOp(nn.Tanh(), x()))

    result = deeper(root)

    assert isinstance(result, CallOp)
    assert result.args[1] is root.args[1]


def test_rewrite_first_branch_only():
    root = add(block(x()), block(x()))

    result = deeper(root)

    assert isinstance(result, CallOp)
    assert len(types(result.args[0])) == 5
    assert result.args[1] is root.args[1]


def test_rewrite_keeps_sharing():
    shared = block(x())
    root = add(shared, shared)

    result = deeper(root)

    assert isinstance(result, CallOp)
    assert result.args[0] is result.args[1]
    assert len(types(result.args[0])) == 5


def test_rewrite_keeps_side_consumer():
    # The side branch reads the middle, which feeds the second `Linear` too.
    middle = SameShapeOp(nn.ReLU(), LayerOp(nn.Linear(4, 8), x()))
    side = SameShapeOp(nn.Tanh(), middle)
    root = add(LayerOp(nn.Linear(8, 4), middle), LayerOp(nn.Linear(8, 4), side))

    result = deeper(root)

    assert isinstance(result, CallOp)
    assert result.args[1] is root.args[1]
    tt.assert_close(result(), root())
