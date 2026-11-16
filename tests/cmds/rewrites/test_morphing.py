# Copyright (c) AIoWay Authors - All Rights Reserved

import pytest
import torch
from torch import nn
from torch import testing as tt

from aioway.cmds import NetMorphLinearOpDeeper, OpRewriter
from aioway.ops import (
    CallOp,
    LayerOp,
    ModuleOp,
    ModuleThunkOp,
    Op,
    SameShapeOp,
    TensorOp,
)


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


# Rule ====


def test_rule_no_match():
    assert NetMorphLinearOpDeeper()(block(x(), nn.Tanh())) is NotImplemented


def test_rule_single_dep():
    # `join` reads both branches, so no chain runs through it.
    first = ModuleThunkOp(func=nn.Linear(4, 4), args=(x(),), kwargs={})
    join = ModuleThunkOp(func=nn.Bilinear(4, 4, 4), args=(first, x()), kwargs={})
    root = ModuleThunkOp(func=nn.Linear(4, 4), args=(join,), kwargs={})

    assert NetMorphLinearOpDeeper(middle=(nn.Bilinear,))(root) is NotImplemented


def test_rewriter_no_match():
    with pytest.raises(LookupError):
        deeper(block(x(), nn.Tanh()))


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
