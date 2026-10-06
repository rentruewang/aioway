# Copyright (c) AIoWay Authors - All Rights Reserved

import typing

import numpy as np
import pytest
import torch
from torch import testing as tt

from aioway.ir import Exec, IndexQuery, Program, TorchFuncDag
from aioway.t import fake_mode, parse_attr


class Graph(typing.NamedTuple):
    prog: Program
    x: torch.Tensor
    y: torch.Tensor
    summed: torch.Tensor
    product: torch.Tensor
    difference: torch.Tensor
    activated: torch.Tensor


@pytest.fixture
def graph() -> Graph:
    """
    Trace down the following operation:


    step 0: summed = add(x, y)
    step 1: product = mul(summed, x)
    step 2: difference = sub(product, summed)
    step 3: activated = relu(difference)
    """

    tracer = TorchFuncDag()

    with fake_mode():
        x, y = torch.zeros(3), torch.zeros(3)

        with tracer.activate():
            summed = torch.add(x, y)
            product = torch.mul(summed, x)
            difference = torch.sub(product, summed)
            activated = torch.relu(difference)

    iset = Program.from_thunk_list(tracer.thunks)
    return Graph(iset, x, y, summed, product, difference, activated)


def test_all(graph: Graph):
    sub = IndexQuery([0, 1, 2, 3]).select(graph.prog)

    assert isinstance(sub, Program)
    assert len(sub) == 4
    assert sub.inputs == {graph.x, graph.y}
    assert sub.outputs == [graph.activated]


def test_prefix(graph: Graph):
    sub = IndexQuery([0, 1]).select(graph.prog)

    assert len(sub) == 2
    assert sub.inputs == {graph.x, graph.y}
    assert sub.outputs == {graph.summed, graph.product}


def test_suffix(graph: Graph):
    sub = IndexQuery([2, 3]).select(graph.prog)

    assert len(sub) == 2
    assert sub.inputs == {graph.product, graph.summed}
    assert sub.outputs == [graph.activated]


def test_single_step(graph: Graph):
    sub = IndexQuery([3]).select(graph.prog)

    assert len(sub) == 1
    assert sub.inputs == [graph.difference]
    assert sub.outputs == [graph.activated]


def test_numpy_idx(graph: Graph):
    sub = IndexQuery(np.array([2, 3])).select(graph.prog)

    assert sub.inputs == {graph.product, graph.summed}
    assert sub.outputs == [graph.activated]


def test_no_depending_on_intermediate(graph: Graph):
    # Step 2 needs `product` from step 1, which is skipped but comes after step 0,
    # this means the subgraph is not complte.
    with pytest.raises(ValueError):
        IndexQuery([0, 2]).select(graph.prog)


def test_no_neg_idx(graph: Graph):
    with pytest.raises(IndexError):
        IndexQuery([-1]).select(graph.prog)


def test_out_of_bounds(graph: Graph):
    with pytest.raises(IndexError):
        IndexQuery([4]).select(graph.prog)


def trace(fn, *shapes: tuple[int, ...]) -> Program:
    "Trace `fn` on fresh fakes of `shapes` into its own instruction set."

    tracer = TorchFuncDag()

    with fake_mode():
        fakes = [torch.zeros(shape) for shape in shapes]

        with tracer.activate():
            fn(*fakes)

    return Program.from_thunk_list(tracer.thunks)


def funcs(iset: Program) -> list:
    return [thunk.func for thunk in iset]


def test_rewrite_with_itself(graph: Graph):
    query = IndexQuery([1, 2])

    result = query.rewrite(graph.prog, query.select(graph.prog))

    assert funcs(result) == funcs(graph.prog)


def test_rewrite_with_itself_setitem(graph: Graph):
    query = IndexQuery([1, 2])
    before = funcs(graph.prog)
    graph.prog[query] = query.select(graph.prog)
    assert before == funcs(graph.prog)
    assert before is not funcs(graph.prog)


def test_rewrite_last(graph: Graph):
    # Step 3 takes one tensor (difference) and returns one.
    replacement = trace(lambda difference: torch.abs(difference), (3,))
    result = IndexQuery([3]).rewrite(graph.prog, replacement)

    assert funcs(result) == [torch.add, torch.mul, torch.sub, torch.abs]


def test_rewrite_last_runs(graph: Graph):
    replacement = trace(lambda difference: torch.abs(difference), (3,))
    result = IndexQuery([3]).rewrite(graph.prog, replacement)

    x_real, y_real = torch.randn(3), torch.randn(3)
    summed = x_real + y_real

    tt.assert_close(Exec(result)(x_real, y_real), torch.abs(summed * x_real - summed))


def test_rewrite_middle(graph: Graph):
    # Steps 1-2 take (summed, x) and return difference. Replace mul/sub with add/sub.
    replacement = trace(
        lambda summed, x: torch.sub(torch.add(summed, x), summed), (3,), (3,)
    )
    result = IndexQuery([1, 2]).rewrite(graph.prog, replacement)

    assert funcs(result) == [torch.add, torch.add, torch.sub, torch.relu]


def test_rewrite_middle_runs(graph: Graph):
    replacement = trace(
        lambda summed, x: torch.sub(torch.add(summed, x), summed), (3,), (3,)
    )
    result = IndexQuery([1, 2]).rewrite(graph.prog, replacement)

    x_real, y_real = torch.randn(3), torch.randn(3)

    # difference becomes (summed + x) - summed == x, so the program computes relu(x).
    tt.assert_close(Exec(result)(x_real, y_real), torch.relu(x_real))


def test_rewrite_keeps_io_attrs(graph: Graph):
    replacement = trace(lambda difference: torch.abs(difference), (3,))
    result = IndexQuery([3]).rewrite(graph.prog, replacement)

    assert len(result.inputs) == len(graph.prog.inputs)
    assert len(result.outputs) == len(graph.prog.outputs)
    assert all(
        parse_attr(got) == parse_attr(want)
        for got, want in zip(result.inputs, graph.prog.inputs)
    )


def test_rewrite_keeps_original(graph: Graph):
    before = funcs(graph.prog)

    IndexQuery([3]).rewrite(
        graph.prog, trace(lambda difference: torch.abs(difference), (3,))
    )

    assert funcs(graph.prog) == before


def test_rewrite_rejects_wrong_shape(graph: Graph):
    # Same arity, but the fake is (4,) where the subnet has (3,).
    replacement = trace(lambda difference: torch.abs(difference), (4,))

    with pytest.raises(ValueError):
        IndexQuery([3]).rewrite(graph.prog, replacement)


def test_rewrite_rejects_wrong_arity(graph: Graph):
    # Two inputs where the subnet has one.
    replacement = trace(lambda lhs, rhs: torch.add(lhs, rhs), (3,), (3,))

    with pytest.raises(ValueError):
        IndexQuery([3]).rewrite(graph.prog, replacement)
