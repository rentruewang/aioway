# Copyright (c) AIoWay Authors - All Rights Reserved

import typing

import numpy as np
import pytest
import torch
from torch import testing as tt

from aioway.cc import Exec, IndexQuery, InstrSet, TorchFuncDag
from aioway.t import fake_mode, parse_attr


class GraphInter(typing.NamedTuple):
    x: torch.Tensor
    y: torch.Tensor
    s: torch.Tensor
    m: torch.Tensor
    d: torch.Tensor
    r: torch.Tensor


@pytest.fixture
def graph():
    """
    Trace down the following operation:


    step 0: s = add(x, y)
    step 1: m = mul(s, x)
    step 2: d = sub(m, s)
    step 3: r = relu(d)
    """

    tracer = TorchFuncDag()

    with fake_mode():
        x, y = torch.zeros(3), torch.zeros(3)

        with tracer.activate():
            s = torch.add(x, y)
            m = torch.mul(s, x)
            d = torch.sub(m, s)
            r = torch.relu(d)

    iset = InstrSet.from_thunk_list(tracer.thunks)
    return iset, GraphInter(x, y, s, m, d, r)


def test_all(graph):
    iset, (x, y, s, m, d, r) = graph
    sub = IndexQuery([0, 1, 2, 3]).select(iset)

    assert isinstance(sub, InstrSet)
    assert len(sub) == 4
    assert sub.inputs == {x, y}
    assert sub.outputs == [r]


def test_prefix(graph):
    iset, (x, y, s, m, d, r) = graph
    sub = IndexQuery([0, 1]).select(iset)

    assert len(sub) == 2
    assert sub.inputs == {x, y}
    assert sub.outputs == [m]


def test_suffix(graph):
    iset, (x, y, s, m, d, r) = graph
    sub = IndexQuery([2, 3]).select(iset)

    assert len(sub) == 2
    assert sub.inputs == {m, s}
    assert sub.outputs == [r]


def test_single_step(graph):
    iset, (x, y, s, m, d, r) = graph
    sub = IndexQuery([3]).select(iset)

    assert len(sub) == 1
    assert sub.inputs == [d]
    assert sub.outputs == [r]


def test_numpy_idx(graph):
    iset, (x, y, s, m, d, r) = graph
    sub = IndexQuery(np.array([2, 3])).select(iset)

    assert sub.inputs == {m, s}
    assert sub.outputs == [r]


def test_no_depending_on_intermediate(graph):
    iset, _ = graph

    # Step 2 needs `m` from step 1, which is skipped but comes after step 0,
    # this means the subgraph is not complte.
    with pytest.raises(ValueError):
        IndexQuery([0, 2]).select(iset)


def test_no_neg_idx(graph):
    iset, _ = graph

    with pytest.raises(IndexError):
        IndexQuery([-1]).select(iset)


def test_out_of_bounds(graph):
    iset, _ = graph

    with pytest.raises(IndexError):
        IndexQuery([4]).select(iset)


def trace(fn, *shapes: tuple[int, ...]) -> InstrSet:
    "Trace `fn` on fresh fakes of `shapes` into its own instruction set."

    tracer = TorchFuncDag()

    with fake_mode():
        fakes = [torch.zeros(shape) for shape in shapes]

        with tracer.activate():
            fn(*fakes)

    return InstrSet.from_thunk_list(tracer.thunks)


def funcs(iset: InstrSet) -> list:
    return [thunk.func for thunk in iset]


def test_rewrite_with_itself(graph):
    iset, _ = graph
    query = IndexQuery([1, 2])

    result = query.rewrite(iset, query.select(iset))

    assert funcs(result) == funcs(iset)


def test_rewrite_last(graph):
    iset, _ = graph

    # Step 3 takes one tensor (d) and returns one.
    replacement = trace(lambda d: torch.abs(d), (3,))
    result = IndexQuery([3]).rewrite(iset, replacement)

    assert funcs(result) == [torch.add, torch.mul, torch.sub, torch.abs]


def test_rewrite_last_runs(graph):
    iset, _ = graph

    replacement = trace(lambda d: torch.abs(d), (3,))
    result = IndexQuery([3]).rewrite(iset, replacement)

    a, b = torch.randn(3), torch.randn(3)
    s = a + b

    tt.assert_close(Exec(result)(a, b), torch.abs(s * a - s))


def test_rewrite_middle(graph):
    iset, _ = graph

    # Steps 1-2 take (s, x) and return d. Replace mul/sub with add/sub.
    replacement = trace(lambda s, x: torch.sub(torch.add(s, x), s), (3,), (3,))
    result = IndexQuery([1, 2]).rewrite(iset, replacement)

    assert funcs(result) == [torch.add, torch.add, torch.sub, torch.relu]


def test_rewrite_middle_runs(graph):
    iset, _ = graph

    replacement = trace(lambda s, x: torch.sub(torch.add(s, x), s), (3,), (3,))
    result = IndexQuery([1, 2]).rewrite(iset, replacement)

    a, b = torch.randn(3), torch.randn(3)

    # d becomes (s + x) - s == x, so the program computes relu(x).
    tt.assert_close(Exec(result)(a, b), torch.relu(a))


def test_rewrite_keeps_io_attrs(graph):
    iset, _ = graph

    replacement = trace(lambda d: torch.abs(d), (3,))
    result = IndexQuery([3]).rewrite(iset, replacement)

    assert len(result.inputs) == len(iset.inputs)
    assert len(result.outputs) == len(iset.outputs)
    assert all(
        parse_attr(got) == parse_attr(want)
        for got, want in zip(result.inputs, iset.inputs)
    )


def test_rewrite_keeps_original(graph):
    iset, _ = graph
    before = funcs(iset)

    IndexQuery([3]).rewrite(iset, trace(lambda d: torch.abs(d), (3,)))

    assert funcs(iset) == before


def test_rewrite_rejects_wrong_shape(graph):
    iset, _ = graph

    # Same arity, but the fake is (4,) where the subnet has (3,).
    replacement = trace(lambda d: torch.abs(d), (4,))

    with pytest.raises(ValueError):
        IndexQuery([3]).rewrite(iset, replacement)


def test_rewrite_rejects_wrong_arity(graph):
    iset, _ = graph

    # Two inputs where the subnet has one.
    replacement = trace(lambda a, b: torch.add(a, b), (3,), (3,))

    with pytest.raises(ValueError):
        IndexQuery([3]).rewrite(iset, replacement)
