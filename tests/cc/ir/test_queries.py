# Copyright (c) AIoWay Authors - All Rights Reserved

import typing

import numpy as np
import pytest
import torch

from aioway.cc import IndexQuery, InstrSet, TorchFuncDag
from aioway.t import fake_mode


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
    sub = IndexQuery([0, 1, 2, 3])(iset)

    assert isinstance(sub, InstrSet)
    assert len(sub) == 4
    assert sub.inputs == [x, y]
    assert sub.outputs == [r]


def test_prefix(graph):
    iset, (x, y, s, m, d, r) = graph
    sub = IndexQuery([0, 1])(iset)

    assert len(sub) == 2
    assert sub.inputs == [x, y]
    assert sub.outputs == [m]


def test_suffix(graph):
    iset, (x, y, s, m, d, r) = graph
    sub = IndexQuery([2, 3])(iset)

    assert len(sub) == 2
    assert sub.inputs == [m, s]
    assert sub.outputs == [r]


def test_single_step(graph):
    iset, (x, y, s, m, d, r) = graph
    sub = IndexQuery([3])(iset)

    assert len(sub) == 1
    assert sub.inputs == [d]
    assert sub.outputs == [r]


def test_numpy_idx(graph):
    iset, (x, y, s, m, d, r) = graph
    sub = IndexQuery(np.array([2, 3]))(iset)

    assert sub.inputs == [m, s]
    assert sub.outputs == [r]


def test_no_depending_on_intermediate(graph):
    iset, _ = graph

    # Step 2 needs `m` from step 1, which is skipped but comes after step 0,
    # this means the subgraph is not complte.
    with pytest.raises(ValueError):
        IndexQuery([0, 2])(iset)


def test_no_neg_idx(graph):
    iset, _ = graph

    with pytest.raises(IndexError):
        IndexQuery([-1])(iset)


def test_out_of_bounds(graph):
    iset, _ = graph

    with pytest.raises(IndexError):
        IndexQuery([4])(iset)
