# Copyright (c) AIoWay Authors - All Rights Reserved

import numpy as np
import pytest
import torch

from aioway.cc import IndexQuery, InstrSet, TorchFuncDag
from aioway.t import fake_mode, parse_attr


@pytest.fixture
def graph():
    """
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
    return iset, (x, y, s, m, d, r)


def test_select_all(graph) -> None:
    # This is running in real mode.
    iset, (x, y, *_, r) = graph
    sub = IndexQuery([0, 1, 2, 3])(iset)

    assert isinstance(sub, InstrSet)
    assert len(sub) == 4
    assert sub.inputs == [x, y]
    assert sub.outputs[0] is r


def test_prefix(graph):
    iset, (x, y, s, m, d, r) = graph
    sub = IndexQuery([0, 1])(iset)

    assert len(sub) == 2
    assert sub.inputs[0] is x
    assert sub.inputs[1] is y
    assert len(sub.outputs) == 1
    assert parse_attr(sub.outputs[0]) == parse_attr(m)


def test_suffix(graph):
    iset, (x, y, s, m, d, r) = graph
    sub = IndexQuery([2, 3])(iset)

    assert len(sub) == 2
    assert sub.inputs[0] is m
    assert sub.inputs[1] is s
    assert sub.outputs[0] is r


def test_single_step(graph):
    iset, (x, y, s, m, d, r) = graph
    sub = IndexQuery([3])(iset)

    assert len(sub) == 1
    assert sub.inputs[0] is d
    assert sub.outputs[0] is r


def test_numpy_indices(graph):
    iset, _ = graph
    sub = IndexQuery(np.array([2, 3]))(iset)

    assert len(sub) == 2


def test_gap_depending_on_intermediate(graph):
    iset, _ = graph

    # Step 2 needs `m` from step 1, which is skipped but comes after step 0.
    with pytest.raises(IndexError):
        IndexQuery([0, 2])(iset)


def test_negative_index(graph):
    iset, _ = graph

    with pytest.raises(IndexError):
        IndexQuery([-1])(iset)


def test_out_of_bounds(graph):
    iset, _ = graph

    with pytest.raises(IndexError):
        IndexQuery([4])(iset)
