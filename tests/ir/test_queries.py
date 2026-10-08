# Copyright (c) AIoWay Authors - All Rights Reserved

import typing
from collections import abc as cabc

import numpy as np
import pytest
import torch
from torch import testing as tt

from aioway.ir import (
    ExactSequential,
    Exec,
    FuncCall,
    IndexQuery,
    Program,
    Query,
    TorchFuncDag,
    TypeSequential,
)
from aioway.t import fake_mode, parse_attr

# Shared utilities ====


def cow_rewrite(query: Query, prog: Program, sub: Program) -> Program:
    "A copy of `prog` with the part picked out by `query` replaced by `sub`."

    prog = prog.copy()
    prog[query] = sub
    return prog


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

    iset = Program.from_instr_list(tracer.thunks)
    return Graph(iset, x, y, summed, product, difference, activated)


# Test query in general ====


def test_all(graph: Graph):
    sub = graph.prog[IndexQuery([0, 1, 2, 3])]

    assert isinstance(sub, Program)
    assert len(sub) == 4
    assert sub.inputs == {graph.x, graph.y}
    assert sub.outputs == [graph.activated]


def test_prefix(graph: Graph):
    sub = graph.prog[IndexQuery([0, 1])]

    assert len(sub) == 2
    assert sub.inputs == {graph.x, graph.y}
    assert sub.outputs == {graph.summed, graph.product}


def test_suffix(graph: Graph):
    sub = graph.prog[IndexQuery([2, 3])]

    assert len(sub) == 2
    assert sub.inputs == {graph.product, graph.summed}
    assert sub.outputs == [graph.activated]


def test_single_step(graph: Graph):
    sub = graph.prog[IndexQuery([3])]

    assert len(sub) == 1
    assert sub.inputs == [graph.difference]
    assert sub.outputs == [graph.activated]


def test_numpy_idx(graph: Graph):
    sub = graph.prog[IndexQuery(np.array([2, 3]))]

    assert sub.inputs == {graph.product, graph.summed}
    assert sub.outputs == [graph.activated]


def test_no_depending_on_intermediate(graph: Graph):
    # Step 2 needs `product` from step 1, which is skipped but comes after step 0,
    # this means the subgraph is not complte.
    with pytest.raises(ValueError):
        graph.prog[IndexQuery([0, 2])]


def test_no_neg_idx(graph: Graph):
    with pytest.raises(IndexError):
        graph.prog[IndexQuery([-1])]


def test_out_of_bounds(graph: Graph):
    with pytest.raises(IndexError):
        graph.prog[IndexQuery([4])]


def trace(fn, *shapes: tuple[int, ...]) -> Program:
    "Trace `fn` on fresh fakes of `shapes` into its own instruction set."

    tracer = TorchFuncDag()

    with fake_mode():
        fakes = [torch.zeros(shape) for shape in shapes]

        with tracer.activate():
            fn(*fakes)

    return Program.from_instr_list(tracer.thunks)


def funcs(iset: Program) -> list:
    return [thunk.func for thunk in iset]


def test_rewrite_with_itself(graph: Graph):
    query = IndexQuery([1, 2])

    result = cow_rewrite(query, graph.prog, graph.prog[query])

    assert funcs(result) == funcs(graph.prog)


def test_rewrite_with_itself_setitem(graph: Graph):
    query = IndexQuery([1, 2])
    before = funcs(graph.prog)
    graph.prog[query] = graph.prog[query]
    assert before == funcs(graph.prog)
    assert before is not funcs(graph.prog)


def test_rewrite_last(graph: Graph):
    # Step 3 takes one tensor (difference) and returns one.
    replacement = trace(lambda difference: torch.abs(difference), (3,))
    result = cow_rewrite(IndexQuery([3]), graph.prog, replacement)

    assert funcs(result) == [torch.add, torch.mul, torch.sub, torch.abs]


def test_rewrite_last_runs(graph: Graph):
    replacement = trace(lambda difference: torch.abs(difference), (3,))
    result = cow_rewrite(IndexQuery([3]), graph.prog, replacement)

    x_real, y_real = torch.randn(3), torch.randn(3)
    summed = x_real + y_real

    tt.assert_close(Exec(result)(x_real, y_real), torch.abs(summed * x_real - summed))


def test_rewrite_middle(graph: Graph):
    # Steps 1-2 take (summed, x) and return difference. Replace mul/sub with add/sub.
    replacement = trace(
        lambda summed, x: torch.sub(torch.add(summed, x), summed), (3,), (3,)
    )
    result = cow_rewrite(IndexQuery([1, 2]), graph.prog, replacement)

    assert funcs(result) == [torch.add, torch.add, torch.sub, torch.relu]


def test_rewrite_middle_runs(graph: Graph):
    replacement = trace(
        lambda summed, x: torch.sub(torch.add(summed, x), summed), (3,), (3,)
    )
    result = cow_rewrite(IndexQuery([1, 2]), graph.prog, replacement)

    x_real, y_real = torch.randn(3), torch.randn(3)

    # difference becomes (summed + x) - summed == x, so the program computes relu(x).
    tt.assert_close(Exec(result)(x_real, y_real), torch.relu(x_real))


def test_rewrite_keeps_io_attrs(graph: Graph):
    replacement = trace(lambda difference: torch.abs(difference), (3,))
    result = cow_rewrite(IndexQuery([3]), graph.prog, replacement)

    assert len(result.inputs) == len(graph.prog.inputs)
    assert len(result.outputs) == len(graph.prog.outputs)
    assert all(
        parse_attr(got) == parse_attr(want)
        for got, want in zip(result.inputs, graph.prog.inputs)
    )


def test_rewrite_keeps_original(graph: Graph):
    before = funcs(graph.prog)

    cow_rewrite(
        IndexQuery([3]),
        graph.prog,
        trace(lambda difference: torch.abs(difference), (3,)),
    )

    assert funcs(graph.prog) == before


def test_rewrite_rejects_wrong_shape(graph: Graph):
    # Same arity, but the fake is (4,) where the subnet has (3,).
    replacement = trace(lambda difference: torch.abs(difference), (4,))

    with pytest.raises(ValueError):
        cow_rewrite(IndexQuery([3]), graph.prog, replacement)


def test_rewrite_rejects_wrong_arity(graph: Graph):
    # Two inputs where the subnet has one.
    replacement = trace(lambda lhs, rhs: torch.add(lhs, rhs), (3,), (3,))

    with pytest.raises(ValueError):
        cow_rewrite(IndexQuery([3]), graph.prog, replacement)


# `aioway.ir.seqs` queries ====


def seq_exact(*pattern) -> ExactSequential:
    def func(thunk: FuncCall, /):
        return thunk.func

    return ExactSequential[FuncCall](pattern, key=func)


def test_seqs_finds_chain(graph: Graph):
    # sub -> relu is a chain.
    sub = graph.prog[seq_exact(torch.sub, torch.relu)]

    assert funcs(sub) == [torch.sub, torch.relu]

    # Can compare this to a list, since the input is ordered.
    assert sub.inputs == [graph.product, graph.summed]
    assert sub.outputs == [graph.activated]


def test_seqs_single_step(graph: Graph):
    indices = seq_exact(torch.mul)(graph.prog)
    assert indices.tolist() == [1]


def test_seqs_requires_chain(graph: Graph):
    # sub reads both `product` and `summed`, so mul -> sub is not a chain.
    with pytest.raises(LookupError):
        graph.prog[seq_exact(torch.mul, torch.sub)]


def test_seqs_whole_program_fail(graph: Graph):
    with pytest.raises(LookupError):
        graph.prog[seq_exact(torch.add, torch.mul, torch.sub, torch.relu)]


def test_seqs_is_ordered(graph: Graph):
    with pytest.raises(LookupError):
        graph.prog[seq_exact(torch.relu, torch.sub)]


def test_seqs_longer_than_program(graph: Graph):
    with pytest.raises(LookupError):
        graph.prog[seq_exact(torch.sub, torch.relu, torch.relu, torch.relu, torch.relu)]


def test_seqs_by_type(graph: Graph):
    # only sub -> relu is a chain.
    query = TypeSequential((cabc.Callable, cabc.Callable), key=lambda thunk: thunk.func)

    assert query(graph.prog).tolist() == [2, 3]


def test_seqs_by_type_no_match(graph: Graph):
    query = TypeSequential((int,), key=lambda thunk: thunk.func)

    with pytest.raises(LookupError):
        query(graph.prog)


def test_seqs_shared_output():
    # `e` feeds both relu and tanh. Each still depends on exp alone.
    def fn(x):
        e = torch.exp(x)
        torch.relu(e)
        torch.tanh(e)

    prog = trace(fn, (3,))

    assert seq_exact(torch.exp, torch.relu)(prog).tolist() == [0, 1]
    assert seq_exact(torch.exp, torch.tanh)(prog).tolist() == [0, 2]

    # relu still reads `e`, so `e` is an extra output of the exp -> tanh part.
    assert len(prog[seq_exact(torch.exp, torch.tanh)].outputs) == 2


def _build_new_program(graph):
    # sub -> relu takes (product, summed) and returns activated.
    replacement = trace(
        lambda product, summed: torch.abs(torch.sub(product, summed)), (3,), (3,)
    )
    return cow_rewrite(seq_exact(torch.sub, torch.relu), graph.prog, replacement)


def test_seqs_rewrite(graph: Graph):
    result = _build_new_program(graph)

    assert funcs(result) == [torch.add, torch.mul, torch.sub, torch.abs]
    assert seq_exact(torch.sub, torch.abs)(result).tolist() == [2, 3]

    with pytest.raises(LookupError):
        seq_exact(torch.sub, torch.relu)(result)


def test_seqs_rewrite_runs(graph: Graph):
    result = _build_new_program(graph)

    x_real, y_real = torch.randn(3), torch.randn(3)
    summed = x_real + y_real

    tt.assert_close(Exec(result)(x_real, y_real), torch.abs(summed * x_real - summed))
