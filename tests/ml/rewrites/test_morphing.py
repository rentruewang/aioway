# Copyright (c) AIoWay Authors - All Rights Reserved

import pytest
import torch
from torch import nn

from aioway.ir import ExactSequential, NnProc, TypeSequential, track_module_thunks
from aioway.ml import NetMorphLinearSeqDeeper
from aioway.t import fake_mode


@pytest.fixture(autouse=True)
def fake():
    "Parameters, inputs and traced outputs are all fakes."

    with fake_mode():
        yield


def trace(module: nn.Module, *shape: int) -> NnProc:
    with track_module_thunks() as hist:
        module(torch.zeros(*shape))

    return hist.procedure


def modules(proc: NnProc) -> list[nn.Module]:
    return [thunk.func for thunk in proc]


def types(proc: NnProc) -> list[type[nn.Module]]:
    return [type(module) for module in modules(proc)]


def by_type(*pattern: type[nn.Module]) -> TypeSequential:
    return TypeSequential(pattern, key=lambda thunk: thunk.func)


def by_module(*pattern: nn.Module) -> ExactSequential:
    return ExactSequential(pattern, key=lambda thunk: thunk.func)


class Block(nn.Module):
    "Linear, a shape-preserving middle, Linear. Not an `nn.Sequential`."

    def __init__(self, *middle: nn.Module) -> None:
        super().__init__()
        self.up = nn.Linear(4, 8)
        self.middle = nn.ModuleList(middle)
        self.down = nn.Linear(8, 4)

    def forward(self, x):
        x = self.up(x)

        for layer in self.middle:
            x = layer(x)

        return self.down(x)


class Net(nn.Module):
    "The `Block` sits in the middle, with a layer on each side."

    def __init__(self, *middle: nn.Module) -> None:
        super().__init__()
        self.stem = nn.Linear(3, 4)
        self.block = Block(*middle)
        self.head = nn.Tanh()

    def forward(self, x):
        return self.head(self.block(self.stem(x)))


class TwoInputs(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.left = nn.Linear(4, 4)
        self.right = nn.Linear(4, 4)
        self.join = nn.Bilinear(4, 4, 4)

    def forward(self, x):
        return self.join(self.left(x), self.right(x))


class Fork(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.linear = nn.Linear(4, 4)
        self.relu = nn.ReLU()
        self.tanh = nn.Tanh()

    def forward(self, x):
        h = self.linear(x)
        return self.relu(h), self.tanh(h)


class SideOutput(nn.Module):
    "`up` also feeds `side`, outside the Linear -> ReLU -> Linear chain."

    def __init__(self) -> None:
        super().__init__()
        self.up = nn.Linear(4, 8)
        self.relu = nn.ReLU()
        self.down = nn.Linear(8, 4)
        self.side = nn.Tanh()

    def forward(self, x):
        h = self.up(x)
        return self.down(self.relu(h)), self.side(h)


# Tracing ====


def test_trace_leaves_only():
    proc = trace(Net(nn.ReLU()), 2, 3)

    assert isinstance(proc, NnProc)
    assert types(proc) == [nn.Linear, nn.Linear, nn.ReLU, nn.Linear, nn.Tanh]


# Querying ====


def test_type_query_whole():
    proc = trace(Block(nn.ReLU()), 2, 4)

    assert by_type(nn.Linear, nn.ReLU, nn.Linear)(proc).tolist() == [0, 1, 2]


def test_type_query_finds_block():
    model = Net(nn.ReLU())
    proc = trace(model, 2, 3)

    sub = proc[by_type(nn.Linear, nn.ReLU, nn.Linear)]

    block = model.block
    assert modules(sub) == [block.up, *block.middle, block.down]
    assert len(sub.inputs) == len(sub.outputs) == 1


def test_exact_query_by_instance():
    model = Net(nn.ReLU())
    proc = trace(model, 2, 3)
    block = model.block

    assert by_module(block.up, *block.middle, block.down)(proc).tolist() == [1, 2, 3]

    # Same type, different instance.
    with pytest.raises(LookupError):
        by_module(nn.Linear(4, 8))(proc)


def test_exact_query_module_called_twice():
    relu = nn.ReLU()
    second = nn.Linear(4, 4)
    proc = trace(nn.Sequential(nn.Linear(4, 4), relu, second, relu), 2, 4)

    assert by_type(nn.Linear, nn.ReLU)(proc).tolist() == [0, 1]
    assert by_module(second, relu)(proc).tolist() == [2, 3]


def test_needs_single_dependency():
    # `join` reads both `left` and `right`.
    proc = trace(TwoInputs(), 2, 4)

    assert by_type(nn.Bilinear)(proc).tolist() == [2]

    with pytest.raises(LookupError):
        by_type(nn.Linear, nn.Bilinear)(proc)


def test_shared_output():
    # `linear` feeds both `relu` and `tanh`. Each still depends on it alone.
    proc = trace(Fork(), 2, 4)

    assert by_type(nn.Linear, nn.ReLU)(proc).tolist() == [0, 1]
    assert by_type(nn.Linear, nn.Tanh)(proc).tolist() == [0, 2]


def test_no_match():
    proc = trace(Block(nn.ReLU()), 2, 4)

    with pytest.raises(LookupError):
        by_type(nn.Linear, nn.Tanh)(proc)


def test_netmorph_rewrite():
    proc = trace(Net(nn.ReLU()), 2, 3)

    result = NetMorphLinearSeqDeeper().map(proc)

    assert types(result) == [
        nn.Linear,
        nn.Linear,
        nn.ReLU,
        nn.Linear,
        nn.ReLU,
        nn.Linear,
        nn.Tanh,
    ]

    inserted = modules(result)[3]
    assert (inserted.in_features, inserted.out_features) == (8, 8)


def test_netmorph_rewrite_longer_middle():
    proc = trace(Net(nn.LayerNorm(8), nn.ReLU()), 2, 3)

    result = NetMorphLinearSeqDeeper(middle=(nn.LayerNorm, nn.ReLU)).map(proc)

    assert types(result) == [
        nn.Linear,
        nn.Linear,
        nn.LayerNorm,
        nn.ReLU,
        nn.Linear,
        nn.LayerNorm,
        nn.ReLU,
        nn.Linear,
        nn.Tanh,
    ]


def test_netmorph_rewrite_keeps_io():
    proc = trace(Net(nn.ReLU()), 2, 3)

    result = NetMorphLinearSeqDeeper().map(proc)

    assert result.inputs.tensors == list(proc.inputs.values())
    assert result.outputs == list(proc.outputs)


def test_netmorph_rewrite_keeps_original():
    proc = trace(Net(nn.ReLU()), 2, 3)
    before = modules(proc)

    NetMorphLinearSeqDeeper().map(proc)

    assert modules(proc) == before
