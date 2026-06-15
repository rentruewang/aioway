# Copyright (c) AIoWay Authors - All Rights Reserved

import typing

import pytest
from torch import nn

from aioway.algos import SupervisedAlgo
from aioway.attrs import Attr
from aioway.compilers import JustLinearEmitter
from aioway.dsets import Stream, TensorListHop
from aioway.hop import Linear, ListHop, NnHop, NnInit, NnLayerHop, TensorHop
from aioway.sinks import Sink
from aioway.tags import AttrTag, TagDict


@pytest.fixture
def input_space():
    return AttrTag.from_attr(Attr.build(dtype="float", shape=[3, 4, 5]))


@pytest.fixture
def output_space():
    return AttrTag.from_attr(Attr.build(dtype="float", shape=[3, 4, 6]))


@pytest.fixture
def input_dataset(input_space: AttrTag):
    class FakeInputDset(Stream):
        def __call__(self, *_):
            return TensorListHop([input_space.to_attr().to_fake_tensor()])

        @typing.override
        def __iter__(self):
            # Not really using this, so we can afford to make a fake one.
            raise NotImplementedError

        @property
        def tags(self):
            return TagDict({input_space.NAME: input_space})

    return FakeInputDset()


@pytest.fixture
def output_dataset(output_space: AttrTag):
    class FakeOutputDset(Sink):
        @typing.override
        def write(self, item):
            # Not really using this, so we can afford to make a fake one.
            raise NotImplementedError

        @property
        def tags(self):
            return TagDict({output_space.NAME: output_space})

    return FakeOutputDset()


@pytest.fixture
def supervised(input_dataset, output_dataset):
    return SupervisedAlgo(input_dataset, output_dataset)


def test_just_linear(input_dataset: Stream, output_dataset: Sink):
    builder = JustLinearEmitter(input_dataset, output_dataset)
    built = builder()
    [tensor_node, linear_node, list_node] = built.dag()
    assert isinstance(linear_node, NnLayerHop)
    assert isinstance(linear_node.module, nn.Linear)
    assert isinstance(linear_node.nn_init, Linear)
    assert isinstance(tensor_node, TensorHop)
    assert isinstance(list_node, ListHop)
    assert linear_node.nn_init.in_features == 5
    assert linear_node.nn_init.out_features == 6

    assert len([node.is_source for node in built.hops])


def test_just_linear_supervised(supervised: SupervisedAlgo):
    original = supervised.just_linear()
    dag = supervised()
    assert len(dag.nodes()) > len(original.nodes())

    for node in dag.deps():
        assert isinstance(node, NnHop)
        assert isinstance(node.nn_init, NnInit)

        # Right now only `nn.MSELoss`.
        assert node.nn_init.NN == nn.MSELoss
