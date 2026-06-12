# Copyright (c) AIoWay Authors - All Rights Reserved

import pytest
import torch
from torch.utils import data

from aioway.hop import TensorHop
from aioway.io import HnswIndex, HnswIndexHop, TensorLoaderHop


@pytest.fixture
def training_data():
    return torch.randn(13, 17)


@pytest.fixture
def testing_data():
    return torch.randn(7, 17)


@pytest.fixture
def hnswlib_index(training_data: torch.Tensor):
    try:
        import hnswlib
    except ImportError:
        pytest.xfail("`hnswlib` not installed")
    else:
        return HnswIndex.from_tensors(training_data)


@pytest.fixture
def query_hop(testing_data: torch.Tensor):
    class SingleTensorDataset(data.Dataset[torch.Tensor]):
        def __len__(self):
            return len(testing_data)

        def __getitem__(self, index):
            return testing_data[index]

    return TensorLoaderHop(dset=SingleTensorDataset())


@pytest.fixture(params=[1, 3, 5])
def k(request: pytest.FixtureRequest):
    return request.param


@pytest.fixture
def hnswlib_index_hop(
    query_hop: TensorHop, k: int, hnswlib_index: HnswIndex, training_data: torch.Tensor
):
    return HnswIndexHop(index=hnswlib_index, source=training_data, query=query_hop, k=k)


def test_index_hop(hnswlib_index_hop: TensorHop, k: int):
    for found in hnswlib_index_hop:
        assert isinstance(found, torch.Tensor)
        assert found.ndim == 2
        assert len(found) == k
