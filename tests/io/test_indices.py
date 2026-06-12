# Copyright (c) AIoWay Authors - All Rights Reserved

import pytest
import torch
from torch.utils import data

from aioway.hop import TensorHop
from aioway.io import FaissIndex, FaissIndexHop, TensorLoaderHop


@pytest.fixture
def training_data():
    return torch.randn(13, 17)


@pytest.fixture
def testing_data():
    return torch.randn(7, 17)


@pytest.fixture
def faiss_index(training_data: torch.Tensor):
    try:
        pass
    except ImportError:
        pytest.xfail("`faiss` not installed")
    else:
        return FaissIndex.from_tensors(training_data)


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
def index_hop(
    query_hop: TensorHop, k: int, faiss_index: FaissIndex, training_data: torch.Tensor
):
    return FaissIndexHop(index=faiss_index, source=training_data, query=query_hop, k=k)


def test_index_hop(index_hop: TensorHop, k: int):
    for found in index_hop:
        assert isinstance(found, torch.Tensor)
        assert found.ndim == 3
        assert found.shape[1] == k
