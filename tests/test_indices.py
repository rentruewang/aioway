# Copyright (c) AIoWay Authors - All Rights Reserved

import pytest
import torch

from aioway.attrs import DType
from aioway.indices import AnnIndexIter, AnnIndexTrainerIter, FaissIndex
from aioway.io import TensorListIter
from aioway.relalg import TensorIter


@pytest.fixture
def training_data():
    return torch.randn(13, 17)


@pytest.fixture
def testing_hop():
    class TestingIter(TensorIter):
        def iterate(self):
            for item in torch.randn(7, 2, 17):
                yield item

    return TestingIter()


@pytest.fixture
def faiss_index_trainer(training_data: torch.Tensor):
    try:
        pass
    except ImportError:
        pytest.xfail("`faiss` not installed")
    else:
        return AnnIndexTrainerIter(
            index=FaissIndex(training_data.shape[-1]),
            data=TensorListIter([training_data]),
        )


@pytest.fixture
def faiss_index(faiss_index_trainer: AnnIndexTrainerIter):
    return _train_index(faiss_index_trainer)


@pytest.fixture(scope="module", params=[1, 3, 5])
def k(request: pytest.FixtureRequest):
    return request.param


@pytest.fixture
def faiss_index_hop(faiss_index: FaissIndex, testing_hop: TensorIter, k: int):
    return AnnIndexIter(faiss_index, testing_hop, k)


def test_index_training(faiss_index_trainer: AnnIndexTrainerIter):
    _train_index(faiss_index_trainer)


def test_index_querying(faiss_index_hop: AnnIndexIter):
    for item in faiss_index_hop:
        assert isinstance(item, torch.Tensor)
        assert DType.parse(item.dtype).family == "int"


def _train_index(index_trainer: AnnIndexTrainerIter):
    assert isinstance(index_trainer, AnnIndexTrainerIter)

    # No samples yet.
    assert not len(index_trainer.index)

    has_trained = False

    for _ in index_trainer:
        has_trained = True

    assert has_trained

    assert len(index_trainer.index)

    return index_trainer.index
