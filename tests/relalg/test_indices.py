# Copyright (c) AIoWay Authors - All Rights Reserved

import pytest
import torch

from aioway._torch import DType
from aioway.io import TensorListExec
from aioway.relalg import AnnIndexExec, AnnIndexTrainerExec, FaissIndex, TensorExec


@pytest.fixture
def training_data():
    return torch.randn(13, 17)


@pytest.fixture
def testing_hop():
    class TestingExec(TensorExec):
        def iterate(self):
            for item in torch.randn(7, 2, 17):
                yield item

    return TestingExec()


@pytest.fixture
def faiss_index_trainer(training_data: torch.Tensor):
    try:
        pass
    except ImportError:
        pytest.xfail("`faiss` not installed")
    else:
        return AnnIndexTrainerExec(
            index=FaissIndex(training_data.shape[-1]),
            data=TensorListExec([training_data]),
        )


@pytest.fixture
def faiss_index(faiss_index_trainer: AnnIndexTrainerExec):
    return _train_index(faiss_index_trainer)


@pytest.fixture(scope="module", params=[1, 3, 5])
def k(request: pytest.FixtureRequest):
    return request.param


@pytest.fixture
def faiss_index_hop(faiss_index: FaissIndex, testing_hop: TensorExec, k: int):
    return AnnIndexExec(faiss_index, testing_hop, k)


def test_index_training(faiss_index_trainer: AnnIndexTrainerExec):
    _train_index(faiss_index_trainer)


def test_index_querying(faiss_index_hop: AnnIndexExec):
    for item in faiss_index_hop:
        assert isinstance(item, torch.Tensor)
        assert DType.parse(item.dtype).family == "int"


def _train_index(index_trainer: AnnIndexTrainerExec):
    assert isinstance(index_trainer, AnnIndexTrainerExec)

    # No samples yet.
    assert not len(index_trainer.index)

    has_trained = False

    for _ in index_trainer:
        has_trained = True

    assert has_trained

    assert len(index_trainer.index)

    return index_trainer.index
