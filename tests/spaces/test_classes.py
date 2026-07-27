# Copyright (c) AIoWay Authors - All Rights Reserved

import pytest
import tensordict as td
import torch

from aioway.attrs import AttrDict
from aioway.spaces import TensorClassSpace


class LossTensorClass(td.TensorClass):
    input: torch.Tensor
    target: torch.Tensor


class LossTensorAllPositive(TensorClassSpace):
    KLASS = LossTensorClass

    def _check_attrs(self, attrs: AttrDict):
        assert attrs.keys() == {"input", "target"}

    def _check_data(self, data: LossTensorClass):
        assert isinstance(data, LossTensorClass)

    def _sample_n(self, n: int):
        return LossTensorClass(torch.randn(n), torch.randn(n))


@pytest.fixture
def loss_space():
    return LossTensorAllPositive()


def test_loss_space(loss_space: LossTensorAllPositive):
    inst = LossTensorClass(torch.randn(3, 5), torch.randn(3, 5))
    assert inst in loss_space
