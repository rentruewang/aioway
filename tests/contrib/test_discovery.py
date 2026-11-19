# Copyright (c) AIoWay Authors - All Rights Reserved

import pytest
import tensordict as td
import torch

from aioway.cmds.nets import IntItemDomain, float_list, int_list
from aioway.contrib.discovery import batch_domain, iter_domain
from aioway.t import ScoreTensorDomain, TDictDomain, sized_of


@pytest.mark.parametrize(
    "tensor, domain",
    [
        (torch.rand(4), ScoreTensorDomain()),
        (torch.rand(4, 3, 2), float_list(3, 2)),
        (torch.arange(4), IntItemDomain()),
        (torch.zeros(4, 3, dtype=torch.long), int_list(3)),
        (torch.zeros(4, 3, dtype=torch.bool), sized_of(3)),
    ],
)
def test_tensor(tensor, domain):
    assert batch_domain(tensor) == domain


def test_tensordict():
    tdict = td.TensorDict(x=torch.rand(4, 3), y=torch.arange(4), batch_size=[4])
    assert batch_domain(tdict) == TDictDomain(
        {"x": float_list(3), "y": IntItemDomain()}
    )


def test_unknown():
    assert batch_domain("not data") is NotImplemented


def test_stream():
    stream = [torch.rand(n, 3) for n in [2, 5, 7]]
    assert iter_domain(stream) == float_list(3)

    with pytest.raises(ValueError):
        iter_domain([torch.rand(2, 3), torch.rand(2, 4)])
