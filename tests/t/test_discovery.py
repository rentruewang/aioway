# Copyright (c) AIoWay Authors - All Rights Reserved

import pytest
import tensordict as td
import torch

from aioway.t import TDictDomain, TensorItemDomain, batch_domain, iter_domain, sized_of


def test_tensor():
    assert batch_domain(torch.rand(4, 3, 2)) == sized_of(3, 2)
    assert batch_domain(torch.rand(4)) == TensorItemDomain()


def test_tensordict():
    tdict = td.TensorDict(x=torch.rand(4, 3), y=torch.rand(4), batch_size=[4])
    assert batch_domain(tdict) == TDictDomain({"x": sized_of(3), "y": sized_of()})


def test_unknown():
    assert batch_domain("not data") is NotImplemented


def test_stream():
    stream = [torch.rand(n, 3) for n in [2, 5, 7]]
    assert iter_domain(stream) == sized_of(3)

    with pytest.raises(ValueError):
        iter_domain([torch.rand(2, 3), torch.rand(2, 4)])
