# Copyright (c) AIoWay Authors - All Rights Reserved

import pytest
import torch
from torch import nn
from torch import testing as tt

from aioway.ops import TorchFuncOpDag, fake_aten_op_dag, track_module_ops
from aioway.t import fake_mode


@pytest.fixture(params=["func", "aten"])
def tracer(request):
    return TorchFuncOpDag().activate() if request.param == "func" else fake_aten_op_dag()


@pytest.fixture(
    params=[
        lambda x, y: torch.mul(torch.add(x, y), x),
        lambda x, y: torch.sum(torch.stack([x, y]), 0),
        lambda x, y: torch.mul(torch.max(torch.stack([x, y]), dim=0).values, y),
    ]
)
def fn(request):
    return request.param


def test_traced_matches_eager(tracer, fn):
    with fake_mode():
        x, y = torch.zeros(3), torch.zeros(3)

        with tracer as dag:
            fn(x, y)

    x, y = torch.randn(3), torch.randn(3)
    tt.assert_close(dag.exec(x, y), fn(x, y))


def test_module_traced_matches_eager():
    model = nn.Sequential(nn.Linear(4, 8), nn.Sequential(nn.ReLU(), nn.Linear(8, 2)))

    with track_module_ops() as hist:
        model(torch.zeros(3, 4))

    x = torch.randn(5, 4)
    tt.assert_close(hist.exec(x), model(x))
