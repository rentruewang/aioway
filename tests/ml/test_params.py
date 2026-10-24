# Copyright (c) AIoWay Authors - All Rights Reserved

import torch
from torch import nn, optim

from aioway.ml import index_module_params, index_optim_params


def test_index_optim_params():
    model = nn.Sequential(nn.Linear(2, 2), nn.Linear(2, 1))
    opt = optim.Adam(
        [
            {"params": model[0].parameters(), "lr": 0.1},
            {"params": model[1].parameters(), "lr": 0.2},
        ]
    )
    model(torch.randn(3, 2)).sum().backward()
    opt.step()

    optim_index = index_optim_params(opt)

    assert isinstance(weight_0 := model[0].weight, torch.Tensor)
    assert isinstance(weight_1 := model[1].weight, torch.Tensor)

    param_0 = optim_index[weight_0]
    param_1 = optim_index[weight_1]

    assert len(optim_index) == 4

    assert param_0.group_idx == 0 and param_1.group_idx == 1
    assert param_1.hyper_params["lr"] == 0.2

    assert "params" not in param_1.hyper_params

    assert param_0.state["exp_avg"].shape == weight_0.shape
    param_0.group["lr"] = 0.5
    assert opt.param_groups[0]["lr"] == 0.5


def test_index_module_params():
    model = nn.Sequential(nn.Linear(2, 2), nn.Linear(2, 1))

    module_index = index_module_params(model)
    assert isinstance(bias := model[1].bias, torch.Tensor)
    info = module_index[bias]

    assert len(module_index) == 4
    assert info.path == "1.bias"
    assert info.owner is model[1]
    assert info.attr == "bias"
