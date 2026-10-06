# Copyright (c) AIoWay Authors - All Rights Reserved

import pytest
import torch

from aioway.ir import FuncCall, Program
from aioway.t import fake_mode


def add(left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
    return left + right


def scale(tensor: torch.Tensor, factor: float) -> torch.Tensor:
    return tensor * factor


@pytest.fixture
def fakes() -> tuple[torch.Tensor, torch.Tensor]:
    with fake_mode():
        return torch.zeros(3), torch.zeros(3)


@pytest.fixture
def thunks(fakes) -> tuple[FuncCall, ...]:
    x0, x1 = fakes

    with fake_mode():
        y = add(x0, x1)
        z = scale(y, factor=2)

    return (
        FuncCall(func=add, args=(x0, x1), kwargs={}, result=y),
        FuncCall(func=scale, args=(y,), kwargs={"factor": 2}, result=z),
    )


@pytest.fixture
def dag(thunks, fakes) -> Program:
    return Program(thunks, inputs=fakes, outputs=[thunks[-1].result])
