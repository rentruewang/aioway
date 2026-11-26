# Copyright (c) AIoWay Authors - All Rights Reserved

import torch

__all__ = ["safe_div"]


def safe_div(num: torch.Tensor, den: torch.Tensor) -> torch.Tensor:
    "`num / den`, `0` where `den` is `0`, same as `torchmetrics`' `zero_division=0`."

    return torch.where(den > 0, num / den, 0.0)
