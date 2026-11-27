# Copyright (c) AIoWay Authors - All Rights Reserved

"Describe the data seen, like `pandas.DataFrame.describe`, but the result is a `Domain`."

import tensordict as td
import torch

__all__ = ["ColStat"]


class ColStat(td.TensorClass):
    "The stats of all values seen, like a column of `pandas.DataFrame.describe`."

    count: torch.Tensor
    "The number of values seen."

    avg: torch.Tensor
    "The mean of the values."

    std_dev: torch.Tensor
    "The sample standard deviation of the values, same as `pandas`."

    low: torch.Tensor
    "The smallest value."

    high: torch.Tensor
    "The largest value."
