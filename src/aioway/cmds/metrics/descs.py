# Copyright (c) AIoWay Authors - All Rights Reserved

"Describe the data seen, like `pandas.DataFrame.describe`, but the result is a `Domain`."

import tensordict as td
import torch

from aioway.t import (
    AnyDomain,
    Domain,
    FloatTensorDomain,
    IntTensorDomain,
    InvalidValueForDomain,
)

__all__ = ["ColStat", "ColStatDomain"]


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


class ColStatDomain(Domain[ColStat]):
    "A `ColStat`: an int `count`, and a float item for each other stat."

    @property
    def __extends__(self) -> list[Domain]:
        return [AnyDomain()]

    def check_type(self, obj: ColStat, /) -> None:
        if not isinstance(obj, ColStat):
            raise InvalidValueForDomain(f"Expected a ColStat, got {type(obj)=}.")

        IntTensorDomain().check_type(obj.count)

        for stat in [obj.avg, obj.std_dev, obj.low, obj.high]:
            FloatTensorDomain().check_type(stat)
