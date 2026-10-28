# Copyright (c) AIoWay Authors - All Rights Reserved

"The explainer based on `captum.concepts`."

import torch
from torch import nn
from torch.utils import data as du

from aioway.ml.tasks import Task


class CaptumConceptTask(Task):
    def __call__(self, dset: du.Dataset, module: nn.Module) -> torch.Tensor:
        raise NotImplementedError
