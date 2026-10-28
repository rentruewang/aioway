# Copyright (c) AIoWay Authors - All Rights Reserved

"The explainer based on `captum.concepts`."

import typing

from torch import nn

from aioway.io import Dset
from aioway.ml.tasks import Task


class CaptumConceptTask(Task):
    def __call__(self, dataset: Dset, module: nn.Module) -> typing.Any:
        raise NotImplementedError
