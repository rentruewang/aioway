# Copyright (c) AIoWay Authors - All Rights Reserved

"The explainer based on `captum`."

import typing
from captum import attr, concept, influence
from torch.utils import data as dutils
from aioway.io import Dset
from torch import nn
from aioway.ml.tasks import Task


class CaptumAttrTask(Task):
    def __call__(self, dataset: Dset, module: nn.Module) -> typing.Any:
        raise NotImplementedError


class CaptumConceptTask(Task):
    def __call__(self, dataset: Dset, module: nn.Module) -> typing.Any:
        raise NotImplementedError


class CaptumInfluenceTask(Task):
    def __call__(self, dataset: Dset, module: nn.Module) -> typing.Any:
        raise NotImplementedError
