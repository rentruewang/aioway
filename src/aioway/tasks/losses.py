# Copyright (c) AIoWay Authors - All Rights Reserved

import torch
import enum
from aioway.tensors import TSpec
import dataclasses as dcls, typing
from torchrl.data import tensor_specs as tspecs


class TaskType(enum.StrEnum):
    REGRESSION = "regression"
    "Where the output is a floating vector."

    CLASSIFICATION = "classification"
    "Where an output can match 1 single class."

    MULTI_CLASS = "multi-class"
    "Where an output can match multiple classes."

    def target_to_input_tspec(self, target: TSpec, /) -> TSpec:
        match self:
            case self.REGRESSION:
                func = self._reg_input

            case self.CLASSIFICATION:
                func = self._clf_input

            case self.MULTI_CLASS:
                func = self._multi_clf_input

        return func(target)

    def _reg_input(self, target: TSpec) -> TSpec:
        return target

    @typing.no_type_check
    def _clf_input(self, target: TSpec) -> TSpec:
        shape = target.shape
        assert dcls.is_dataclass(target)
        return dcls.replace(target, shape=torch.Size(*shape, -1))

    @typing.no_type_check
    def _multi_clf_input(self, target: TSpec) -> TSpec:
        shape = target.shape
        assert dcls.is_dataclass(target)
        return tspecs.Unbounded(shape=torch.Size(*shape, -1))
