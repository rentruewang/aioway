# ---
# jupyter:
#   jupytext:
#     formats: py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.19.5
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

import typing

import lightning as L
import torch
from torch import nn, optim
from torch.utils import data as dutils
from torchrl.data import tensor_specs as tspecs

from aioway.dsets import DatasetIdxDset, Dset, InputTarget, MnistDataset
from aioway.emits import ClfLogitHead, linear_regression
from aioway.trainers import StaticTrainer, TrainCfg
from aioway.tspecs import as_tspec


# %%
class MnistFlat(MnistDataset):
    def __getitem__(self, idx) -> InputTarget:
        item = super().__getitem__(idx)
        return InputTarget(input=item.input.flatten(), target=item.target)

    @property
    def input_tspec(self) -> tspecs.Unbounded:
        return tspecs.Unbounded(shape=torch.Size([784]))


# %%
@typing.no_type_check
def train_test_split[D: Dset](dataset: D, test_ratio: float) -> tuple[D, D]:
    assert 0 <= test_ratio <= 1

    test_samples = round(len(dataset) * test_ratio)
    train_samples = len(dataset) - test_samples

    def wrap_dset(dset):
        return DatasetIdxDset(dset, dataset.__tspec__(), dataset.collate_fn)

    train_dset, test_dset = map(
        wrap_dset,
        dutils.random_split(dataset, [train_samples, test_samples]),
    )

    return train_dset, test_dset


# %%
def main(batch_size: int):
    fabric = L.Fabric()

    dset = MnistFlat()
    train_dset, test_dset = train_test_split(dset, 0.1)

    module = ClfLogitHead(linear_regression)(
        as_tspec(dset.input_tspec), as_tspec(dset.target_tspec)
    )
    optimizer = optim.AdamW(module.parameters())
    loss_func = nn.CrossEntropyLoss()

    cfg = TrainCfg(batch_size=batch_size, fabric=fabric)
    trainer = StaticTrainer(cfg, module, optimizer, loss_func)

    for pred in trainer.train_dataset_epoch(train_dset):
        print(pred.loss)
    for pred in trainer.validate_dataset_epoch(test_dset):
        print(pred.loss)


# %%
main(1024)
