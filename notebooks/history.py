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

# %%
import torch
from rich import pretty
from torch import nn

# %%
pretty.install()

# %%
from aioway.cc import track_module_thunks
from aioway.cc.ir.queries import IndexQuery
from aioway.t import fake_mode

# %%
with fake_mode():
    module = nn.Sequential(
        linear1 := nn.Linear(3, 5),
        relu := nn.ReLU(),
        seq2 := nn.Sequential(
            linear3 := nn.Linear(5, 7),
            linear4 := nn.Linear(7, 9),
        ),
    )
    input = torch.randn(100, 3)

# %%
with track_module_thunks() as hist:
    output = module(input)

hist

# %%
list(hist.dag())

# %%
list(IndexQuery([0, 1, 2]).select(hist.dag()))
