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
from aioway.t import fake_mode

# %%
with fake_mode():
    module = nn.Sequential(
        nn.Linear(3, 5),
        nn.ReLU(),
        nn.Sequential(
            nn.Linear(5, 7),
            nn.Linear(7, 9),
        ),
    )
    input = torch.randn(100, 3)

# %%
with track_module_thunks() as hist:
    output = module(input)

hist
