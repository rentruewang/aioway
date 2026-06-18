# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.19.3
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %%
import torch
from torch import nn

# %%
from aioway.modes import NnFwdThunk, NnInitThunk, PrintNnFwd, PrintNnInit, mode_off

# %%
with PrintNnInit()():
    NnInitThunk(nn.Linear, 3, 5)()
    NnInitThunk(nn.Dropout)()

# %%
t = torch.randn(7, 3)

with PrintNnInit()(), PrintNnFwd()():
    linear = NnInitThunk(nn.Linear, 3, 5)()
    dropout = NnInitThunk(nn.Dropout)()

    print()
    print("fwd")
    print()

    t = NnFwdThunk(linear, t)()
    t = NnFwdThunk(dropout, t)()

# %%
t = torch.randn(7, 3)


with PrintNnInit()(), mode_off(), PrintNnFwd()():
    linear = NnInitThunk(nn.Linear, 3, 5)()
    dropout = NnInitThunk(nn.Dropout)()

    print("fwd, this should be the first statement in the cell's output")
    print()

    t = NnFwdThunk(linear, t)()
    t = NnFwdThunk(dropout, t)()

# %% [markdown]
# Note that outside contexts of `mode_off` (no init calls in second cell) is disabled. This is consistent with how `torch`'s dispatch mode and function mode works.

# %%
t = torch.randn(7, 3)


with PrintNnInit()(), PrintNnFwd()(), mode_off():
    linear = NnInitThunk(nn.Linear, 3, 5)()
    dropout = NnInitThunk(nn.Dropout)()

    print("This should be the ONLY statement in the cell's output")
    print()

    t = NnFwdThunk(linear, t)()
    t = NnFwdThunk(dropout, t)()

# %% [markdown]
# Perfect. This means we copied torch modes' mechanism beautifully.

# %%
