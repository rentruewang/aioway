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

from aioway._torch import *
from aioway.fate import *
from aioway.modes import *

# %%
with torch_fake_mode():
    a = torch.randn(3, 4)
    b = torch.randn(3, 4)

# %%
with fake_fn() as hists, PrintTorchDisp()():
    a + b

# %%
with fake_fn() as hists, PrintTorchDisp()():
    c = a + b
    d = a + c
    e = a + d
    f = d + b
    g = e + f

hists.dispatch

# %%
with fake_fn(), PrintTorchDisp()():
    3 - a

# %%
with fake_fn(), PrintTorchDisp()():
    a[a > 0]

# %%
with fake_fn(), PrintTorchDisp()():
    torch.stack([a, a, a])

# %%
with fake_fn(), PrintTorchDisp()():
    torch.cat([a, a, a], dim=-1)
