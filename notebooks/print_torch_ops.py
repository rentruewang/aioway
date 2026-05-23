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

# %%
from aioway._torch import torch_fake_mode
from aioway.modes import PrintTorchDisp, PrintTorchFunc

# %%
a = torch.randn(3, 4)
b = torch.randn(1, 1)

# %%
with PrintTorchDisp().enter(), PrintTorchFunc().enter():
    c = a + b

# %%
c

# %%
with torch_fake_mode():
    a = torch.randn(3, 4)
    b = torch.randn(1, 1)

# %%
with PrintTorchDisp().enter(), PrintTorchFunc().enter():
    c = a + b

# %%
c
