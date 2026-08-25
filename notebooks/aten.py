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

from aioway._utils import *
from aioway.modes import *
from aioway.schemas import *

# %%
dispatch_print = PrintTorchDisp()
function_print = PrintTorchFunc()

# %%
with fake_mode():
    a = torch.randn(3, 4)
    b = torch.randn(3, 4)

# %%
with fake_fn() as hists, dispatch_print.activate():
    a + b

# %%
with fake_fn() as hists, dispatch_print.activate():
    c = a + b
    d = a + c
    e = a + d
    f = d + b
    g = e + f

hists.dispatch

# %%
with fake_fn(), dispatch_print.activate():
    3 - a

# %%
with fake_fn(), dispatch_print.activate():
    a[a > 0]

# %%
with fake_fn(), dispatch_print.activate():
    torch.stack([a, a, a])

# %%
with fake_fn(), dispatch_print.activate():
    torch.cat([a, a, a], dim=-1)

# %%
with fake_mode():
    a = torch.randn(5).requires_grad_()
    b = torch.randn(5).requires_grad_()

# %%
s = (a + 2 * b).sum()

# %%
with dispatch_print.activate(), function_print.activate():
    s.backward()
