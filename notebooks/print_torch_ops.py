# ---
# jupyter:
#   jupytext:
#     formats: py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.19.1
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %%
import torch

# %%
from aioway.fake import torch_fake_mode
from aioway.fn import PrintTDis, PrintTFunc

# %%
a = torch.randn(3, 4)
b = torch.randn(1, 1)

# %%
with PrintTDis().ctx(), PrintTFunc().ctx():
    c = a + b

# %%
c

# %%
with torch_fake_mode():
    a = torch.randn(3, 4)
    b = torch.randn(1, 1)

# %%
with PrintTDis().ctx(), PrintTFunc().ctx():
    c = a + b

# %%
c
