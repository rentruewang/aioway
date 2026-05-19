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

from aioway._logging import *
from aioway.fake import *
from aioway.fate import *
from aioway.fn import *

# %%
with torch_fake_mode():
    a = torch.randn(3, 4)
    b = torch.randn(3, 4)

# %%
with fake_fn() as hists, PrintTorDis().enter():
    a + b

# %%
with fake_fn(), PrintTorDis().enter():
    3 - a

# %%
with fake_fn(), PrintTorDis().enter():
    a[a > 0]
