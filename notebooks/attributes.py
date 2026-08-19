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
import pickle

# %%
import torch

# %%
from aioway.schemas import Attr

# %%

# %%
t = torch.randn(5, 7, 9).to(torch.float16).requires_grad_()

# %%
a = Attr.parse(t)
a

# %%
a.__getstate__()

# %%
pickle.dumps(a)

# %%
a == {"shape": [5, 7, 9], "dtype": "float16", "requires_grad": True}
