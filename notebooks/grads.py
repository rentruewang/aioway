# ---
# jupyter:
#   jupytext:
#     formats: py:percent
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

from aioway.fake import TorchDispMode


# %%
class PrintTorchGradEnabled(TorchDispMode):
    def run(self, thunk):
        print(torch.is_grad_enabled())
        return thunk()


# %%
with PrintTorchGradEnabled()():
    a = torch.tensor(3)
    b = torch.tensor(4)
    a + b

# %%
with PrintTorchGradEnabled()(), torch.set_grad_enabled(False):
    a = torch.tensor(3)
    b = torch.tensor(4)
    a + b
