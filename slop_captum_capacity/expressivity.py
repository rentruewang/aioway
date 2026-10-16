"""Expressivity bottleneck: would a layer do better with more units?

After TINY (Verbockhaven et al., "Growing Tiny Networks: Spotting Expressivity Bottlenecks and Fixing Them
Optimally", TMLR 2024). Plain PyTorch, works on any model built from nn.Linear layers with ReLUs between them.

A *site* is a place where units could be added:

    a = input of `layer`                    (n, d_in)
    h = relu(layer(a)) = input of `next`    (n, width)
    z = next(h)                             (n, d_out)    v = -dLoss/dz: where the loss wants z to move

    rest     = v - h @ pinv(h) @ v          the part of v that no change to `next`'s weights can produce
    headroom = top singular value² of S^-1/2 · a.T @ rest / n,  S = a.T @ a / n
             = how fast the loss starts falling if the best new unit is added to `layer`

Usage:

    sites = [Site("fc1", "fc2"), Site("fc2", "fc3")]
    res = measure(model, sites, batches)            # batches: iterable of (x, y); sums are accumulated
    res["fc2"].headroom
    bigger = grow(model, sites[1], res["fc2"])      # a copy with 2 more units in fc2 (output unchanged until scaled)
    check(model, sites[1], res["fc2"], fit_batches, check_batches)   # measured loss drop on other data
"""
import copy
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = ["Site", "SiteResult", "linear_sites", "measure", "grow", "check", "mean_loss"]


@dataclass
class Site:
    layer: str  # the nn.Linear whose width is tested, e.g. "fc2"
    next: str   # the nn.Linear that reads its output after a ReLU, e.g. "fc3"


@dataclass
class SiteResult:
    unreachable: float   # share of |v|² that retuning `next` can't produce (≈ 1 once `next` has converged)
    headroom: float      # first-order loss decrease rate of the best new unit
    gains: torch.Tensor  # the same for the 2nd, 3rd, ... best new unit
    alpha: torch.Tensor  # (d_in + 1, k) in-weights of the new units; last row = bias
    omega: torch.Tensor  # (k, d_out) out-weights of the new units
    n: int               # examples used


def linear_sites(model):
    """Every pair of consecutive nn.Linear layers, in registration order (fine for plain MLPs)."""
    names = [name for name, m in model.named_modules() if isinstance(m, nn.Linear)]
    return [Site(a, b) for a, b in zip(names, names[1:])]


def _with_ones(t):
    return torch.cat([t, torch.ones(len(t), 1, dtype=t.dtype)], dim=1)


def measure(model, sites, batches, loss_fn=F.cross_entropy, k=8):
    """Accumulate the sums over `batches` (iterable of (x, y)), then compute each site's bottleneck."""
    model.eval()
    sums = {s.layer: {} for s in sites}
    n = 0
    for x, y in batches:
        store, hooks = {s.layer: {} for s in sites}, []
        for s in sites:
            hooks.append(model.get_submodule(s.layer).register_forward_hook(
                lambda m, inp, out, s=s: store[s.layer].update(a=inp[0])))
            hooks.append(model.get_submodule(s.next).register_forward_hook(
                lambda m, inp, out, s=s: store[s.layer].update(h=inp[0], z=out)))
        try:
            with torch.enable_grad():
                out = model(x)
                loss = loss_fn(out, y, reduction="sum")  # sum: each row of v is one example's own gradient
                grads = torch.autograd.grad(loss, [store[s.layer]["z"] for s in sites])
        finally:
            for hook in hooks:
                hook.remove()
        for s, g in zip(sites, grads):
            a = _with_ones(store[s.layer]["a"].detach().flatten(1).double())
            h = _with_ones(store[s.layer]["h"].detach().flatten(1).double())
            v = -g.detach().flatten(1).double()
            for key, val in [("aa", a.T @ a), ("hh", h.T @ h), ("hv", h.T @ v), ("av", a.T @ v), ("ah", a.T @ h),
                             ("vv", (v * v).sum())]:
                sums[s.layer][key] = sums[s.layer].get(key, 0) + val
        n += len(x)

    results = {}
    for s in sites:
        S = sums[s.layer]
        dW = torch.linalg.pinv(S["hh"], hermitian=True, rtol=1e-10) @ S["hv"]  # best change to `next`'s weights
        rest_sq = S["vv"] - (dW * S["hv"]).sum()                              # |v - h dW|²
        N = (S["av"] - S["ah"] @ dW) / n                                      # a.T @ rest / n
        ev, U = torch.linalg.eigh(S["aa"] / n)
        keep = ev > 1e-6 * ev.max()                                           # drop empty directions (dead units)
        s_inv_half = U[:, keep] @ torch.diag(ev[keep] ** -0.5) @ U[:, keep].T
        Uk, lam, Vk = torch.linalg.svd(s_inv_half @ N, full_matrices=False)
        kk = min(k, len(lam))
        Uk, lam, Vk = Uk[:, :kk], lam[:kk], Vk[:kk]
        results[s.layer] = SiteResult(
            unreachable=float(rest_sq / S["vv"]), headroom=float(lam[0] ** 2), gains=(lam ** 2).float(),
            alpha=(s_inv_half @ Uk * lam.sqrt()).float(), omega=(Vk * lam.sqrt()[:, None]).float(), n=n)
    return results


def grow(model, site, result, unit=0, scale=1.0):
    """A copy of `model` with 2 more units in `site.layer`: relu(u)·scale·ω and relu(-u)·(-scale·ω), u = a·α.
    Together they add exactly scale·u·ω to `next`'s pre-activation (needs a ReLU between the two layers)."""
    model = copy.deepcopy(model)
    L, N = model.get_submodule(site.layer), model.get_submodule(site.next)
    alpha, omega = result.alpha[:, unit], result.omega[unit]
    new_L = nn.Linear(L.in_features, L.out_features + 2)
    new_N = nn.Linear(N.in_features + 2, N.out_features)
    with torch.no_grad():
        new_L.weight.copy_(torch.cat([L.weight, alpha[None, :-1], -alpha[None, :-1]]))
        new_L.bias.copy_(torch.cat([L.bias, alpha[-1:], -alpha[-1:]]))
        new_N.weight.copy_(torch.cat([N.weight, scale * omega[:, None], -scale * omega[:, None]], dim=1))
        new_N.bias.copy_(N.bias)
    for name, new in [(site.layer, new_L), (site.next, new_N)]:
        parent, _, child = name.rpartition(".")
        setattr(model.get_submodule(parent) if parent else model, child, new)
    return model


@torch.no_grad()
def mean_loss(model, batches, loss_fn=F.cross_entropy):
    model.eval()
    total, n = 0.0, 0
    for x, y in batches:
        total += loss_fn(model(x), y, reduction="sum").item()
        n += len(x)
    return total / n


def check(model, site, result, fit_batches, check_batches, loss_fn=F.cross_entropy,
          scales=(0.003, 0.01, 0.03, 0.1, 0.3, 1, 3, 10, 30, 100)):
    """Add the best new unit pair, pick its size on `fit_batches`, report the loss drop on `check_batches`."""
    fit_batches, check_batches = list(fit_batches), list(check_batches)
    best = min(scales, key=lambda s: mean_loss(grow(model, site, result, scale=s), fit_batches, loss_fn))
    before = mean_loss(model, check_batches, loss_fn)
    after = mean_loss(grow(model, site, result, scale=best), check_batches, loss_fn)
    return dict(before=before, after=after, drop=before - after, scale=best)
