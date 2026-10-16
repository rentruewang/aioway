# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Captum tour: features, model, training
#
# **Goal**: use Captum to find the parts of a model that have **not converged** or **need more capacity**.
#
# The test bed has a known answer: **16 features** with known roles, and **two separate models**, trained on the same
# data, that differ only in their middle layer `fc2`:
#
# ```python
# narrow = Student(width=3)    # fc2 has 3 units: deliberately too few (the "broken" model)
# wide   = Student(width=64)   # fc2 has 64 units: the healthy control
# ```
#
# Most plots show both models side by side. A tool works if it flags `fc2` in **narrow** and nothing in **wide**.
#
# Every plot uses one of two frames: the **16 features** (same x-axis everywhere) or the **layers and units** of the
# models (same names and colours everywhere). Each section starts with **pseudo-code** of what its code cells do, so
# the code cells themselves can be skipped.
#
# | Part | § | Captum |
# |---|---|---|
# | 1 Feature space | 1 | (data only) |
# | 2 The two models | 2–3 | (architecture, training) |
# | 3 Feature attribution | 4–11 | `attr` primary methods, `NoiseTunnel`, `ClassSummarizer`, `DataLoaderAttribution`, `metrics` |
# | 4 Model attribution | 12–16 | `attr` layer and neuron methods |
# | 5 Training examples | 17–20 | `influence` |
# | 6 Capacity probes | 21–24 | `module`, `concept`, `robust`; expressivity bottleneck (plain PyTorch) |
# | 7 Verdict | 25–27 | all of the above |
#
# **Run**: `pip install torch captum matplotlib pandas scikit-learn`, then open this file as a notebook (VS Code,
# PyCharm, or Jupyter with jupytext) or run `python captum_capacity_probe.py`. About 3 minutes on a laptop CPU.
# Tested with torch 2.14 and captum 0.9.0.

# %%
import copy
import math
import os
import tempfile
import time
import uuid
import warnings
from dataclasses import dataclass

import captum
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import FancyArrowPatch, Rectangle
from sklearn.linear_model import LogisticRegression
from torch.utils.data import DataLoader, Dataset, TensorDataset

from captum.attr import (
    ClassSummarizer,
    CommonStats,
    DataLoaderAttribution,
    Deconvolution,
    DeepLift,
    DeepLiftShap,
    FeatureAblation,
    FeaturePermutation,
    GradientShap,
    GuidedBackprop,
    InputXGradient,
    IntegratedGradients,
    InternalInfluence,
    KernelShap,
    LayerActivation,
    LayerConductance,
    LayerDeepLift,
    LayerDeepLiftShap,
    LayerFeatureAblation,
    LayerFeaturePermutation,
    LayerGradientShap,
    LayerGradientXActivation,
    LayerIntegratedGradients,
    LayerLRP,
    Lime,
    LRP,
    NeuronConductance,
    NeuronDeconvolution,
    NeuronDeepLift,
    NeuronDeepLiftShap,
    NeuronFeatureAblation,
    NeuronGradient,
    NeuronGradientShap,
    NeuronGuidedBackprop,
    NeuronIntegratedGradients,
    NoiseTunnel,
    Occlusion,
    Saliency,
    ShapleyValues,
    ShapleyValueSampling,
)
from captum.concept import TCAV, Classifier, Concept
from captum.influence import (
    NaiveInfluenceFunction,
    SimilarityInfluence,
    TracInCP,
    TracInCPFast,
    TracInCPFastRandProj,
)
from captum.influence._utils.nearest_neighbors import NearestNeighbors
from captum.metrics import (
    infidelity,
    infidelity_perturb_func_decorator,
    sensitivity_max,
)
from captum.module import BinaryConcreteStochasticGates, GaussianStochasticGates
from captum.robust import FGSM, PGD, AttackComparator, MinParamPerturbation

try:
    from IPython.display import display
except ImportError:  # plain script
    display = print

warnings.filterwarnings("ignore")
pd.set_option("display.precision", 3)
pd.set_option("display.width", 200)
T_START = time.time()
WORKDIR = tempfile.mkdtemp(
    prefix="captum_tour_"
)  # Captum writes checkpoints / activations here
print(f"torch {torch.__version__} | captum {captum.__version__}")

# %%
# Plot style. Colours: one per layer, one per model; blue/red = sign of an attribution.
SURFACE, INK, INK2, GRID, BAND, NEUTRAL = (
    "#fcfcfb",
    "#0b0b0b",
    "#52514e",
    "#e6e5e1",
    "#efeeea",
    "#a3a29b",
)
POS, NEG = "#2a78d6", "#e34948"
LAYER_COLOR = {"fc1": "#2a78d6", "fc2": "#eb6834", "fc3": "#1baf7a"}
MODEL_COLOR = {"narrow": "#6250d6", "wide": "#008300"}
BLUES = LinearSegmentedColormap.from_list(
    "blues",
    [
        "#f7f9fc",
        "#cde2fb",
        "#9ec5f4",
        "#6da7ec",
        "#3987e5",
        "#256abf",
        "#184f95",
        "#0d366b",
    ],
)
DIVERGING = LinearSegmentedColormap.from_list(
    "div", ["#b22c2c", "#e34948", "#f0efec", "#2a78d6", "#184f95"]
)
plt.rcParams.update(
    {
        "figure.facecolor": SURFACE,
        "axes.facecolor": SURFACE,
        "savefig.facecolor": SURFACE,
        "axes.edgecolor": INK2,
        "axes.labelcolor": INK2,
        "xtick.color": INK2,
        "ytick.color": INK2,
        "text.color": INK,
        "axes.titlecolor": INK,
        "axes.titlesize": 10.5,
        "axes.titleweight": "bold",
        "axes.labelsize": 9.5,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "legend.fontsize": 9,
        "legend.frameon": False,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "axes.grid.axis": "y",
        "grid.color": GRID,
        "grid.linewidth": 0.8,
        "axes.axisbelow": True,
        "lines.linewidth": 2,
        "lines.markersize": 6,
        "figure.dpi": 110,
        "figure.constrained_layout.use": True,
    }
)

FEATURES = (
    [f"s{i}" for i in range(8)]
    + [f"c{i}" for i in range(4)]
    + [f"n{i}" for i in range(4)]
)
GROUPS = {"signal": (0, 8), "copies": (8, 12), "noise": (12, 16)}
N_FEATURES = len(FEATURES)


def feature_axis(ax, ticklabels=True, group_labels=False):
    """The x-axis of every feature plot: s0..s7 | c0..c3 | n0..n3 (copies and noise shaded)."""
    ax.set_xlim(-0.6, N_FEATURES - 0.4)
    ax.set_xticks(range(N_FEATURES), FEATURES if ticklabels else [])
    ax.grid(axis="x", visible=False)
    for name, (a, b) in GROUPS.items():
        if name != "signal":
            ax.axvspan(a - 0.5, b - 0.5, color=BAND, lw=0, zorder=0)
    if group_labels:
        top = ax.secondary_xaxis("top")
        top.set_xticks([(a + b - 1) / 2 for a, b in GROUPS.values()], list(GROUPS))
        top.tick_params(length=0, labelsize=8.5, labelcolor=INK2)
        top.spines["top"].set_visible(False)


def feature_heatmap(
    ax, values, row_labels, cmap=BLUES, vmin=None, vmax=None, group_labels=True
):
    """Heatmap whose columns are the 16 features."""
    im = ax.imshow(
        values, aspect="auto", cmap=cmap, vmin=vmin, vmax=vmax, interpolation="nearest"
    )
    ax.set_xticks(range(N_FEATURES), FEATURES)
    ax.set_yticks(range(len(row_labels)), row_labels)
    ax.grid(False)
    for x in (7.5, 11.5):
        ax.axvline(x, color=SURFACE, lw=3)
    if group_labels:
        top = ax.secondary_xaxis("top")
        top.set_xticks([(a + b - 1) / 2 for a, b in GROUPS.values()], list(GROUPS))
        top.tick_params(length=0, labelsize=8.5, labelcolor=INK2)
        top.spines["top"].set_visible(False)
    return im


def end_label(ax, x, y, text):
    ax.annotate(
        text,
        (x, y),
        xytext=(5, 0),
        textcoords="offset points",
        va="center",
        fontsize=9,
        color=INK2,
    )


def end_labels(ax, items):
    """Label line ends [(x, y, text)], nudged apart so they never overlap. Call after the y-scale is set."""
    log = ax.get_yscale() == "log"
    f, inv = (np.log10, lambda v: 10**v) if log else (lambda v: v, lambda v: v)
    lo, hi = ax.get_ylim()
    gap, placed = (f(hi) - f(lo)) * 0.07, []
    for x, y, text in sorted(items, key=lambda t: f(t[1])):
        yy = f(y) if not placed else max(f(y), placed[-1] + gap)
        placed.append(yy)
        end_label(ax, x, inv(yy), text)


def entropy_count(v):
    """exp(entropy) of a non-negative vector: how many entries carry the mass."""
    v = torch.as_tensor(v).double().clamp_min(0)
    if v.sum() <= 0:
        return 0.0
    p = v / v.sum()
    p = p[p > 0]
    return float(torch.exp(-(p * p.log()).sum()))


def spearman(a, b):
    ra, rb = a.argsort().argsort().double(), b.argsort().argsort().double()
    return float(torch.corrcoef(torch.stack([ra, rb]))[0, 1])


def share(attr):
    """Mean |attribution| per column, as a share of the total."""
    v = attr.abs().mean(0)
    return v / v.sum()


# %% [markdown]
# ## Part 1 · Feature space
#
# ## 1. The 16 input features
#
# | group | features | role |
# |---|---|---|
# | signal | `s0`–`s7` | the label depends on them; `s0` matters most, `s7` least |
# | copies | `c0`–`c3` | noisy copies of `s0`–`s3` (correlation 0.9); the label does not use them |
# | noise | `n0`–`n3` | pure noise |
#
# ```python
# s = randn(N, 8);  n = randn(N, 4)
# c = 0.9 * s[:, :4] + 0.44 * randn(N, 4)     # noisy copies of s0..s3
# x = [s, c, n]                               # (N, 16)
# y = argmax(teacher(s))                      # 10 balanced classes; the hidden teacher only sees s0..s7
#
# for j in range(16):                         # ground truth importance
#     truth[j] = share of labels that change when column j of x is shuffled
# ```

# %%
N_CLASSES, N_TRAIN, RHO = 10, 8000, 0.9
STRENGTH = [
    1.0,
    0.85,
    0.7,
    0.6,
    0.5,
    0.4,
    0.3,
    0.2,
]  # how strongly the teacher uses each signal


def make_data(n, seed):
    g = torch.Generator().manual_seed(seed)
    s = torch.randn(n, 8, generator=g)
    c = RHO * s[:, :4] + math.sqrt(1 - RHO**2) * torch.randn(n, 4, generator=g)
    noise = torch.randn(n, 4, generator=g)
    return torch.cat([s, c, noise], dim=1)


class Teacher(nn.Module):
    """The hidden rule that labels the data. It only looks at s0..s7."""

    def __init__(self):
        super().__init__()
        self.l1, self.l2 = nn.Linear(8, 64), nn.Linear(64, N_CLASSES)
        with torch.no_grad():
            self.l1.weight.mul_(torch.tensor(STRENGTH))

    def forward(self, x):
        return self.l2(torch.tanh(self.l1(x[:, :8])))


@torch.no_grad()
def label_flip_rate(model, x, y):
    """Per feature: share of predictions that change when that feature is shuffled across examples."""
    out = []
    for j in range(x.shape[1]):
        xp = x.clone()
        xp[:, j] = xp[
            torch.randperm(len(x), generator=torch.Generator().manual_seed(j)), j
        ]
        out.append((model(xp).argmax(-1) != y).float().mean().item())
    return torch.tensor(out)


torch.manual_seed(0)
teacher = Teacher()
X = make_data(N_TRAIN, seed=0)
with torch.no_grad():
    logits = teacher(X)
    shift = torch.zeros(N_CLASSES)
    for _ in range(
        800
    ):  # nudge the class biases until the 10 classes are equally common
        freq = (
            torch.bincount((logits + shift).argmax(-1), minlength=N_CLASSES).float()
            / N_TRAIN
        )
        shift -= 0.02 * (freq - 1 / N_CLASSES) * N_CLASSES
    teacher.l2.bias += shift
    order = label_flip_rate(teacher, X, teacher(X).argmax(-1))[:8].argsort(
        descending=True
    )
    teacher.l1.weight.copy_(
        teacher.l1.weight[:, order]
    )  # rename the signals so that s0 matters most
    Y = teacher(X).argmax(-1)
TRUTH = label_flip_rate(teacher, X, Y)
TRUTH_SHARE = TRUTH / TRUTH.sum()
print("examples per class:", torch.bincount(Y).tolist())

fig, axes = plt.subplots(
    1, 2, figsize=(12, 4.3), gridspec_kw=dict(width_ratios=[1, 1.4])
)
ax = axes[0]
im = ax.imshow(np.corrcoef(X.numpy().T), cmap=DIVERGING, vmin=-1, vmax=1)
ax.set_xticks(range(N_FEATURES), FEATURES, fontsize=8)
ax.set_yticks(range(N_FEATURES), FEATURES, fontsize=8)
ax.grid(False)
for v in (7.5, 11.5):
    ax.axvline(v, color=SURFACE, lw=2)
    ax.axhline(v, color=SURFACE, lw=2)
fig.colorbar(im, ax=ax, shrink=0.85, label="correlation")
ax.set_title("Correlation between features")
ax = axes[1]
ax.bar(range(N_FEATURES), TRUTH.numpy(), color=NEUTRAL, width=0.7)
feature_axis(ax, group_labels=True)
ax.set_ylabel("labels that change\nwhen the feature is shuffled")
ax.set_title("Ground truth importance")
plt.show()

# %% [markdown]
# * Each copy `cᵢ` correlates 0.9 with `sᵢ`; all other pairs are independent.
# * Shuffling `s0` changes 45% of the labels, `s7` 16%. Copies and noise change none.

# %% [markdown]
# ## Part 2 · The two models
#
# ## 2. Architecture
#
# Both models come from the same class; only the width of `fc2` differs:
#
# ```python
# narrow = Student(width=3)     # fc2: 64 → 3    the planted bottleneck
# wide   = Student(width=64)    # fc2: 64 → 64   the healthy control
#
# # forward pass for N examples (the same code in both models)
# h1 = relu(fc1(x))       # (N, 64)
# h2 = relu(fc2(h1))      # (N, 3) in narrow, (N, 64) in wide    <- "fc2 output"
# h3 = relu(fc3(h2))      # (N, 64)
# scores = fc4(h3)        # (N, 10): one score per class
# ```
#
# * A **unit** is one column of a layer's output: `fc2` unit 0 is `h2[:, 0]`.
# * "**narrow fc2** vs **wide fc2**" always means the same layer in the two models.
# * The narrow model has to squeeze everything it knows about an example into 3 numbers at `fc2`.
#
# In the diagram, box height grows with the number of units.

# %%
HIDDEN = 64
LAYERS = ["fc1", "fc2", "fc3"]
WIDTH = {"narrow": 3, "wide": 64}


class Student(nn.Module):
    def __init__(self, width):
        super().__init__()
        # ReLUs are separate modules: DeepLift, GuidedBackprop, Deconvolution and LRP hook into them.
        self.fc1, self.act1 = nn.Linear(N_FEATURES, HIDDEN), nn.ReLU()
        self.fc2, self.act2 = nn.Linear(HIDDEN, width), nn.ReLU()
        self.fc3, self.act3 = nn.Linear(width, HIDDEN), nn.ReLU()
        self.fc4 = nn.Linear(HIDDEN, N_CLASSES)

    def forward(self, x):
        x = self.act1(self.fc1(x))
        x = self.act2(self.fc2(x))
        x = self.act3(self.fc3(x))
        return self.fc4(x)


def draw_architecture(ax, width, title, notes=None, flag=None):
    """Boxes = layers (height ~ sqrt(units)). `notes`: text under each box. `flag`: layer to outline."""
    layers = [
        ("input", N_FEATURES, "16 features"),
        ("fc1", HIDDEN, f"{HIDDEN} units"),
        ("fc2", width, f"{width} units"),
        ("fc3", HIDDEN, f"{HIDDEN} units"),
        ("fc4", N_CLASSES, "10 class scores"),
    ]
    gap = 2.5
    for i, (name, n, label) in enumerate(layers):
        x, h = i * gap, 0.3 * math.sqrt(n)
        ax.add_patch(
            Rectangle(
                (x - 0.5, -h / 2),
                1.0,
                h,
                facecolor=LAYER_COLOR.get(name, NEUTRAL),
                edgecolor=INK if name == flag else SURFACE,
                lw=3 if name == flag else 1,
            )
        )
        ax.text(
            x, h / 2 + 0.1, name, ha="center", va="bottom", fontsize=10.5, weight="bold"
        )
        ax.text(x, -h / 2 - 0.1, label, ha="center", va="top", fontsize=9, color=INK2)
        if notes and name in notes:
            ax.text(
                x,
                -1.75,
                notes[name],
                ha="center",
                va="top",
                fontsize=8.5,
                linespacing=1.45,
                weight="bold" if name == flag else "normal",
            )
        if i < len(layers) - 1:
            ax.add_patch(
                FancyArrowPatch(
                    (x + 0.6, 0),
                    (x + gap - 0.6, 0),
                    arrowstyle="-|>",
                    mutation_scale=12,
                    color=INK2,
                    lw=1.2,
                )
            )
            if i > 0:
                ax.text(
                    x + gap / 2,
                    0.08,
                    "ReLU",
                    ha="center",
                    va="bottom",
                    fontsize=8,
                    color=INK2,
                )
    ax.set_xlim(-1.3, (len(layers) - 1) * gap + 1.3)
    ax.set_ylim(-4.2 if notes else -1.8, 1.6)
    ax.set_title(title, loc="left")
    ax.axis("off")


fig, axes = plt.subplots(2, 1, figsize=(10, 5.2))
for ax, name in zip(axes, WIDTH):
    draw_architecture(ax, WIDTH[name], f"{name} model: fc2 has {WIDTH[name]} units")
plt.show()
display(
    pd.DataFrame(
        {
            name: {
                layer: sum(
                    p.numel() for p in Student(w).get_submodule(layer).parameters()
                )
                for layer in LAYERS + ["fc4"]
            }
            for name, w in WIDTH.items()
        }
    ).T.rename_axis("weights")
)

# %% [markdown]
# ## 3. Training
#
# ```python
# for model in [narrow, wide]:                   # the same recipe for both
#     for epoch in 1..60:
#         for x, y in batches of 128:
#             loss = cross_entropy(model(x), y)
#             loss.backward();  adam.step()
#         checkpoints[model][epoch] = copy(model.weights)    # later sections compare checkpoints
# ```
#
# Shaded windows: the "early" and "late" checkpoints used for the verdicts in §25.

# %%
BATCH, EPOCHS, LR = 128, 60, 1e-3


@torch.no_grad()
def loss_acc(model, x=X, y=Y):
    model.eval()
    out = model(x)
    return F.cross_entropy(out, y).item(), (out.argmax(-1) == y).float().mean().item()


def train(width, seed=1):
    """Returns {epoch: state_dict} and the per-epoch loss / accuracy."""
    torch.manual_seed(seed)
    model = Student(width)
    opt = torch.optim.Adam(model.parameters(), lr=LR)
    ckpts, hist = {0: copy.deepcopy(model.state_dict())}, [loss_acc(model)]
    for epoch in range(1, EPOCHS + 1):
        model.train()
        for idx in torch.randperm(N_TRAIN).split(BATCH):
            loss = F.cross_entropy(model(X[idx]), Y[idx])
            opt.zero_grad()
            loss.backward()
            opt.step()
        ckpts[epoch] = copy.deepcopy(model.state_dict())
        hist.append(loss_acc(model))
    return ckpts, pd.DataFrame(hist, columns=["loss", "acc"])


def load(width, state):
    model = Student(width)
    model.load_state_dict(state)
    return model.eval()


t0 = time.time()
CKPTS, HIST = {}, {}
for name, w in WIDTH.items():
    CKPTS[name], HIST[name] = train(w)
MODELS = {name: load(WIDTH[name], CKPTS[name][EPOCHS]) for name in WIDTH}
NARROW, WIDE = MODELS["narrow"], MODELS["wide"]
print(f"trained both models in {time.time() - t0:.0f}s")

fig, axes = plt.subplots(1, 2, figsize=(11, 3.4))
for name in WIDTH:
    h = HIST[name]
    for ax, col in zip(axes, ["loss", "acc"]):
        ax.plot(h.index, h[col], color=MODEL_COLOR[name])
        end_label(ax, h.index[-1], h[col].iloc[-1], f"{name} {h[col].iloc[-1]:.3f}")
for ax in axes:
    for lo, hi, text, x, ha in [
        (1, 3, "early", 3.6, "left"),
        (40, 60, "late", 50, "center"),
    ]:
        ax.axvspan(lo, hi, color=BAND, lw=0, zorder=0)
        ax.text(
            x,
            0.03,
            text,
            transform=ax.get_xaxis_transform(),
            ha=ha,
            fontsize=8.5,
            color=INK2,
        )
    ax.set_xlabel("epoch")
    ax.set_xlim(0, 75)
axes[0].set_yscale("log")
axes[0].set_ylabel("training loss (log)")
axes[1].set_ylabel("training accuracy")
axes[0].set_title("Narrow stalls, wide keeps improving")
plt.show()

# %% [markdown]
# ## Part 3 · Feature attribution (`captum.attr`, primary methods)
#
# **Question**: which input features does a model use?
#
# ```python
# attr = Method(wide).attribute(x, target=y)     # x: (N, 16)  →  attr: (N, 16)
# attr[i, j]    # how much feature j pushed example i's score for its true class y[i]
# ```
#
# Each figure has one row per method:
#
# * **left**: `attr` for one example, signed (blue pushes the score up, red pushes it down)
# * **right**: `mean(|attr|)` over 256 probe examples, as a share, with the ground truth as black ticks
#
# §4–§10 use the wide model. §11 compares narrow and wide.

# %%
PROBE = torch.randperm(N_TRAIN, generator=torch.Generator().manual_seed(2))[:256]
PX, PY = X[PROBE], Y[PROBE]  # the probe set used by every tool
with torch.no_grad():
    both_right = (NARROW(PX).argmax(-1) == PY) & (WIDE(PX).argmax(-1) == PY)
EX = int(both_right.nonzero()[0])  # the example shown in the "one example" panels
DIST = X[
    torch.randperm(N_TRAIN, generator=torch.Generator().manual_seed(3))[:16]
]  # baselines for *Shap methods
FEATURE_METHODS = (
    {}
)  # every Part 3 method: name -> (explain(x, target), output units, number of examples)


def run_family(methods, n=None):
    """methods = {name: (explain(x, target), units)}; units: 'score', 'grad' or None. Returns attributions."""
    out = {}
    for name, (explain, units) in methods.items():
        FEATURE_METHODS[name] = (explain, units, n)
        x, y = (PX, PY) if n is None else (PX[:n], PY[:n])
        out[name] = explain(x, target=y).detach()
    return out


def plot_feature_methods(attrs, title):
    names = list(attrs)
    fig, axes = plt.subplots(
        len(names) + 1,
        2,
        figsize=(11.5, 0.95 * len(names) + 2.3),
        gridspec_kw=dict(height_ratios=[1.25] + [1] * len(names)),
    )
    axes[0, 0].bar(range(N_FEATURES), PX[EX].numpy(), color=NEUTRAL, width=0.7)
    axes[0, 0].axhline(0, color=INK2, lw=0.8)
    axes[0, 0].set_title(
        f"Example #{EX} (true class {int(PY[EX])}): input values", loc="left"
    )
    axes[0, 1].bar(range(N_FEATURES), TRUTH_SHARE.numpy(), color=NEUTRAL, width=0.7)
    axes[0, 1].set_title("Ground truth: share of importance", loc="left")
    for i, name in enumerate(names, start=1):
        local = attrs[name][EX].numpy()
        ax = axes[i, 0]
        ax.bar(
            range(N_FEATURES),
            local,
            color=[POS if v >= 0 else NEG for v in local],
            width=0.7,
        )
        ax.axhline(0, color=INK2, lw=0.8)
        lim = np.abs(local).max() * 1.15 + 1e-12
        ax.set_ylim(-lim, lim)
        ax.set_ylabel(
            name, rotation=0, ha="right", va="center", fontsize=9.5, color=INK
        )
        ax = axes[i, 1]
        ax.bar(range(N_FEATURES), share(attrs[name]).numpy(), color=POS, width=0.7)
        ax.hlines(
            TRUTH_SHARE.numpy(),
            np.arange(N_FEATURES) - 0.38,
            np.arange(N_FEATURES) + 0.38,
            color=INK,
            lw=1.5,
            label="ground truth",
        )
        ax.set_ylim(0, 0.3)
    for i, row in enumerate(axes):
        for ax in row:
            feature_axis(ax, ticklabels=(i == len(names)), group_labels=(i == 0))
            ax.set_yticks([])
            ax.spines["left"].set_visible(False)
    axes[1, 0].set_title(
        "this example (blue +, red −)",
        loc="left",
        fontsize=9,
        color=INK2,
        weight="normal",
    )
    axes[1, 1].set_title(
        "all examples: mean |attribution|, share",
        loc="left",
        fontsize=9,
        color=INK2,
        weight="normal",
    )
    axes[1, 1].legend(loc="upper right", fontsize=8.5)
    fig.suptitle(title, fontweight="bold", color=INK)
    plt.show()


# %% [markdown]
# ## 4. Gradient methods
#
# ```python
# g = d scores[y] / d x                    # (N, 16): slope of the true-class score per feature
#
# Saliency(model)        → g               # abs=False keeps the sign
# InputXGradient(model)  → g * x
# GuidedBackprop(model)  → g, but each ReLU passes back only positive gradients
# Deconvolution(model)   → g, but each ReLU passes back relu(gradient), ignoring the forward pass
# ```

# %%
attrs_grad = run_family(
    {
        "Saliency": (
            lambda x, target: Saliency(WIDE).attribute(x, target=target, abs=False),
            "grad",
        ),
        "InputXGradient": (
            lambda x, target: InputXGradient(WIDE).attribute(x, target=target),
            "score",
        ),
        "GuidedBackprop": (
            lambda x, target: GuidedBackprop(WIDE).attribute(x, target=target),
            "grad",
        ),
        "Deconvolution": (
            lambda x, target: Deconvolution(WIDE).attribute(x, target=target),
            "grad",
        ),
    }
)
plot_feature_methods(attrs_grad, "Gradient methods (wide model)")


def same_class_spread(a):
    """How much attributions differ between examples of the same class (0 = identical)."""
    return np.mean(
        [
            (a[PY == c].std(0).sum() / a[PY == c].abs().mean(0).sum()).item()
            for c in range(N_CLASSES)
        ]
    )


print(
    "spread between examples of the same class:",
    {k: round(same_class_spread(v), 3) for k, v in attrs_grad.items()},
)

# %% [markdown]
# * All four rank the signals roughly in ground-truth order and give copies and noise little weight.
# * InputXGradient = Saliency × input value, so the sign flips wherever the input is negative (`s0` here).
# * `Deconvolution` gives every example of a class exactly the same attribution (spread 0): in a ReLU network its
#   backward pass does not depend on the input at all.

# %% [markdown]
# ## 5. Reference-based methods
#
# These compare `x` with a **baseline** ("feature absent"). Here the baseline is all zeros, the feature means.
#
# ```python
# IntegratedGradients → (x - baseline) * mean of g at 32 points on the line from baseline to x
# DeepLift            → same goal in one backward pass, using differences from the baseline at each ReLU
# DeepLiftShap        → mean of DeepLift over 16 baselines (real examples)
# GradientShap        → IntegratedGradients with a random baseline and a random point on the line, averaged
# LRP                 → pass the score backwards layer by layer, split in proportion to each input's share
#
# completeness:  attr[i].sum() ≈ score(x[i]) - score(baseline)
# ```

# %%
torch.manual_seed(0)
attrs_ref = run_family(
    {
        "IntegratedGradients": (
            lambda x, target: IntegratedGradients(WIDE).attribute(
                x, target=target, n_steps=32
            ),
            "score",
        ),
        "DeepLift": (
            lambda x, target: DeepLift(WIDE).attribute(x, target=target),
            "score",
        ),
        "DeepLiftShap": (
            lambda x, target: DeepLiftShap(WIDE).attribute(
                x, baselines=DIST, target=target
            ),
            "score",
        ),
        "GradientShap": (
            lambda x, target: GradientShap(WIDE).attribute(
                x, baselines=DIST, target=target, n_samples=20
            ),
            "score",
        ),
        "LRP": (lambda x, target: LRP(WIDE).attribute(x, target=target), "score"),
    }
)
plot_feature_methods(attrs_ref, "Reference-based methods (wide model)")

# %% [markdown]
# **Completeness**: the attributions should add up to `score(input) − score(baseline)`.
# `return_convergence_delta=True` reports the gap for IntegratedGradients; it shrinks as `n_steps` grows.

# %%
with torch.no_grad():
    s_x = WIDE(PX).gather(1, PY[:, None]).squeeze(1)
    s_zero = WIDE(torch.zeros_like(PX)).gather(1, PY[:, None]).squeeze(1)
    s_dist = WIDE(DIST).mean(0)[PY]
targets = {
    "IntegratedGradients": s_x - s_zero,
    "DeepLift": s_x - s_zero,
    "DeepLiftShap": s_x - s_dist,
    "GradientShap": s_x - s_dist,
    "LRP": s_x,
}
display(
    pd.DataFrame(
        {
            k: {
                "compared with": (
                    "score(x)" if k == "LRP" else "score(x) − score(baseline)"
                ),
                "relative gap": (
                    (attrs_ref[k].sum(1) - t).abs().mean() / t.abs().mean()
                ).item(),
            }
            for k, t in targets.items()
        }
    ).T
)
for n_steps in [4, 16, 64]:
    _, delta = IntegratedGradients(WIDE).attribute(
        PX, target=PY, n_steps=n_steps, return_convergence_delta=True
    )
    print(
        f"IntegratedGradients n_steps={n_steps:>2}: median |convergence_delta| / |score gap| = "
        f"{(delta.abs() / (s_x - s_zero).abs()).median().item():.3f}"
    )
print(
    "max |LRP − InputXGradient| =",
    f"{(attrs_ref['LRP'] - attrs_grad['InputXGradient']).abs().max().item():.1e}",
)

# %% [markdown]
# * All five match the ground-truth ranking closely.
# * IntegratedGradients, DeepLift and DeepLiftShap add up almost exactly. GradientShap only does on average (13% gap
#   here), because it samples random baselines and random points on the path.
# * `LRP` is not complete here (biases absorb part of the score) and equals `InputXGradient` exactly: with Captum's
#   default rules, LRP on a plain ReLU network reduces to gradient × input.

# %% [markdown]
# ## 6. Perturbation methods
#
# No gradients: change the input, rerun the model, record how much the score drops.
#
# ```python
# for j in range(16):
#     FeatureAblation[:, j]    = score(x) - score(x with column j set to 0)
#     FeaturePermutation[:, j] = score(x) - score(x with column j shuffled across examples)
#
# Occlusion: like FeatureAblation, but sets a window of 2 neighbouring columns to 0
#            (made for images and sequences)
# ```

# %%
torch.manual_seed(0)
attrs_pert = run_family(
    {
        "FeatureAblation": (
            lambda x, target: FeatureAblation(WIDE).attribute(
                x, baselines=0.0, target=target
            ),
            "score",
        ),
        "FeaturePermutation": (
            lambda x, target: FeaturePermutation(WIDE).attribute(x, target=target),
            "score",
        ),
        "Occlusion (window 2)": (
            lambda x, target: Occlusion(WIDE).attribute(
                x, sliding_window_shapes=(2,), strides=1, baselines=0.0, target=target
            ),
            "score",
        ),
    }
)
plot_feature_methods(attrs_pert, "Perturbation methods (wide model)")

# %% [markdown]
# * Over all examples, FeatureAblation and FeaturePermutation agree with the gradient-based methods. For a single
#   example FeaturePermutation is erratic: the result depends on which value the shuffle happened to swap in.
# * Occlusion blurs neighbours together: each window covers two adjacent features, so `c0` picks up some of the credit
#   of `s7`, its neighbour.

# %% [markdown]
# ## 7. Shapley values and local surrogate models
#
# ```python
# ShapleyValueSampling[:, j] = average over 25 random orders of adding features:
#                              score(features added before j, plus j) - score(features added before j)
# ShapleyValues              = the exact version, over all subsets. Only feasible for a few features, so
#                              feature_mask groups the 16 into 4: s0-s3, s4-s7, copies, noise
#
# Lime:        masks z = 200 random on/off patterns over the features
#              fit a small linear model  score(x, features off where z = 0) ≈ w · z;   attr = w
# KernelShap:  Lime with weights on the masks that turn w into Shapley values
# ```
#
# Lime and KernelShap fit a model per example, so they run on 32 examples.

# %%
GROUP_MASK = torch.tensor([[0] * 4 + [1] * 4 + [2] * 4 + [3] * 4])
torch.manual_seed(0)
attrs_shap = run_family(
    {
        "ShapleyValueSampling": (
            lambda x, target: ShapleyValueSampling(WIDE).attribute(
                x, baselines=0.0, target=target, n_samples=25
            ),
            "score",
        ),
        "ShapleyValues (4 groups)": (
            lambda x, target: ShapleyValues(WIDE).attribute(
                x, baselines=0.0, target=target, feature_mask=GROUP_MASK
            ),
            None,
        ),
    }
)
attrs_shap.update(
    run_family(
        {
            "Lime (32 ex.)": (
                lambda x, target: Lime(WIDE).attribute(
                    x, baselines=0.0, target=target, n_samples=200
                ),
                "score",
            ),
            "KernelShap (32 ex.)": (
                lambda x, target: KernelShap(WIDE).attribute(
                    x, baselines=0.0, target=target, n_samples=200
                ),
                "score",
            ),
        },
        n=32,
    )
)
plot_feature_methods(attrs_shap, "Shapley values and surrogate models (wide model)")
group_values = attrs_shap["ShapleyValues (4 groups)"][:, [0, 4, 8, 12]]
print(
    "exact Shapley value per group, mean |value|:",
    dict(
        zip(
            ["s0-s3", "s4-s7", "copies", "noise"],
            group_values.abs().mean(0).numpy().round(2),
        )
    ),
)

# %% [markdown]
# * Sampled Shapley values match the ground truth as well as Integrated Gradients does.
# * The exact group Shapley values put almost all credit on the two signal groups.
# * Lime and KernelShap are noisier: each example's 200 random on/off samples are a small budget for 16 features.

# %% [markdown]
# ## 8. `NoiseTunnel`: smoothing any method
#
# ```python
# runs = [Saliency(wide).attribute(x + noise_k, target=y) for k in range(25)]    # noise sd 0.3
# smoothgrad    = mean(runs)
# smoothgrad_sq = mean(runs ** 2)
# vargrad       = var(runs)
# ```

# %%
torch.manual_seed(0)
nt = NoiseTunnel(Saliency(WIDE))
attrs_nt = {"Saliency": attrs_grad["Saliency"]}
attrs_nt.update(
    run_family(
        {
            f"NoiseTunnel {kind}": (
                lambda x, target, kind=kind: nt.attribute(
                    x, nt_type=kind, nt_samples=25, stdevs=0.3, target=target, abs=False
                ),
                "grad" if kind == "smoothgrad" else None,
            )
            for kind in ["smoothgrad", "smoothgrad_sq", "vargrad"]
        }
    )
)
plot_feature_methods(attrs_nt, "NoiseTunnel around Saliency (wide model)")

# %% [markdown]
# * `smoothgrad` keeps the sign and is close to plain Saliency: this model's gradients are already smooth.
# * `smoothgrad_sq` and `vargrad` are always positive. They measure how strong or how variable the gradient is
#   around the input, not which way it pushes.

# %% [markdown]
# ## 9. Summaries over many examples: `ClassSummarizer` and `DataLoaderAttribution`
#
# ```python
# summary = ClassSummarizer(CommonStats())        # running mean / var / min / max, kept per class
# for x, y in loader:
#     summary.update(IntegratedGradients(wide).attribute(x, target=y), y)
# summary.class_summaries[c]["mean"]              # (16,): mean attribution for class c   → heatmap row
#
# # DataLoaderAttribution(FeatureAblation(model)): one number per feature for the whole dataset
# for j in range(16):
#     drop[j] = accuracy(all data) - accuracy(all data with column j set to 0)     → bar chart
# ```

# %%
summary = ClassSummarizer(CommonStats())
ig = IntegratedGradients(WIDE)
for xb, yb in DataLoader(TensorDataset(X[:4000], Y[:4000]), batch_size=500):
    summary.update(
        ig.attribute(xb, target=yb, n_steps=16).detach(), yb.tolist()
    )  # labels as plain ints
class_mean = torch.stack(
    [summary.class_summaries[c][0]["mean"] for c in range(N_CLASSES)]
)

EVAL_LOADER = DataLoader(TensorDataset(X[:4000], Y[:4000]), batch_size=1000)


def count_correct(total, out, inputs):  # `reduce`: accumulate over batches
    correct = (out.argmax(-1) == inputs[1]).sum().float()
    return correct if total is None else total + correct


def accuracy_drop(model):
    """Accuracy lost when each feature is set to 0 for the whole dataset (metric before − after)."""
    dla = DataLoaderAttribution(FeatureAblation(model))
    # input_roles: 0 = attribute this input, 2 = do not pass it to the model (the labels).
    # (The captum 0.9 docstring swaps roles 1 and 2; the code uses 2 = "not passed".)
    attr = dla.attribute(
        EVAL_LOADER,
        input_roles=(0, 2),
        baselines=(torch.zeros(1, N_FEATURES),),
        reduce=count_correct,
        to_metric=lambda c: (c / len(EVAL_LOADER.dataset)).reshape(1),
        return_input_shape=False,
    )
    return attr.detach().reshape(-1)


drop = {"teacher (ceiling)": accuracy_drop(teacher), "wide": accuracy_drop(WIDE)}
print(
    "accuracy lost when s0 is removed:",
    {k: round(v[0].item(), 3) for k, v in drop.items()},
)

fig, axes = plt.subplots(
    2, 1, figsize=(11, 6.6), gridspec_kw=dict(height_ratios=[1.6, 1])
)
ax = axes[0]
lim = class_mean.abs().max().item()
im = feature_heatmap(
    ax,
    class_mean.numpy(),
    [f"class {c}" for c in range(N_CLASSES)],
    cmap=DIVERGING,
    vmin=-lim,
    vmax=lim,
)
fig.colorbar(im, ax=ax, shrink=0.9, label="mean IG attribution")
ax.set_title(
    "ClassSummarizer: mean Integrated Gradients per class (wide model)", loc="left"
)
ax = axes[1]
for k, (name, v) in enumerate(drop.items()):
    ax.bar(
        np.arange(N_FEATURES) + (k - 0.5) * 0.38,
        v.numpy(),
        width=0.38,
        label=name,
        color=NEUTRAL if name.startswith("teacher") else MODEL_COLOR["wide"],
    )
feature_axis(ax)
ax.set_ylabel("accuracy lost\nwhen removed")
ax.set_title(
    "DataLoaderAttribution: accuracy lost when a feature is removed from the dataset",
    loc="left",
)
ax.legend(loc="upper right")
plt.show()

# %% [markdown]
# * Each class relies on its own signed mix of signals; copies and noise are near zero for every class.
# * Removing `s0` costs the wide model 32 accuracy points. The teacher's 34 points are the most any model
#   could lose from it.

# %% [markdown]
# ## 10. All feature methods side by side, and which to trust (`captum.metrics`)
#
# ```python
# infidelity:   mask = a random 25% of the features
#               error = (attr[mask].sum() - (score(x) - score(x with mask set to 0))) ** 2
#               lower = attributions predict real removals better ("more faithful")
#
# sensitivity:  x2 = x + noise of at most ±0.1
#               change = max |attr(x2) - attr(x)| / |attr(x)|
#               lower = the explanation is more stable
# ```
#
# Both on 64 examples (Lime and KernelShap: infidelity only, on their 32). Squared and grouped outputs get no infidelity.

# %%
ALL = {**attrs_grad, **attrs_ref, **attrs_pert, **attrs_shap, **attrs_nt}


@infidelity_perturb_func_decorator(multiply_by_inputs=True)
def remove_random_features(inputs):
    return inputs * (torch.rand_like(inputs) > 0.25).float()


@infidelity_perturb_func_decorator(multiply_by_inputs=False)
def remove_random_features_grad(
    inputs,
):  # same, for gradient-type outputs (not in score units)
    return inputs * (torch.rand_like(inputs) > 0.25).float()


t0 = time.time()
rows = []
for name, (explain, units, n) in FEATURE_METHODS.items():
    k = min(64, n or 64)
    x, y, attr = PX[:k], PY[:k], ALL[name][:k]
    torch.manual_seed(0)
    inf = np.nan
    if units is not None:
        perturb = (
            remove_random_features if units == "score" else remove_random_features_grad
        )
        inf = (
            infidelity(
                WIDE, perturb, x, attr, target=y, n_perturb_samples=20, normalize=True
            )
            .mean()
            .item()
        )
    sens = (
        np.nan
        if n
        else sensitivity_max(
            explain, x, target=y, perturb_radius=0.1, n_perturb_samples=5
        )
        .mean()
        .item()
    )
    rows.append(dict(method=name, infidelity=inf, sensitivity=sens))
metrics = pd.DataFrame(rows).set_index("method")
print(f"metrics in {time.time() - t0:.0f}s")

names = list(FEATURE_METHODS)
fig, axes = plt.subplots(
    1,
    3,
    figsize=(13, 0.32 * len(names) + 1.9),
    sharey=True,
    gridspec_kw=dict(width_ratios=[5, 1.3, 1.3]),
)
values = torch.stack([TRUTH_SHARE] + [share(ALL[n]) for n in names]).numpy()
im = feature_heatmap(axes[0], values, ["ground truth"] + names, vmin=0, vmax=0.25)
axes[0].axhline(0.5, color=SURFACE, lw=3)
axes[0].set_title("Share of importance per feature", loc="left")
fig.colorbar(im, ax=axes[0], shrink=0.6, label="share")
for ax, col, label in [
    (axes[1], "infidelity", "infidelity\n(lower = more faithful)"),
    (axes[2], "sensitivity", "sensitivity\n(lower = more stable)"),
]:
    v = metrics[col].reindex(names).to_numpy()
    ax.barh(np.arange(1, len(names) + 1), np.nan_to_num(v), color=POS, height=0.6)
    for i, val in enumerate(v, start=1):
        if np.isnan(val):
            ax.text(0, i, " n/a", va="center", fontsize=8, color=INK2)
    ax.set_xlabel(label)
    ax.grid(axis="x", visible=True)
    ax.grid(axis="y", visible=False)
plt.show()
display(metrics.sort_values("infidelity"))

# %% [markdown]
# * Every method ranks the features about right on this model. They differ in **faithfulness** (how well attributions
#   predict what happens when features are really removed). FeatureAblation and the Shapley-based methods (sampled,
#   Lime, KernelShap) are best, then IntegratedGradients and DeepLift. Occlusion and FeaturePermutation score worst
#   because they measure something else (windows of neighbours, shuffled values). GuidedBackprop is the worst gradient method.
# * **Stability**: FeaturePermutation and GradientShap vary most, because they use random shuffles and random baselines.
#   Deconvolution is perfectly stable only because it ignores the input (§4).

# %% [markdown]
# ## 11. Feature view of capacity and convergence
#
# ```python
# for model in [teacher, wide, narrow]:
#     drop[model] = accuracy lost when each feature is removed         # DataLoaderAttribution, §9
# captured = drop[model] / drop[teacher]       # share of each signal's information the model uses
#
# for epoch in [1, 2, 3, 5, 10, ..., 60]:       # heatmaps: one row per checkpoint
#     row[epoch] = drop, computed with the weights saved at that epoch
# ```
#
# The teacher's loss is the ceiling: the most any model could get out of that feature.

# %%
drop["narrow"] = accuracy_drop(NARROW)
EPOCH_SNAPSHOTS = [1, 2, 3, 5, 10, 15, 20, 30, 40, 50, 60]
t0 = time.time()
drop_by_epoch = {
    name: torch.stack(
        [accuracy_drop(load(WIDTH[name], CKPTS[name][e])) for e in EPOCH_SNAPSHOTS]
    )
    for name in WIDTH
}
print(f"{2 * len(EPOCH_SNAPSHOTS)} checkpoints in {time.time() - t0:.0f}s")

fig, axes = plt.subplots(
    1, 2, figsize=(12.5, 3.6), gridspec_kw=dict(width_ratios=[2.2, 1])
)
ax = axes[0]
for k, name in enumerate(["teacher (ceiling)", "wide", "narrow"]):
    ax.bar(
        np.arange(N_FEATURES) + (k - 1) * 0.27,
        drop[name].numpy(),
        width=0.27,
        label=name,
        color=MODEL_COLOR.get(name, NEUTRAL),
    )
feature_axis(ax, group_labels=True)
ax.set_ylabel("accuracy lost when removed")
ax.legend(loc="upper right")
ax.set_title("How much each model gets out of each feature", loc="left")
ax = axes[1]
for k, name in enumerate(["wide", "narrow"]):
    captured = (drop[name][:8] / drop["teacher (ceiling)"][:8]).numpy()
    ax.bar(
        np.arange(8) + (k - 0.5) * 0.38,
        captured * 100,
        width=0.38,
        color=MODEL_COLOR[name],
        label=name,
    )
ax.axhline(100, color=INK2, lw=1, ls=":")
ax.set_xticks(range(8), FEATURES[:8])
ax.grid(axis="x", visible=False)
ax.set_ylim(0, 110)
ax.set_ylabel("% of the ceiling")
ax.set_title("Share of each signal captured", loc="left")
plt.show()
for name in ["wide", "narrow"]:
    captured = (drop[name][:8] / drop["teacher (ceiling)"][:8] * 100).numpy()
    print(
        f"{name}: captures {captured.min():.0f}–{captured.max():.0f}% of each signal's information "
        f"(s5–s7: {captured[5:].mean():.0f}%)"
    )

fig, axes = plt.subplots(1, 2, figsize=(12.5, 3.8), sharey=True)
vmax = max(v.max().item() for v in drop_by_epoch.values())
for ax, name in zip(axes, ["wide", "narrow"]):
    im = feature_heatmap(
        ax,
        drop_by_epoch[name].numpy(),
        [str(e) for e in EPOCH_SNAPSHOTS],
        vmin=0,
        vmax=vmax,
    )
    ax.set_title(
        f"{name}: accuracy lost when a feature is removed, per checkpoint", loc="left"
    )
axes[0].set_ylabel("epoch")
fig.colorbar(im, ax=axes, shrink=0.9, label="accuracy lost")
plt.show()

# %% [markdown]
# * **Capacity**: the wide model captures 83–94% of each signal's information. The narrow model captures 32–74% and
#   does worst on the weak signals `s5`–`s7`: with 3 units in `fc2` it cannot carry all eight.
# * **Convergence**: the wide model's rows stop changing after about epoch 30, so its use of the features has
#   converged. The narrow model's rows are paler and still creep up slowly.
# * In the first epochs both models lean on the copies `c0`–`c3`, then drop them as they learn the exact signals.

# %% [markdown]
# ## Part 4 · Model attribution (`captum.attr`, layer and neuron methods)
#
# **Question**: which layers and units carry the signal, and is any of them full?
#
# ```python
# LayerMethod(model, model.fc2).attribute(x, target=y)              # (N, 3) in narrow, (N, 64) in wide:
#                                                                   # one number per example per fc2 unit
# NeuronMethod(model, model.fc2).attribute(x, neuron_selector=u)    # (N, 16): which features drive fc2 unit u
# ```
#
# Each plot shows both models. Layer and unit names follow §2.

# %% [markdown]
# ## 12. `LayerActivation`: what the units output
#
# ```python
# h2 = LayerActivation(model, model.fc2).attribute(x)     # = fc2 output: (N, 3) narrow, (N, 64) wide
#
# plot 1:  relu(h2) for 100 examples (10 per class), narrow and wide side by side
#          rows = examples, columns = fc2 units, colour = output (0 = off)
#
# plot 2:  for layer in [fc1, fc2, fc3]:
#              width_used = effective rank of the layer's output / its number of units   # 1 = all dimensions used
#              dead       = share of units whose output is <= 0 for 99% of examples
# ```


# %%
def layer_output(model, layer, x=PX):
    return LayerActivation(model, model.get_submodule(layer)).attribute(x).detach()


def width_used(model, layer):
    a = layer_output(model, layer).double()
    eig = torch.linalg.svdvals(a - a.mean(0)) ** 2
    return entropy_count(eig) / a.shape[1]


def dead_fraction(model, layer):
    return float(
        ((layer_output(model, layer) <= 0).float().mean(0) >= 0.99).float().mean()
    )


rows_by_class = torch.cat(
    [(PY == c).nonzero().squeeze(1)[:10] for c in range(N_CLASSES)]
)
fig, axes = plt.subplots(
    1, 2, figsize=(12.5, 4.8), gridspec_kw=dict(width_ratios=[1.2, 5])
)
for ax, name in zip(axes, ["narrow", "wide"]):
    a = layer_output(MODELS[name], "fc2")[rows_by_class].clamp_min(0)  # after the ReLU
    a = a[:, a.mean(0).argsort(descending=True)]
    im = ax.imshow(a.numpy(), aspect="auto", cmap=BLUES, interpolation="nearest")
    ax.set_yticks(
        np.arange(N_CLASSES) * 10 + 4.5, [f"class {c}" for c in range(N_CLASSES)]
    )
    for c in range(1, N_CLASSES):
        ax.axhline(c * 10 - 0.5, color=SURFACE, lw=1.5)
    ax.grid(False)
    ax.set_xticks(range(a.shape[1]) if a.shape[1] <= 3 else [0, 15, 31, 47, 63])
    ax.set_xlabel(f"its {a.shape[1]} fc2 units, most active first")
    ax.set_title(f"{name} model: fc2 ({a.shape[1]} units)", loc="left")
    fig.colorbar(im, ax=ax, shrink=0.8, label="output after ReLU (0 = off)")
axes[0].set_ylabel("100 probe examples, 10 per class")
fig.suptitle(
    "LayerActivation on fc2: one row per example, one column per unit",
    fontweight="bold",
    color=INK,
)
plt.show()

usage = pd.DataFrame(
    [
        dict(
            model=name,
            layer=layer,
            width_used=width_used(m, layer),
            dead=dead_fraction(m, layer),
        )
        for name, m in MODELS.items()
        for layer in LAYERS
    ]
)
fig, axes = plt.subplots(1, 2, figsize=(10, 3))
for ax, col, label in [
    (axes[0], "width_used", "width used (1 = all)"),
    (axes[1], "dead", "dead units (share)"),
]:
    for k, name in enumerate(["narrow", "wide"]):
        v = usage[usage.model == name][col].to_numpy()
        ax.bar(
            np.arange(3) + (k - 0.5) * 0.38,
            v,
            width=0.38,
            color=MODEL_COLOR[name],
            label=name,
        )
    ax.set_xticks(range(3), LAYERS)
    ax.grid(axis="x", visible=False)
    ax.set_ylabel(label)
    ax.set_ylim(0, 1)
axes[0].legend(loc="upper right")
plt.show()

# %% [markdown]
# * Narrow `fc2` gives each class an on/off pattern over only 3 units, and its output uses most of those 3 dimensions.
#   Some classes have to share a pattern: classes 0 and 1 are both near zero on all three units.
# * Wide `fc2` uses about a tenth of its 64 dimensions. Only `fc3` has dead units (10–15%, in both models).

# %% [markdown]
# ## 13. Layer attribution: how much each unit contributes to the score
#
# ```python
# attr = LayerConductance(model, model.fc2).attribute(x, target=y)    # (N, units of fc2)
#                                                                     # IntegratedGradients flowing through each unit
# importance[u] = mean(|attr[:, u]|)                                  # one number per unit
#
# plot: one row per method = importance / max(importance), units sorted by LayerConductance
#       ρ = rank agreement with LayerConductance
# ```
#
# The other methods, same output shape: `LayerGradientXActivation` (unit output × its gradient), `LayerIntegratedGradients`
# (path in the layer's own output space), `InternalInfluence` (gradient along the path, no × output), and `LayerDeepLift`,
# `LayerDeepLiftShap`, `LayerGradientShap`, `LayerLRP` (the §5 methods, stopped at this layer).


# %%
def unit_importance(model, layer, n_steps=32):
    """Mean |LayerConductance| per unit of `layer`."""
    return (
        LayerConductance(model, model.get_submodule(layer))
        .attribute(PX, target=PY, n_steps=n_steps)
        .detach()
        .abs()
        .mean(0)
    )


LAYER_METHODS = {
    "LayerActivation (no attribution)": lambda m, l: LayerActivation(m, l).attribute(
        PX
    ),
    "LayerGradientXActivation": lambda m, l: LayerGradientXActivation(m, l).attribute(
        PX, target=PY
    ),
    "LayerConductance": lambda m, l: LayerConductance(m, l).attribute(
        PX, target=PY, n_steps=32
    ),
    "LayerIntegratedGradients": lambda m, l: LayerIntegratedGradients(m, l).attribute(
        PX, target=PY, n_steps=32
    ),
    "InternalInfluence": lambda m, l: InternalInfluence(m, l).attribute(
        PX, target=PY, n_steps=32
    ),
    "LayerDeepLift": lambda m, l: LayerDeepLift(m, l).attribute(PX, target=PY),
    "LayerDeepLiftShap": lambda m, l: LayerDeepLiftShap(m, l).attribute(
        PX, baselines=DIST, target=PY
    ),
    "LayerGradientShap": lambda m, l: LayerGradientShap(m, l).attribute(
        PX, baselines=DIST, target=PY, n_samples=20
    ),
    "LayerLRP": lambda m, l: LayerLRP(m, l).attribute(PX, target=PY),
}
torch.manual_seed(0)
layer_scores = {
    name: {
        meth: f(MODELS[name], MODELS[name].fc2).detach().abs().mean(0)
        for meth, f in LAYER_METHODS.items()
    }
    for name in WIDTH
}

fig, axes = plt.subplots(1, 2, figsize=(13, 3.9), gridspec_kw=dict(width_ratios=[6, 1]))
for ax, name in zip(axes, ["wide", "narrow"]):
    scores = layer_scores[name]
    order = scores["LayerConductance"].argsort(descending=True)
    grid = torch.stack([s[order] / s.max() for s in scores.values()]).numpy()
    im = ax.imshow(
        grid, aspect="auto", cmap=BLUES, vmin=0, vmax=1, interpolation="nearest"
    )
    ax.grid(False)
    if name == "wide":
        ax.set_yticks(
            range(len(scores)),
            [
                f"{meth}  ρ={spearman(s, scores['LayerConductance']):.2f}"
                for meth, s in scores.items()
            ],
        )
    else:
        ax.set_yticks([])
    ax.set_xticks(range(WIDTH[name]) if WIDTH[name] <= 3 else [0, 15, 31, 47, 63])
    ax.set_xlabel(f"its {WIDTH[name]} fc2 units, most important first")
    ax.set_title(
        f"{name} model: fc2 ({WIDTH[name]} units)"
        + (", same rows" if name == "narrow" else ""),
        loc="left",
    )
fig.colorbar(im, ax=axes, shrink=0.8, label="÷ row max")
plt.show()

util = pd.DataFrame(
    {
        name: {
            layer: entropy_count(unit_importance(m, layer))
            / m.get_submodule(layer).out_features
            for layer in LAYERS
        }
        for name, m in MODELS.items()
    }
)
display(util.rename_axis("units carrying importance (share)"))

# %% [markdown]
# * The path methods (Conductance, LayerIntegratedGradients, LayerDeepLift and the Shap variants) rank the wide units almost
#   identically. Raw activation size barely predicts importance.
# * In the narrow model every method gives each of the 3 units about the same weight: all are fully used.

# %% [markdown]
# ## 14. Layer perturbation: what breaks when a unit is switched off
#
# ```python
# for layer in [fc1, fc2, fc3]:
#     for each unit u:
#         ablation[u]    = mean |score(x) - score(x with unit u's output set to 0)|    # LayerFeatureAblation
#         permutation[u] = mean |score(x) - score(x with unit u's output shuffled)|    # LayerFeaturePermutation
#
# plot: costs sorted from highest to lowest, one line per layer
# ```


# %%
def true_class_score(model, y=PY):
    # captum 0.9's LayerFeaturePermutation rejects tensor targets, so return the score directly
    return lambda x: model(x).gather(1, y[:, None]).squeeze(1)


unit_cost = {}
for name, model in MODELS.items():
    for layer in LAYERS:
        mod = model.get_submodule(layer)
        unit_cost[("LayerFeatureAblation", name, layer)] = (
            LayerFeatureAblation(model, mod)
            .attribute(PX, target=PY)
            .detach()
            .abs()
            .mean(0)
        )
        torch.manual_seed(0)
        unit_cost[("LayerFeaturePermutation", name, layer)] = (
            LayerFeaturePermutation(true_class_score(model), mod)
            .attribute(PX)
            .detach()
            .abs()
            .mean(0)
        )

fig, axes = plt.subplots(2, 2, figsize=(11, 6), sharey=True)
for i, method in enumerate(["LayerFeatureAblation", "LayerFeaturePermutation"]):
    for j, name in enumerate(["narrow", "wide"]):
        ax = axes[i, j]
        for layer in LAYERS:
            v = (
                unit_cost[(method, name, layer)]
                .sort(descending=True)
                .values.clamp_min(1e-3)
                .numpy()
            )
            x = (np.arange(len(v)) + 0.5) / len(v)
            ax.plot(
                x,
                v,
                color=LAYER_COLOR[layer],
                marker="o" if len(v) <= 3 else None,
                label=layer,
            )
        ax.set_yscale("log")
        ax.set_title(f"{method}: {name}", loc="left")
        if i == 1:
            ax.set_xlabel("units, most costly first (position within the layer)")
        if j == 0:
            ax.set_ylabel("score change when\none unit is removed")
axes[0, 0].legend(loc="lower left")
plt.show()

# %% [markdown]
# * Removing one narrow `fc2` unit moves the score about 20× as much as removing an `fc1` or `fc3` unit:
#   the layer has no spare units.
# * In the wide model, `fc2` units are as cheap to lose as any other. Ablation and permutation agree.

# %% [markdown]
# ## 15. Neuron attribution: which input features drive one unit
#
# ```python
# attr = NeuronIntegratedGradients(model, model.fc2).attribute(x, neuron_selector=u)    # (N, 16)
# plot 1: for the busiest fc2 unit, one row per method = share of mean |attr| per feature
#
# # NeuronConductance: only the part of the class score that flows through unit u, so it needs target
# attr = NeuronConductance(model, model.fc2).attribute(x, neuron_selector=u, target=y)
# plot 2: one row per fc2 unit;   load[u] = sum over features of mean |attr|
# ```
#
# Plot 1 methods: `NeuronGradient`, `NeuronIntegratedGradients`, `NeuronDeepLift`, `NeuronDeepLiftShap`, `NeuronGradientShap`,
# `NeuronGuidedBackprop`, `NeuronDeconvolution`, `NeuronFeatureAblation` (the Part 3 ideas, applied to one unit's output) and
# `NeuronConductance`. This joins the two frames: a unit of the model, broken down by input feature.

# %%
NEURON_METHODS = {
    "NeuronGradient": lambda m, l, j: NeuronGradient(m, l).attribute(
        PX, neuron_selector=j
    ),
    "NeuronIntegratedGradients": lambda m, l, j: NeuronIntegratedGradients(
        m, l
    ).attribute(PX, neuron_selector=j, n_steps=32),
    "NeuronDeepLift": lambda m, l, j: NeuronDeepLift(m, l).attribute(
        PX, neuron_selector=j
    ),
    "NeuronDeepLiftShap": lambda m, l, j: NeuronDeepLiftShap(m, l).attribute(
        PX, neuron_selector=j, baselines=DIST
    ),
    "NeuronGradientShap": lambda m, l, j: NeuronGradientShap(m, l).attribute(
        PX, neuron_selector=j, baselines=DIST, n_samples=20
    ),
    "NeuronGuidedBackprop": lambda m, l, j: NeuronGuidedBackprop(m, l).attribute(
        PX, neuron_selector=j
    ),
    "NeuronDeconvolution": lambda m, l, j: NeuronDeconvolution(m, l).attribute(
        PX, neuron_selector=j
    ),
    "NeuronFeatureAblation": lambda m, l, j: NeuronFeatureAblation(m, l).attribute(
        PX, neuron_selector=j
    ),
    "NeuronConductance": lambda m, l, j: NeuronConductance(m, l).attribute(
        PX, neuron_selector=j, target=PY, n_steps=32
    ),
}
busiest = {name: int(layer_scores[name]["LayerConductance"].argmax()) for name in WIDTH}
torch.manual_seed(0)
neuron_maps = {
    name: torch.stack(
        [
            share(f(MODELS[name], MODELS[name].fc2, busiest[name]).detach())
            for f in NEURON_METHODS.values()
        ]
    )
    for name in WIDTH
}

fig, axes = plt.subplots(1, 2, figsize=(13, 3.9), sharey=True)
for ax, name in zip(axes, ["narrow", "wide"]):
    im = feature_heatmap(
        ax, neuron_maps[name].numpy(), list(NEURON_METHODS), vmin=0, vmax=0.35
    )
    ax.set_title(f"{name}: fc2 unit {busiest[name]} (its busiest)", loc="left")
fig.colorbar(im, ax=axes, shrink=0.8, label="share")
plt.show()

# %% [markdown]
# * All nine agree on what drives the unit, except GuidedBackprop and Deconvolution (as in §4).
# * NeuronConductance differs slightly from the rest: it counts only the part of the unit's output that reaches the
#   class score.
#
# **Every `fc2` unit × feature**, with `NeuronConductance` (narrow: all 3 units; wide: its 12 busiest; same colour scale).
# Each unit's **load** = the total class score that flows through it.


# %%
def unit_feature_map(model, units):
    nc = NeuronConductance(model, model.fc2)
    return torch.stack(
        [
            nc.attribute(PX, neuron_selector=int(j), target=PY, n_steps=16)
            .detach()
            .abs()
            .mean(0)
            for j in units
        ]
    )


maps = {
    "narrow": unit_feature_map(NARROW, range(3)),
    "wide": unit_feature_map(
        WIDE, layer_scores["wide"]["LayerConductance"].argsort(descending=True)[:12]
    ),
}
vmax = max(m.max().item() for m in maps.values())
fig, axes = plt.subplots(
    2, 1, figsize=(10, 6.2), gridspec_kw=dict(height_ratios=[3, 12])
)
for ax, name in zip(axes, ["narrow", "wide"]):
    im = feature_heatmap(
        ax,
        maps[name].numpy(),
        [f"load {v:.1f}" for v in maps[name].sum(1)],
        vmin=0,
        vmax=vmax,
        group_labels=(name == "narrow"),
    )
    ax.set_title(f"{name}: fc2 units", loc="left")
fig.colorbar(im, ax=axes, shrink=0.8, label="mean |NeuronConductance|")
plt.show()
print({name: f"mean load per unit {maps[name].sum(1).mean():.1f}" for name in WIDTH})

# %% [markdown]
# * Each narrow `fc2` unit carries about 3.5× the load of even the busiest wide units, and every one of them mixes
#   most of the eight signals.
# * In the wide model the work is spread over many units, each carrying a lighter mix.

# %% [markdown]
# ## 16. Model view of convergence: are the units' roles still changing?
#
# ```python
# for epoch in 0, 2, 4, ..., 60:
#     imp[epoch]   = LayerConductance importance per unit (§13), with that epoch's weights
#     drift[epoch] = 1 - cosine(imp[epoch], imp[epoch - 2])        # 0 = unit roles unchanged
# ```

# %%
EPOCH_GRID = list(range(0, EPOCHS + 1, 2))


def conductance_drift(name):
    rows, prev = [], {}
    for e in EPOCH_GRID:
        model = load(WIDTH[name], CKPTS[name][e])
        for layer in LAYERS:
            imp = unit_importance(model, layer, n_steps=16)
            if layer in prev:
                rows.append(
                    dict(
                        epoch=e,
                        layer=layer,
                        drift=1 - F.cosine_similarity(imp, prev[layer], dim=0).item(),
                    )
                )
            prev[layer] = imp
    return pd.DataFrame(rows)


t0 = time.time()
drift = {name: conductance_drift(name) for name in WIDTH}
print(
    f"conductance at {len(EPOCH_GRID)} checkpoints x 2 models in {time.time() - t0:.0f}s"
)
fig, axes = plt.subplots(1, 2, figsize=(11, 3.3), sharey=True)
for ax, name in zip(axes, ["narrow", "wide"]):
    for layer in LAYERS:
        d = drift[name][drift[name].layer == layer]
        ax.plot(
            d.epoch, d.drift.clip(lower=1e-7), color=LAYER_COLOR[layer], label=layer
        )
    ax.axhline(0.02, color=INK2, lw=1, ls=":")
    ax.set_yscale("log")
    ax.set_xlabel("epoch")
    ax.set_title(name, loc="left")
axes[0].set_ylabel("change in unit importance\nbetween checkpoints")
axes[0].legend(loc="upper right", ncol=3)
plt.show()

# %% [markdown]
# Unit roles settle within a few epochs in every layer of both models. Drift catches a run stopped very early, but it
# cannot tell a converged layer from a stuck one. Part 5 can.

# %% [markdown]
# ## Part 5 · Training examples (`captum.influence`)
#
# **Question**: which training examples is each layer still being pushed by?
#
# ## 17. `TracInCP` restricted to one layer: how hard training still pushes it
#
# ```python
# g[i] = gradient of example i's loss w.r.t. fc2's weights     # 195 numbers in narrow, 4160 in wide
# tracin = TracInCP(model, data, checkpoints, layers=["fc2"])
# influence[i, j]   = g[i] · g[j]                             # does example i help example j?
# self_influence[i] = g[i] · g[i]                             # how hard example i still pushes fc2
# push = mean(self_influence) / number of fc2 weights         # per layer, per checkpoint   → plot
# ```
#
# A converging layer: the push shrinks. A stuck layer: it stays high or grows.


# %%
def save_state(state):
    path = os.path.join(WORKDIR, f"{uuid.uuid4().hex}.pt")
    torch.save(state, path)
    return path


def load_ckpt(model, path):
    model.load_state_dict(torch.load(path, weights_only=True))
    return 1.0  # learning-rate weight for this checkpoint


def layer_push(width, state, layer):
    """Self-influence of every probe example on one layer's parameters."""
    path = save_state(state)
    tracin = TracInCP(
        Student(width),
        TensorDataset(PX, PY),
        checkpoints=[path],
        checkpoints_load_func=load_ckpt,
        layers=[layer],
        loss_fn=nn.CrossEntropyLoss(reduction="sum"),
        batch_size=64,
        sample_wise_grads_per_batch=True,
    )  # batched per-example gradients (Linear/Conv layers)
    out = tracin.self_influence((PX, PY))
    os.remove(path)
    return out


def n_params(width, layer):
    return sum(p.numel() for p in Student(width).get_submodule(layer).parameters())


t0 = time.time()
push = pd.DataFrame(
    [
        dict(
            model=name,
            epoch=e,
            layer=layer,
            per_param=layer_push(w, CKPTS[name][e], layer).mean().item()
            / n_params(w, layer),
        )
        for name, w in WIDTH.items()
        for e in EPOCH_GRID
        for layer in LAYERS
    ]
)
print(
    f"TracInCP at {len(EPOCH_GRID)} checkpoints x 3 layers x 2 models in {time.time() - t0:.0f}s"
)
fig, axes = plt.subplots(1, 2, figsize=(11, 3.4), sharey=True)
for ax, name in zip(axes, ["narrow", "wide"]):
    for layer in LAYERS:
        d = push[(push.model == name) & (push.layer == layer)]
        ax.plot(d.epoch, d.per_param, color=LAYER_COLOR[layer])
    ax.set_yscale("log")
    ax.set_xlabel("epoch")
    ax.set_xlim(0, 68)
    ax.set_title(name, loc="left")
for ax, name in zip(axes, ["narrow", "wide"]):
    last = push[(push.model == name) & (push.epoch == EPOCHS)]
    end_labels(ax, [(EPOCHS, v, layer) for layer, v in zip(last.layer, last.per_param)])
axes[0].set_ylabel("push per parameter\n(mean self-influence)")
plt.show()

# %% [markdown]
# * Narrow `fc2`: pushed 6–9× harder per parameter than its neighbours, and the push keeps growing all run.
# * Wide model: the push stays flat or falls. `fc1`'s push per parameter is high in both models (it sees the raw
#   features), so a high push alone is not a capacity signal. It has to come with saturation and a stalled loss (§25).

# %% [markdown]
# ## 18. Which examples: `self_influence`, `influence(k=…)`, `TracInCPFast`, `TracInCPFastRandProj`
#
# ```python
# plot: mean self_influence on fc2 per class, narrow vs wide
#
# tracin.influence(example, k=5, proponents=True)     # 5 training examples with the largest g[i] · g[example]
# tracin.influence(example, k=5, proponents=False)    # 5 most negative: they pull the other way
# TracInCPFast(model, final_fc_layer=model.fc4, ...)  # the same with only the last layer's gradients (cheap)
# TracInCPFastRandProj(..., projection_dim=8)         # + random projection and nearest-neighbour search,
#                                                     #   for huge training sets
# ```
#
# `TracInCPFastRandProj`'s default index needs the `annoy` package, so a brute-force one is passed in.

# %%
si = {name: layer_push(WIDTH[name], CKPTS[name][EPOCHS], "fc2") for name in WIDTH}
fig, ax = plt.subplots(figsize=(9, 3.2))
for k, name in enumerate(["narrow", "wide"]):
    v = [si[name][PY == c].mean().item() for c in range(N_CLASSES)]
    ax.bar(
        np.arange(N_CLASSES) + (k - 0.5) * 0.38,
        v,
        width=0.38,
        color=MODEL_COLOR[name],
        label=name,
    )
ax.set_yscale("log")
ax.set_xticks(range(N_CLASSES), [f"class {c}" for c in range(N_CLASSES)])
ax.grid(axis="x", visible=False)
ax.set_ylabel("mean self-influence on fc2")
ax.set_title("Which classes still push fc2", loc="left")
ax.legend(loc="upper right")
plt.show()

narrow_path = save_state(CKPTS["narrow"][EPOCHS])
tracin = TracInCP(
    NARROW,
    TensorDataset(PX, PY),
    [narrow_path],
    checkpoints_load_func=load_ckpt,
    layers=["fc2"],
    loss_fn=nn.CrossEntropyLoss(reduction="sum"),
    batch_size=64,
    sample_wise_grads_per_batch=True,
)
hard = int(si["narrow"].argmax())
with torch.no_grad():
    pred = int(NARROW(PX[hard : hard + 1]).argmax())
print(
    f"hardest example for narrow fc2: #{hard}, true class {int(PY[hard])}, predicted {pred}"
)
rows = []
for kind, proponents in [("helps (proponent)", True), ("hurts (opponent)", False)]:
    res = tracin.influence(
        (PX[hard : hard + 1], PY[hard : hard + 1]), k=6, proponents=proponents
    )
    for idx, score in zip(res.indices[0].tolist(), res.influence_scores[0].tolist()):
        if idx != hard:  # an example always helps itself most
            rows.append(
                dict(
                    kind=kind,
                    example=idx,
                    its_class=int(PY[idx]),
                    influence=round(score, 1),
                )
            )
display(pd.DataFrame(rows).groupby("kind").head(5).set_index(["kind", "example"]))


class BruteForceNN(NearestNeighbors):
    """Exact nearest neighbours by dot product, instead of the default annoy index."""

    def setup(self, data):
        self.data = data.reshape(len(data), -1)

    def get_nearest_neighbors(self, query, k):
        scores, idx = (query.reshape(len(query), -1) @ self.data.T).topk(k, dim=1)
        return idx, scores


t0 = time.time()
fast = TracInCPFast(
    NARROW,
    NARROW.fc4,
    TensorDataset(PX, PY),
    [narrow_path],
    checkpoints_load_func=load_ckpt,
    loss_fn=nn.CrossEntropyLoss(reduction="sum"),
    batch_size=64,
)
proj = TracInCPFastRandProj(
    NARROW,
    NARROW.fc4,
    TensorDataset(PX, PY),
    [narrow_path],
    checkpoints_load_func=load_ckpt,
    loss_fn=nn.CrossEntropyLoss(reduction="sum"),
    batch_size=64,
    nearest_neighbors=BruteForceNN(),
    projection_dim=8,
)
queries = (PX[:32], PY[:32])
top_fast = fast.influence(queries, k=5).indices
top_proj = proj.influence(queries, k=5).indices
overlap = np.mean(
    [len(set(a.tolist()) & set(b.tolist())) / 5 for a, b in zip(top_fast, top_proj)]
)
print(
    f"TracInCPFast vs TracInCP(fc2) self-influence ranking: Spearman {spearman(fast.self_influence((PX, PY)), si['narrow']):.2f}"
)
print(
    f"TracInCPFastRandProj top-5 proponents overlap with TracInCPFast: {overlap:.0%}  ({time.time() - t0:.1f}s)"
)

# %% [markdown]
# * Every class still pushes the narrow `fc2`, some far more than others. In the wide model the push is far smaller
#   for every class.
# * The examples that most help or hurt the hardest example mostly belong to other classes. With only 3 units, classes
#   have to share `fc2`'s directions, so fitting one class moves the others.
# * `TracInCPFast` (last layer only) ranks examples much like the full per-layer TracInCP, at a fraction of the cost.
#   `TracInCPFastRandProj` with only 8 projection dimensions finds about a quarter of the same top-5 proponents. Raise
#   `projection_dim` for more accuracy.

# %% [markdown]
# ## 19. `NaiveInfluenceFunction`: classic influence functions
#
# ```python
# H = Hessian of the training loss w.r.t. fc2's weights      # 195 × 195 in narrow
# influence[i, j] = g[i] · H⁻¹ · g[j]                        # TracInCP is the same without H⁻¹
# ```
#
# "Naive" = it forms `H` explicitly, so it is used on the narrow `fc2` only (Koh & Liang, 2017).

# %%
t0 = time.time()
nif = NaiveInfluenceFunction(
    NARROW,
    TensorDataset(PX, PY),
    narrow_path,
    checkpoints_load_func=load_ckpt,
    layers=["fc2"],
    loss_fn=nn.CrossEntropyLoss(reduction="none"),
    batch_size=64,
    projection_dim=None,
)
si_nif = nif.self_influence((PX, PY))
print(
    f"NaiveInfluenceFunction in {time.time() - t0:.1f}s; "
    f"Spearman with TracInCP self-influence: {spearman(si_nif, si['narrow']):.2f}"
)
fig, ax = plt.subplots(figsize=(9, 3))
for v, label, style in [
    (si_nif, "NaiveInfluenceFunction", "bar"),
    (si["narrow"], "TracInCP", "tick"),
]:
    per_class = torch.stack([v[PY == c].mean() for c in range(N_CLASSES)])
    per_class = (per_class / per_class.sum()).numpy()
    if style == "bar":
        ax.bar(
            range(N_CLASSES),
            per_class,
            color=MODEL_COLOR["narrow"],
            width=0.6,
            label=label,
        )
    else:
        ax.hlines(
            per_class,
            np.arange(N_CLASSES) - 0.3,
            np.arange(N_CLASSES) + 0.3,
            color=INK,
            lw=2,
            label=label,
        )
ax.set_xticks(range(N_CLASSES), [f"class {c}" for c in range(N_CLASSES)])
ax.grid(axis="x", visible=False)
ax.set_ylabel("share of self-influence")
ax.set_title("Narrow fc2: influence functions vs TracInCP, per class", loc="left")
ax.legend(loc="upper right")
plt.show()

# %% [markdown]
# The two rank individual examples almost identically. The Hessian shifts the weight between classes a little (more
# on class 5, less on classes 3 and 9). On a model this size, TracInCP's cheaper gradient dot products give the same picture.

# %% [markdown]
# ## 20. `SimilarityInfluence`: nearest training examples inside each layer
#
# ```python
# for layer in [fc1, fc2, fc3, fc4]:
#     a_train = layer output for 2000 training examples
#     a_x     = layer output for one example
#     nearest = the 10 training examples with the highest cosine(a_x, a_train)
#     same_label = share of nearest with the same label as the example          → bar chart
# ```

# %%
SIM_LAYERS = ["fc1", "fc2", "fc3", "fc4"]
knn = {}
for name, model in MODELS.items():
    # captum 0.9: with the default cosine metric, pass more than one layer
    simi = SimilarityInfluence(
        model,
        SIM_LAYERS,
        TensorDataset(X[:2000]),
        os.path.join(WORKDIR, f"sim_{name}"),
        model_id=name,
        batch_size=500,
    )
    found = simi.influence(PX[:64], top_k=10)
    knn[name] = [
        (Y[:2000][found[layer][0]] == PY[:64, None]).float().mean().item()
        for layer in SIM_LAYERS
    ]
fig, ax = plt.subplots(figsize=(7, 3))
for k, name in enumerate(["narrow", "wide"]):
    ax.bar(
        np.arange(4) + (k - 0.5) * 0.38,
        knn[name],
        width=0.38,
        color=MODEL_COLOR[name],
        label=name,
    )
ax.set_xticks(range(4), SIM_LAYERS)
ax.grid(axis="x", visible=False)
ax.set_ylim(0, 1)
ax.set_ylabel("nearest 10 neighbours\nwith the same label")
ax.legend(loc="upper left", ncol=2)
ax.set_title("Do the layers keep the classes apart?", loc="left")
plt.show()

# %% [markdown]
# In the narrow `fc2`, under half of each example's nearest neighbours share its label: the bottleneck squeezes
# different classes onto the same activation patterns. Wide `fc2` keeps them apart as well as the other layers do.

# %% [markdown]
# ## Part 6 · Capacity probes
#
# ## 21. `captum.module` stochastic gates: what the model can do without
#
# ```python
# gates = BinaryConcreteStochasticGates(n, reg_weight=price)    # n learnable on/off gates
# h, penalty = gates(h)                         # h * gates;  penalty = price × expected number of open gates
# loss = cross_entropy(scores, y) + penalty     # fine-tune for 10 epochs
# kept = gates.get_gate_active_probs() > 0.5    # the values worth their price
# ```
#
# * plot 1: gates on the **16 input features** (feature selection), `BinaryConcrete` and `Gaussian` versions
# * plot 2: gates on the outputs of `fc1`, `fc2`, `fc3`, at three prices


# %%
class InputGated(nn.Module):
    def __init__(self, student, gates):
        super().__init__()
        self.m, self.gates, self.l0 = student, gates, 0.0

    def forward(self, x):
        x, self.l0 = self.gates(x)
        return self.m(x)


class HiddenGated(nn.Module):
    def __init__(self, student, make_gates):
        super().__init__()
        self.m, self.l0 = student, 0.0
        self.gates = nn.ModuleDict(
            {
                layer: make_gates(student.get_submodule(layer).out_features)
                for layer in LAYERS
            }
        )

    def forward(self, x):
        m, l0 = self.m, 0.0
        for layer, act in zip(LAYERS, [m.act1, m.act2, m.act3]):
            x, penalty = self.gates[layer](act(m.get_submodule(layer)(x)))
            l0 = l0 + penalty
        self.l0 = l0
        return m.fc4(x)


def finetune_with_gates(gated, epochs=10):
    opt = torch.optim.Adam(
        [
            {"params": gated.m.parameters(), "lr": 1e-3},
            {"params": gated.gates.parameters(), "lr": 0.1},
        ]
    )
    for _ in range(epochs):
        gated.train()
        for idx in torch.randperm(N_TRAIN).split(BATCH):
            loss = F.cross_entropy(gated(X[idx]), Y[idx]) + gated.l0
            opt.zero_grad()
            loss.backward()
            opt.step()
    return gated.eval()


GATES = {
    "BinaryConcreteStochasticGates": BinaryConcreteStochasticGates,
    "GaussianStochasticGates": GaussianStochasticGates,
}
t0 = time.time()
input_open = {}
for kind, cls in GATES.items():
    for name in WIDTH:
        torch.manual_seed(0)
        g = finetune_with_gates(
            InputGated(
                load(WIDTH[name], CKPTS[name][EPOCHS]), cls(N_FEATURES, reg_weight=0.03)
            )
        )
        input_open[(kind, name)] = g.gates.get_gate_active_probs().detach()
PRICES = [0.005, 0.01, 0.02]
hidden_kept = []
for name in WIDTH:
    for price in PRICES:
        torch.manual_seed(0)
        g = finetune_with_gates(
            HiddenGated(
                load(WIDTH[name], CKPTS[name][EPOCHS]),
                lambda n: BinaryConcreteStochasticGates(n, reg_weight=price),
            )
        )
        hidden_kept += [
            dict(
                model=name,
                price=price,
                layer=layer,
                kept=(g.gates[layer].get_gate_active_probs() > 0.5)
                .float()
                .mean()
                .item(),
            )
            for layer in LAYERS
        ]
hidden_kept = pd.DataFrame(hidden_kept)
print(f"gated fine-tunes in {time.time() - t0:.0f}s")

fig, axes = plt.subplots(2, 1, figsize=(10, 4.6), sharex=True)
for ax, kind in zip(axes, GATES):
    for k, name in enumerate(["narrow", "wide"]):
        ax.bar(
            np.arange(N_FEATURES) + (k - 0.5) * 0.38,
            input_open[(kind, name)].numpy(),
            width=0.38,
            color=MODEL_COLOR[name],
            label=name,
        )
    feature_axis(ax, ticklabels=(ax is axes[-1]), group_labels=(ax is axes[0]))
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("P(gate open)")
    ax.set_title(f"{kind} on the inputs (price 0.03)", loc="left")
axes[0].legend(loc="upper right")
plt.show()

fig, axes = plt.subplots(1, 2, figsize=(10, 3.3), sharey=True)
for ax, name in zip(axes, ["narrow", "wide"]):
    for layer in LAYERS:
        d = hidden_kept[(hidden_kept.model == name) & (hidden_kept.layer == layer)]
        ax.plot(d.price, d.kept, marker="o", color=LAYER_COLOR[layer])
    ax.set_xscale("log")
    ax.set_xticks(PRICES, [str(p) for p in PRICES])
    ax.xaxis.set_minor_formatter(plt.NullFormatter())
    ax.set_xlim(0.004, 0.03)
    ax.set_ylim(0, 1.05)
    ax.set_xlabel("price per open unit")
    ax.set_title(f"{name}: BinaryConcreteStochasticGates on hidden units", loc="left")
    last = hidden_kept[(hidden_kept.model == name) & (hidden_kept.price == PRICES[-1])]
    end_labels(ax, [(PRICES[-1], v, layer) for layer, v in zip(last.layer, last.kept)])
axes[0].set_ylabel("units kept (share)")
plt.show()

# %% [markdown]
# * On the inputs, both gate types keep all eight signals and close all the noise. One or two copies stay open: a copy
#   carries almost the same information as its signal, so keeping either costs the same price.
# * On the hidden units, narrow `fc2` keeps all 3 units at every price, while `fc1` and `fc3` shed 25–70% of theirs. In the
#   wide model, `fc2` sheds units like the other layers. The narrow `fc2` cannot afford to lose anything.

# %% [markdown]
# ## 22. `TCAV` (`captum.concept`): which layers still carry each feature
#
# ```python
# concept "s7 high" = examples with s7 > 1;   random = ordinary examples       # one concept per feature
# for layer in [fc1, fc2, fc3]:
#     a_c, a_r = layer output for the concept / random examples
#     accuracy = logistic regression test accuracy, a_c vs a_r        # plot 1 (0.5 = the layer lost it)
#     cav = mean(a_c) - mean(a_r)                                    # the concept's direction
#     tcav_score = share of class-0 examples where (d score[0] / d layer output) · cav > 0      # plot 2
# ```
#
# The classifier is pluggable (`Classifier`). Captum 0.9's default reports ~50% accuracy for every two-concept set
# (it takes an argmax over a single column), so a small one is plugged in.


# %%
class ConceptCAV(Classifier):
    """Readability = logistic-regression test accuracy. Direction = difference of the two sets' mean activations."""

    def __init__(self):
        self._classes, self._weights = None, None

    def train_and_eval(self, dataloader, test_split_ratio=0.33, **kwargs):
        xs, ys = zip(*dataloader)
        x, y = torch.cat(xs), torch.cat(ys)
        perm = torch.randperm(len(y), generator=torch.Generator().manual_seed(0))
        x, y = x[perm], y[perm]
        n_test = int(len(y) * test_split_ratio)
        lr = LogisticRegression(max_iter=2000).fit(
            x[n_test:].numpy(), y[n_test:].numpy()
        )
        self._classes = [int(c) for c in lr.classes_]
        means = torch.stack([x[y == c].mean(0) for c in self._classes])
        self._weights = means - means.mean(0)
        return {"accs": torch.tensor(lr.score(x[:n_test].numpy(), y[:n_test].numpy()))}

    def weights(self):
        return self._weights

    def classes(self):
        return self._classes


class Rows(Dataset):
    def __init__(self, t):
        self.t = t

    def __len__(self):
        return len(self.t)

    def __getitem__(self, i):
        return self.t[i]


pool = make_data(60000, seed=7)
concepts = [
    Concept(
        j, f"{f} high", DataLoader(Rows(pool[pool[:, j] > 1][:1000]), batch_size=500)
    )
    for j, f in enumerate(FEATURES)
]
randoms = [
    Concept(
        100 + k,
        f"random {k}",
        DataLoader(Rows(make_data(1000, seed=200 + k)), batch_size=500),
    )
    for k in range(3)
]
TCAV_CLASS = 0
X_CLASS = X[Y == TCAV_CLASS][:200]  # examples of the class whose score TCAV tests
cav_acc, tcav_score = {}, {}
t0 = time.time()
for name, model in MODELS.items():
    tcav = TCAV(
        model,
        layers=LAYERS,
        model_id=name,
        classifier=ConceptCAV(),
        save_path=os.path.join(WORKDIR, "cav"),
    )
    sets = [[c, r] for c in concepts for r in randoms] + [
        [randoms[1], randoms[0]],
        [randoms[2], randoms[0]],
        [randoms[2], randoms[1]],
    ]
    scores = tcav.interpret(X_CLASS, experimental_sets=sets, target=TCAV_CLASS)
    for c, r in sets:
        for layer in LAYERS:
            cav_acc[(name, layer, c.name, r.name)] = float(
                tcav.cavs[f"{c.id}-{r.id}"][layer].stats["accs"]
            )
            tcav_score[(name, layer, c.name, r.name)] = scores[f"{c.id}-{r.id}"][layer][
                "sign_count"
            ][0].item()
print(f"TCAV in {time.time() - t0:.0f}s")

fig, axes = plt.subplots(1, 2, figsize=(13, 2.6), sharey=True)
for ax, name in zip(axes, ["narrow", "wide"]):
    grid = np.array(
        [
            [
                np.mean([cav_acc[(name, layer, c.name, r.name)] for r in randoms])
                for c in concepts
            ]
            for layer in LAYERS
        ]
    )
    im = feature_heatmap(ax, grid, LAYERS, vmin=0.5, vmax=1.0)
    ax.set_title(
        f"{name}: CAV accuracy ('is feature f high?' readable from the layer)",
        loc="left",
    )
fig.colorbar(im, ax=axes, shrink=0.9, label="accuracy (0.5 = lost)")
plt.show()

# reference for the TCAV score: the model's own input gradient for the class
xt = X_CLASS.clone().requires_grad_(True)
WIDE(xt)[:, TCAV_CLASS].sum().backward()
rises = (xt.grad > 0).float().mean(0)
s = np.array(
    [[tcav_score[("wide", "fc1", c.name, r.name)] for r in randoms] for c in concepts]
)
random_pairs = [
    tcav_score[("wide", "fc1", a.name, b.name)]
    for a, b in [
        (randoms[1], randoms[0]),
        (randoms[2], randoms[0]),
        (randoms[2], randoms[1]),
    ]
]
fig, ax = plt.subplots(figsize=(10.5, 3.3))
ax.bar(
    range(N_FEATURES),
    s.mean(1),
    color=MODEL_COLOR["wide"],
    width=0.6,
    label="TCAV score (mean over 3 random sets, whisker = range)",
)
ax.errorbar(
    range(N_FEATURES),
    s.mean(1),
    yerr=[s.mean(1) - s.min(1), s.max(1) - s.mean(1)],
    fmt="none",
    ecolor=INK2,
    capsize=3,
    lw=1,
)
ax.hlines(
    rises.numpy(),
    np.arange(N_FEATURES) - 0.35,
    np.arange(N_FEATURES) + 0.35,
    color=INK,
    lw=2,
    label=f"model's own gradient: share of class-{TCAV_CLASS} examples whose score rises with the feature",
)
feature_axis(ax)
ax.set_ylim(0, 1.32)
ax.set_yticks([0, 0.25, 0.5, 0.75, 1])
ax.set_ylabel("TCAV score")
ax.set_title(
    f"Wide model, fc1: TCAV score for class {TCAV_CLASS} per concept", loc="left"
)
ax.legend(loc="upper center", fontsize=8.5, ncol=1)
plt.show()
print("TCAV scores of random concepts against each other:", np.round(random_pairs, 2))

# %% [markdown]
# * **Where features survive**: `fc1` still carries every feature in both models, noise included. In the narrow model
#   the weak signals `s5`–`s7` are close to chance from `fc2` on: they were lost in the bottleneck. Noise is gone there
#   too. Wide `fc2` still carries every feature, even the noise: it has room to spare.
# * **TCAV scores** match the model's own gradient direction for all eight signals, with tight whiskers. Concepts are
#   sets of examples, so correlated features travel together: the "`c1` high" examples also have `s1` high, so `c1`
#   gets `s1`'s score.
# * The noise concepts land anywhere from 0 to 1 depending on the random set, and random concepts score 0 or 1 against
#   each other. That is what an unreliable TCAV score looks like: always repeat against several random sets.

# %% [markdown]
# ## 23. `captum.robust`: adversarial checks
#
# ```python
# FGSM:  x_adv = x + ε * sign(d loss / d x)             # one step that hurts the true class most
# PGD:   10 steps of size ε/4, kept within ±ε of x;   mask = attack only the chosen features
# MinParamPerturbation:  smallest ε for which model(x_adv) is wrong      (binary search)
# AttackComparator:      accuracy under each attack, or under any input transform (here: removing feature groups)
# ```

# %%
EPS = [0.05, 0.1, 0.2, 0.4]
fig, axes = plt.subplots(
    1, 2, figsize=(12.5, 3.5), gridspec_kw=dict(width_ratios=[1, 2])
)
for name, model in MODELS.items():
    fgsm, pgd = FGSM(model), PGD(model)
    clean = loss_acc(model, PX, PY)[1]
    for attack, ls in [("FGSM", "-"), ("PGD", "--")]:
        accs = [clean]
        for eps in EPS:
            if attack == "FGSM":
                xa = fgsm.perturb(PX, epsilon=eps, target=PY)
            else:
                xa = pgd.perturb(
                    PX, radius=eps, step_size=eps / 4, step_num=10, target=PY
                )
            accs.append(loss_acc(model, xa, PY)[1])
        axes[0].plot(
            [0] + EPS,
            accs,
            ls=ls,
            marker="o",
            color=MODEL_COLOR[name],
            label=f"{name}, {attack}",
        )
    per_feature = []
    for j in range(N_FEATURES):  # attack one feature at a time
        mask = torch.zeros(1, N_FEATURES)
        mask[0, j] = 1
        per_feature.append(
            loss_acc(
                model,
                pgd.perturb(
                    PX, radius=0.5, step_size=0.1, step_num=10, target=PY, mask=mask
                ),
                PY,
            )[1]
        )
    k = list(WIDTH).index(name)
    axes[1].bar(
        np.arange(N_FEATURES) + (k - 0.5) * 0.38,
        per_feature,
        width=0.38,
        color=MODEL_COLOR[name],
        label=name,
    )
    axes[1].axhline(clean, color=MODEL_COLOR[name], lw=1, ls=":")
axes[0].set_xlabel("ε (max change per feature)")
axes[0].set_ylabel("accuracy under attack")
axes[0].set_title("FGSM and PGD", loc="left")
axes[0].legend(fontsize=8)
feature_axis(axes[1], group_labels=True)
axes[1].set_ylim(0.5, 1.02)
axes[1].set_ylabel("accuracy")
axes[1].set_title("PGD on one feature only (ε = 0.5, dotted = clean)", loc="left")
axes[1].legend(loc="upper left", ncol=2)
plt.show()


def accuracy_metric(out, target):
    return (out.argmax(-1) == target).float().mean()


def remove(cols):
    def transform(inputs):
        inputs = inputs.clone()
        inputs[:, cols] = 0.0
        return inputs

    return transform


rows = {}
for name, model in MODELS.items():
    fgsm = FGSM(model)
    search = MinParamPerturbation(
        model,
        attack=fgsm,
        arg_name="epsilon",
        arg_min=0.0,
        arg_max=2.0,
        arg_step=0.01,
        mode="binary",
    )
    eps_needed = [
        search.evaluate(
            PX[i : i + 1], target=PY[i : i + 1], attack_kwargs={"target": PY[i : i + 1]}
        )[1]
        for i in range(48)
    ]
    eps_needed = [
        e for e in eps_needed if e is not None and e > 0
    ]  # skip examples already wrong
    comparator = AttackComparator(forward_func=model, metric=accuracy_metric)
    comparator.add_attack(
        fgsm,
        "FGSM ε=0.1",
        attack_kwargs={"epsilon": 0.1},
        additional_attack_arg_names=["target"],
    )
    for label, cols in [
        ("remove s0-s3", [0, 1, 2, 3]),
        ("remove s4-s7", [4, 5, 6, 7]),
        ("remove copies", [8, 9, 10, 11]),
        ("remove noise", [12, 13, 14, 15]),
    ]:
        comparator.add_attack(remove(cols), label)
    result = comparator.evaluate(PX, target=PY)
    rows[name] = {
        "MinParamPerturbation: median ε to flip": float(np.median(eps_needed)),
        **{
            k: float(v["mean"] if isinstance(v, dict) else v) for k, v in result.items()
        },
    }
display(pd.DataFrame(rows).rename(index={"Original": "accuracy (no attack)"}))

# %% [markdown]
# * Attacking one feature at a time shows where each model is vulnerable: through the signals (most through `s0`–`s3`),
#   not through copies or noise.
# * A better fit is not a more robust one: the wide model loses accuracy faster under FGSM and PGD and flips at a
#   smaller ε. Robustness checks answer a different question than capacity.
# * `AttackComparator` with "remove a feature group" transforms: removing the copies costs the narrow model about
#   5 points and the wide model 1. The narrow model still leans on the copies.

# %% [markdown]
# ## 24. Expressivity bottleneck: would more units help? (plain PyTorch, not Captum)
#
# Captum explains what the model already does; it can't say what a layer is **missing**. This section asks that
# directly, following TINY (Verbockhaven et al., *Growing Tiny Networks*, TMLR 2024). It needs one pass over 2048
# training examples (16 minibatches): no checkpoints, no retraining.
#
# The question for a layer, e.g. `fc2`: the loss wants `fc3`'s pre-activation to move by `v`. How much of that move
# can **no** weight change reach, because `fc2` doesn't give `fc3` enough directions? And how much would new `fc2`
# units recover?
#
# ```python
# for layer in [fc1, fc2, fc3]:            # e.g. fc2:
#     a = input of layer                   # h1          (n, 64)
#     h = output of layer                  # h2          (n, 3) or (n, 64)
#     v = -grad(loss, next pre-activation) # fc3_pre     (n, 64): where the loss wants it to move, per example
#
#     # 1. retune the next layer's weights as well as possible
#     dW   = pinv(h) @ v                   # (width, 64): the best change to fc3's weights
#     rest = v - h @ dW                    # the part no change to fc3's weights can produce
#     unreachable = sum(rest**2) / sum(v**2)
#
#     # 2. what one new unit in this layer could do (it reads a, writes into fc3_pre)
#     S, N = a.T @ a / n,  a.T @ rest / n
#     lam, U, V = svd(S^-1/2 @ N)
#     headroom = lam[0]**2                 # how fast the loss starts falling if the best new unit is added
#     alpha, omega = S^-1/2 @ U[:, 0] * sqrt(lam[0]),  V[0] * sqrt(lam[0])     # its in- and out-weights
#
#     # 3. check: really add it, as 2 ReLU units  relu(a@alpha)*omega + relu(-a@alpha)*(-omega),
#     #    pick its size on these 2048 examples, then measure the loss on 2048 other training examples
# ```
#
# * plot 1: per layer, both models: unreachable share, headroom, and the measured loss drop from adding the unit
# * plot 2: headroom at every epoch: how early the flag appears


# %%
def with_ones(t):
    return torch.cat(
        [t, torch.ones(len(t), 1)], dim=1
    )  # the constant column stands for the bias


def forward_parts(model, x, extra=None):
    """Layer inputs [x, h1, h2, h3, scores] and pre-activations [z1..z4].
    `extra = (layer, fn)` adds fn(input of layer, output of layer) to the next layer's pre-activation.
    """
    hs, zs = [x], []
    steps = [
        (model.fc1, model.act1),
        (model.fc2, model.act2),
        (model.fc3, model.act3),
        (model.fc4, None),
    ]
    for i, (linear, act) in enumerate(steps):
        z = linear(hs[-1])
        if extra is not None and i == LAYERS.index(extra[0]) + 1:
            z = z + extra[1](hs[-2], hs[-1])
        zs.append(z)
        hs.append(act(z) if act is not None else z)
    return hs, zs


def bottleneck(model, x, y):
    """Per layer: share of the wanted move out of reach, and the best new unit (first order)."""
    hs, zs = forward_parts(model, x)
    for z in zs:
        z.retain_grad()
    model.zero_grad()
    F.cross_entropy(
        zs[-1], y, reduction="sum"
    ).backward()  # sum: each row of v is one example's own gradient
    n, out = len(x), {}
    for i, layer in enumerate(LAYERS):
        a = with_ones(hs[i].detach()).double()  # input of the layer
        h = with_ones(
            hs[i + 1].detach()
        ).double()  # output of the layer = input of the next one
        v = (
            -zs[i + 1].grad.detach().double()
        )  # where the next pre-activation wants to move
        rest = v - h @ torch.linalg.lstsq(h, v).solution
        ev, U = torch.linalg.eigh(a.T @ a / n)
        keep = ev > 1e-6 * ev.max()  # drop empty directions (dead units)
        s_inv_half = U[:, keep] @ torch.diag(ev[keep] ** -0.5) @ U[:, keep].T
        Uk, lam, Vk = torch.linalg.svd(s_inv_half @ (a.T @ rest / n))
        out[layer] = dict(
            unreachable=float((rest**2).sum() / (v**2).sum()),
            headroom=float(lam[0] ** 2),
            alpha=(s_inv_half @ Uk[:, 0] * lam[0].sqrt()).float(),
            omega=(Vk[0] * lam[0].sqrt()).float(),
        )
    return out


SCALES = [0.003, 0.01, 0.03, 0.1, 0.3, 1, 3, 10, 30, 100]


@torch.no_grad()
def grown_loss(model, x, y, layer, unit, scale):
    """Loss with 2 extra ReLU units in `layer`: relu(u)*omega + relu(-u)*(-omega), u = a @ alpha."""

    def add(a, h):
        u = with_ones(a) @ unit["alpha"]
        return scale * (F.relu(u) - F.relu(-u))[:, None] * unit["omega"]

    return F.cross_entropy(forward_parts(model, x, (layer, add))[1][-1], y).item()


def unit_check(model, layer, unit):
    """Size the new units on the fit examples, report the loss drop on the check examples."""
    best = min(SCALES, key=lambda s: grown_loss(model, FIT_X, FIT_Y, layer, unit, s))
    return loss_acc(model, CHECK_X, CHECK_Y)[0] - grown_loss(
        model, CHECK_X, CHECK_Y, layer, unit, best
    )


_order = torch.randperm(N_TRAIN, generator=torch.Generator().manual_seed(4))
FIT_X, FIT_Y = X[_order[:2048]], Y[_order[:2048]]  # 16 minibatches of 128
CHECK_X, CHECK_Y = X[_order[2048:4096]], Y[_order[2048:4096]]  # other training examples

t0 = time.time()
BN = {name: bottleneck(MODELS[name], FIT_X, FIT_Y) for name in WIDTH}
bn_table = pd.DataFrame(
    [
        dict(
            model=name,
            layer=layer,
            unreachable=BN[name][layer]["unreachable"],
            headroom=BN[name][layer]["headroom"],
            loss_drop=unit_check(MODELS[name], layer, BN[name][layer]),
            loss=loss_acc(MODELS[name], CHECK_X, CHECK_Y)[0],
        )
        for name in WIDTH
        for layer in LAYERS
    ]
).set_index(["model", "layer"])
bn_table["drop_%"] = 100 * bn_table.loss_drop / bn_table.loss
print(f"bottleneck + check in {time.time() - t0:.1f}s")
display(bn_table.round(4))

fig, axes = plt.subplots(1, 3, figsize=(12, 3.4))
xs = np.arange(len(LAYERS))
for ax, col, label in [
    (axes[0], "unreachable", "unreachable share of v"),
    (axes[1], "headroom", "headroom (log)"),
    (axes[2], "loss_drop", "loss drop, other examples"),
]:
    for k, name in enumerate(["narrow", "wide"]):
        vals = [bn_table.loc[(name, layer), col] for layer in LAYERS]
        if col == "headroom":
            vals = np.maximum(vals, 1e-6)
        ax.bar(
            xs + (k - 0.5) * 0.38, vals, width=0.38, color=MODEL_COLOR[name], label=name
        )
    ax.set_xticks(
        xs, [f"new {layer} unit" if col != "unreachable" else layer for layer in LAYERS]
    )
    ax.grid(axis="x", visible=False)
    ax.set_ylabel(label)
axes[0].set_ylim(0, 1.05)
axes[0].set_title("1. Out of reach for the next layer", loc="left")
axes[1].set_yscale("log")
axes[1].set_title("2. Headroom of the best new unit", loc="left")
axes[2].axhline(0, color=INK2, lw=1)
axes[2].set_title("3. Measured: add 2 units, no training", loc="left")
for k, name in enumerate(
    ["narrow", "wide"]
):  # mark the layers where the headroom is exactly 0
    for j, layer in enumerate(LAYERS):
        if bn_table.loc[(name, layer), "headroom"] <= 1e-6:
            axes[1].text(
                j + (k - 0.5) * 0.38,
                1.2e-6,
                "0",
                ha="center",
                va="bottom",
                fontsize=9,
                color=INK2,
            )
axes[2].legend(loc="upper right")
plt.show()

# %% [markdown]
# **How early does it show?** The same computation on the checkpoint of every epoch (one pass over the same 2048
# examples each time).

# %%
BN_EPOCHS = [0, 1, 2, 3, 5, 10, 20, 30, 40, 50, 60]
bn_hist = {
    (name, e): bottleneck(load(WIDTH[name], CKPTS[name][e]), FIT_X, FIT_Y)
    for name in WIDTH
    for e in BN_EPOCHS
}

fig, axes = plt.subplots(1, 2, figsize=(11, 3.4), sharey=True)
for ax, name in zip(axes, ["narrow", "wide"]):
    items = []
    for layer in LAYERS:
        ys = [max(bn_hist[(name, e)][layer]["headroom"], 1e-6) for e in BN_EPOCHS]
        ax.plot(BN_EPOCHS, ys, marker="o", ms=4, color=LAYER_COLOR[layer])
        items.append(
            (BN_EPOCHS[-1], ys[-1], layer if ys[-1] > 1e-6 else f"{layer} (0)")
        )
    ax.set_yscale("log")
    ax.set_xlim(-1, 70)
    ax.set_xlabel("epoch")
    ax.set_title(f"{name} model: headroom of a new unit, per layer", loc="left")
    end_labels(ax, items)
axes[0].set_ylabel("headroom (log)")
plt.show()

# %% [markdown]
# * **Plot 1, left: `rest` alone says nothing.** 96–99% of `v` is out of reach in every layer of both models.
#   That is expected: once a layer's weights have converged, their gradient `h.T @ v` is about zero, so `v` has
#   almost nothing in the directions `h` offers. What matters is whether new units can reach the rest.
# * **Plot 1, middle: headroom flags only narrow `fc2`.** It is 16× above narrow `fc1` and over 200× above every
#   layer of the wide model. Narrow `fc3` has none: a new `fc3` unit could only read `fc2`'s 3 numbers, which is
#   where the information was already lost.
# * **Plot 1, right: the prediction holds.** Adding the suggested 2 units to narrow `fc2`, with no training, lowers the
#   loss on other training examples by 10%. Anywhere else the change is zero or negative: there the new unit only
#   fits noise in the 2048 examples it was sized on.
# * **Plot 2: it shows within a few epochs.** An untrained model has headroom everywhere (wide `fc3` starts as high as
#   narrow `fc2`). In the wide model it then falls in every layer. In the narrow model `fc2` leads from epoch 0 and
#   keeps rising while the loss falls; by epoch 3 it is 10× above every layer of the wide model.
# * Use at least ~1000 examples (8 or more minibatches; you can accumulate the sums `a.T @ a`, `h.T @ v`, ... batch by
#   batch). With 512, the estimate of the new unit's 65 × 64 weights picks up noise, and the check fails.
# * It is a first-order, one-unit test: it finds the layer where *one more direction* pays off now. It doesn't say
#   how wide the layer should end up (§26).

# %% [markdown]
# ## Part 7 · Verdict
#
# ## 25. One verdict per layer
#
# ```python
# for layer in [fc1, fc2, fc3], from checkpoints at epochs 40, 50, 60:
#     util, drift     ← LayerConductance (§13, §16)
#     rank            ← width used, LayerActivation (§12)
#     E_rel, E_trend  ← TracInCP push vs the median layer, and last / first checkpoint (§17)
# loss_trend = probe loss, last / first checkpoint
#
# if drift >= 0.02 or (loss_trend < 0.75 and E_trend >= 0.8):    → NOT CONVERGED
# elif util >= 0.7 and rank >= 0.7 and E_rel >= 2 and E_trend >= 0.8:      → CAPACITY-BOUND
# elif many idle units:                                           → UNDER-USED
# else:                                       → converging normally (push shrinking) or settled
# ```
#
# The thresholds are heuristics set on this toy problem. Use them to rank layers, and recalibrate on your own model.


# %%
@dataclass
class Thresholds:
    loss_improving: float = 0.75  # probe loss last / first below this = still improving
    drift: float = 0.02  # 1 - cos of unit importance between the last two checkpoints
    push_shrinking: float = 0.80  # E_trend below this = push shrinking
    saturated_util: float = 0.70
    saturated_rank: float = 0.70
    idle: float = 0.30
    push_rel: float = 2.0  # push per parameter vs the median layer


def verdict(r, th, loss_trend):
    improving = loss_trend < th.loss_improving
    shrinking = r["E_trend"] < th.push_shrinking
    saturated = r["util"] >= th.saturated_util and r["rank"] >= th.saturated_rank
    pulled = r["E_rel"] >= th.push_rel and not shrinking
    if r["drift"] >= th.drift or (improving and not shrinking):
        return "NOT CONVERGED" + (", saturated" if saturated else "")
    if saturated and pulled:
        return "CAPACITY-BOUND"
    if r["idle"] >= th.idle:
        return "UNDER-USED"
    if saturated:
        return "saturated, still improving" if shrinking else "saturated, settled"
    if pulled:
        return "large flat push"
    return "converging normally" if shrinking else "settled"


def diagnose(width, states, th=Thresholds()):
    """One row per layer from checkpoints `states` (oldest first, at least 2)."""
    first, prev, last = (
        load(width, states[0]),
        load(width, states[-2]),
        load(width, states[-1]),
    )
    loss_trend = loss_acc(last, PX, PY)[0] / loss_acc(first, PX, PY)[0]
    push_first = {l: layer_push(width, states[0], l).mean().item() for l in LAYERS}
    push_last = {l: layer_push(width, states[-1], l).mean().item() for l in LAYERS}
    per_param = {l: push_last[l] / n_params(width, l) for l in LAYERS}
    median = float(np.median(list(per_param.values())))
    rows = []
    for l in LAYERS:
        imp_prev, imp = unit_importance(prev, l), unit_importance(last, l)
        r = dict(
            layer=l,
            util=entropy_count(imp) / len(imp),
            idle=float((imp < 0.05 * imp.mean()).float().mean()),
            rank=width_used(last, l),
            drift=1 - F.cosine_similarity(imp, imp_prev, dim=0).item(),
            E_rel=per_param[l] / median,
            E_trend=push_last[l] / push_first[l],
        )
        r["verdict"] = verdict(r, th, loss_trend)
        rows.append(r)
    out = pd.DataFrame(rows).set_index("layer")
    out.attrs["loss_trend"] = loss_trend
    return out


SCENARIOS = {
    "wide, early (epochs 1, 2, 3)": ("wide", [1, 2, 3]),
    "wide, late (epochs 40, 50, 60)": ("wide", [40, 50, 60]),
    "narrow, late (epochs 40, 50, 60)": ("narrow", [40, 50, 60]),
}
reports = {}
for title, (name, epochs) in SCENARIOS.items():
    reports[title] = diagnose(WIDTH[name], [CKPTS[name][e] for e in epochs])
    print(
        f"\n{title}: probe loss last/first = {reports[title].attrs['loss_trend']:.2f}"
    )
    display(reports[title].round(3))

# %% [markdown]
# **Everything on the architecture.** Late checkpoints, one line per tool. Input box: the share of the signals'
# information captured (§11).


# %%
def notes_for(name):
    report = reports[f"{name}, late (epochs 40, 50, 60)"]
    captured = (drop[name][:8] / drop["teacher (ceiling)"][:8]) * 100
    notes = {
        "input": f"signals captured\n{captured.min():.0f}–{captured.max():.0f}%",
        "fc4": f"train accuracy\n{HIST[name].acc.iloc[-1]:.2f}",
    }
    for i, layer in enumerate(LAYERS):
        cost = unit_cost[("LayerFeatureAblation", name, layer)].mean().item()
        kept = hidden_kept[
            (hidden_kept.model == name)
            & (hidden_kept.layer == layer)
            & (hidden_kept.price == 0.01)
        ].kept.item()
        weak = np.mean(
            [
                cav_acc[(name, layer, f"{f} high", r.name)]
                for f in ["s5", "s6", "s7"]
                for r in randoms
            ]
        )
        notes[layer] = (
            f"width used {report.loc[layer, 'rank']:.2f}\n"
            f"1-unit ablation {cost:.2f}\n"
            f"push {report.loc[layer, 'E_rel']:.1f}× median\n"
            f"same-label kNN {knn[name][i]:.2f}\n"
            f"gates kept {kept:.0%}\n"
            f"s5–s7 readable {weak:.2f}\n"
            f"headroom {BN[name][layer]['headroom']:.4f}\n"
            f"{report.loc[layer, 'verdict']}"
        )
    return notes


fig, axes = plt.subplots(2, 1, figsize=(11.5, 10))
for ax, name in zip(axes, ["narrow", "wide"]):
    report = reports[f"{name}, late (epochs 40, 50, 60)"]
    flagged = [l for l in LAYERS if report.loc[l, "verdict"] == "CAPACITY-BOUND"]
    draw_architecture(
        ax,
        WIDTH[name],
        f"{name} model",
        notes=notes_for(name),
        flag=flagged[0] if flagged else None,
    )
plt.show()

# %% [markdown]
# * **wide, early**: every layer is *not converged*. The loss is still falling fast and every layer's push grew.
#   Drift is already small, so drift alone would have missed this run.
# * **wide, late**: every layer is *converging normally*: the loss still falls and the push shrinks. `fc1`'s push is
#   high but shrinking, so it is not flagged.
# * **narrow, late**: the loss has stalled and only `fc2` is *capacity-bound*. Every tool agrees: full width, costly
#   units, heavy push, mixed-up classes, no gate closes, weak signals lost, and the only real headroom (§24).

# %% [markdown]
# ## 26. Does widening the flagged layer help?
#
# ```python
# for width in [3, 6, 10, 16, 32, 64]:                # 3 and 64 are the two models above
#     model = train(Student(width))
#     verdict = diagnose(model, checkpoints at epochs 40, 50, 60)
#     headroom = bottleneck(model)["fc2"]                       # §24
# plot: final training loss, fc2 push, fc2 width used and fc2 headroom, against width
# ```
#
# A capacity-bound verdict is a hypothesis: widening that layer should lower the **training** loss.

# %%
t0 = time.time()
sweep = []
for w in [3, 6, 10, 16, 32, 64]:
    if w in WIDTH.values():
        name = [k for k, v in WIDTH.items() if v == w][0]
        ckpts, hist = CKPTS[name], HIST[name]
    else:
        ckpts, hist = train(w)
    r = diagnose(w, [ckpts[e] for e in [40, 50, 60]])
    grown = load(w, ckpts[EPOCHS])
    b = bottleneck(grown, FIT_X, FIT_Y)["fc2"]
    sweep.append(
        dict(
            width=w,
            train_loss=hist.loss.iloc[-1],
            loss_trend=r.attrs["loss_trend"],
            fc2_headroom=b["headroom"],
            fc2_unit_drop=unit_check(grown, "fc2", b),
            fc2_rank=r.loc["fc2", "rank"],
            fc2_E_rel=r.loc["fc2", "E_rel"],
            fc2_E_trend=r.loc["fc2", "E_trend"],
            fc2_verdict=r.loc["fc2", "verdict"],
            other_layers=", ".join(sorted(set(r.loc[["fc1", "fc3"], "verdict"]))),
        )
    )
sweep = pd.DataFrame(sweep).set_index("width")
print(f"width sweep in {time.time() - t0:.0f}s")
display(sweep.round(3))

MARKER = {
    "capacity-bound": dict(fillstyle="full", markerfacecolor=LAYER_COLOR["fc2"]),
    "not converged": dict(
        fillstyle="left", markerfacecolor=LAYER_COLOR["fc2"], markerfacecoloralt=SURFACE
    ),
    "has headroom": dict(fillstyle="full", markerfacecolor=SURFACE),
}
group = sweep.fc2_verdict.map(
    lambda v: (
        "capacity-bound"
        if v.startswith("CAPACITY")
        else "not converged" if v.startswith("NOT") else "has headroom"
    )
)
fig, axes = plt.subplots(1, 4, figsize=(14, 3.4), sharex=True)
for ax, col, label, log in [
    (axes[0], "train_loss", "final training loss (log)", True),
    (axes[1], "fc2_E_rel", "fc2 push vs median layer", False),
    (axes[2], "fc2_rank", "fc2 width used", False),
    (axes[3], "fc2_headroom", "fc2 headroom (log, §24)", True),
]:
    ax.plot(sweep.index, sweep[col], color=LAYER_COLOR["fc2"], lw=1.5, zorder=1)
    for g, style in MARKER.items():
        sel = group == g
        if sel.any():
            ax.plot(
                sweep.index[sel],
                sweep[col][sel],
                ls="none",
                marker="o",
                ms=9,
                color=LAYER_COLOR["fc2"],
                markeredgewidth=1.5,
                zorder=2,
                label=g,
                **style,
            )
    ax.set_xscale("log", base=2)
    ax.set_xticks(sweep.index, [str(w) for w in sweep.index])
    ax.xaxis.set_minor_formatter(plt.NullFormatter())
    ax.grid(axis="x", visible=False)
    ax.set_xlabel("fc2 width")
    ax.set_ylabel(label)
    if log:
        ax.set_yscale("log")
axes[1].axhline(Thresholds().push_rel, color=INK2, lw=1, ls=":")
axes[2].axhline(Thresholds().saturated_rank, color=INK2, lw=1, ls=":")
axes[0].legend(title="fc2 verdict", title_fontsize=9)
plt.show()

# %% [markdown]
# * Widths 3 and 6 are flagged *capacity-bound*, and widening them does lower the training loss (0.47 → 0.12 → 0.08).
# * From width 10 the flag turns off: `fc2` no longer saturates, and the remaining runs just need more training.
# * `fc2` headroom (§24) falls steadily with width, 250× from 3 to 64, from one pass over the final model. Only at
#   width 3 does its suggested unit pay off at once (loss −0.048). At width 6 it gains just 0.001, although retraining
#   at width 10 lowers the loss by 0.04: a one-unit, first-order test underestimates what a retrain finds.

# %% [markdown]
# ## 27. Using this on your own model
#
# ```python
# states = checkpoints from late in training, oldest first      # e.g. epochs 40, 50, 60
# probe  = 200–500 training examples, the same at every checkpoint
#
# 1. features:  DataLoaderAttribution(FeatureAblation(model)) at each checkpoint          # §11
#               still rising = still learning
# 2. layers:    TracInCP push per layer (§17), LayerActivation width used (§12),
#               LayerFeatureAblation per unit (§14)
# 3. confirm:   stochastic gates on the suspect layer (§21), TCAV readability (§22),
#               expressivity bottleneck (§24): one pass, no checkpoints, gives the new unit's weights
#               then widen that layer and retrain: training loss should drop (§26)
# ```
#
# Keep in mind:
#
# * Checkpoints should be 5–30% of the run apart.
# * For `(N, T, D)` outputs (transformers), units are on dim 2. For other tasks, change the target and the loss.
# * Run with `model.eval()` (dropout off, BatchNorm frozen).
#
# **Captum features not shown**, and why:
#
# | Feature | Why not here |
# |---|---|
# | `LayerGradCam`, `GuidedGradCam` | need convolutional feature maps |
# | `LLMAttribution`, `LLMGradientAttribution`, `RemoteLLMAttribution`, `TextTemplateInput`, `TextTokenInput` | text generation |
# | `ImageMaskInput` | image segments |
# | `configure_interpretable_embedding_layer`, `TokenReferenceBase` | embedding layers |
# | `ProductBaselines` | sampled baselines for structured (mostly text) inputs |
# | `captum.attr.visualization` | plotting helpers for images, text and time series |

# %%
print(f"Total runtime: {time.time() - T_START:.0f}s")
