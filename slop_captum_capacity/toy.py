"""Shared setup for the Captum capacity notebooks (toy problem).

Imports, plot style, plot helpers, the data (16 features, hidden teacher), the two student models (narrow and
wide fc2), training with a disk cache, the probe set, and the measurement functions several notebooks reuse.

Notebooks start with `from toy import *`.
"""
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

from expressivity import Site, check, grow, linear_sites, mean_loss, measure

from captum.attr import (
    ClassSummarizer, CommonStats, DataLoaderAttribution, Deconvolution, DeepLift, DeepLiftShap,
    FeatureAblation, FeaturePermutation, GradientShap, GuidedBackprop, InputXGradient,
    IntegratedGradients, InternalInfluence, KernelShap, LayerActivation, LayerConductance,
    LayerDeepLift, LayerDeepLiftShap, LayerFeatureAblation, LayerFeaturePermutation,
    LayerGradientShap, LayerGradientXActivation, LayerIntegratedGradients, LayerLRP, Lime, LRP,
    NeuronConductance, NeuronDeconvolution, NeuronDeepLift, NeuronDeepLiftShap,
    NeuronFeatureAblation, NeuronGradient, NeuronGradientShap, NeuronGuidedBackprop,
    NeuronIntegratedGradients, NoiseTunnel, Occlusion, Saliency, ShapleyValues, ShapleyValueSampling,
)
from captum.concept import TCAV, Classifier, Concept
from captum.influence import (
    NaiveInfluenceFunction, SimilarityInfluence, TracInCP, TracInCPFast, TracInCPFastRandProj,
)
from captum.influence._utils.nearest_neighbors import NearestNeighbors
from captum.metrics import infidelity, infidelity_perturb_func_decorator, sensitivity_max
from captum.module import BinaryConcreteStochasticGates, GaussianStochasticGates
from captum.robust import FGSM, PGD, AttackComparator, MinParamPerturbation

try:
    from IPython.display import display
except ImportError:  # plain script
    display = print

warnings.filterwarnings("ignore")
pd.set_option("display.precision", 3)
pd.set_option("display.width", 200)
WORKDIR = tempfile.mkdtemp(prefix="captum_tour_")  # Captum writes checkpoints / activations here


# Plot style. Colours: one per layer, one per model; blue/red = sign of an attribution.
SURFACE, INK, INK2, GRID, BAND, NEUTRAL = "#fcfcfb", "#0b0b0b", "#52514e", "#e6e5e1", "#efeeea", "#a3a29b"
POS, NEG = "#2a78d6", "#e34948"
LAYER_COLOR = {"fc1": "#2a78d6", "fc2": "#eb6834", "fc3": "#1baf7a"}
MODEL_COLOR = {"narrow": "#6250d6", "wide": "#008300"}
BLUES = LinearSegmentedColormap.from_list(
    "blues", ["#f7f9fc", "#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"])
DIVERGING = LinearSegmentedColormap.from_list("div", ["#b22c2c", "#e34948", "#f0efec", "#2a78d6", "#184f95"])
plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "axes.edgecolor": INK2, "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2,
    "text.color": INK, "axes.titlecolor": INK, "axes.titlesize": 10.5, "axes.titleweight": "bold",
    "axes.labelsize": 9.5, "xtick.labelsize": 9, "ytick.labelsize": 9,
    "legend.fontsize": 9, "legend.frameon": False,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "axes.grid.axis": "y", "grid.color": GRID, "grid.linewidth": 0.8,
    "axes.axisbelow": True, "lines.linewidth": 2, "lines.markersize": 6,
    "figure.dpi": 110, "figure.constrained_layout.use": True,
})

FEATURES = [f"s{i}" for i in range(8)] + [f"c{i}" for i in range(4)] + [f"n{i}" for i in range(4)]
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


def feature_heatmap(ax, values, row_labels, cmap=BLUES, vmin=None, vmax=None, group_labels=True):
    """Heatmap whose columns are the 16 features."""
    im = ax.imshow(values, aspect="auto", cmap=cmap, vmin=vmin, vmax=vmax, interpolation="nearest")
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
    ax.annotate(text, (x, y), xytext=(5, 0), textcoords="offset points", va="center", fontsize=9, color=INK2)


def end_labels(ax, items):
    """Label line ends [(x, y, text)], nudged apart so they never overlap. Call after the y-scale is set."""
    log = ax.get_yscale() == "log"
    f, inv = (np.log10, lambda v: 10 ** v) if log else (lambda v: v, lambda v: v)
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


# ---------------------------------------------------------------- data

N_CLASSES, N_TRAIN, RHO = 10, 8000, 0.9
STRENGTH = [1.0, 0.85, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2]  # how strongly the teacher uses each signal


def make_data(n, seed):
    g = torch.Generator().manual_seed(seed)
    s = torch.randn(n, 8, generator=g)
    c = RHO * s[:, :4] + math.sqrt(1 - RHO ** 2) * torch.randn(n, 4, generator=g)
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
        xp[:, j] = xp[torch.randperm(len(x), generator=torch.Generator().manual_seed(j)), j]
        out.append((model(xp).argmax(-1) != y).float().mean().item())
    return torch.tensor(out)


torch.manual_seed(0)
teacher = Teacher()
X = make_data(N_TRAIN, seed=0)
with torch.no_grad():
    logits = teacher(X)
    shift = torch.zeros(N_CLASSES)
    for _ in range(800):  # nudge the class biases until the 10 classes are equally common
        freq = torch.bincount((logits + shift).argmax(-1), minlength=N_CLASSES).float() / N_TRAIN
        shift -= 0.02 * (freq - 1 / N_CLASSES) * N_CLASSES
    teacher.l2.bias += shift
    order = label_flip_rate(teacher, X, teacher(X).argmax(-1))[:8].argsort(descending=True)
    teacher.l1.weight.copy_(teacher.l1.weight[:, order])  # rename the signals so that s0 matters most
    Y = teacher(X).argmax(-1)
TRUTH = label_flip_rate(teacher, X, Y)
TRUTH_SHARE = TRUTH / TRUTH.sum()


# ---------------------------------------------------------------- the two models

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
    layers = [("input", N_FEATURES, "16 features"), ("fc1", HIDDEN, f"{HIDDEN} units"),
              ("fc2", width, f"{width} units"), ("fc3", HIDDEN, f"{HIDDEN} units"),
              ("fc4", N_CLASSES, "10 class scores")]
    gap = 2.5
    for i, (name, n, label) in enumerate(layers):
        x, h = i * gap, 0.3 * math.sqrt(n)
        ax.add_patch(Rectangle((x - 0.5, -h / 2), 1.0, h, facecolor=LAYER_COLOR.get(name, NEUTRAL),
                               edgecolor=INK if name == flag else SURFACE, lw=3 if name == flag else 1))
        ax.text(x, h / 2 + 0.1, name, ha="center", va="bottom", fontsize=10.5, weight="bold")
        ax.text(x, -h / 2 - 0.1, label, ha="center", va="top", fontsize=9, color=INK2)
        if notes and name in notes:
            ax.text(x, -1.75, notes[name], ha="center", va="top", fontsize=8.5, linespacing=1.45,
                    weight="bold" if name == flag else "normal")
        if i < len(layers) - 1:
            ax.add_patch(FancyArrowPatch((x + 0.6, 0), (x + gap - 0.6, 0), arrowstyle="-|>",
                                         mutation_scale=12, color=INK2, lw=1.2))
            if i > 0:
                ax.text(x + gap / 2, 0.08, "ReLU", ha="center", va="bottom", fontsize=8, color=INK2)
    ax.set_xlim(-1.3, (len(layers) - 1) * gap + 1.3)
    ax.set_ylim(-4.2 if notes else -1.8, 1.6)
    ax.set_title(title, loc="left")
    ax.axis("off")


# ---------------------------------------------------------------- training

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



CACHE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "cache")


def trained(width):
    """train(width), cached on disk: the first notebook trains, the others load."""
    path = os.path.join(CACHE, f"toy_width{width}.pt")
    if os.path.exists(path):
        return torch.load(path, weights_only=False)
    os.makedirs(CACHE, exist_ok=True)
    out = train(width)
    torch.save(out, path)
    return out


CKPTS, HIST = {}, {}
for _name, _w in WIDTH.items():
    CKPTS[_name], HIST[_name] = trained(_w)
MODELS = {name: load(WIDTH[name], CKPTS[name][EPOCHS]) for name in WIDTH}
NARROW, WIDE = MODELS["narrow"], MODELS["wide"]

# the probe set used by every Captum tool, and baselines for the *Shap methods
PROBE = torch.randperm(N_TRAIN, generator=torch.Generator().manual_seed(2))[:256]
PX, PY = X[PROBE], Y[PROBE]
DIST = X[torch.randperm(N_TRAIN, generator=torch.Generator().manual_seed(3))[:16]]



# ---------------------------------------------------------------- measurements reused by several notebooks

EVAL_LOADER = DataLoader(TensorDataset(X[:4000], Y[:4000]), batch_size=1000)


def count_correct(total, out, inputs):  # `reduce`: accumulate over batches
    correct = (out.argmax(-1) == inputs[1]).sum().float()
    return correct if total is None else total + correct


def accuracy_drop(model):
    """Accuracy lost when each feature is set to 0 for the whole dataset (metric before − after)."""
    dla = DataLoaderAttribution(FeatureAblation(model))
    # input_roles: 0 = attribute this input, 2 = do not pass it to the model (the labels).
    # (The captum 0.9 docstring swaps roles 1 and 2; the code uses 2 = "not passed".)
    attr = dla.attribute(EVAL_LOADER, input_roles=(0, 2), baselines=(torch.zeros(1, N_FEATURES),),
                         reduce=count_correct, to_metric=lambda c: (c / len(EVAL_LOADER.dataset)).reshape(1),
                         return_input_shape=False)
    return attr.detach().reshape(-1)


def layer_output(model, layer, x=PX):
    return LayerActivation(model, model.get_submodule(layer)).attribute(x).detach()


def width_used(model, layer):
    a = layer_output(model, layer).double()
    eig = torch.linalg.svdvals(a - a.mean(0)) ** 2
    return entropy_count(eig) / a.shape[1]


def dead_fraction(model, layer):
    return float(((layer_output(model, layer) <= 0).float().mean(0) >= 0.99).float().mean())


def unit_importance(model, layer, n_steps=32):
    """Mean |LayerConductance| per unit of `layer`."""
    return LayerConductance(model, model.get_submodule(layer)).attribute(PX, target=PY, n_steps=n_steps).detach().abs().mean(0)


def true_class_score(model, y=PY):
    # captum 0.9's LayerFeaturePermutation rejects tensor targets, so return the score directly
    return lambda x: model(x).gather(1, y[:, None]).squeeze(1)


def unit_cost_of(model, layer, method="LayerFeatureAblation"):
    """Mean |true-class score change| when each unit of `layer` is switched off (or shuffled)."""
    mod = model.get_submodule(layer)
    if method == "LayerFeatureAblation":
        return LayerFeatureAblation(model, mod).attribute(PX, target=PY).detach().abs().mean(0)
    torch.manual_seed(0)
    return LayerFeaturePermutation(true_class_score(model), mod).attribute(PX).detach().abs().mean(0)


EPOCH_GRID = list(range(0, EPOCHS + 1, 2))

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
    tracin = TracInCP(Student(width), TensorDataset(PX, PY), checkpoints=[path], checkpoints_load_func=load_ckpt,
                      layers=[layer], loss_fn=nn.CrossEntropyLoss(reduction="sum"), batch_size=64,
                      sample_wise_grads_per_batch=True)  # batched per-example gradients (Linear/Conv layers)
    out = tracin.self_influence((PX, PY))
    os.remove(path)
    return out


def n_params(width, layer):
    return sum(p.numel() for p in Student(width).get_submodule(layer).parameters())


SIM_LAYERS = ["fc1", "fc2", "fc3", "fc4"]


def same_label_knn(name, model):
    """Per layer: share of each probe example's 10 nearest training examples (cosine) with the same label."""
    # captum 0.9: with the default cosine metric, pass more than one layer
    simi = SimilarityInfluence(model, SIM_LAYERS, TensorDataset(X[:2000]), os.path.join(WORKDIR, f"sim_{name}"),
                               model_id=name, batch_size=500)
    found = simi.influence(PX[:64], top_k=10)
    return [(Y[:2000][found[layer][0]] == PY[:64, None]).float().mean().item() for layer in SIM_LAYERS]


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
        self.gates = nn.ModuleDict({layer: make_gates(student.get_submodule(layer).out_features) for layer in LAYERS})

    def forward(self, x):
        m, l0 = self.m, 0.0
        for layer, act in zip(LAYERS, [m.act1, m.act2, m.act3]):
            x, penalty = self.gates[layer](act(m.get_submodule(layer)(x)))
            l0 = l0 + penalty
        self.l0 = l0
        return m.fc4(x)


def finetune_with_gates(gated, epochs=10):
    opt = torch.optim.Adam([{"params": gated.m.parameters(), "lr": 1e-3}, {"params": gated.gates.parameters(), "lr": 0.1}])
    for _ in range(epochs):
        gated.train()
        for idx in torch.randperm(N_TRAIN).split(BATCH):
            loss = F.cross_entropy(gated(X[idx]), Y[idx]) + gated.l0
            opt.zero_grad()
            loss.backward()
            opt.step()
    return gated.eval()


GATES = {"BinaryConcreteStochasticGates": BinaryConcreteStochasticGates, "GaussianStochasticGates": GaussianStochasticGates}


def hidden_gates_kept(prices):
    """Fine-tune each model with BinaryConcrete gates on fc1-fc3 outputs; share of units kept per price."""
    rows = []
    for name in WIDTH:
        for price in prices:
            torch.manual_seed(0)
            g = finetune_with_gates(HiddenGated(load(WIDTH[name], CKPTS[name][EPOCHS]),
                                                lambda n: BinaryConcreteStochasticGates(n, reg_weight=price)))
            rows += [dict(model=name, price=price, layer=layer,
                          kept=(g.gates[layer].get_gate_active_probs() > 0.5).float().mean().item()) for layer in LAYERS]
    return pd.DataFrame(rows)


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
        lr = LogisticRegression(max_iter=2000).fit(x[n_test:].numpy(), y[n_test:].numpy())
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



_pool = make_data(60000, seed=7)
concepts = [Concept(j, f"{f} high", DataLoader(Rows(_pool[_pool[:, j] > 1][:1000]), batch_size=500))
            for j, f in enumerate(FEATURES)]
randoms = [Concept(100 + k, f"random {k}", DataLoader(Rows(make_data(1000, seed=200 + k)), batch_size=500))
           for k in range(3)]
TCAV_CLASS = 0
X_CLASS = X[Y == TCAV_CLASS][:200]  # examples of the class whose score TCAV tests


def run_tcav(name, model, sets):
    """TCAV on fc1-fc3. Returns {(model, layer, concept, random): CAV accuracy} and the same for TCAV scores."""
    tcav = TCAV(model, layers=LAYERS, model_id=name, classifier=ConceptCAV(), save_path=os.path.join(WORKDIR, "cav"))
    scores = tcav.interpret(X_CLASS, experimental_sets=sets, target=TCAV_CLASS)
    acc, score = {}, {}
    for c, r in sets:
        for layer in LAYERS:
            acc[(name, layer, c.name, r.name)] = float(tcav.cavs[f"{c.id}-{r.id}"][layer].stats["accs"])
            score[(name, layer, c.name, r.name)] = scores[f"{c.id}-{r.id}"][layer]["sign_count"][0].item()
    return acc, score


# expressivity bottleneck (bottleneck.py) on the toy models
SITES = [Site("fc1", "fc2"), Site("fc2", "fc3"), Site("fc3", "fc4")]  # layer whose width is tested -> layer reading it
_order = torch.randperm(N_TRAIN, generator=torch.Generator().manual_seed(4))
FIT = [(X[i], Y[i]) for i in _order[:2048].split(BATCH)]           # 16 minibatches of 128
CHECK = [(X[i], Y[i]) for i in _order[2048:4096].split(512)]       # other training examples
