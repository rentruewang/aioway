"""
catalog.py: every method in the original list, as a typed signature over the types of data.py, grouped by
signature. The interface only: bodies are left to the implementation (torch, torchrl, your own).

  structure   convert    C[a] → C′[a]          change the container, keep what the elements are
              map        C[a] → C[b]           change each element on its own, keep the container
              aggregate  C[h] → C[h]           each element reads the elements it is related to
              select     C[a] → C[a]           keep, drop, reorder or reweight elements
              fold       C[a] → a              reduce a container to one value
              scan       Seq[X] → Seq[Y]       carry a state along a sequence: Scan[X, S, Y]
  learning    split      C → Example           hide part of a container (Holes); what was hidden is the target
              score      Prediction × C → ℝ    how bad a prediction is: every loss, every surrogate objective
  partial     fill       Holes[C] → C          fill the holes: generation is split undone; state Holes[C]
              walk       PartialTree → Path    grow the known part of a tree, then read a path; state PartialTree
              unroll     PartialSeq[S] → S     step any state: search distributions, rounds of a method
              reshape    Partial → Partial     change how a partial container goes on: its options, its model
  models      adapt      Module → Module       change a model

A method that carries a state says so in its type: Step[S] is one step of a PartialSeq over S (a chooser is a
Step[PartialTree], a schedule a Step[Holes]); Scan[X, S, Y] carries S along a stream (an optimizer carries the
moments). The state column of the printed catalog names it, and what the method keeps there.

Each entry is model-agnostic (any model is an opaque function to it, or it uses none) or model-dependent (it is
a layer, or it opens a model up). Phenomena are observed rather than computed; the meta-primitives are
properties of the groups; the well-known works are programs of entries (program.py).

@method(group, name, state=...) registers a stub defined here: its signature comes from its annotations, its
explanation from its docstring. alias(...) registers one that torch already has.

    python catalog.py             prints the catalog, one line per method
    python catalog.py --explain   with what each one does
"""

from __future__ import annotations

import inspect
import re
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any

import torch
from torch import Tensor, nn

from data import (
    Chooser, Container, Dist, Environment, Example, Graph, Grid, Holes, Model, Predictor, PartialSeq, PartialTree, Path, Prediction, Product, Scan, Schedule, Seq,
    Set, Split, Step, Value,
)

# ═══ Registry ═════════════════════════════════════════════════════════════════════════════════════


@dataclass(frozen=True)
class Group:
    name: str
    family: str
    signature: str
    explanation: str
    side: str  # the usual side of its entries


GROUPS: dict[str, Group] = {g.name: g for g in (
    Group("convert", "structure", "C[a] → C′[a]", "change the container, keep what the elements are", "agnostic"),
    Group("map", "structure", "C[a] → C[b]", "change each element on its own, keep the container", "model"),
    Group("aggregate", "structure", "C[h] → C[h]", "each element reads the elements it is related to", "model"),
    Group("select", "structure", "C[a] → C[a]", "keep, drop, reorder or reweight elements", "agnostic"),
    Group("fold", "structure", "C[a] → a", "reduce a container to one value", "agnostic"),
    Group("scan", "structure", "Seq[X] → Seq[Y]", "carry a state along a sequence", "agnostic"),
    Group("split", "learning", "C → Example", "hide part of a container; what was hidden (or a teacher's answer) is the target", "agnostic"),
    Group("score", "learning", "Prediction × C → ℝ", "how bad a prediction is: every loss, every surrogate objective", "agnostic"),
    Group("fill", "partial", "Holes[C] → C", "fill the holes, a schedule deciding which and how fast: a split, undone", "agnostic"),
    Group("walk", "partial", "PartialTree → Path", "grow the known part of a tree, a chooser deciding which leaves to keep", "agnostic"),
    Group("unroll", "partial", "PartialSeq[S] → S", "step a state until done", "agnostic"),
    Group("reshape", "partial", "Partial → Partial", "change how a partial container goes on: its options, its model", "agnostic"),
    Group("adapt", "models", "Module → Module", "change a model", "model"),
)}
FAMILIES = {"structure": "STRUCTURE · data → data", "learning": "LEARNING · data → a training signal",
            "partial": "PARTIAL CONTAINERS · known in part, and how to get the rest", "models": "MODELS · model → model"}


@dataclass(frozen=True)
class Entry:
    group: str          # a key of GROUPS, or "phenomenon" / "meta"
    name: str
    signature: str      # its type; for "meta", the groups it is about
    state: str          # the type of the state it carries, and what it keeps there ("" when it carries none)
    side: str           # "agnostic" | "model" | "" (phenomena, meta)
    function: str       # a name defined here, or a dotted path (comma-separated)
    explanation: str
    obj: Any = None     # the stub, when defined here


ENTRIES: list[Entry] = []


def _top_level(args: str) -> list[str]:
    """Split "A, B[C, D], E" at the commas that are not inside brackets."""
    out, depth, start = [], 0, 0
    for i, ch in enumerate(args):
        depth += 1 if ch == "[" else -1 if ch == "]" else 0
        if ch == "," and depth == 0:
            out.append(args[start:i].strip())
            start = i + 1
    return out + [args[start:].strip()]


COLLAPSE = {"Model": "Model", "Callable": "fn", "Scan": "Scan", "PartialTree": "PartialTree", "PartialSeq": "PartialSeq", "Step": "Step",
            "dict": "dict", "Schedule": "Schedule"}


def _short(annotation: str) -> str:
    """A compact type: Container → C, nn.Module → Module, and Model[...], Callable[...], PartialTree[...] without
    their arguments."""
    s, out, i = annotation.replace("nn.Module", "Module").replace("torch.", ""), "", 0
    while i < len(s):
        m = re.match(r"(\w+)\[", s[i:])
        if m and m.group(1) in COLLAPSE and (i == 0 or not (s[i - 1].isalnum() or s[i - 1] == "_")):
            depth, j = 0, i + len(m.group(1))
            while True:
                depth += 1 if s[j] == "[" else -1 if s[j] == "]" else 0
                j += 1
                if depth == 0:
                    break
            out, i = out + COLLAPSE[m.group(1)], j
        else:
            out, i = out + s[i], i + 1
    return re.sub(r"\bContainer\b", "C", out)


ALIASES = {"Chooser": "Step[PartialTree]"}


def _signature(obj: Any) -> tuple[str, str]:
    """(signature, state type) from a stub's annotations. A step or a scan is shown as what it does, whatever its
    settings: Step[S] is S → S with state S, Scan[X, S, Y] is Seq[X] → Seq[Y] with state S. A layer (Model[A, B] built
    from settings) is A → B. Anything else: its required positional parameters → what it returns."""
    sig = inspect.signature(obj)
    ret = ALIASES.get(str(sig.return_annotation), str(sig.return_annotation))
    params = [p for p in sig.parameters.values()
              if p.kind == p.VAR_POSITIONAL or (p.kind in (p.POSITIONAL_ONLY, p.POSITIONAL_OR_KEYWORD) and p.default is inspect.Parameter.empty)]
    settings_only = all(str(p.annotation) in ("int", "float", "bool", "str") for p in params)
    for wrapper in ("Step", "Scan", "Model"):
        if ret.startswith(wrapper + "[") and (wrapper != "Model" or settings_only):
            args = _top_level(ret[len(wrapper) + 1 : -1])
            if wrapper == "Step":
                return f"{_short(args[0])} → {_short(args[0])}", _short(args[0])
            if wrapper == "Scan":
                return f"Seq[{_short(args[0])}] → Seq[{_short(args[2])}]", _short(args[1])
            return f"{_short(args[0])} → {_short(args[1])}", ""
    ins = " × ".join(_short(str(p.annotation)) + ("…" if p.kind == p.VAR_POSITIONAL else "") for p in params) or "()"
    return f"{ins} → {_short(ret)}", ""


def _explanation(obj: Any) -> str:
    doc = inspect.getdoc(obj)
    if not doc:
        raise TypeError(f"{obj!r}: write its explanation in the docstring")
    return " ".join(doc.split("\n\n")[0].split())


def method[T](group: str, name: str, *, state: str = "", side: str = "") -> Callable[[T], T]:
    """Register a stub defined here. state says what the state keeps, when the type carries one."""

    def register(obj: T) -> T:
        signature, state_type = _signature(obj)
        keeps = f"{state_type} — {state}" if state_type and state else state_type or state
        ENTRIES.append(Entry(group, name, signature, keeps, side or GROUPS[group].side, getattr(obj, "__name__", repr(obj)), _explanation(obj), obj))
        return obj

    return register


def alias(group: str, name: str, function: str, explanation: str, signature: str = "", state: str = "", side: str = "") -> None:
    """A method torch (or this file) already names; there is no stub to read."""
    g = GROUPS[group]
    ENTRIES.append(Entry(group, name, signature or g.signature, state, side or g.side, function, explanation))


def phenomenon(name: str, explanation: str) -> None:
    ENTRIES.append(Entry("phenomenon", name, "", "", "", "", explanation))


def meta(name: str, groups: str, explanation: str) -> None:
    ENTRIES.append(Entry("meta", name, groups, "", "", "", explanation))


def entry_of(f: Any) -> Entry | None:
    """The entry a function belongs to (a program prints each instruction's group from it)."""
    f = getattr(f, "func", f)  # functools.partial
    by_name = getattr(f, "__name__", None)
    return next((e for e in ENTRIES if e.group in GROUPS and (e.obj is f or (e.obj is None and by_name in [p.strip() for p in e.function.split(",")]))), None)


# ═══ convert: C[a] → C′[a] ════════════════════════════════════════════════════════════════════════


@method("convert", "Tokenization")
def tokenize(texts: list[str], *, length: int) -> Seq:
    """Text → a sequence of class ids (UTF-8 bytes, or BPE merges of them), padded to one length."""
    ...


@method("convert", "Patching")
def patchify(x: Grid, *, size: int) -> Grid:
    """An image → a coarser grid of p×p patches, each one element (a reshape); a linear map after it is a patch
    embedding."""
    ...


@method("convert", "Flattening")
def flatten(x: Grid) -> Seq:
    """A grid → a sequence in raster order: the lattice is forgotten, learned positions say where each element was."""
    ...


@method("convert", "Multimodal sequence")
def concat(*parts: Seq) -> Seq:
    """Several sequences of the same width as one (image patches, then caption tokens): one network reads them all."""
    ...


alias("convert", "In-context learning (few-shot prompts)", "concat",
      "solved examples, then the query, as one prompt: the model adapts by reading them", "Seq… → Seq")


@method("convert", "Data augmentation: random flip")
def random_flip(x: Grid, *, p: float = 0.5) -> Grid:
    """Mirror each image with probability p: its elements move, none changes."""
    ...


@method("convert", "Data augmentation: random crop")
def random_crop(x: Grid, *, padding: int = 2) -> Grid:
    """Pad, and cut back at a random offset: a random translation."""
    ...


@method("convert", "Experience (transitions)")
def transitions(path: Path, returns: Seq | None = None) -> Value:
    """The steps of walked paths as a set of transitions (state, action, reward, next state, done, and the action's
    log-probability), their order dropped: what a replay buffer holds and RL losses read."""
    ...


@method("convert", "Minibatches")
def batches(data: Container, *, size: int, steps: int) -> Seq:
    """A dataset → a sequence of random minibatches: the stream training scans."""
    ...


@method("convert", "A path from a tree")
def path_to(tree: PartialTree[Any, Any], leaf: Tensor) -> Path:
    """The steps from the root to a leaf of a tree's known part: the result of a walk."""
    ...


# ═══ map: C[a] → C[b] ═════════════════════════════════════════════════════════════════════════════


@method("map", "Network (each layer chosen by the types)")
def network(x: Container, y: Container, *, d: int = 64, depth: int = 2, mixer: Callable[[int], Any] | None = None,
            mlp: Callable[[int], Any] | None = None) -> Predictor:
    """embed → position → [aggregate, MLP] × depth → pool → readout, each slot chosen from the input and target types
    (embed_for, aggregate_for, pool_for, readout_for); mixer and mlp replace a slot's choice (a scan, experts)."""
    ...


@method("map", "Embedding models (a container to one vector)")
def encoder(x: Container, *, dim: int) -> Model[Container, Value]:
    """A network from any container to one vector per example: the towers of CLIP, a reward model, a critic."""
    ...


@method("map", "Embeddings")
def embed_for(x: Container, *, d: int) -> Model[Container, Container]:
    """By what the elements hold: classes → a row of a lookup table (one more for [MASK]); real values → a linear map."""
    ...


@method("map", "Readout")
def readout_for(y: Container, *, d: int) -> Model[Container, Container]:
    """By what the target holds: a logit per class, a value per real number, one readout per field of a record."""
    ...


alias("map", "Linear layer + nonlinearity (MLP)", "nn.Linear, nn.GELU, nn.Sequential", "the same small network on every element", "C[h] → C[h]")
alias("map", "Gating (SwiGLU)", "nn.SiLU", "silu(gate(h)) · up(h), then down: one branch decides how much of the other passes", "C[h] → C[h]")
alias("map", "Normalization", "nn.RMSNorm, nn.LayerNorm", "rescale each element's features (BatchNorm aggregates over the batch instead)", "C[h] → C[h]")
alias("map", "Dropout", "nn.Dropout", "zero random features while training", "C[h] → C[h]")


@method("map", "Positional encoding: learned")
def learned_positions(x: Container, *, d: int) -> Model[Container, Container]:
    """A learned vector added at each position of a Seq, a Grid or a Product; a Set, a Graph or a Tree gets none."""
    ...


@method("map", "Positional encoding: sinusoidal")
def sinusoidal_positions(x: Container, *, d: int) -> Model[Container, Container]:
    """Fixed sin/cos features of each position at geometrically spaced frequencies, added."""
    ...


@method("map", "Positional encoding: rotary (RoPE)")
def rope(x: Tensor, positions: Tensor) -> Tensor:
    """Rotate pairs of query and key features by angles proportional to the position, so that attention depends only
    on relative offsets."""
    ...


@method("map", "Mixture of experts")
def moe(d: int, *, experts: int = 8, k: int = 2) -> Model[Container, Container]:
    """A router sends each element to its top-k experts (MLPs) and sums their gate-weighted outputs."""
    ...


@method("map", "Vector quantization")
def vector_quantize(x: Container, codebook: Tensor) -> Container:
    """Each element → the id of its nearest codebook entry (real → classes), with a straight-through gradient."""
    ...


@method("map", "Straight-through estimator", side="agnostic")
def straight_through(hard: Tensor, soft: Tensor) -> Tensor:
    """Forward the hard value, backpropagate through the soft one: a gradient for a choice that has none."""
    ...


@method("map", "Latent space / bottleneck")
def autoencoder(x: Container, *, latent: int) -> Model[Container, Container]:
    """encode to a smaller container (fewer positions, few channels), decode back: the latent must keep what matters,
    and generation can happen there (latent diffusion). Its .encode and .decode are the two halves."""
    ...


@method("map", "Conditioning", side="agnostic")
def condition(x: Container, label: Value) -> Container:
    """Attach a label to every element (its one-hot vector next to each element's values; a none label for guidance)."""
    ...


@method("map", "Sparse autoencoders (superposition)")
def sparse_autoencoder(d: int, *, features: int) -> Model[Container, Container]:
    """Activations → a wide, sparse code → activations, trained by reconstruction plus an L1 penalty: each code unit
    tends to be one feature."""
    ...


@method("map", "Linear probes (features as directions)")
def linear_probe(acts: Value, labels: Value) -> Model[Value, Value]:
    """A linear readout fitted on frozen activations: is a feature readable along one direction?"""
    ...


@method("map", "Data augmentation: color jitter", side="agnostic")
def color_jitter(x: Grid, *, strength: float = 0.2) -> Grid:
    """Scale each image's brightness and contrast by random factors."""
    ...


@method("map", "Data augmentation: additive noise", side="agnostic")
def gaussian_noise(x: Container, *, sigma: float = 0.05) -> Container:
    """x + σ·ε at every element."""
    ...


@method("map", "Label smoothing", side="agnostic")
def label_smoothing(y: Value, *, eps: float = 0.1) -> Value:
    """A class → a distribution over classes, (1 − ε)·one_hot + ε / n: never pushed to infinite confidence."""
    ...


@method("map", "Attribution (input × gradient)", side="agnostic")
def attribution(model: Predictor, x: Container) -> Container:
    """How much each input element pushed the output: the input times the output's gradient, on any differentiable
    model."""
    ...


@method("map", "Compute-optimal allocation (Chinchilla)", side="agnostic")
def chinchilla(compute: float) -> tuple[float, float]:
    """A compute budget → (parameters, tokens): C ≈ 6·N·D with D ≈ 20 tokens per parameter."""
    ...


# ═══ aggregate: C[h] → C[h] ═══════════════════════════════════════════════════════════════════════


@method("aggregate", "Choosing the aggregation from the container")
def aggregate_for(x: Container, *, d: int) -> Model[Container, Container]:
    """What each element reads, from the container: Graph, Tree → message passing · Product → axial · causal Seq →
    causal attention (a selective scan when long) · large Grid → convolution · else attention over all."""
    ...


@method("aggregate", "Attention (self, causal)")
def attention(d: int, *, causal: bool = False) -> Model[Container, Container]:
    """Each element reads every other (or every earlier one), weighted by query·key similarity."""
    ...


@method("aggregate", "Cross-attention")
def cross_attention(d: int) -> Model[tuple[Container, Container], Container]:
    """Queries from one container, keys and values from another (text reading an image)."""
    ...


@method("aggregate", "Convolution")
def conv(d: int, *, size: int = 3) -> Model[Grid, Grid]:
    """Each element reads its neighbours on the lattice, with the same weights at every offset."""
    ...


@method("aggregate", "Recurrence (RNN / LSTM / GRU)", state="a hidden vector, carried along the sequence")
def recurrence(d: int) -> Scan[Tensor, Tensor, Tensor]:
    """Each element reads a state carried from the earlier ones, updated by a gated nonlinear function."""
    ...


@method("aggregate", "State-space models / linear recurrences", state="s_t = a_t·s_{t−1} + b_t, run as a parallel scan")
def selective_scan(d: int) -> Scan[Tensor, Tensor, Tensor]:
    """A linear recurrence whose coefficients come from the input (S4, Mamba, linear attention): every earlier element
    read, in linear time."""
    ...


@method("aggregate", "Message passing (GNN)")
def message_passing(d: int) -> Model[Graph, Graph]:
    """Each node reads its own features and the mean along its incoming edges; in a tree, children pass up to parents."""
    ...


@method("aggregate", "Axial (products of structures)")
def axial(d: int, *, parts: tuple[str, ...] = ("Seq", "Grid")) -> Model[Product, Product]:
    """One aggregation per factor of a product (frames, then pixels), the other factors as batch; summed."""
    ...


@method("aggregate", "Encoder–decoder hierarchy (U-Net)")
def unet(d: int, *, levels: int = 2) -> Model[Grid, Grid]:
    """Down to a coarse grid and back up, with skip connections: each element reads a wide neighbourhood."""
    ...


@method("aggregate", "Retrieval (attention over a memory)")
def retrieval(k: int) -> Model[tuple[Container, Set], Container]:
    """Each element reads the k most similar entries of an external memory, softmax-weighted."""
    ...


@method("aggregate", "Data augmentation: mixup", side="agnostic")
def mixup(x: Container, y: Value, *, alpha: float = 0.2) -> tuple[Container, Value]:
    """Each example blended with a random partner of the batch, labels by the same λ: the batch as a Set, paired at
    random."""
    ...


@method("aggregate", "Data augmentation: cutmix", side="agnostic")
def cutmix(x: Grid, y: Value) -> tuple[Grid, Value]:
    """A random box pasted from a partner image; the labels mix by its area."""
    ...


@method("aggregate", "Baseline: the batch mean", side="agnostic")
def batch_baseline(returns: Value) -> Value:
    """Each return minus the batch's mean: an advantage with less variance and no bias."""
    ...


@method("aggregate", "GRPO (group-relative advantages)", side="agnostic")
def group_relative(rewards: Value, *, n: int) -> Value:
    """Each reward standardized within its group of n samples from one prompt: an advantage without a critic."""
    ...


# ═══ select: C[a] → C[a] ══════════════════════════════════════════════════════════════════════════


@method("select", "Data filtering")
def quality_filter(x: Container, scores: Value, *, keep: float = 0.5) -> Container:
    """Keep the best-scoring fraction (a quality classifier, a perplexity, a verifier)."""
    ...


@method("select", "Data deduplication")
def dedup(x: Container) -> Container:
    """Keep one copy of each distinct example (exact, or near: MinHash)."""
    ...


@method("select", "Data mixture weights")
def mix_sources(sources: list[Container], weights: list[float], *, n: int) -> Container:
    """Draw from several sources in fixed proportions."""
    ...


@method("select", "Curriculum learning")
def curriculum(x: Container, difficulty: Value) -> Container:
    """Order the examples from easy to hard: training reads them in that order."""
    ...


@method("select", "Data augmentation: token dropout")
def token_dropout(x: Seq, *, p: float = 0.1) -> Seq:
    """Drop a random fraction of the positions; the rest keep their order."""
    ...


@method("select", "Continual learning (replay)", state="a buffer: a Set of past examples, bounded")
def replay(*, size: int, k: int) -> Scan[Container, Set, Container]:
    """Each new batch, with k old examples from the buffer mixed in, so old tasks aren't forgotten."""
    ...


@method("select", "Weight decay (parameter groups)")
def decay_groups(model: nn.Module, *, weight_decay: float = 0.1) -> list[dict[str, Any]]:
    """The parameters to decay (the weight matrices) apart from the rest (norms, biases, embeddings)."""
    ...


@method("select", "Parameterization (μP)")
def mup_groups(model: nn.Module, *, width: int, base_width: int) -> list[dict[str, Any]]:
    """Hidden matrices reweighted by base_width / width, so a learning rate tuned narrow carries over wide."""
    ...


# ═══ fold: C[a] → a ═══════════════════════════════════════════════════════════════════════════════


@method("fold", "Pooling (mean / last position)")
def pool_for(x: Container) -> Model[Container, Value]:
    """A container's features → one vector per example: a causal Seq keeps its last position, the one that read
    everything; anything else, the mean."""
    ...


@method("fold", "Expected return (the RL objective)")
def discounted_return(path: Path, *, gamma: float = 1.0) -> Value:
    """Σ γ^t·r_t along each path: what RL maximizes. A walk can't be differentiated through, so its gradient comes
    from scoring the path (reinforce)."""
    ...


@method("fold", "Best-of-N with a verifier")
def best(paths: Path, scores: Value, *, n: int) -> Path:
    """The best-scoring of each group of n paths (scored by a verifier or a reward model)."""
    ...


@method("fold", "Self-consistency (majority vote)")
def majority(answers: Value, *, n: int) -> Value:
    """The most frequent answer of each group of n samples."""
    ...


@method("fold", "Early stopping")
def early_stop(curve: Seq, *, patience: int = 3) -> bool:
    """Whether the validation loss has stopped improving for `patience` evaluations."""
    ...


@method("fold", "Scaling laws (power-law fit)")
def fit_power_law(sizes: Seq, losses: Seq) -> tuple[float, float]:
    """loss ≈ a·size^(−b), fitted on the logs."""
    ...


# ═══ scan: Seq[X] → Seq[Y] ════════════════════════════════════════════════════════════════════════


@method("scan", "Training (a scan over batches)", state="Module — the parameters and the optimizer's moments")
def train[M](model: M, objective: Callable[..., Tensor], *data: Container, steps: int, optimizer: str = "adamw") -> M:
    """Carry the parameters along a stream of minibatches of the data: each one, the objective, its gradient, the
    optimizer's update. When the objective walks for its own data (RL, self-play), the scan closes into an unroll."""
    ...


@method("scan", "Scan")
def scan(s: Scan[Any, Any, Any], xs: Seq) -> tuple[Any, Seq]:
    """Carry a Scan's state along a sequence of inputs: the final state, and the outputs."""
    ...


@method("scan", "Optimizers (SGD, momentum, Adam, Muon)", state="the moments of every parameter")
def optimizer(kind: str = "adamw", *, lr: float = 3e-4) -> Scan[Tensor, Tensor, Tensor]:
    """Gradients in, updates out: SGD carries nothing, momentum a velocity, Adam two moments, Muon orthogonalizes."""
    ...


@method("scan", "Learning-rate schedules", state="the step count")
def warmup_cosine(*, warmup: int, total: int) -> Scan[int, int, float]:
    """A learning rate for each step: linear warmup, then cosine decay."""
    ...


@method("scan", "Training step (clipping, mixed precision)", state="the parameters and the optimizer's moments")
def train_step(*, max_grad_norm: float = 1.0, precision: torch.dtype | None = None) -> Scan[Container, nn.Module, Tensor]:
    """One step: the objective under autocast, backward, clip the gradient norm, the optimizer's update."""
    ...


@method("scan", "EMA of weights", state="the averaged weights")
def ema(*, decay: float = 0.999) -> Scan[nn.Module, nn.Module, nn.Module]:
    """A running average of the weights training produces: smoother, and the slow target of self-distillation and DQN."""
    ...


@method("scan", "Constrained objectives (Lagrangian)", state="the multiplier λ ≥ 0")
def lagrangian(*, lr: float = 0.1) -> Scan[tuple[Tensor, Tensor], float, Tensor]:
    """(loss f, constraint g) in, f + λ·(g − c) out; λ rises while the constraint is violated (dual ascent)."""
    ...


@method("scan", "Returns-to-go", state="the return from here on (scanned backward)")
def returns_to_go(rewards: Seq, *, gamma: float = 1.0) -> Seq:
    """G_t = r_t + γ·G_{t+1}: what each step led to."""
    ...


@method("scan", "Self-play", state="the outcome for the player to move")
def negamax(rewards: Seq) -> Seq:
    """One policy plays both sides; each move's return is the outcome for whoever made it: G_t = r_t − G_{t+1}."""
    ...


@method("scan", "Value functions, Bellman, GAE", state="the advantage from here on (scanned backward)")
def gae(rewards: Seq, values: Seq, *, gamma: float = 0.99, lmbda: float = 0.95) -> Seq:
    """A critic's TD errors r + γV(s′) − V(s), averaged backward with weights (γλ)^k."""
    ...


alias("scan", "Parallel scan", "torch.cumsum", "s_t = a_t·s_{t−1} + b_t for every t at once, in logarithmic depth: how a recurrence trains in parallel")

# ═══ split: C → Example ═══════════════════════════════════════════════════════════════════════════


@method("split", "Choosing the split from the data")
def split_for(x: Container) -> Split:
    """Classes in order → shift · other classes → mask · real values → noise."""
    ...


@method("split", "Supervised")
def labelled(x: Container, y: Container) -> Example:
    """The data comes split: the input with nothing hidden, the label as the target."""
    ...


@method("split", "Self-supervised: next-token prediction")
def shift(x: Seq) -> Example:
    """Each element is predicted from the ones before it (a causal Seq): the target at i is the element at i + 1."""
    ...


@method("split", "Self-supervised: masked prediction (BERT, MAE, MaskGIT)")
def mask(x: Container, *, ratio: float | None = None) -> Example:
    """Random elements at level 1 ([MASK] for classes, 0 for real values): 15% is BERT, 75% MAE, a random ratio per
    example MaskGIT and masked diffusion."""
    ...


@method("split", "Diffusion / score matching / flow matching")
def noise(x: Container) -> Example:
    """Every element at one random level t (x_t = (1 − t)·x + t·ε): the target is the velocity, the noise or the
    clean data, rescalings of one another."""
    ...


@method("split", "Diffusion forcing")
def noise_each(x: Seq) -> Example:
    """A random level for each element: next-token prediction and diffusion are its two extremes."""
    ...


@method("split", "Self-supervised: future-frame prediction")
def next_frame(x: Product) -> Example:
    """The frames up to t as the input, the frames from t + 1 as the target: shift, on the frames of a video."""
    ...


@method("split", "Reconstruction (autoencoder)")
def reconstruction(x: Container) -> Example:
    """Nothing hidden; the target is the input itself, so the model's bottleneck is what makes it hard."""
    ...


@method("split", "Contrastive views")
def two_views(x: Container, augment: Callable[[Container], Container]) -> Example:
    """Two random augmentations of the same examples, each the other's target (CLIP pairs an image with its caption)."""
    ...


@method("split", "Knowledge distillation")
def distillation(x: Container, teacher: Predictor) -> Example:
    """The target is a teacher's (softened) prediction on the same input."""
    ...


@method("split", "Self-distillation (BYOL / DINO / JEPA)")
def self_distillation(x: Container, teacher: Predictor, augment: Callable[[Container], Container]) -> Example:
    """One view as the input; the target is a slow (EMA) copy's embedding of another view: agree with an older self."""
    ...


@method("split", "Instruction tuning / SFT")
def sft(x: Seq, *, prompt_len: int) -> Example:
    """shift, with only the response as targets: the prompt is read, never predicted."""
    ...


@method("split", "TD learning (value regression)")
def td_target(batch: Value, critic: Predictor, *, gamma: float = 0.99) -> Example:
    """V(s) learns toward r + γ·V(s′): a target bootstrapped from the critic's own estimate."""
    ...


@method("split", "Q-learning (DQN)")
def q_target(batch: Value, q: Predictor, *, gamma: float = 0.99) -> Example:
    """Q(s, a) learns toward r + γ·max Q(s′, ·) at the action taken (the weight picks it out)."""
    ...


@method("split", "World-model targets (dynamics)")
def dynamics(batch: Value) -> Example:
    """From each transition, the input is the state and the action; the target what followed (next state, reward,
    done). A model of it is a learned follow (imagined)."""
    ...


@method("split", "Amortized search (AlphaZero targets)")
def search_targets(path: Path, search: PartialTree[Any, Any]) -> Example:
    """For each state a walk visited: what search made of it (its visit distribution) and what followed (the return)."""
    ...


# ═══ score: Prediction × C → ℝ ════════════════════════════════════════════════════════════════════


@method("score", "Objective (a split, then its loss)")
def objective(split: Split) -> Callable[..., Tensor]:
    """The loss of a model on a batch: split it into an Example, predict from the input, score with the loss the
    target picks (labelled data: split = labelled)."""
    ...


@method("score", "Loss = negative log-likelihood of the target")
def loss(pred: Prediction, target: Container, weight: Tensor | None = None) -> Tensor:
    """−log p(target), chosen by what the target holds: classes → cross-entropy · bits → BCE · real → squared error ·
    distribution → KL · record → the sum of its fields' losses."""
    ...


@method("score", "Cross-entropy")
def cross_entropy(logits: Tensor, target: Container, weight: Tensor | None = None) -> Tensor:
    """Classes, or a distribution over them (soft labels)."""
    ...


@method("score", "Binary cross-entropy")
def binary_cross_entropy(logits: Tensor, target: Container, weight: Tensor | None = None) -> Tensor:
    """Bits."""
    ...


@method("score", "Squared error")
def squared_error(pred: Tensor, target: Container, weight: Tensor | None = None) -> Tensor:
    """Real values: a unit Gaussian's negative log-likelihood, up to a constant."""
    ...


@method("score", "KL divergence")
def kl(pred: Dist, target: Dist, weight: Tensor | None = None) -> Tensor:
    """A distribution against a reference (a prior, a reference policy)."""
    ...


@method("score", "Multi-task training")
def per_field(pred: Prediction, target: Container, weight: Tensor | None = None) -> Tensor:
    """A record: one loss per field, each chosen by what the field holds, summed."""
    ...


@method("score", "Log-probability")
def log_prob(pred: Prediction, x: Container) -> Value:
    """log p(x) for each element under a prediction, unreduced: what losses and estimators are built from."""
    ...


@method("score", "Teacher forcing (the log-probability of a path)")
def path_log_prob(tree: PartialTree[Any, Any], path: Path) -> Value:
    """Σ log options(node)(edge) along a path; on the data's own path, −1/length of it is the next-token loss."""
    ...


@method("score", "Sequence log-probability")
def sequence_log_prob(model: Predictor, x: Seq, *, start: int = 1) -> Value:
    """log p(x[start:] | the rest before it), per sequence: teacher forcing on whole sequences, in one pass."""
    ...


@method("score", "Contrastive (InfoNCE / CLIP)")
def info_nce(a: Value, b: Value, *, temperature: float = 0.07) -> Tensor:
    """Which b goes with each a, among the batch: cross-entropy over similarities, the matching pair as the class."""
    ...


@method("score", "Reward models (Bradley–Terry)")
def bradley_terry(chosen: Value, rejected: Value) -> Tensor:
    """−log σ(r(chosen) − r(rejected)): the likelihood that the preferred answer wins."""
    ...


@method("score", "Preferences (DPO)")
def dpo(reference: Predictor, *, beta: float = 0.1) -> Callable[..., Tensor]:
    """The loss of (policy, chosen, rejected): Bradley–Terry on β·(the policy's log-ratio to the reference, chosen minus
    rejected). Preferences as classification: no reward model, no walk."""
    ...


@method("score", "KL-regularized reward (RLHF)")
def kl_regularized_reward(reward: Value, log_prob: Value, ref_log_prob: Value, *, beta: float = 0.1) -> Value:
    """The reward, minus how far the policy has moved from a reference: use it as the return."""
    ...


@method("score", "VAE (evidence lower bound)")
def elbo(encoder: Model[Container, Dist], decoder: Model[Tensor, Prediction], x: Container) -> Tensor:
    """The reconstruction from a latent drawn from the posterior (rsample: reparameterization), plus KL(posterior ‖ prior)."""
    ...


@method("score", "GAN (two players)")
def gan(generator: Model[Tensor, Container], discriminator: Predictor, real: Container, z: Tensor) -> tuple[Tensor, Tensor]:
    """The discriminator tells real from generated; the generator wants its samples called real. Two scores, for two
    players (alternate their steps)."""
    ...


@method("score", "Normalizing flows", side="model")
def flow_log_prob(x: Container, transforms: list[Any]) -> Value:
    """The exact log-likelihood through invertible maps (change of variables)."""
    ...


@method("score", "Policy gradients (REINFORCE)")
def reinforce(path: Path, advantage: Value) -> Tensor:
    """−E[log p(path)·A]: a walked path's log-probability, weighted by how much better than usual it went."""
    ...


@method("score", "PPO clipped surrogate")
def ppo_clip(log_prob: Value, old_log_prob: Value, advantage: Value, *, clip: float = 0.2) -> Tensor:
    """The policy gradient with the ratio π / π_old clipped: it stops pushing once the policy has moved far."""
    ...


@method("score", "Pathwise estimator (through a critic)")
def pathwise(policy: Dist, critic: Model[Tensor, Tensor]) -> Tensor:
    """−E[Q(a)] with a = rsample(): differentiated through the sample into a learned critic (DDPG, SAC)."""
    ...


@method("score", "Maximum-entropy RL (entropy bonus)")
def entropy_bonus(options: Dist, *, alpha: float = 0.01) -> Tensor:
    """−α·H(options): keeps the policy exploring."""
    ...


@method("score", "Meta-learning (MAML)")
def maml_loss(model: nn.Module, objective: Callable[..., Tensor], support: Container, query: Container) -> Tensor:
    """Bilevel: the query loss of the weights after an inner step (an unroll) on the support set, differentiated through."""
    ...


@method("score", "Metrics (accuracy)")
def accuracy(pred: Tensor, target: Container) -> Tensor:
    """Reported and selected on, never differentiated: a loss stands in for it in training."""
    ...


@method("score", "Retrieval accuracy")
def retrieval_accuracy(a: Value, b: Value) -> Tensor:
    """How often each a is closest to its own b among the batch (matching images to captions)."""
    ...


alias("score", "Backpropagation / autodiff", "torch.autograd", "the gradient of any score", "ℝ → ∇")

# ═══ fill: Holes[C] → C ═══════════════════════════════════════════════════════════════════════════


@method("fill", "Fill", state="Holes[C] — the levels fall, as the schedule decides")
def fill[C: Container](holes: Holes[C], model: Predictor, schedule: Schedule, *, steps: int) -> C:
    """Generation as a split undone: at each step the model predicts every hole, the schedule lowers some levels with
    it; when every level is 0, the container is known."""
    ...


@method("fill", "Blank (where generation starts)")
def blank[C: Container](like: C) -> Holes[C]:
    """Every element unknown: all [MASK] for classes, pure noise for real values."""
    ...


@method("fill", "Autoregressive order", state="Holes[C] — known up to i")
def in_order(holes: Holes[Container], pred: Prediction) -> Holes[Container]:
    """Fill the first hole: one element at a time, in order (the holes view of decoding; walk gives the tree view)."""
    ...


@method("fill", "Unmasking (MaskGIT, masked diffusion)", state="Holes[C] — some holes at 1, the rest known")
def most_confident(holes: Holes[Container], pred: Prediction) -> Holes[Container]:
    """Fill the holes the model is surest of, more each step (a cosine schedule), until none is left."""
    ...


@method("fill", "Sampling a flow (diffusion)", state="Holes[C] — every element at level t")
def together(holes: Holes[Container], pred: Prediction) -> Holes[Container]:
    """Lower every level by one step along the predicted velocity (Euler): noise, undone."""
    ...


@method("fill", "Diffusion forcing sampling", state="Holes[C] — a level for each element")
def per_element(holes: Holes[Container], pred: Prediction) -> Holes[Container]:
    """Each element follows its own schedule (the near future clean, the far future noisy): between in_order and together."""
    ...


@method("fill", "Energy-based models (Langevin)", state="a sample, moving")
def langevin(energy: Predictor, *, step_size: float = 0.01) -> Step[Container]:
    """x ← x − η·∇E(x) + √(2η)·ε: samples of exp(−energy), with the gradient from autograd."""
    ...


# ═══ walk: PartialTree → Path ═════════════════════════════════════════════════════════════════════


@method("walk", "Walk", state="PartialTree — its known part, grown by the chooser")
def walk(tree: PartialTree[Any, Any], choose: Chooser, *, steps: int) -> Path:
    """Unroll the chooser on the tree, then read the path to the best leaf of each root."""
    ...


@method("walk", "An LM as a tree")
def lm_tree(model: Predictor, prompt: Seq) -> PartialTree[Seq, Tensor]:
    """A node is a sequence (Holes: known up to its end), its options the next-token distribution, an edge appends."""
    ...


@method("walk", "MDP (an environment as a tree)")
def env_tree(env: Environment, policy: Predictor | None = None) -> PartialTree[Any, Tensor]:
    """A node is a state, its options the policy's actions (uniform with none), an edge leads where the environment
    steps to, with a reward and whether it ended."""
    ...


@method("walk", "Sampling", state="PartialTree — one leaf per root")
def sample(tree: PartialTree[Any, Any]) -> PartialTree[Any, Any]:
    """Expand each kept leaf by one edge drawn from its options."""
    ...


@method("walk", "Greedy decoding", state="PartialTree — one leaf per root")
def greedy(tree: PartialTree[Any, Any]) -> PartialTree[Any, Any]:
    """Expand each kept leaf by its likeliest edge."""
    ...


@method("walk", "Beam search", state="k leaves per root and their log-probabilities")
def beam(k: int) -> Chooser:
    """Expand every kept leaf; keep the k likeliest children of each root."""
    ...


@method("walk", "Several paths per root (best-of-N, groups)", state="n independent leaves per root")
def independent(n: int, choose: Chooser) -> Chooser:
    """n paths from each root side by side, each grown by its own chooser: n samples per prompt (best-of-N, GRPO)."""
    ...


@method("walk", "Teacher forcing (walking the data's path)", state="one leaf per root, on the data's path")
def forced(edges: Seq) -> Chooser:
    """Expand each leaf by the next given edge: scoring the path walked is teacher forcing."""
    ...


@method("walk", "Speculative decoding", state="the accepted leaf, and a draft model's guesses past it")
def speculative(draft: PartialTree[Any, Any], *, k: int = 4) -> Chooser:
    """The draft walks k edges ahead greedily, the tree's model checks them in one pass, the agreeing prefix is kept:
    greedy decoding, in fewer of the big model's passes."""
    ...


alias("walk", "Chain-of-thought / longer reasoning", "walk", "a longer walk before the answer; the gains come from training the walk (R1)")

# ═══ unroll: PartialSeq[S] → S ════════════════════════════════════════════════════════════════════


@method("unroll", "Unroll", state="S — any state")
def unroll(seq: PartialSeq[Any], *, steps: int) -> list[Any]:
    """The list a PartialSeq stands for: start, step(start), step(step(start)), …"""
    ...


@method("unroll", "Iteration (rounds of a program)", state="whatever the rounds pass on: models, data, a buffer")
def loop(round: Callable[..., Any], x: Any, *, times: int) -> Any:
    """Rounds of collecting and learning (policy iteration, self-play, expert iteration): an unroll over their result."""
    ...


@method("unroll", "Evolution strategies (no gradient)", state="a search distribution over the parameters")
def evolution_strategy(fitness: Callable[[Tensor], Tensor], *, population: int = 32) -> Step[Dist]:
    """Draw a population (a Set of parameters) from the search distribution, score them, move it toward the better
    ones: it works when nothing is differentiable (CMA-ES keeps a covariance too)."""
    ...


@method("unroll", "Two-player games (alternating steps)", state="both players' parameters")
def alternate(*steps: Step[nn.Module]) -> Step[tuple[nn.Module, ...]]:
    """No single quantity goes down: each player takes a step on its own loss in turn (GANs, adversarial training)."""
    ...


# ═══ reshape: Partial → Partial ═══════════════════════════════════════════════════════════════════


@method("reshape", "Sampling temperature")
def temperature(tree: PartialTree[Any, Any], t: float) -> PartialTree[Any, Any]:
    """The options' logits divided by t: below 1 sharper (toward greedy), above 1 flatter."""
    ...


@method("reshape", "Top-k sampling")
def top_k(tree: PartialTree[Any, Any], k: int) -> PartialTree[Any, Any]:
    """Only the k likeliest edges stay possible."""
    ...


@method("reshape", "Top-p (nucleus) sampling")
def top_p(tree: PartialTree[Any, Any], p: float) -> PartialTree[Any, Any]:
    """Only the likeliest edges whose probabilities add up to p stay possible."""
    ...


@method("reshape", "A condition, fixed")
def given(model: Predictor, condition: Value | None) -> Predictor:
    """A conditional model with its condition fixed (None: no condition): the two models guidance mixes."""
    ...


@method("reshape", "Classifier-free guidance")
def guided(cond: Predictor, uncond: Predictor, *, w: float) -> Predictor:
    """u + w·(c − u): extrapolate from the unconditional prediction toward the conditional one, for a tree's options or
    a fill's model."""
    ...


@method("reshape", "Exploration vs exploitation (ε-greedy)")
def epsilon_greedy(tree: PartialTree[Any, Any], eps: float) -> PartialTree[Any, Any]:
    """With probability ε a uniform edge, else the likeliest."""
    ...


@method("reshape", "Tool use")
def with_tool(tree: PartialTree[Any, Any], call: int, tool: Callable[[Any], Tensor]) -> PartialTree[Any, Any]:
    """When a walk emits the call token, follow writes the tool's answer (a calculator, a search, an interpreter)."""
    ...


@method("reshape", "KV caching", state="each known node keeps its keys and values")
def cached(tree: PartialTree[Any, Any]) -> PartialTree[Any, Any]:
    """The tree's known nodes keep what the model computed for them, so expanding a node costs one position."""
    ...


@method("reshape", "World models (a learned follow)")
def imagined(tree: PartialTree[Any, Any], world: Model[Container, Container]) -> PartialTree[Any, Any]:
    """The environment's follow (and reward, done) replaced by a learned model of them: walks, searches and plans in
    imagination (Dreamer, MuZero)."""
    ...


@method("reshape", "Planning (random shooting / MPC)", state="the random plans tried from each node")
def planned(tree: PartialTree[Any, Any], world: PartialTree[Any, Any] | None = None, *, horizon: int = 5, samples: int = 64) -> PartialTree[Any, Any]:
    """Options that look ahead: random action sequences simulated in a world, all the probability on the first action
    of the best."""
    ...


@method("reshape", "MCTS (tree search)", state="a search tree per node: priors, visit counts, values")
def mcts(tree: PartialTree[Any, Any], *, simulations: int = 64, two_player: bool = False) -> PartialTree[Any, Any]:
    """Options improved by search: grow a tree by PUCT (the options as priors), back up rewards and values, return the
    visit counts. Walk it to search; train toward it to amortize the search (AlphaZero)."""
    ...


@method("reshape", "Classifier-free guidance training")
def droppable(model: Predictor, *, p: float = 0.1) -> Predictor:
    """The condition dropped with probability p while training, so one model is both conditional and unconditional."""
    ...


# ═══ adapt: Module → Module ═══════════════════════════════════════════════════════════════════════


@method("adapt", "Residual connections")
def residual(f: nn.Module) -> nn.Module:
    """h + f(norm h): the layer edits the stream instead of replacing it."""
    ...


@method("adapt", "Pretrain → finetune (transfer)", side="agnostic")
def transfer(source: nn.Module, target: nn.Module) -> nn.Module:
    """Start a new model from a trained one: copy the weights whose names and shapes match (the body), keep the new head."""
    ...


@method("adapt", "Freezing", side="agnostic")
def freeze(model: nn.Module, trainable: Callable[[str], bool]) -> nn.Module:
    """Train only the parameters whose name passes a test."""
    ...


@method("adapt", "Frozen copy (reference / target network)", side="agnostic")
def frozen_copy(model: nn.Module) -> nn.Module:
    """A copy that no longer trains: DPO's and RLHF's reference, DQN's target network."""
    ...


@method("adapt", "LoRA")
def lora(model: nn.Module, *, rank: int = 8) -> nn.Module:
    """W + B·A on chosen linear layers, the base frozen: the finetuning update, kept low-rank."""
    ...


@method("adapt", "Adapters")
def adapter(layer: nn.Module, *, bottleneck: int = 8) -> nn.Module:
    """A frozen layer followed by a small bottleneck MLP that trains alone, starting at zero."""
    ...


@method("adapt", "Prompt tuning")
def prompt_tuning(model: nn.Module, *, n: int) -> nn.Module:
    """Learned vectors in front of a frozen model's input; only they train."""
    ...


@method("adapt", "Initialization", side="agnostic")
def init(model: nn.Module) -> nn.Module:
    """Fan-in normal for matrices, zeros for output projections: each residual block starts as the identity."""
    ...


@method("adapt", "Quantization (int8 / int4)")
def quantize(model: nn.Module, *, bits: int = 8) -> nn.Module:
    """Weights rounded to 2^(bits−1) − 1 levels a side per row: computing as the low-precision model would."""
    ...


alias("adapt", "Low precision: inference", "quantize", "int8 / int4 weights")
alias("adapt", "Low precision: training", "torch.autocast", "bf16 (or fp8) compute, float32 master weights", side="agnostic")
alias("adapt", "Parallelism", "torch.distributed",
      "data, sharded, tensor, pipeline, expert, sequence: DDP and FSDP wrap a module (fully_shard); DTensor places its weights", side="agnostic")
alias("adapt", "Hardware-aware kernels", "torch.compile", "torch.compile(module) fuses its ops; attention picks flash kernels by itself")


@method("adapt", "Reading activations", side="agnostic")
def observe(model: nn.Module, at: nn.Module) -> nn.Module:
    """A model that returns what one of its modules computed: the activations probes, SAEs and circuits study."""
    ...


@method("adapt", "Activation patching (circuits)", side="agnostic")
def patch(model: nn.Module, at: nn.Module, replace: Callable[[Tensor], Tensor]) -> nn.Module:
    """A model with one module's output replaced (by another input's activations): which components carry a behaviour?"""
    ...


@method("adapt", "Steering along a direction", side="agnostic")
def steer(model: nn.Module, at: nn.Module, direction: Tensor, *, alpha: float = 1.0) -> nn.Module:
    """α·direction added to one module's output: features are directions, and moving along one changes the behaviour."""
    ...


# ═══ Phenomena: observed, not computed ════════════════════════════════════════════════════════════

phenomenon("Implicit regularization of SGD", "a property of training dynamics, seen in eval curves")
phenomenon("Double descent / benign overfitting", "seen by sweeping model or data size")
phenomenon("Grokking", "delayed generalization in long runs")
phenomenon("Superposition", "more features than dimensions; measured with sparse autoencoders")
phenomenon("General methods + scale beat hand-engineering", "an empirical lesson (the bitter lesson), not an op")

# ═══ Meta-primitives: each is a property of a group ═══════════════════════════════════════════════

meta("Bake in symmetry by sharing weights", "aggregate", "the same weights wherever the container relates elements: conv on a grid, attention on a set, messages on a graph")
meta("Let the input decide the computation", "aggregate, map", "attention weights, scan coefficients and expert choices computed from the data")
meta("Predict the hidden part from the visible part", "split, fill", "every self-supervised signal is a split into Holes; generation fills them back")
meta("Turn one hard step into many easy ones", "fill, walk, unroll", "a token, a denoising step, an update at a time: a state, stepped")
meta("Learning is compression", "map, split", "bottlenecks, codebooks and students: the target, from a smaller description")
meta("Bootstrap from yourself", "split", "targets made by your own (older) model: TD, self-distillation, self-play, AlphaZero")
meta("Amortize, or search", "map, walk, reshape", "one forward pass that answers at once, or a tree grown by search")
meta("Every loss is a likelihood", "score", "loss(pred, target): what the target holds picks which")
meta("Not every objective is a loss", "fold, score", "a return is a fold of a path; its gradient comes from scoring the path, or there is none")
meta("Structure decides the computation", "aggregate", "the container picks what each element reads")

# ═══ Printing ═════════════════════════════════════════════════════════════════════════════════════

NEW = {"Fill", "Autoregressive order", "Unmasking (MaskGIT, masked diffusion)", "Sampling a flow (diffusion)", "Diffusion forcing",
       "Diffusion forcing sampling", "Training (a scan over batches)", "Optimizers (SGD, momentum, Adam, Muon)", "KV caching", "A path from a tree",
       "Sampling", "Several paths per root (best-of-N, groups)", "Teacher forcing (walking the data's path)", "Supervised", "Scan", "Log-probability", "Retrieval accuracy", "Blank (where generation starts)", "A condition, fixed", "Objective (a split, then its loss)", "Minibatches",
       "Classifier-free guidance training", "Recurrence (RNN / LSTM / GRU)", "State-space models / linear recurrences", "Walk"}


def print_catalog(explain: bool = False) -> None:
    """Grouped by family and group, one line per method: name: signature · state. explain adds what each one does."""
    for family, title in FAMILIES.items():
        print(f"\n══ {title} ══")
        for g in (g for g in GROUPS.values() if g.family == family):
            print(f"\n{g.name}  {g.signature}\n  {g.explanation}")
            for e in (e for e in ENTRIES if e.group == g.name):
                mark = "★ " if e.name in NEW else ""
                state = f"  · state: {e.state}" if e.state else ""
                print(f"  - {mark}{e.name}: {e.signature}{state}")
                if explain:
                    print(f"      {e.explanation}")
    print("\n══ META-PRIMITIVES · properties of the groups ══\n")
    for e in (e for e in ENTRIES if e.group == "meta"):
        print(f"  - {e.name}: {e.signature}" + (f"\n      {e.explanation}" if explain else ""))
    print("\n══ PHENOMENA · observed, not computed ══\n")
    for e in (e for e in ENTRIES if e.group == "phenomenon"):
        print(f"  - {e.name}" + (f"\n      {e.explanation}" if explain else ""))
    rows = [e for e in ENTRIES if e.group in GROUPS]
    print(f"\n{len(rows)} methods in {len(GROUPS)} groups, {sum(bool(e.state) for e in rows)} carrying a state · "
          f"{sum(e.side == 'agnostic' for e in rows)} model-agnostic, {sum(e.side == 'model' for e in rows)} model-dependent")


if __name__ == "__main__":
    import sys

    print_catalog(explain="--explain" in sys.argv)
