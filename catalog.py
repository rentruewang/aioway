"""
catalog.py: every method in the original list, as a typed signature over the types of data.py. Two kinds, as a
database splits them. Expressions take batches (a Container, in memory), build values and change nothing: queries.
Statements take datasets, or change something in place: writes, and the blocking work that needs every batch. The
interface only: bodies are left to the implementation (torch, torchrl, your own).

EXPRESSIONS · on a batch, nothing changes
  structure   encode     A ⇄ C[classes]        write raw data in a vocabulary and read it back (a codec)
              convert    C[a] → C′[a]          change the container, keep what the elements are
              map        C[a] → C[b]           change each element on its own, keep the container
              aggregate  C[h] → C[h]           each element reads the elements it is related to
              select     C[a] → C[a]           keep, drop, reorder or reweight elements
              fold       C[a] → a              reduce a container to one value
              scan       Seq[X] → Seq[Y]       carry a state along a sequence: Scan[X, S, Y]
  learning    split      C → Example           hide part of a container (Holes); what was hidden is the target
              score      Prediction × C → ℝ    how bad a prediction is: every loss, every surrogate objective
  partial     fill       Holes[C] → C          fill the holes: generation is split undone; state Holes[C]
              walk       PartialTree → Path    grow the known part of a tree, then read a path; state PartialTree
              unroll     PartialSeq[S] → S     step any state: search distributions
              reshape    Partial → Partial     change how a partial container goes on: its options, its model
  lookup      search     Index × Value → Set   the k nearest elements of an indexed set to each query
  models      wrap       Module → Module′      a new model around an old one, which stays as it was

STATEMENTS · on a dataset, or changing something in place (!)
  update      fit        X! × Dataset          fit something to a dataset: a model by a loss, a codec by its rule
              assign     Weights!              set parameter values directly: copied in, initialized, averaged
              alter      Module!               change what the parameters are: add, freeze, quantize, shard
  store       index      Index! × …            learn an index's structure, insert into it, delete from it
              buffer     Buffer! × C           add a batch to a buffer that is read as a dataset
  dataset     source     … → Dataset           where batches come from: memory, an environment, an expr per batch
              curate     Dataset → Dataset     keep, drop or reorder examples looking at all of them (blocking)
              reduce     Dataset → a           one value from a whole dataset: a metric, a statistic, a batch

A statement is never a value inside an expression: in a script (program.py) it is a line of its own, and its
arguments are expressions or other dataset statements, as a SQL write or CREATE INDEX sits at the top of a
statement and a FROM may nest. Update and store statements change their argument marked ! and return None; dataset
statements are pure but read every batch. Weights is the subtype of Module whose architecture an op leaves alone.

A method that carries a state says so in its type: Step[S] is one step of a PartialSeq over S (a chooser is a
Step[PartialTree], a schedule a Step[Holes]); Scan[X, S, Y] carries S along a sequence (an optimizer carries the
moments: fit runs it across steps). The state column of the printed catalog names it, and what the method keeps there.

Each entry is model-agnostic (any model is an opaque function to it, or it uses none) or model-dependent (it is
a layer, or it opens a model up). Phenomena are observed rather than computed; the meta-primitives are
properties of the groups; the well-known works are programs of entries (program.py).

@method(group, name, state=...) registers an expression stub defined here, @statement(group, name, changes=i) a
statement (i: the argument it changes in place, None for a dataset statement): the signature comes from the
annotations, the explanation from the docstring. alias(...) registers one that torch already has.

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
    Buffer, Chooser, Codec, Container, Dataset, Dist, Environment, Example, Graph, Grid, Holes, Index, Model, Predictor, PartialSeq, PartialTree, Path,
    Prediction, Product, Scan, Schedule, Seq, Set, Split, Step, Value, Weights,
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
    Group("encode", "structure", "A ⇄ C[classes]", "write raw data in a vocabulary and read it back: a codec, fitted or fixed", "model"),
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
    Group("search", "lookup", "Index × Value → Set", "the k elements of an indexed set nearest each query", "agnostic"),
    Group("wrap", "models", "Module → Module′", "a new model around an old one, which stays as it was", "agnostic"),
    Group("fit", "update", "X! × Dataset", "fit something to a dataset: a model by a loss's gradient, a codec by its own rule", "agnostic"),
    Group("assign", "update", "Weights!", "set parameter values directly, without a gradient: copied in, initialized, averaged", "agnostic"),
    Group("alter", "update", "Module!", "change what the parameters are: add some, freeze some, quantize, shard", "model"),
    Group("index", "store", "Index! × …", "learn an index's structure from a sample, insert into it, delete from it", "agnostic"),
    Group("buffer", "store", "Buffer! × C", "add a batch to a buffer, which statements read as a dataset", "agnostic"),
    Group("source", "dataset", "… → Dataset", "where batches come from: memory, an environment, an expression applied to each batch", "agnostic"),
    Group("curate", "dataset", "Dataset → Dataset", "keep, drop or reorder examples, looking at all of them (blocking)", "agnostic"),
    Group("reduce", "dataset", "Dataset → a", "one value from a whole dataset: a metric over every batch, a statistic, one batch", "agnostic"),
)}
FAMILIES = {"structure": "STRUCTURE · data → data", "learning": "LEARNING · data → a training signal",
            "partial": "PARTIAL CONTAINERS · known in part, and how to get the rest",
            "lookup": "LOOKUP · a set known by how to search it", "models": "MODELS · model → a new model",
            "update": "UPDATE · a model or a codec, changed in place", "store": "STORE · an index or a buffer, kept up to date",
            "dataset": "DATASET · read every batch, change nothing"}
FRAMEWORKS = {"expr": ("EXPRESSIONS · on a batch, nothing changes", ["structure", "learning", "partial", "lookup", "models"]),
              "stmt": ("STATEMENTS · on a dataset, or changing something in place (!)", ["update", "store", "dataset"])}
STATEMENTS = {g.name for g in GROUPS.values() if g.family in FRAMEWORKS["stmt"][1]}


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
    changes: int | None = None  # an update or store statement: which argument it changes in place


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
            "dict": "dict", "Schedule": "Schedule", "Codec": "Codec"}


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
    return re.sub(r"\bContainer\b", "C", out.replace("[Any]", ""))


ALIASES = {"Chooser": "Step[PartialTree]"}


def _signature(obj: Any, changes: int | None = None) -> tuple[str, str]:
    """(signature, state type) from a stub's annotations. A step or a scan is shown as what it does, whatever its
    settings: Step[S] is S → S with state S, Scan[X, S, Y] is Seq[X] → Seq[Y] with state S. A layer (Model[A, B] built
    from settings) is A → B. Anything else: its required positional parameters → what it returns (a statement
    returns nothing, so it is its parameters alone, the changed one marked !)."""
    sig = inspect.signature(obj)
    bounds = {t.__name__: getattr(t.__bound__, "__name__", t.__name__) for t in getattr(obj, "__type_params__", ()) if t.__bound__ is not None}

    def short(annotation: str) -> str:  # a bounded type parameter is shown as its bound: frozen_copy[M: Module] is Module → Module
        return _short(re.sub(r"\b\w+\b", lambda m: bounds.get(m.group(), m.group()), annotation))

    ret = ALIASES.get(str(sig.return_annotation), str(sig.return_annotation))
    params = [p for p in sig.parameters.values()
              if p.kind == p.VAR_POSITIONAL or (p.kind in (p.POSITIONAL_ONLY, p.POSITIONAL_OR_KEYWORD) and p.default is inspect.Parameter.empty)]
    settings_only = all(str(p.annotation) in ("int", "float", "bool", "str") for p in params)
    for wrapper in ("Step", "Scan", "Model", "Codec"):
        if ret.startswith(wrapper + "[") and (wrapper not in ("Model", "Codec") or settings_only or wrapper == "Codec"):
            args = _top_level(ret[len(wrapper) + 1 : -1])
            if wrapper == "Codec":
                return f"{short(args[0])} ⇄ {short(args[1])}", ""
            if wrapper == "Step":
                return f"{short(args[0])} → {short(args[0])}", short(args[0])
            if wrapper == "Scan":
                return f"Seq[{short(args[0])}] → Seq[{short(args[2])}]", short(args[1])
            return f"{short(args[0])} → {short(args[1])}", ""
    ins = " × ".join(short(str(p.annotation)) + ("…" if p.kind == p.VAR_POSITIONAL else "") + ("!" if i == changes else "")
                     for i, p in enumerate(params)) or "()"
    return (ins if changes is not None else f"{ins} → {short(ret)}"), ""


def _explanation(obj: Any) -> str:
    doc = inspect.getdoc(obj)
    if not doc:
        raise TypeError(f"{obj!r}: write its explanation in the docstring")
    return " ".join(doc.split("\n\n")[0].split())


def method[T](group: str, name: str, *, state: str = "", side: str = "") -> Callable[[T], T]:
    """Register an expression stub defined here. state says what the state keeps, when the type carries one."""

    def register(obj: T) -> T:
        if group in STATEMENTS:
            raise TypeError(f"{name}: {group} is a group of statements (@statement)")
        signature, state_type = _signature(obj)
        keeps = f"{state_type} — {state}" if state_type and state else state_type or state
        ENTRIES.append(Entry(group, name, signature, keeps, side or GROUPS[group].side, getattr(obj, "__name__", repr(obj)), _explanation(obj), obj))
        return obj

    return register


def statement[T](group: str, name: str, *, changes: int | None = None, side: str = "") -> Callable[[T], T]:
    """Register a statement stub defined here. An update or store statement changes argument number changes in
    place and returns None; a dataset statement (changes=None) changes nothing and returns what it read."""

    def register(obj: T) -> T:
        if group not in STATEMENTS:
            raise TypeError(f"{name}: {group} is a group of expressions (@method)")
        if (changes is not None) != (_signature(obj)[0].rpartition(" → ")[2] == "None"):
            raise TypeError(f"{name}: a statement that changes its argument returns None; one that changes nothing returns a value")
        ENTRIES.append(Entry(group, name, _signature(obj, changes)[0], "", side or GROUPS[group].side, getattr(obj, "__name__", repr(obj)),
                             _explanation(obj), obj, changes))
        return obj

    return register


def alias(group: str, name: str, function: str, explanation: str, signature: str = "", state: str = "", side: str = "") -> None:
    """A method torch (or this file) already names; there is no stub to read."""
    g = GROUPS[group]
    ENTRIES.append(Entry(group, name, signature or g.signature, state, side or g.side, function, explanation))


def changes(f: Any) -> int | None:
    """Which argument f changes in place, when it is an update or store statement (None otherwise)."""
    e = entry_of(f)
    return e.changes if e is not None else None


def is_statement(f: Any) -> bool:
    """Whether f is a statement: it reads a dataset, or changes something; either way, never inside an expression."""
    e = entry_of(f)
    return e is not None and e.group in STATEMENTS


def phenomenon(name: str, explanation: str) -> None:
    ENTRIES.append(Entry("phenomenon", name, "", "", "", "", explanation))


def meta(name: str, groups: str, explanation: str) -> None:
    ENTRIES.append(Entry("meta", name, groups, "", "", "", explanation))


def entry_of(f: Any) -> Entry | None:
    """The entry a function belongs to (a program prints each instruction's group from it)."""
    f = getattr(f, "func", f)  # functools.partial
    by_name = getattr(f, "__name__", None)
    return next((e for e in ENTRIES if e.group in GROUPS and (e.obj is f or (e.obj is None and by_name in [p.strip() for p in e.function.split(",")]))), None)


# ═══ encode: A ⇄ C[classes] ═══════════════════════════════════════════════════════════════════════


@method("encode", "Tokenization (BPE, SentencePiece, bytes)")
def tokenizer(*, kind: str = "bpe", vocab: int = 32_000) -> Codec[list[str], Seq]:
    """Text ⇄ a sequence of ids, built empty: BPE merges, fitted to a corpus by fit_codec (bytes alone: a fixed
    vocabulary of 256, nothing to fit). The tokenizer decides the content type: Seq of classes(vocab)."""
    ...


@method("encode", "Image tokenizer (VQ-VAE, VQGAN)")
def image_tokenizer(images: Grid, *, codes: int = 1024) -> Codec[Grid, Grid]:
    """An image ⇄ a coarser grid of codebook ids: an encoder, vector quantization, a decoder, trained to reconstruct."""
    ...


@method("encode", "Audio codec (EnCodec, residual quantization)")
def audio_codec(audio: Seq, *, codebooks: int = 8, codes: int = 1024) -> Codec[Seq, Product]:
    """A waveform ⇄ frames × codebooks of ids: each codebook quantizes what the previous ones left (residual VQ)."""
    ...


@method("encode", "Encode")
def encode(codec: Codec[Any, Any], x: Any) -> Container:
    """Raw data → its codes (tokenize)."""
    ...


@method("encode", "Decode")
def decode(codec: Codec[Any, Any], codes: Container) -> Any:
    """Codes → raw data (detokenize)."""
    ...


# ═══ convert: C[a] → C′[a] ════════════════════════════════════════════════════════════════════════


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
def retrieval(k: int) -> Model[tuple[Container, Index], Container]:
    """Each element reads the k nearest entries of an external memory (an Index), softmax-weighted."""
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
def quality_filter[C: Container](x: C, scores: Value, *, threshold: float) -> C:
    """Keep the examples of a batch scoring above a threshold (a classifier's, a verifier's, a perplexity). Keeping
    the best fraction needs every batch: keep_best."""
    ...


@method("select", "Data augmentation: token dropout")
def token_dropout(x: Seq, *, p: float = 0.1) -> Seq:
    """Drop a random fraction of the positions; the rest keep their order."""
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


@method("scan", "Scan")
def scan(s: Scan[Any, Any, Any], xs: Seq) -> tuple[Any, Seq]:
    """Carry a Scan's state along a sequence of inputs: the final state, and the outputs."""
    ...


@method("scan", "Optimizers (SGD, momentum, Adam, Muon)", state="the moments of every parameter")
def optimizer(kind: str = "adamw", *, lr: float = 3e-4) -> Scan[Tensor, Tensor, Tensor]:
    """Gradients in, updates out: SGD carries nothing, momentum a velocity, Adam two moments, Muon orthogonalizes. A
    value; fit carries its state from step to step."""
    ...


@method("scan", "Learning-rate schedules", state="the step count")
def warmup_cosine(*, warmup: int, total: int) -> Scan[int, int, float]:
    """A learning rate for each step: linear warmup, then cosine decay."""
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


alias("unroll", "Iteration (rounds of a program)", "repeat",
      "rounds of collecting and learning (policy iteration, self-play, expert iteration): a script's repeat (program.py), "
      "since each round's fit is a statement")


@method("unroll", "Evolution strategies (no gradient)", state="a search distribution over the parameters")
def evolution_strategy(fitness: Callable[[Tensor], Tensor], *, population: int = 32) -> Step[Dist]:
    """Draw a population (a Set of parameters) from the search distribution, score them, move it toward the better
    ones: it works when nothing is differentiable (CMA-ES keeps a covariance too)."""
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


# ═══ search: Index × Value → Set ══════════════════════════════════════════════════════════════════


@method("search", "Index (an empty one)")
def index(like: Value, *, kind: str = "flat") -> Index:
    """An empty index for vectors like these (their size): flat (exact), IVF (clusters), HNSW (a neighbour graph), PQ
    (compressed codes), LSH (hash buckets), inverted (BM25), sum tree (by priority). Statements train and fill it."""
    ...


@method("search", "Nearest-neighbour search")
def search(index: Index, queries: Value, *, k: int = 10) -> Set:
    """The k nearest elements to each query, with their scores: exact for flat, approximate for the rest."""
    ...


@method("search", "Retrieval-augmented generation (retrieve)")
def retrieve(index: Index, documents: Set, query: Seq, embed: Predictor, *, k: int = 4) -> Seq:
    """The k documents nearest the query's embedding, put in front of it as one prompt (then walk the LM's tree)."""
    ...


@method("search", "kNN-LM (a memory of contexts)")
def knn_options(tree: PartialTree[Any, Any], memory: Index, embed: Predictor, *, weight: float = 0.25) -> PartialTree[Any, Any]:
    """Options mixed with the next tokens of the k nearest stored contexts: an LM that can look things up."""
    ...


@method("search", "Hard-negative mining")
def hard_negatives(index: Index, queries: Value, positives: Value, *, k: int = 8) -> Set:
    """The nearest elements that are not the match: harder negatives for a contrastive loss."""
    ...


# ═══ wrap: Module → Module′ (pure) ════════════════════════════════════════════════════════════════


@method("wrap", "Residual connections", side="model")
def residual(f: nn.Module) -> nn.Module:
    """h + f(norm h): the layer edits the stream instead of replacing it."""
    ...


@method("wrap", "Frozen copy (reference / target network)")
def frozen_copy[M: nn.Module](model: M) -> M:
    """A copy that no longer trains: DPO's and RLHF's reference, DQN's target network."""
    ...


alias("wrap", "Low precision: training", "torch.autocast", "bf16 (or fp8) compute, float32 master weights")
alias("wrap", "Hardware-aware kernels", "torch.compile", "torch.compile(module) fuses its ops; attention picks flash kernels by itself", side="model")


@method("wrap", "Reading activations")
def observe(model: nn.Module, at: nn.Module) -> nn.Module:
    """A model that returns what one of its modules computed: the activations probes, SAEs and circuits study."""
    ...


@method("wrap", "Activation patching (circuits)")
def patch(model: nn.Module, at: nn.Module, replace: Callable[[Tensor], Tensor]) -> nn.Module:
    """A model with one module's output replaced (by another input's activations): which components carry a behaviour?"""
    ...


@method("wrap", "Steering along a direction")
def steer(model: nn.Module, at: nn.Module, direction: Tensor, *, alpha: float = 1.0) -> nn.Module:
    """α·direction added to one module's output: features are directions, and moving along one changes the behaviour."""
    ...


# ════════════════════════════════════════════════════════════════════════════════════════════════
# STATEMENTS · on a dataset, or changing something in place (!)
# ════════════════════════════════════════════════════════════════════════════════════════════════

# ═══ fit: X! × Dataset ════════════════════════════════════════════════════════════════════════════


@statement("fit", "Training", changes=0)
def train(model: nn.Module, objective: Callable[..., Tensor], data: Dataset[Any] | None, *, steps: int,
          optimizer: str | Scan[Tensor, Any, Tensor] = "adamw", schedule: Scan[int, int, float] | None = None,
          max_grad_norm: float | None = 1.0, precision: torch.dtype | None = None) -> None:
    """Descend the objective over the dataset's batches: each step, objective(model, *batch) (under autocast to
    precision), its gradient (clipped to max_grad_norm), the optimizer's update of the parameters, in place, at the
    schedule's rate. In training mode only here, so a forward's own effects (BatchNorm statistics) are training's
    too. With no data, the objective makes its own each step (it walks the model: RL, self-play)."""
    ...


alias("fit", "Training step (clipping, mixed precision)", "train", "train's settings: autocast the objective, clip the gradient norm")
alias("fit", "Two-player games (alternating fits)", "repeat, train",
      "no single quantity goes down: each player's fit on its own loss, in turn, inside a repeat (GANs, adversarial training)")


@statement("fit", "Codec fitting (BPE merges, a vocabulary)", changes=0)
def fit_codec(codec: Codec[Any, Any], data: Dataset[Any]) -> None:
    """Fit a codec to a corpus by its own rule, in place: BPE merges the most frequent pairs until the vocabulary is
    full, a unigram model prunes. (A learned codec, VQ-VAE or EnCodec, is a model: train.)"""
    ...


# ═══ assign: Weights! ═════════════════════════════════════════════════════════════════════════════


@statement("assign", "Pretrain → finetune (transfer)", changes=1)
def transfer(source: Weights, target: Weights) -> None:
    """Start a model from a trained one: copy into it the weights whose names and shapes match (the body), keep its head."""
    ...


@statement("assign", "Initialization", changes=0)
def init(model: Weights) -> None:
    """Fan-in normal for matrices, zeros for output projections: each residual block starts as the identity."""
    ...


@statement("assign", "EMA of weights", changes=0)
def ema(average: Weights, model: Weights, *, decay: float = 0.999) -> None:
    """average ← decay·average + (1 − decay)·model, in place: after each round or step, a smoother copy, and the slow
    target of self-distillation and DQN."""
    ...


# ═══ alter: Module! ═══════════════════════════════════════════════════════════════════════════════


@statement("alter", "Freezing", side="agnostic", changes=0)
def freeze(model: nn.Module, trainable: Callable[[str], bool]) -> None:
    """Train only the parameters whose name passes a test."""
    ...


@statement("alter", "LoRA", changes=0)
def lora(model: nn.Module, *, rank: int = 8) -> None:
    """W + B·A on chosen linear layers, the base frozen: the finetuning update, kept low-rank."""
    ...


@statement("alter", "Adapters", changes=0)
def adapter(model: nn.Module, *, bottleneck: int = 8) -> None:
    """After chosen frozen layers, a small bottleneck MLP that trains alone, starting at zero."""
    ...


@statement("alter", "Prompt tuning", changes=0)
def prompt_tuning(model: nn.Module, *, n: int) -> None:
    """Learned vectors in front of a frozen model's input; only they train."""
    ...


@statement("alter", "Quantization (int8 / int4)", changes=0)
def quantize(model: Weights, *, bits: int = 8) -> None:
    """Weights rounded to 2^(bits−1) − 1 levels a side per row: computing as the low-precision model would."""
    ...


alias("alter", "Low precision: inference", "quantize", "int8 / int4 weights")
alias("alter", "Parallelism", "torch.distributed",
      "data, sharded, tensor, pipeline, expert, sequence: fully_shard shards a module in place; DTensor places its weights", side="agnostic")

# ═══ index: Index! × … ════════════════════════════════════════════════════════════════════════════


@statement("index", "Index training (IVF centroids, PQ codebooks)", changes=0)
def train_index(index: Index, sample: Value) -> None:
    """Learn the index's structure from a sample of the vectors, in place: k-means centroids for IVF, codebooks for
    PQ (flat and HNSW have nothing to learn). CREATE INDEX, before any rows."""
    ...


@statement("index", "Index insertion", changes=0)
def insert(index: Index, data: Dataset[Set]) -> None:
    """Add every batch of the dataset to the index, in place (FAISS add, INSERT). An index over an expression's output
    (embed(x)) goes stale when that expression's model is trained again: build it again."""
    ...


@statement("index", "Index deletion", changes=0)
def delete(index: Index, ids: Tensor) -> None:
    """Remove elements from the index by id, in place (FAISS remove_ids, DELETE)."""
    ...


# ═══ buffer: Buffer! × C ══════════════════════════════════════════════════════════════════════════


@statement("buffer", "Continual learning (replay)", changes=0)
def extend[C: Container](buffer: Buffer[C], batch: C) -> None:
    """Add a batch to a bounded buffer, in place, dropping the oldest (or the lowest priority): past experience to train
    on again (continual learning, off-policy RL, prioritized replay)."""
    ...


# ═══ source: … → Dataset ══════════════════════════════════════════════════════════════════════════


@statement("source", "Minibatches")
def batches(*data: Container, size: int, shuffle: bool = True) -> Dataset[Any]:
    """Containers in memory as a dataset: random minibatches of size examples, the containers sliced together (a
    tuple per batch when there are several)."""
    ...


@statement("source", "Collector (walks of a live policy)")
def collector(env: Environment, policy: Predictor, *, steps: int, batch: int) -> Dataset[Path]:
    """Walks of the environment with the policy as it is when each batch is drawn: on-policy data, read inside the
    fit that changes the policy (torchrl's collectors)."""
    ...


@statement("source", "Replay buffer (empty)")
def replay_buffer(*, size: int, priority: bool = False) -> Buffer[Any]:
    """An empty buffer of at most size examples, read by sampling (uniformly, or by priority); extend fills it."""
    ...


@statement("source", "Data mixture weights")
def mix(*datasets: Dataset[Any], weights: list[float]) -> Dataset[Any]:
    """Batches drawn from several datasets in proportion to their weights (web, code, books): the mixture as a dataset."""
    ...


@statement("source", "An expression on each batch")
def each(f: Callable[..., Any], data: Dataset[Any]) -> Dataset[Any]:
    """f applied to every batch, once per batch as the dataset batches it (never re-batched, since f may depend on
    what a batch holds): how a statement reads an expression's output, embeddings to index, features to probe."""
    ...


# ═══ curate: Dataset → Dataset ════════════════════════════════════════════════════════════════════


@statement("curate", "Data deduplication")
def dedup[C: Container](data: Dataset[C]) -> Dataset[C]:
    """Drop exact copies across the whole dataset (by hash): the first of each kept."""
    ...


@statement("curate", "Near-duplicate removal (MinHash / LSH)")
def drop_near_duplicates[C: Container](data: Dataset[C], *, threshold: float = 0.8) -> Dataset[C]:
    """Drop examples whose MinHash signatures collide with an earlier one's above a similarity threshold: dedup beyond
    exact copies."""
    ...


@statement("curate", "Data filtering by fraction")
def keep_best[C: Container](data: Dataset[C], score: Callable[[C], Value], *, keep: float = 0.5) -> Dataset[C]:
    """The best-scoring fraction of the whole dataset: a quantile of every score first, then a threshold per batch."""
    ...


@statement("curate", "Curriculum learning")
def curriculum[C: Container](data: Dataset[C], difficulty: Callable[[C], Value]) -> Dataset[C]:
    """The dataset sorted by difficulty, easy first: a global sort, so it reads every batch before the first comes out."""
    ...


# ═══ reduce: Dataset → a ══════════════════════════════════════════════════════════════════════════


@statement("reduce", "Evaluation (a metric over a dataset)")
def evaluate(metric: Callable[..., Tensor], data: Dataset[Any]) -> Tensor:
    """A per-batch metric (accuracy, a loss, perplexity), averaged over every batch, weighted by their sizes."""
    ...


@statement("reduce", "Accumulate (a scan across batches)")
def accumulate(s: Scan[Any, Any, Any], data: Dataset[Any]) -> Any:
    """A Scan's state carried across a dataset's batches, its final state returned: normalization statistics, a
    histogram, any metric that is not a mean."""
    ...


@statement("reduce", "First batch")
def peek[C](data: Dataset[C]) -> C:
    """The first batch of a dataset: what an expression can see of it, its type, to build a network or an index for."""
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
meta("Amortize, or search", "map, walk, reshape, search", "one forward pass that answers at once, or a tree grown by search")
meta("Every loss is a likelihood", "score", "loss(pred, target): what the target holds picks which")
meta("Not every objective is a loss", "fold, score", "a return is a fold of a path; its gradient comes from scoring the path, or there is none")
meta("Structure decides the computation", "aggregate", "the container picks what each element reads")

# ═══ Printing ═════════════════════════════════════════════════════════════════════════════════════

NEW = {"Tokenization (BPE, SentencePiece, bytes)", "Image tokenizer (VQ-VAE, VQGAN)", "Audio codec (EnCodec, residual quantization)", "Encode", "Decode", "Index (an empty one)", "Index training (IVF centroids, PQ codebooks)", "Index insertion", "Index deletion", "Codec fitting (BPE merges, a vocabulary)", "Collector (walks of a live policy)", "Replay buffer (empty)", "An expression on each batch", "Data filtering by fraction", "Evaluation (a metric over a dataset)", "Accumulate (a scan across batches)", "First batch", "Nearest-neighbour search", "Retrieval-augmented generation (retrieve)", "kNN-LM (a memory of contexts)",
       "Near-duplicate removal (MinHash / LSH)", "Hard-negative mining", "Fill", "Autoregressive order", "Unmasking (MaskGIT, masked diffusion)", "Sampling a flow (diffusion)", "Diffusion forcing",
       "Diffusion forcing sampling", "Training", "Optimizers (SGD, momentum, Adam, Muon)", "KV caching", "A path from a tree",
       "Sampling", "Several paths per root (best-of-N, groups)", "Teacher forcing (walking the data's path)", "Supervised", "Scan", "Log-probability", "Retrieval accuracy", "Blank (where generation starts)", "A condition, fixed", "Objective (a split, then its loss)", "Minibatches",
       "Classifier-free guidance training", "Recurrence (RNN / LSTM / GRU)", "State-space models / linear recurrences", "Walk"}


def _paint(code: str) -> Callable[[str], str]:
    """An ANSI colour, or nothing when the output isn't a terminal (or NO_COLOR is set)."""
    import os
    import sys

    on = (sys.stdout.isatty() or os.environ.get("FORCE_COLOR")) and not os.environ.get("NO_COLOR")
    return (lambda t: f"\033[{code}m{t}\033[0m") if on else (lambda t: t)


def print_catalog(explain: bool = False) -> None:
    """Grouped by framework, family and group, one line per method: name: signature · state. explain adds what each one does."""
    bold, group, sig, state_c, dim, star = _paint("1"), _paint("1;36"), _paint("33"), _paint("35"), _paint("2"), _paint("32")
    for heading, families in FRAMEWORKS.values():
        print(bold(f"\n━━━━ {heading} ━━━━"))
        for family in families:
            print(bold(f"\n══ {FAMILIES[family]} ══"))
            for g in (g for g in GROUPS.values() if g.family == family):
                print(f"\n{group(g.name)}  {sig(g.signature)}\n  {dim(g.explanation)}")
                for e in (e for e in ENTRIES if e.group == g.name):
                    mark = star("★ ") if e.name in NEW else ""
                    state = f"  {dim('·')} {state_c('state: ' + e.state)}" if e.state else ""
                    print(f"  - {mark}{bold(e.name)}: {sig(e.signature)}{state}")
                    if explain:
                        print(dim(f"      {e.explanation}"))
    print(bold("\n══ META-PRIMITIVES · properties of the groups ══\n"))
    for e in (e for e in ENTRIES if e.group == "meta"):
        print(f"  - {bold(e.name)}: {group(e.signature)}" + (dim(f"\n      {e.explanation}") if explain else ""))
    print(bold("\n══ PHENOMENA · observed, not computed ══\n"))
    for e in (e for e in ENTRIES if e.group == "phenomenon"):
        print(f"  - {bold(e.name)}" + (dim(f"\n      {e.explanation}") if explain else ""))
    rows = [e for e in ENTRIES if e.group in GROUPS]
    print(dim(f"\n{len(rows)} methods in {len(GROUPS)} groups, {sum(bool(e.state) for e in rows)} carrying a state · "
              f"{sum(e.group in STATEMENTS for e in rows)} statements ({sum(e.changes is not None for e in rows)} changing their argument) · "
              f"{sum(e.side == 'agnostic' for e in rows)} model-agnostic, {sum(e.side == 'model' for e in rows)} model-dependent"))


if __name__ == "__main__":
    import sys

    print_catalog(explain="--explain" in sys.argv)
