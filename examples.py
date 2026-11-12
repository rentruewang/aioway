"""
examples.py: use cases written against the interface (catalog.py), each a config like the catalog's methods: a
@dataclass(frozen=True) whose fields are its settings and whose __call__ takes the data. They show how the methods fit together; they run once the methods
are implemented.

The catalog's rule decides what kind each one is. A use case that trains or reads a whole dataset is a statement
(Stmt); one that changes what it is given is an update, a Fit like Train, and returns None; one that only reads is
an expression (Expr). The examples are not registered in the catalog: they have no title.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import torch
from sklearn.cluster import DBSCAN, KMeans
from sklearn.decomposition import PCA
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from torch import Tensor, nn

import catalog as C
from catalog import Expr, Fit, Stmt
from data import Codec, Container, Dataset, Environment, Grid, Holes, Indexed, Model, Path, Predictor, Set, Subset, Temporal, Value

# ═══ Statements that build a model, train it, and return it ═══════════════════════════════════════


@dataclass(frozen=True)
class Classify(Stmt):
    """Any container → a label: the types pick the layers, the label picks the loss. A statement (it trains), though
    it changes nothing it is given: the model is new."""

    steps: int = 1000
    batch: int = 64

    def __call__(self, x: Container, labels: Value, /) -> Predictor:
        model = C.Network()(x, labels)
        C.Train(steps=self.steps)(model, C.Objective(split=C.Supervised()), C.Batches(size=self.batch)(x, labels))
        return model


@dataclass(frozen=True)
class Generate(Stmt):
    """Learn any container by hiding parts of it, then generate n by filling the holes back in. The type picks the
    split (NextToken, Mask or Noise); AllAtOnce fills every hole at once."""

    steps: int = 1000
    fill_steps: int = 16
    n: int = 8

    def __call__(self, data: Container, /) -> Container:
        split = C.AutoSplit()
        ex = split(data)
        model = C.Network()(ex.input.known, ex.target)
        C.Train(steps=self.steps)(model, C.Objective(split=split), C.Batches(size=64)(data))
        return C.Fill(steps=self.fill_steps)(C.Blank()(data[: self.n]), model, C.AllAtOnce())


@dataclass(frozen=True)
class LanguageModel(Stmt):
    """Next-token prediction over a dataset, then generation as walking the LM's tree with a beam."""

    steps: int = 1000
    beam: int = 4
    length: int = 32

    def __call__(self, corpus: Dataset[Temporal], prompt: Temporal, /) -> Path:
        ex = C.NextToken()(C.Peek()(corpus))
        lm = C.Network()(ex.input.known, ex.target)
        C.Train(steps=self.steps)(lm, C.Objective(split=C.NextToken()), corpus)
        return C.Walk(steps=self.length)(C.LMTree()(lm, prompt), C.Beam(k=self.beam))


@dataclass(frozen=True)
class Plan(Stmt):
    """Learn what follows each action from experience, then act by planning inside that model."""

    steps: int = 1000
    horizon: int = 50

    def __call__(self, env: Environment, experience: Path, /) -> Path:
        batch = C.Transitions()(experience)
        ex = C.Dynamics()(batch)
        world = C.Network()(ex.input.known, ex.target)
        C.Train(steps=self.steps)(world, C.Objective(split=C.Dynamics()), C.Batches(size=64)(batch))
        tree = C.EnvTree()(env)
        return C.Walk(steps=self.horizon)(C.MPC()(tree, C.WorldModel()(tree, world)), C.Greedy())


@dataclass(frozen=True)
class SearchDocuments(Stmt):
    """CREATE INDEX, then look up: training and filling the index are statements, searching it an expression."""

    kind: str = "ivf"
    k: int = 10

    def __call__(self, embed: Predictor, documents: Set, queries: Value, /) -> Set:
        vectors = Value(embed(documents))
        memory = C.NewIndex(kind=self.kind)(vectors)
        C.TrainIndex()(memory, vectors)
        C.Insert()(memory, C.Batches(size=1024, shuffle=False)(vectors))
        return memory.search(queries, self.k)[0]


@dataclass(frozen=True)
class Curation(Stmt):
    """Mix the sources, drop the copies, keep the better fraction, easy first: each step reads every batch. What comes
    out is a Subset of the mixture, and its repr says which steps chose it."""

    weights: list[float]
    keep: float = 0.5

    def __call__(self, *sources: Indexed[Temporal], quality: Callable[[Temporal], Value]) -> Subset[Temporal]:
        data = C.NearDedup()(C.Dedup()(C.Mix(weights=self.weights)(*sources)))
        return C.Curriculum()(C.KeepBest(keep=self.keep)(data, quality), quality)


# ═══ Updates: they change the model they are given, and return None ═══════════════════════════════


@dataclass(frozen=True)
class ReinforcementLearning(Fit[Predictor, [Environment]]):
    """Rounds of walking the environment's tree with the policy and learning from the paths (REINFORCE with a
    batch-mean baseline). The policy changes in place."""

    rounds: int = 100
    steps: int = 10
    horizon: int = 50
    gamma: float = 0.99

    def __call__(self, policy: Predictor, env: Environment, /) -> None:
        def objective(policy: Predictor) -> Tensor:
            path = C.Walk(steps=self.horizon)(C.EnvTree()(env, policy), C.Sample())
            return C.Reinforce()(path, C.MeanBaseline()(C.DiscountedReturn(gamma=self.gamma)(path)))

        for _ in range(self.rounds):
            C.Train(steps=self.steps)(policy, objective, None)


@dataclass(frozen=True)
class Align(Fit[Predictor, [Temporal, Temporal]]):
    """Preferences, without a reward model: DPO against a frozen copy of the policy as it was. The policy changes in
    place."""

    steps: int = 500
    beta: float = 0.1
    batch: int = 8

    def __call__(self, policy: Predictor, chosen: Temporal, rejected: Temporal, /) -> None:
        reference = C.FrozenCopy()(policy)
        C.Train(steps=self.steps)(policy, C.DPO(reference=reference, beta=self.beta), C.Batches(size=self.batch)(chosen, rejected))


@dataclass(frozen=True)
class Finetune(Fit[nn.Module, [Dataset[Temporal]]]):
    """A pretrained model, frozen, with a low-rank update trained on new data. The model changes in place: LoRA adds
    the update, Train fits it."""

    rank: int = 8
    steps: int = 500

    def __call__(self, base: nn.Module, data: Dataset[Temporal], /) -> None:
        C.LoRA(rank=self.rank)(base)
        C.Train(steps=self.steps)(base, C.Objective(split=C.NextToken()), data)


# ═══ Expressions: they only read ══════════════════════════════════════════════════════════════════


@dataclass(frozen=True)
class Inpaint(Expr):
    """Fill only some holes: the known pixels stay at level 0."""

    steps: int = 16

    def __call__(self, model: Predictor, image: Grid, missing: Tensor, /) -> Grid:
        return C.Fill(steps=self.steps)(Holes(image, missing.float()), model, C.AllAtOnce())


@dataclass(frozen=True)
class DecodeWithSearch(Expr):
    """The same tree, walked four ways: one state type (PartialTree), different choosers and reshapes."""

    steps: int = 32

    def __call__(self, lm: Predictor, prompt: Temporal, verify: Callable[[Tensor], Tensor], /) -> dict[str, Any]:
        tree = C.LMTree()(lm, prompt)
        walk = C.Walk(steps=self.steps)
        sampled = walk(C.TopP(p=0.9)(C.Temperature(t=0.8)(tree)), C.Sample())
        n = walk(tree, C.NPaths(n=8, choose=C.Sample()))
        return {"sampled": sampled, "beam": walk(tree, C.Beam(k=4)), "best of 8": C.BestOfN(n=8)(n, Value(verify(n.end))),
                "searched": walk(C.MCTS(simulations=64)(tree), C.Greedy())}


@dataclass(frozen=True)
class Interpret(Expr):
    """Read a layer's activations, steer along a direction, and attribute the output to the input: all reads."""

    alpha: float = 1.0

    def __call__(self, model: Predictor, layer: nn.Module, x: Container, direction: Tensor, /) -> dict[str, Any]:
        return {"activations": C.Observe()(model, layer)(x), "steered": C.Steer(alpha=self.alpha)(model, layer, direction)(x),
                "attribution": C.Attribution()(x, model)}


# ═══ Statements that fit a new model to what a model computes ════════════════════════════════════


@dataclass(frozen=True)
class Probe(Stmt):
    """Is a feature readable along one direction? Fit a linear probe to a layer's activations (closed form), and a
    sparse autoencoder to decompose them. A statement: both are fitted, though the model is only read."""

    features: int = 1024
    steps: int = 1000

    def __call__(self, model: Predictor, layer: nn.Module, x: Container, labels: Value, /) -> dict[str, Any]:
        acts = Value(C.Observe()(model, layer)(x))
        probe = C.LinearProbe()(acts, labels)
        C.LeastSquares()(probe, C.Batches(size=1024)(acts, labels))
        sae = C.SAE(d=acts.data.shape[-1], features=self.features)()
        C.Train(steps=self.steps)(sae, C.Objective(split=C.Reconstruction()), C.Batches(size=256)(acts))
        return {"probe": probe, "features": sae}


# ═══ Classical machine learning: scikit-learn's estimators, made, fitted, then called ═════════════


@dataclass(frozen=True)
class Tabular(Stmt):
    """Feature vectors and a target: features standardized by the training data only, then gradient-boosted trees,
    logistic (or linear) regression and k nearest neighbours, each scored on held-out data. The target's type picks
    classification or regression for the last two; the estimator says which for the trees."""

    trees: int = 200
    k: int = 5

    def __call__(self, x: Value, y: Value, test_x: Value, test_y: Value, /) -> dict[str, Tensor]:
        scaler = C.Standardize(StandardScaler())()
        C.FitEstimator()(scaler, C.Batches(size=1024)(x))
        x, test_x = scaler.transform(x), scaler.transform(test_x)
        data = C.Batches(size=1024)(x, y)
        boosted = C.Trees(HistGradientBoostingClassifier(max_iter=self.trees))()
        C.FitEstimator()(boosted, data)
        linear = C.Linear()(x, y)
        C.Train(steps=500)(linear, C.Objective(split=C.Supervised()), data)
        memory = C.NewIndex()(x)
        C.Insert()(memory, C.Batches(size=1024, shuffle=False)(x))
        return {"boosted trees": C.Accuracy()(boosted.predict(test_x).data, test_y), "linear": C.Accuracy()(linear(test_x), test_y),
                "kNN": C.Accuracy()(C.KNN(k=self.k)(memory, test_x, y), test_y)}


@dataclass(frozen=True)
class Cluster(Stmt):
    """No labels: project onto the principal components, find k clusters there, and the dense regions without a k
    (DBSCAN, which labels only the data it is fitted to: a dataset statement)."""

    dim: int = 16
    k: int = 10

    def __call__(self, x: Value, /) -> dict[str, Tensor]:
        pca = C.PCA(PCA(n_components=self.dim))()
        C.FitEstimator()(pca, C.Batches(size=1024)(x))
        z = pca.transform(x)
        kmeans = C.KMeans(KMeans(n_clusters=self.k))()
        C.FitEstimator()(kmeans, C.Batches(size=1024)(z))
        return {"k-means": kmeans.predict(z).data, "density": C.DBSCAN(DBSCAN(eps=0.5))(C.Batches(size=1024, shuffle=False)(z))}


@dataclass(frozen=True)
class CrossValidate(Stmt):
    """k folds, each class in the same proportion in each: a new network trained on each fold's train part and scored
    on its test part. Each part is a Subset, so the run can log which examples each fold trained on."""

    k: int = 5
    steps: int = 500

    def __call__(self, x: Container, labels: Value, /) -> dict[str, Any]:
        loss = C.Objective(split=C.Supervised())
        folds = C.KFold(StratifiedKFold(n_splits=self.k, shuffle=True), label=lambda batch: batch[1].data)
        scores: list[Tensor] = []
        for train, test in folds(C.Batches(size=64)(x, labels)):
            model = C.Network()(x, labels)
            C.Train(steps=self.steps)(model, loss, train)
            scores.append(C.Evaluate()(test, loss, model))
        return {"mean": torch.stack(scores).mean(), "folds": scores}


# ═══ More well-known algorithms ═══════════════════════════════════════════════════════════════════


@dataclass(frozen=True)
class GPT(Stmt):
    """GPT end to end: a tokenizer fitted to the raw text, a causal Transformer trained on next-token prediction with
    warmup and cosine decay, then nucleus sampling at a temperature, with a KV cache."""

    vocab: int = 32_000
    steps: int = 1000
    length: int = 64

    def __call__(self, texts: Dataset[list[str]], prompt: list[str], /) -> list[str]:
        codec = C.Tokenizer(vocab=self.vocab)()
        C.FitCodec()(codec, texts)
        corpus: Dataset[Temporal] = C.MapBatches()(codec.encode, texts)
        ex = C.NextToken()(C.Peek()(corpus))
        lm = C.Network(d=256, depth=6)(ex.input.known, ex.target)
        schedule = C.WarmupCosine(warmup=self.steps // 10, total=self.steps)
        C.Train(steps=self.steps, schedule=schedule)(lm, C.Objective(split=C.NextToken()), corpus)
        tree = C.KVCache()(C.TopP(p=0.9)(C.Temperature(t=0.8)(C.LMTree()(lm, codec.encode(prompt)))))
        return codec.decode(C.Walk(steps=self.length)(tree, C.Sample()).end)


@dataclass(frozen=True)
class BERT(Stmt):
    """BERT: pretrain an encoder by predicting 15% masked tokens, then fine-tune a classifier that starts from its
    weights (Transfer copies the body, keeps the new head)."""

    ratio: float = 0.15
    steps: int = 1000

    def __call__(self, corpus: Dataset[Temporal], texts: Temporal, labels: Value, /) -> Predictor:
        ex = C.Mask(ratio=self.ratio)(C.Peek()(corpus))
        encoder = C.Network(d=256, depth=6)(ex.input.known, ex.target)
        C.Train(steps=self.steps)(encoder, C.Objective(split=C.Mask(ratio=self.ratio)), corpus)
        classifier = C.Network(d=256, depth=6)(texts, labels)
        C.Transfer()(classifier, encoder)
        C.Train(steps=self.steps // 10)(classifier, C.Objective(split=C.Supervised()), C.Batches(size=32)(texts, labels))
        return classifier


@dataclass(frozen=True)
class InstructionTuning(Fit[Predictor, [Dataset[Temporal]]]):
    """Instruction tuning with LoRA: a low-rank update on a frozen base, the loss on the response only (the prompt is
    read, never predicted). The model changes in place."""

    prompt_len: int
    rank: int = 8
    steps: int = 300

    def __call__(self, base: Predictor, chats: Dataset[Temporal], /) -> None:
        C.LoRA(rank=self.rank)(base)
        C.Train(steps=self.steps)(base, C.Objective(split=C.SFT(prompt_len=self.prompt_len)), chats)


@dataclass(frozen=True)
class Distill(Stmt):
    """Knowledge distillation: a small student trained toward a large teacher's softened predictions."""

    steps: int = 500

    def __call__(self, teacher: Predictor, x: Container, y: Container, /) -> Predictor:
        student = C.Network(d=64, depth=2)(x, y)

        def objective(model: Predictor, batch: Container) -> Tensor:
            ex = C.Distillation()(batch, teacher)
            return C.Loss()(model(ex.input.known), ex.target)

        C.Train(steps=self.steps)(student, objective, C.Batches(size=64)(x))
        return student


@dataclass(frozen=True)
class SpeculativeDecoding(Expr):
    """Speculative decoding: a small model drafts k tokens, the big one checks them in one pass. The same output as
    the big model's greedy decoding, in fewer of its passes."""

    k: int = 4
    length: int = 64

    def __call__(self, big: Predictor, small: Predictor, prompt: Temporal, /) -> Temporal:
        draft = C.LMTree()(small, prompt)
        return C.Walk(steps=self.length)(C.LMTree()(big, prompt), C.Speculative(draft=draft, k=self.k)).end


@dataclass(frozen=True)
class SelfConsistency(Expr):
    """Self-consistency: n chains of thought per prompt, and the most frequent final answer."""

    n: int = 8
    length: int = 256

    def __call__(self, lm: Predictor, prompts: Temporal, answer: Callable[[Any], Value], /) -> Value:
        paths = C.Walk(steps=self.length)(C.LMTree()(lm, prompts), C.NPaths(n=self.n, choose=C.Sample()))
        return C.MajorityVote(n=self.n)(answer(paths.end))


@dataclass(frozen=True)
class MAE(Stmt):
    """Masked autoencoder: an image as a sequence of patches, 75% of them hidden and predicted from the rest."""

    patch: int = 16
    ratio: float = 0.75
    steps: int = 1000

    def __call__(self, images: Grid, /) -> Predictor:
        patches = C.Flatten()(C.Patchify(size=self.patch)(images))
        ex = C.Mask(ratio=self.ratio)(patches)
        model = C.Network(d=256, depth=8)(ex.input.known, ex.target)
        C.Train(steps=self.steps)(model, C.Objective(split=C.Mask(ratio=self.ratio)), C.Batches(size=256)(patches))
        return model


def augment(x: Any) -> Any:
    """A random crop, flip and colour jitter: called twice, two views of the same images."""
    return C.ColorJitter(strength=0.4)(C.RandomFlip()(C.RandomCrop(padding=4)(x)))


@dataclass(frozen=True)
class SimCLR(Stmt):
    """SimCLR: two augmented views of each image, embedded so that each finds its partner among the batch."""

    dim: int = 128
    steps: int = 1000

    def __call__(self, images: Grid, /) -> Model[Container, Value]:
        encoder = C.Encoder(dim=self.dim)(images)

        def objective(model: Model[Container, Value], x: Grid) -> Tensor:
            ex = C.Contrastive()(x, augment)
            return C.InfoNCE()(model(ex.input.known), model(ex.target))

        C.Train(steps=self.steps)(encoder, objective, C.Batches(size=256)(images))
        return encoder


@dataclass(frozen=True)
class FlowMatching(Stmt):
    """Class-conditional flow matching with classifier-free guidance: learn the velocity from noise to data with the
    label dropped 10% of the time, then integrate the guided velocity from pure noise."""

    w: float = 3.0
    steps: int = 1000
    fill_steps: int = 50

    def __call__(self, images: Grid, labels: Value, /) -> Grid:
        ex = C.Noise()(images)
        velocity = C.Network(d=256, depth=8)(C.Condition()(ex.input.known, labels), ex.target)

        def objective(model: Predictor, x: Grid, y: Value) -> Tensor:
            noised = C.Noise()(x)
            return C.Loss()(model(C.Condition()(noised.input.known, y)), noised.target)

        C.Train(steps=self.steps)(C.CFGDropout(p=0.1)(velocity), objective, C.Batches(size=128)(images, labels))
        guided = C.CFG(w=self.w)(C.Given()(velocity, labels), C.Given()(velocity, None))
        return C.Fill(steps=self.fill_steps)(C.Blank()(images), guided, C.AllAtOnce())


@dataclass(frozen=True)
class MaskGIT(Stmt):
    """MaskGIT: images as grids of codebook ids (a fitted image tokenizer), trained to fill random masks; generated by
    unmasking the surest ids first, a few more each step."""

    steps: int = 1000
    fill_steps: int = 12

    def __call__(self, images: Grid, codec: Codec[Grid, Grid], /) -> Grid:
        tokens = codec.encode(images)
        ex = C.Mask()(tokens)
        model = C.Network(d=256, depth=8)(ex.input.known, ex.target)
        C.Train(steps=self.steps)(model, C.Objective(split=C.Mask()), C.Batches(size=128)(tokens))
        return codec.decode(C.Fill(steps=self.fill_steps)(C.Blank()(tokens), model, C.MostConfident()))
