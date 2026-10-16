# Captum capacity probes

Find the parts of a model that have not converged or need more capacity: a Captum tour on a toy problem with a
known answer, plus a plain-PyTorch "expressivity bottleneck" measure that also runs on other models.

| File | What |
|---|---|
| `01_data_and_models.py` | §1–3 the 16 features, the two models (narrow and wide `fc2`), training |
| `02_feature_attribution.py` | §4–11 Captum primary methods, `NoiseTunnel`, summaries, `captum.metrics` |
| `03_model_attribution.py` | §12–16 layer and neuron methods |
| `04_training_examples.py` | §17–20 `captum.influence` |
| `05_capacity_probes.py` | §21–23 `captum.module` gates, `TCAV`, `captum.robust` |
| `06_expressivity_bottleneck.py` | §24 would more units help? (`expressivity.py`) |
| `07_verdict.py` | §25–27 one verdict per layer, width sweep, recipe |
| `08_digits_bottleneck.py` | `expressivity.py` on scikit-learn digits with an `nn.Sequential` MLP, incl. growing |
| `toy.py` | shared setup for 01–07 (data, models, cached training, helpers) |
| `expressivity.py` | the bottleneck measure: `measure`, `check`, `grow`, `linear_sites` |
| `run_all.py` | runs the notebooks and saves them with outputs to `executed/*.ipynb` |

The notebooks are jupytext percent files: open them as notebooks (VS Code, PyCharm, Jupyter with jupytext) or run
them with `python`. `executed/` holds the same notebooks already run, with all outputs.

```
pip install torch captum matplotlib pandas scikit-learn
python run_all.py                     # all notebooks, about 3 minutes on a laptop CPU
python run_all.py 08_digits_bottleneck.py
```

Each file runs on its own. The first one to run trains the two toy models and caches them in `cache/`.
Tested with torch 2.14 and captum 0.9.0.
