# Copyright (c) AIoWay Authors - All Rights Reserved

from .explainers import Explainer, explainer_dcls

from captum import attr


@explainer_dcls
class SaliencyExplainer(Explainer):
    _CAPTUM_CLASS = attr.Saliency

    abs: bool = False


@explainer_dcls
class IntegratedGradientsExplainer(Explainer):
    _CAPTUM_CLASS = attr.IntegratedGradients

    n_steps: int = 32


@explainer_dcls
class DeepLiftExplainer(Explainer):
    _CAPTUM_CLASS = attr.DeepLift


@explainer_dcls
class FeatureAblationExplainer(Explainer):
    _CAPTUM_CLASS = attr.FeatureAblation

    perturbations_per_eval: int = 8


@explainer_dcls
class KernelShapExplainer(Explainer):
    _CAPTUM_CLASS = attr.KernelShap

    n_samples: int = 128
