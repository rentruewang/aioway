# Copyright (c) AIoWay Authors - All Rights Reserved

from .expls import Expl, expl_dcls

from captum import attr


__all__ = [
    "SaliencyExpl",
    "IntegratedGradientsExpl",
    "DeepLiftExpl",
    "FeatureAblationExpl",
    "KernelShapExpl",
]


@expl_dcls
class SaliencyExpl(Expl):
    _CAPTUM_CLASS = attr.Saliency

    abs: bool = False


@expl_dcls
class IntegratedGradientsExpl(Expl):
    _CAPTUM_CLASS = attr.IntegratedGradients

    n_steps: int = 32


@expl_dcls
class DeepLiftExpl(Expl):
    _CAPTUM_CLASS = attr.DeepLift


@expl_dcls
class FeatureAblationExpl(Expl):
    _CAPTUM_CLASS = attr.FeatureAblation

    perturbations_per_eval: int = 8


@expl_dcls
class KernelShapExpl(Expl):
    _CAPTUM_CLASS = attr.KernelShap

    n_samples: int = 128
