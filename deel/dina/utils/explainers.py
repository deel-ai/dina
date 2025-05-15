"""
Explainer utility modules for the explainers used for DINA

Provides access to a set of predefined explainers for evaluation,
some directly from the Xplique library and others that has been
adapted from the original code (but optimized for this repo).
"""

from enum import Enum
import tensorflow as tf

from xplique.attributions import (
    Saliency, GradCAM, GradCAMPP, VarGrad, SmoothGrad, SquareGrad,
    IntegratedGradients, Rise, GradientInput, KernelShap, Occlusion
)

from ..dina_attributions import Lime, HsicAttributionMethod, SobolAttributionMethod

class DINAExplainers(Enum):
    """
    Enum class for the explainers used for DINA.
    """
    SALIENCY = "Saliency"
    GRADCAM = "GradCAM"
    GRADCAMPP = "GradCAMPP"
    VARGRAD = "VarGrad"
    SMOOTHGRAD = "SmoothGrad"
    SQUAREGRAD = "SquareGrad"
    INTEGRATEDGRADIENTS = "IntegratedGradients"
    RISE = "Rise"
    GRADIENTINPUT = "GradientInput"
    KERNELSHAP = "KernelShap"
    OCCLUSION = "Occlusion"
    LIME = "Lime"
    HSICATTRIBUTIONMETHOD = "HsicAttributionMethod"
    SOBOLATTRIBUTIONMETHOD = "SobolAttributionMethod"

EXPLAINER_ARGS = {
    DINAExplainers.SALIENCY.value: {"reducer": None},
    DINAExplainers.GRADCAM.value: {},
    DINAExplainers.GRADCAMPP.value: {},
    DINAExplainers.VARGRAD.value: {"nb_samples": 100, "noise": 0.62, "reducer": None},
    DINAExplainers.SMOOTHGRAD.value: {"nb_samples": 175, "noise": 0.11, "reducer": None},
    DINAExplainers.SQUAREGRAD.value: {"nb_samples": 194, "noise": 0.63, "reducer": None},
    DINAExplainers.INTEGRATEDGRADIENTS.value: {"baseline_value": "mean", "steps": 100, "reducer": None},
    DINAExplainers.GRADIENTINPUT.value: {"reducer": None},
    DINAExplainers.RISE.value: {"nb_samples": 8000, "grid_size": 10, "mask_value": 0.0},
    DINAExplainers.KERNELSHAP.value: {"nb_samples": 400},
    DINAExplainers.OCCLUSION.value: {"patch_size": 5, "patch_stride": 3, "occlusion_value": 0.0,},
    DINAExplainers.LIME.value: {"nb_samples": 2000, "distance_mode": "cosine", "kernel_width": 46.0},
    DINAExplainers.HSICATTRIBUTIONMETHOD.value: {"grid_size": 11, "estimator_batch_size": 768},
    DINAExplainers.SOBOLATTRIBUTIONMETHOD.value: {"grid_size": 11}
}

@tf.function
def dino_operator(model, inputs, targets):
    scores = tf.reduce_sum(model(inputs) * targets, axis=-1)
    return scores

def get_explainer_args(explainer_name: str):
    """
    Retrieve the arguments for a given explainer.

    Args:
        explainer_name (str): Name of the explainer (must match one of DINAExplainers values).

    Returns:
        dict: Arguments for the explainer.
    """
    # Check if explainer_name is a valid value in the Enum
    assert explainer_name in [explainer.value for explainer in DINAExplainers], \
        f"Explainer '{explainer_name}' not found in DINAExplainers. Should be one of {[explainer.value for explainer in DINAExplainers]}"
    
    return EXPLAINER_ARGS[explainer_name]

def get_explainer(explainer_name: str, model: tf.keras.Model, batch_size:int, explainer_args: dict = None):
    """
    Retrieve a predefined explainer by name.

    Args:
        explainer_name (str): Name of the explainer (must match one of DINAExplainers values).
        model (tf.keras.Model): The model one is explaining.
        batch_size (int): Number of samples to evaluate at once.

    Returns:
        object: Instantiated explainer.
    """
    # Check if explainer_name is a valid value in the Enum
    assert explainer_name in [explainer.value for explainer in DINAExplainers], \
        f"Explainer '{explainer_name}' not found in DINAExplainers. Should be one of {[explainer.value for explainer in DINAExplainers]}"

    # Map of explainer names to their respective classes
    explainer_map = {
        DINAExplainers.SALIENCY.value: lambda: Saliency(model=model, batch_size=batch_size, **explainer_args),
        DINAExplainers.GRADCAM.value: lambda: GradCAM(model=model, batch_size=batch_size, **explainer_args),
        DINAExplainers.GRADCAMPP.value: lambda: GradCAMPP(model=model, batch_size=batch_size, **explainer_args),
        DINAExplainers.VARGRAD.value: lambda: VarGrad(model=model, batch_size=batch_size, **explainer_args),
        DINAExplainers.SMOOTHGRAD.value: lambda: SmoothGrad(model=model, batch_size=batch_size, **explainer_args),
        DINAExplainers.SQUAREGRAD.value: lambda: SquareGrad(model=model, batch_size=batch_size, **explainer_args),
        DINAExplainers.INTEGRATEDGRADIENTS.value: lambda: IntegratedGradients(model=model, batch_size=batch_size, **explainer_args),
        DINAExplainers.RISE.value: lambda: Rise(model=model, batch_size=batch_size, **explainer_args),
        DINAExplainers.GRADIENTINPUT.value: lambda: GradientInput(model=model, batch_size=batch_size, **explainer_args),
        DINAExplainers.KERNELSHAP.value: lambda: KernelShap(model=model, batch_size=batch_size, **explainer_args),
        DINAExplainers.OCCLUSION.value: lambda: Occlusion(model=model, batch_size=batch_size, **explainer_args),
        DINAExplainers.LIME.value: lambda: Lime(model=model, batch_size=batch_size, **explainer_args),
        DINAExplainers.HSICATTRIBUTIONMETHOD.value: lambda: HsicAttributionMethod(model=model, batch_size=batch_size, **explainer_args),
        DINAExplainers.SOBOLATTRIBUTIONMETHOD.value: lambda: SobolAttributionMethod(model=model, batch_size=batch_size, **explainer_args)
    }
    return explainer_map[explainer_name]()
