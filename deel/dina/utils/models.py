"""
Model utility module.

Provides access to a set of predefined vision models for evaluation, all sourced from
the Keras CV Attention Models (Kecam) library:
https://github.com/leondgarse/keras_cv_attention_models

Each model is instantiated with a softmax classifier head.
"""

from enum import Enum

import tensorflow as tf

from keras_cv_attention_models import (
    convnext, resnest, maxvit, mlp_family, beit, resnet_family,
    efficientnet, inceptionnext
)

class InvestigatedModels(Enum):
    """
    Enum class for the supported model names.
    """
    BEITV2 = "BeitV2"
    CONVNEXTV2 = "ConvNeXtV2"
    DINOV2 = "DinoV2"
    EFFICIENTNETV2 = "EfficientNetV2"
    INCEPTIONNEXT = "InceptionNeXt"
    MAXVIT = "MaxVIT"
    MLPMIXER = "MLPMixer"
    RESNEST = "ResNest50"
    RESNET50 = "ResNet50"

def get_model(model_name: str) -> tf.keras.Model:
    """
    Retrieve a predefined Keras model by name.

    Args:
        model_name (str): Name of the model (must match one of InvestigatedModels values).

    Returns:
        tf.keras.Model: Instantiated model with softmax classifier head.

    Raises:
        AssertionError: If the model_name is not in InvestigatedModels.
        ValueError: If the model_name is unknown.
    """
    model_activation = "softmax"  # Default activation function
    # Check if model_name is a valid value in the Enum
    assert model_name in [model.value for model in InvestigatedModels], \
        f"Model '{model_name}' not found in InvestigatedModels. Should be one of {[model.value for model in InvestigatedModels]}"
    
    model_map = {
        InvestigatedModels.BEITV2.value: lambda: beit.BeitV2BasePatch16(classifier_activation=model_activation),
        InvestigatedModels.CONVNEXTV2.value: lambda: convnext.ConvNeXtV2Base(classifier_activation=model_activation),
        InvestigatedModels.DINOV2.value: lambda: beit.DINOv2_ViT_Small14(input_shape=(224, 224, 3), classifier_activation=model_activation),
        InvestigatedModels.EFFICIENTNETV2.value: lambda: efficientnet.EfficientNetV2B0(classifier_activation=model_activation),
        InvestigatedModels.INCEPTIONNEXT.value: lambda: inceptionnext.InceptionNeXtSmall(classifier_activation=model_activation),
        InvestigatedModels.MAXVIT.value: lambda: maxvit.MaxViT_Small(classifier_activation=model_activation),
        InvestigatedModels.MLPMIXER.value: lambda: mlp_family.MLPMixerL16(classifier_activation=model_activation),
        InvestigatedModels.RESNEST.value: lambda: resnest.ResNest50(classifier_activation=model_activation),
        InvestigatedModels.RESNET50.value: lambda: resnet_family.ResNet50D(classifier_activation=model_activation),
    }

    return model_map[model_name]()

def compute_dataset_mean(preprocess_ds: tf.data.Dataset):
    """
    Compute the mean pixel value of the dataset.
    """
    dataset = preprocess_ds
    sum_pixels = tf.zeros((3,), dtype=tf.float32)  # Assuming images have 3 channels (RGB)
    total_pixels = 0
    for batch in dataset:
        images = batch[0]  # Assuming batch[0] is images
        batch_size = tf.shape(images)[0]
        height = tf.shape(images)[1]
        width = tf.shape(images)[2]
        sum_pixels += tf.reduce_sum(images, axis=[0, 1, 2])  # Sum over batch, height, width
        total_pixels += batch_size * height * width
    mean_value = sum_pixels / tf.cast(total_pixels, tf.float32)
    mean_value = mean_value.numpy().tolist()
    mean_value = [float(x) for x in mean_value]  # Convert to standard Python floats
    return mean_value  # Returns a list of per-channel means