from abc import ABC, abstractmethod

import tensorflow as tf
from xplique.attributions.base import BlackBoxExplainer, WhiteBoxExplainer
from xplique.types import Callable, Optional, Union, List

from .base import BaseAttributionMetric


def ssim(a: tf.Tensor, b: tf.Tensor, batched: bool = False, **kwargs) -> tf.Tensor:
    """
    Calculate the Structural Similarity Index Measure (SSIM) between two images (or pairs of images)
    using pure TensorFlow to benefit from GPU acceleration. Handles zero dynamic range by returning 1.0.
    """
    # Additional parameters or defaults
    filter_size = kwargs.get("win_size", 11)
    filter_sigma = kwargs.get("filter_sigma", 1.5)
    k1 = kwargs.get("k1", 0.01)
    k2 = kwargs.get("k2", 0.03)

    def _ssim_pair(image_a, image_b):
        # Compute dynamic range for this pair
        stacked = tf.stack([image_a, image_b], axis=0)
        max_point = tf.reduce_max(stacked)
        min_point = tf.reduce_min(stacked)
        data_range = tf.cast(tf.abs(max_point - min_point), image_a.dtype)
        # If data_range is zero, images are identical constant => SSIM should be 1
        # Use a small epsilon to avoid zero division OR return 1 directly
        safe_data_range = tf.where(data_range > 0,
                                   data_range,
                                   tf.constant(1.0, dtype=image_a.dtype))
        ssim_val = tf.image.ssim(
            image_a, image_b, max_val=safe_data_range,
            filter_size=filter_size,
            filter_sigma=filter_sigma,
            k1=k1,
            k2=k2,
        )
        # If original data_range was zero, override to 1.0
        return tf.where(data_range > 0, ssim_val, tf.constant(1.0, dtype=image_a.dtype))

    if batched:
        ssim_values = tf.map_fn(lambda pair: _ssim_pair(pair[0], pair[1]), (a, b), dtype=a.dtype)
        return ssim_values
    else:
        return _ssim_pair(a, b)


def spearman(a: tf.Tensor, b: tf.Tensor) -> tf.Tensor:
    """
    Calculate the Spearman rank correlation coefficient between two tensors,
    using pure TensorFlow so it can run on GPU.

    Args:
        a (tf.Tensor): If batched=False, shape [F]; if batched=True, shape [B, F].
        b (tf.Tensor): Same shape as `a`.
        batched (bool): Whether to compute one correlation per-row (True) or a single correlation (False).

    Returns:
        tf.Tensor: If batched=False, a scalar tf.Tensor; if batched=True, shape [B].
    """
    # 1) Compute integer “ranks” via argsort‑of‑argsort
    # a, b: [B, F]
    rank_a = tf.argsort(tf.argsort(a, axis=1), axis=1)
    rank_b = tf.argsort(tf.argsort(b, axis=1), axis=1)
    # cast to float for covariance/std
    rank_a = tf.cast(rank_a, tf.float32)
    rank_b = tf.cast(rank_b, tf.float32)

    # 2) Row‑wise demean
    mean_a = tf.reduce_mean(rank_a, axis=1, keepdims=True)  # [B,1]
    mean_b = tf.reduce_mean(rank_b, axis=1, keepdims=True)  # [B,1]
    da = rank_a - mean_a   # [B,F]
    db = rank_b - mean_b   # [B,F]

    # 3) Covariance and standard deviations per row
    cov = tf.reduce_mean(da * db, axis=1)                   # [B]
    std_a = tf.math.reduce_std(rank_a, axis=1)              # [B]
    std_b = tf.math.reduce_std(rank_b, axis=1)              # [B]

    # 4) Spearman = Pearson( rank_a, rank_b ) per row
    return cov / (std_a * std_b)


class RandomLogitMetric(BaseAttributionMetric):
    """
    Tests whether explanations change when target logits are randomized.
    """
    def __init__(self,
                 model: Callable,
                 explainer: Union[BlackBoxExplainer, WhiteBoxExplainer],
                 batch_size: Optional[int] = 64,
                 seed: int = 42):
        """
        Args:
            model (Callable): The model for which the explanations have been computed.
            explainer (Union[BlackBoxExplainer, WhiteBoxExplainer]): The explainer used to generate explanations.
            batch_size (Optional[int]): The batch size for evaluation.
            seed (int): The random seed for reproducibility.
        """
        super().__init__(model=model, batch_size=batch_size)
        self.explainer = explainer
        self.seed = seed

    def _set_params(self, metric_loader: tf.data.Dataset):
        """
        Initialize parameters that depend on the data shape.

        Args:
            metric_loader (tf.data.Dataset): Loader for evaluation batches.
        """
        # Grab one batch to infer num_classes
        _, pred_batch, _ = next(iter(metric_loader))
        # If pred_batch is one‑hot or probabilities, its last dim is n_classes
        self.n_classes = int(pred_batch.shape[-1])
        tf.random.set_seed(self.seed)

    def _batch_evaluate(self,
                        inputs: tf.Tensor,
                        targets: tf.Tensor,
                        explanations: tf.Tensor) -> tf.Tensor:
        """
        For each sample:
          1) Determine its “true” class as argmax(targets).
          2) Draw a random off‑class uniformly from {0,…,C−1}∖{true}.
          3) Compute a perturbed explanation for that off‑class.
          4) Return SSIM(original, perturbed) per sample.

        Args:
            inputs (tf.Tensor): Input batch.
            targets (tf.Tensor): One-hot class predicted labels.
            explanations (tf.Tensor): Attribution maps corresponding to inputs.


        Returns:
            tf.Tensor: SSIM scores for each sample.
        """
        # 1) Get batch size and true class indices
        batch_size = tf.shape(inputs)[0]
        true_class = tf.argmax(targets, axis=-1, output_type=tf.int32)  # (batch,)

        # 2) Sample off‑class by the “shift trick”
        k = self.n_classes - 1
        rnd = tf.random.uniform([batch_size], minval=0, maxval=k, dtype=tf.int32)
        off_class = tf.where(rnd >= true_class, rnd + 1, rnd)  # (batch,)

        # 3) One‑hot encode and explain
        off_one_hot = tf.one_hot(off_class, depth=self.n_classes, dtype=tf.float32)
        explanations_perturbed = self.explainer.explain(inputs=inputs, targets=off_one_hot)

        # 4) Compute SSIM on the **image‑shaped** attributions
        score = ssim(explanations, explanations_perturbed, batched=True)

        return score


class ModelRandomizationStrategy(ABC):
    """
    Interface for model randomization strategies.
    """
    @abstractmethod
    def randomize(self, model: Callable) -> tf.keras.Model:
        """
        Randomize the model parameters.
        """
        NotImplementedError()


class ProgressiveLayerRandomization(ModelRandomizationStrategy):
    """
    Randomizes the model's parameters layer by layer starting from the last layer.

    Args:
        stop_layer (Union[str, int, float, List]): The layer at which to stop randomizing.;
            This can be a string (layer name), an integer (layer index), a float (percentage ;
            of layers to randomize), or a list of layer names or indices.
        reverse (bool): If True, randomizes layers in reverse order (from first to last).
    """
    def __init__(self, stop_layer: Union[str, int, float, List], reverse: bool = False):
        # Validate stop_layer type
        if isinstance(stop_layer, (str, int, float)):
            pass
        elif isinstance(stop_layer, list):
            for elem in stop_layer:
                if not isinstance(elem, (str, int)):
                    raise TypeError("List elements must be str or int.")
        else:
            raise TypeError("stop_layer must be str, int, float, or list.")

        self.stop_layer = stop_layer
        self.reverse = reverse

    def randomize(self, model: Callable) -> tf.keras.Model:
        """
        Randomizes the model parameters layer by layer.

        Args: 
            model (Callable): The model to randomize.
        """
        # Get all layers in the model
        layers = model.layers

        # Determine layer order
        layer_list = layers[::-1] if self.reverse else layers
        n = len(layer_list)

        # Find stop_index in this order
        if isinstance(self.stop_layer, str):
            # find first matching name in layer_list
            idxs = [i for i, l in enumerate(layer_list) if l.name == self.stop_layer]
            if not idxs:
                raise ValueError(f"Layer '{self.stop_layer}' not found.")
            stop_index = idxs[0]
        elif isinstance(self.stop_layer, int):
            stop_index = self.stop_layer
        elif isinstance(self.stop_layer, float):
            stop_index = int(n * self.stop_layer)
        else:  # list
            indices = []
            for elem in self.stop_layer:
                if isinstance(elem, str):
                    idxs = [i for i, l in enumerate(layer_list) if l.name == elem]
                    if not idxs:
                        raise ValueError(f"Layer '{elem}' not found.")
                    indices.append(idxs[0])
                else:
                    indices.append(elem)
            stop_index = min(indices)

        # Randomize weights up to stop_index (exclusive)
        for layer in layer_list[:stop_index]:
            new_weights = [tf.random.uniform(w.shape, dtype=w.dtype) for w in layer.get_weights()]
            layer.set_weights(new_weights)

        return model


class ModelRandomizationMetric(BaseAttributionMetric):
    """
    Tests whether explanations change when model parameters are randomized.
    This metric uses a randomization strategy to randomize the model parameters

    Args:
        model (Callable): The model to be evaluated.
        explainer (Union[BlackBoxExplainer, WhiteBoxExplainer]): The explainer used to generate explanations.
        randomization_strategy (ModelRandomizationStrategy): The strategy used to randomize the model parameters.
        batch_size (Optional[int]): The batch size for evaluation.
        seed (int): The random seed for reproducibility.
    """
    def __init__(self,
                 model: Callable,
                 explainer: Union[BlackBoxExplainer, WhiteBoxExplainer],
                 randomization_strategy: ModelRandomizationStrategy = ProgressiveLayerRandomization(0.25),
                 batch_size: Optional[int] = 64,
                 seed: int = 42):
        super().__init__(model=model, batch_size=batch_size)
        self.randomization_strategy = randomization_strategy
        self.explainer = explainer
        self.seed = seed

    def _set_params(self, metric_loader: tf.data.Dataset):
        """
        Initialize parameters that depend on the data shape.

        Args:
            metric_loader (tf.data.Dataset): Loader for evaluation batches.
        """
        tf.random.set_seed(self.seed)

        # Infer n_classes from loader predictions
        _, pred_batch, _ = next(iter(metric_loader))
        self.n_classes = int(pred_batch.shape[-1])

        # Randomize model once
        self.model = self.randomization_strategy.randomize(self.model)

    def _batch_evaluate(self,
                        inputs: tf.Tensor,
                        targets: tf.Tensor,
                        explanations: tf.Tensor) -> tf.Tensor:
        """
        For each sample:
          1) Determine its “true” class as argmax(targets).
          2) Compute a perturbed explanation for the true class.
          3) Return spearman(original, perturbed) per sample.

        Args:
            inputs (tf.Tensor): Input batch.
            targets (Optional[tf.Tensor]): One-hot encoded true labels.
            explanations (tf.Tensor): Attribution maps corresponding to inputs.

        Returns:
            tf.Tensor: SSIM scores for each sample.
        """
        # 1) Get true class indices
        true_class = tf.argmax(targets, axis=-1, output_type=tf.int32)

        # 2) One‑hot encode and explain
        true_one_hot = tf.one_hot(true_class, depth=self.n_classes, dtype=tf.float32)
        explanations_perturbed = self.explainer.explain(inputs, true_one_hot)

        # 3) Reduce explanations if needed
        if len(explanations.shape) == 4:
            explanations = tf.reduce_mean(explanations, axis=-1)
            explanations_perturbed = tf.reduce_mean(explanations_perturbed, axis=-1)

        # 4) Compute the spearman correlation on the **image‑shaped** attributions
        explanations = tf.reshape(explanations, (inputs.shape[0], -1))
        explanations_perturbed = tf.reshape(explanations_perturbed, (inputs.shape[0], -1))
        score = spearman(explanations, explanations_perturbed)

        return score
