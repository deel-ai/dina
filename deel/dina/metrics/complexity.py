"""
Attribution complexity metrics

Re-implementations in TensorFlow for efficiency, adapted from the Quantus library:
https://github.com/understandable-machine-intelligence-lab/Quantus/blob/main/quantus/metrics/complexity/complexity.py

References:
    - Complexity metric: https://arxiv.org/abs/2005.00631
    - Sparseness metric: https://proceedings.mlr.press/v119/chalasani20a.html
"""
import tensorflow as tf

from .base import BaseAttributionMetric


class Complexity(BaseAttributionMetric):
    """
    Complexity metric for attributions based on the entropy of the explanation.
    """
    def __init__(self, model: tf.keras.Model, batch_size: int = 32):
        """
        Args:
            model (tf.keras.Model): Model for which we computed explanations.
            batch_size (Optional[int]): Batch size for evaluation.
        """
        super().__init__(model, batch_size)

    def _batch_evaluate(self,
                       inputs: tf.Tensor,
                       targets: tf.Tensor = None,
                       explanations: tf.Tensor = None) -> tf.Tensor:
        """
        Evaluate the complexity score for the batch.

        Args:
            inputs (tf.Tensor): Input batch (unused).
            targets (tf.Tensor): Target labels (unused).
            explanations (tf.Tensor): Explanation maps (B, H, W) or (B, H, W, C).

        Returns:
            tf.Tensor: Entropy score per sample (shape: [B]).
        """
        b = inputs.shape[0]
        if len(explanations.shape) == 4:
            explanations = tf.reduce_mean(explanations, axis=-1)
        flattened_explanations = tf.abs(tf.reshape(explanations, (b, -1)))
        # Compute the entropy of the explanations
        entropy = self._compute_entropy(flattened_explanations)
        return entropy

    @staticmethod
    @tf.function
    def _compute_entropy(flattened_explanations: tf.Tensor) -> tf.Tensor:
        """
        Compute the entropy of the explanations.

        Args:
            flattened_explanations (tf.Tensor): Explanation tensors flattened per sample (B, N).

        Returns:
            tf.Tensor: Entropy per sample (shape: [B]).
        """
        norm = tf.reduce_sum(flattened_explanations, axis=-1, keepdims=True)
        probs = flattened_explanations / (norm + 1e-8)
        entropy = -tf.reduce_sum(probs * tf.math.log(probs + 1e-8), axis=-1)
        return entropy

    def _set_params(self, metric_loader: tf.data.Dataset):
        """
        This metric doesn't require any specific parameters to be set.
        """
        pass


class Sparseness(BaseAttributionMetric):
    """
    Sparseness metric for attributions based on the Gini index of the explanations.
    """
    def __init__(self, model: tf.keras.Model, batch_size: int = 32):
        """
        Args:
            model (tf.keras.Model): Model for which we computed explanations.
            batch_size (Optional[int]): Batch size for evaluation.
        """
        super().__init__(model, batch_size)

    def _batch_evaluate(self,
                       inputs: tf.Tensor,
                       targets: tf.Tensor = None,
                       explanations: tf.Tensor = None) -> tf.Tensor:
        """
        Evaluate the sparseness score for the batch.

        Args:
            inputs (tf.Tensor): Input batch (unused).
            targets (tf.Tensor): Target labels (unused).
            explanations (tf.Tensor): Explanation maps (B, H, W) or (B, H, W, C).

        Returns:
            tf.Tensor: Gini index per sample (shape: [B]).
        """
        b = inputs.shape[0]
        if len(explanations.shape) == 4:
            explanations = tf.reduce_mean(explanations, axis=-1)
        flattened_explanations = tf.abs(tf.reshape(explanations, (b, -1)))

        # Normalize the explanations
        norm = tf.reduce_sum(flattened_explanations, axis=-1, keepdims=True)
        flattened_explanations = flattened_explanations / (norm + 1e-8)
        n = flattened_explanations.shape[1]

        # Compute the Gini index of the explanations
        sorted_explanations = tf.sort(flattened_explanations, axis=-1) + 1e-7
        features = tf.stack([tf.range(1, n + 1, dtype=tf.float32) for _ in range(b)], axis=0)
        gini_index = tf.reduce_sum((2 * features - n - 1) * sorted_explanations, axis=-1) / \
                     (tf.reduce_sum(sorted_explanations, axis=-1) * n)

        return gini_index

    def _set_params(self, metric_loader: tf.data.Dataset):
        """
        This metric doesn't require any specific parameters to be set.
        """
        pass
