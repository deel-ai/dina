"""
Metrics for evaluating model fidelity using input perturbations.

Implements Deletion, Insertion and MuFidelity metrics, adapted from the Xplique library,
with efficient batch processing for tf.data.Dataset.

References:
    - Petsiuk et al., RISE: Randomized Input Sampling for Explanation of Black-box Models (2018).
      https://arxiv.org/pdf/1806.07421.pdf
    - Bhatt & al., Evaluating and Aggregating Feature-based Model Explanations (2020).
      https://arxiv.org/abs/2005.00631 (def. 3)
"""

from typing import Callable, Optional, Union
from inspect import isfunction

import tensorflow as tf

from .base import BaseAttributionMetric

class CausalFidelity(BaseAttributionMetric):
    """
    Base class for Deletion and Insertion metrics.

    Attributes:
        causal_mode (str): "insertion" or "deletion".
        baseline_mode (Union[float, Callable]): Value or function to create the baseline.
        nb_steps (int): Number of perturbation steps.
        max_percentage_perturbed (float): Fraction of features to perturb.
    """

    def __init__(
        self,
        model: tf.keras.Model,
        batch_size: Optional[int] = 64,
        causal_mode: str = "deletion",
        baseline_mode: Union[float, Callable] = 0.0,
        nb_steps: int = 10,
        max_percentage_perturbed: float = 1.0,
    ):
        """
        Args:
            model (tf.keras.Model): Model for which we computed explanations.
            batch_size (Optional[int]): Batch size for evaluation.
            causal_mode (str): If 'insertion', the path is baseline to original image;
                for "deletion", the path is original image to baseline.
            baseline_mode (Union[float, Callable]): Baseline value or callable.
            nb_steps (int): Number of steps between the start and the end state.
            max_percentage_perturbed (float): Maximum percentage of the input perturbed. (0, 1].
        """
        super().__init__(model, batch_size)
        assert causal_mode in ["insertion", "deletion"], f"causal_mode must be in ['insertion', 'deletion'] but got {causal_mode}."
        self.causal_mode = causal_mode
        self.baseline_mode = baseline_mode

        assert 0.0 < max_percentage_perturbed <= 1.0, (
            "`max_percentage_perturbed` must be in (0, 1]."
        )
        self._max_percentage_perturbed = max_percentage_perturbed
        self._nb_steps = nb_steps

    @property
    def nb_steps(self) -> int:
        return self._nb_steps

    @nb_steps.setter
    def nb_steps(self, value: int):
        self._nb_steps = value

    @property
    def max_percentage_perturbed(self) -> float:
        return self._max_percentage_perturbed

    @max_percentage_perturbed.setter
    def max_percentage_perturbed(self, value: float):
        assert 0.0 < value <= 1.0, "`max_percentage_perturbed` must be in (0, 1]."
        self._max_percentage_perturbed = value

    def _set_params(self, metric_loader: tf.data.Dataset):
        """
        Initialize parameters that depend on the data shape.

        Args:
            metric_loader (tf.data.Dataset): Loader for evaluation batches.
        """
        input_batch, _, __ = next(iter(metric_loader))
        # If the input has channels (colored image), they are all occluded at the same time
        self.has_channels = len(input_batch.shape) > 3
        if self.has_channels:
            self.nb_features = tf.reduce_prod(input_batch.shape[1:-1])
        else:
            self.nb_features = tf.reduce_prod(input_batch.shape[1:])
        # Calculate the maximum number of perturbed features
        max_nb_perturbed = tf.cast(
            tf.floor(tf.cast(self.nb_features, tf.float32) * self.max_percentage_perturbed), 
            tf.int32
        )
        # Generate the steps with TensorFlow
        self.steps = tf.linspace(0., tf.cast(max_nb_perturbed - 1, tf.float32), self.nb_steps)
        self.step_size = self.steps[1] - self.steps[0]

    @staticmethod
    @tf.function
    def _batch_rank_matrix(batch_tensor: tf.Tensor) -> tf.Tensor:
        """
        Ranks features in the explanations tensor for each batch element, high-to-low.

        Args:
            batch_tensor (tf.Tensor): Input tensor to rank.

        Returns:
            tf.Tensor: Tensor of same shape with feature ranks.
        """
        # Get the shape of the original tensor
        original_shape = tf.shape(batch_tensor)
        batch_size = original_shape[0]
        # Step 1: Flatten each 2D matrix in the batch to shape [B, H*W]
        flat_tensor = tf.reshape(batch_tensor, [batch_size, -1])
        # Step 2: Get the sorted indices for each matrix in the batch
        sorted_indices = tf.argsort(flat_tensor, axis=-1, direction="DESCENDING")
        # Step 3: Create ranks based on sorted indices for each batch element
        ranks = tf.zeros_like(flat_tensor, dtype=tf.int32)
        batch_indices = tf.reshape(tf.range(batch_size), [-1, 1])  # Shape [B, 1]
        batch_indices = tf.tile(batch_indices, [1, tf.shape(flat_tensor)[1]])  # Shape [B, H*W]
        scatter_indices = tf.stack([batch_indices, sorted_indices], axis=-1)
        ranks = tf.tensor_scatter_nd_update(
            ranks, 
            scatter_indices,
            tf.tile(tf.expand_dims(tf.range(tf.shape(flat_tensor)[1]), 0), [batch_size, 1])
        )
        # Step 4: Reshape the ranks back to the original [B, H, W] shape
        ranked_tensor = tf.reshape(ranks, original_shape)
        return ranked_tensor

    @tf.function
    def _apply_perturbations_and_predict(
        self,
        inputs: tf.Tensor,
        targets: tf.Tensor,
        baselines: tf.Tensor,
        explanations: tf.Tensor,
    ) -> tf.Tensor:
        """
        Apply perturbations to the inputs and compute model predictions simultaneously.

        Args:
            inputs (tf.Tensor): Original input tensor, shape (batch, ...).
            targets (tf.Tensor): One-hot encoding of target class, shape (batch, nb_classes).
            baselines (tf.Tensor): Baseline tensor, same shape as inputs.
            explanations (tf.Tensor): Explanation tensor for perturbation order.

        Returns:
            tf.Tensor: Model predictions at each step, shape (nb_steps, batch_size, nb_classes).
        """
        # Step 1: Rank the explanations to get ranks for each element
        ranks = self._batch_rank_matrix(explanations)
        # Expand ranks if inputs have a channel dimension
        ranks = tf.cond(
            tf.equal(tf.rank(inputs), 4),  # Check if inputs have a channel dimension
            lambda: tf.expand_dims(ranks, axis=-1),  # Expand to [B, H, W, 1]
            lambda: ranks  # No expansion needed for [B, H, W]
        )
        ranks = tf.cast(ranks, tf.float32)
        # Step 2: Define a function to apply perturbations and get predictions for a single step
        def perturb_and_predict_for_step(step):
            # Mask to determine which values to keep from the original inputs
            mask = tf.less_equal(ranks, step)
            # Apply the mask to inputs and baselines to get the perturbed matrix
            perturbed = tf.where(mask, inputs, baselines)
            # Get predictions for the perturbed inputs
            predictions = tf.reduce_sum(self.model(perturbed) * targets, axis=-1)
            return predictions
        # Step 3: Use tf.map_fn to iterate over steps and get predictions
        predictions = tf.map_fn(perturb_and_predict_for_step, self.steps, dtype=tf.float32)
        return predictions

    def _compute_auc(self, predictions: tf.Tensor) -> tf.Tensor:
        """
        Compute area under the probability curve using the trapezoidal rule.

        Args:
            predictions (tf.Tensor): Predictions at each step.

        Returns:
            tf.Tensor: AUC score for each sample.
        """
        predictions_next = tf.slice(predictions, [1, 0], [self.nb_steps - 1, -1])
        predictions_prev = tf.slice(predictions, [0, 0], [self.nb_steps - 1, -1])
        auc_values = tf.reduce_mean((predictions_next + predictions_prev) / 2, axis=0) # works because we are evenly spaced and look at the percentage of perturbed features
        return auc_values

    def _batch_evaluate(self,
                        inputs: tf.Tensor,
                        targets: Optional[tf.Tensor] = None,
                        explanations: tf.Tensor = None) -> tf.Tensor:
        """
        Evaluate the causal score for a batch of inputs, targets and explanations.

        Args:
            inputs (tf.Tensor): Input batch.
            targets (Optional[tf.Tensor]): One-hot encoded predicted labels.
            explanations (tf.Tensor): Attribution maps corresponding to inputs.

        Returns:
            tf.Tensor: Metric score, area over the deletion (lower is better);
                or insertion (higher is better) curve.
        """
        # Step 1: Get the baselines based on the baseline_mode
        if isfunction(self.baseline_mode):
            baselines = self.baseline_mode(inputs)
        else:
            baselines = tf.ones_like(inputs, dtype=tf.float32) * self.baseline_mode
        # Step 2: Reverse the baselines and inputs if the causal mode is 'deletion'
        if self.causal_mode == "deletion":
            inputs, baselines = baselines, inputs
        # Step 3: Reduce explanations if needed
        if len(explanations.shape) == 4:
            explanations = tf.reduce_mean(explanations, axis=-1)
        # Step 4: Get the predictions for each step
        predictions = self._apply_perturbations_and_predict(inputs, targets, baselines, explanations)
        # Step 5: Compute the area under the curve for each sample
        scores = self._compute_auc(predictions)
        return scores

class Deletion(CausalFidelity):
    """
    The deletion metric measures the drop in the probability of a class as important pixels (given
    by the saliency map) are gradually removed from the image. A sharp drop, and thus a small
    area under the probability curve, are indicative of a good explanation.

    Ref. Petsiuk & al., RISE: Randomized Input Sampling for Explanation of Black-box Models (2018).
    https://arxiv.org/pdf/1806.07421.pdf
    """
    def __init__(
        self,
        model: tf.keras.Model,
        batch_size: Optional[int] = 64,
        baseline_mode: Union[float, Callable] = 0.0,
        nb_steps: int = 10,
        max_percentage_perturbed: float = 1.0,
    ):
        """
        Args:
            model (tf.keras.Model): Model for which explanations are computed.
            batch_size (Optional[int]): Batch size for evaluation.
            baseline_mode (Union[float, Callable]): Baseline value or callable.
            nb_steps (int): Number of steps between start and end state.
            max_percentage_perturbed (float): Maximum percent of the input perturbed.
        """
        super().__init__(
            model=model, batch_size=batch_size,
            causal_mode="deletion", baseline_mode=baseline_mode,
            nb_steps=nb_steps, max_percentage_perturbed=max_percentage_perturbed
        )

class Insertion(CausalFidelity):
    """
    The insertion metric, on the other hand, captures the importance of the pixels in terms of
    their ability to synthesize an image and is measured by the rise in the probability of the
    class of interest as pixels are added according to the generated importance map.

    Ref. Petsiuk & al., RISE: Randomized Input Sampling for Explanation of Black-box Models (2018).
    https://arxiv.org/pdf/1806.07421.pdf
    """
    def __init__(
        self,
        model: tf.keras.Model,
        batch_size: Optional[int] = 64,
        baseline_mode: Union[float, Callable] = 0.0,
        nb_steps: int = 10,
        max_percentage_perturbed: float = 1.0,
    ):
        """
        Args:
            model (tf.keras.Model): Model for which explanations are computed.
            batch_size (Optional[int]): Batch size for evaluation.
            baseline_mode (Union[float, Callable]): Baseline value or callable.
            nb_steps (int): Number of steps between start and end state.
            max_percentage_perturbed (float): Maximum percent of the input perturbed.
        """
        super().__init__(
            model=model, batch_size=batch_size,
            causal_mode="insertion", baseline_mode=baseline_mode,
            nb_steps=nb_steps, max_percentage_perturbed=max_percentage_perturbed
        )

@tf.function
def get_rank(y_pred):
    """Calculate the rank of the predictions."""
    # Use argsort twice to get ranks starting from 1
    rank = tf.argsort(tf.argsort(y_pred, axis=-1, direction="ASCENDING"), axis=-1) + 1
    return tf.cast(rank, tf.float32)

@tf.function
def spearman_correlation(y_true, y_pred):
    """Compute Spearman rank correlation."""
    # Calculate ranks for predicted values
    y_true_rank = get_rank(y_true)
    y_pred_rank = get_rank(y_pred)
    # Compute covariance of the ranks
    cov = tfp.stats.covariance(y_true_rank, y_pred_rank, sample_axis=1, event_axis=None)
    # Compute standard deviations of the ranks
    sd_x = tfp.stats.stddev(y_true_rank, sample_axis=1)
    sd_y = tfp.stats.stddev(y_pred_rank, sample_axis=1)
    # Calculate Spearman correlation
    spearman_corr = cov / (sd_x * sd_y)
    # Reduce to a single average value across the batch
    return spearman_corr

class MuFidelity(BaseAttributionMetric):
    """
    Used to compute the fidelity correlation metric. This metric ensure there is a correlation
    between a random subset of pixels and their attribution score. For each random subset
    created, we set the pixels of the subset at a baseline state and obtain the prediction score.
    This metric measures the correlation between the drop in the score and the importance of the
    explanation.

    Reference:
        Bhatt et al., "Evaluating and Aggregating Feature-based Model Explanations" (2020)
        https://arxiv.org/abs/2005.00631 (Definition 3)

    Note:
        For medium or high-dimensional inputs, it is recommended to use grid-based superpixels
        via the `grid_size` parameter to obtain spatially coherent perturbations.

    Args:
        model (Callable): Model for which explanations are computed.
        batch_size (int): Number of samples to evaluate per batch.
        grid_size (Optional[int]): Size of the grid for superpixel-based perturbation masks.
        subset_percent (float): Percent of the image or time-series that will be set to baseline.
        baseline_mode (Union[Callable, float]): Value or function that defines the baseline.
        nb_samples (int): Number of different subsets to use for measuring correlation.
    """
    def __init__(self,
                 model: Callable,
                 batch_size: int = 64,
                 grid_size: Optional[int] = 8,
                 subset_percent: float = 0.2,
                 baseline_mode: Union[Callable, float] = 0.0,
                 nb_samples: int = 200):
        super().__init__(model=model, batch_size=batch_size)
        self.grid_size = grid_size
        self.subset_percent = tf.cast(subset_percent, tf.float32)
        self.baseline_mode = baseline_mode
        self.nb_samples = nb_samples

    def _set_params(self, metric_loader: tf.data.Dataset):
        """
        Initialize parameters that depend on the data shape.

        Args:
            metric_loader (tf.data.Dataset): Loader for evaluation batches.
        """
        input_batch, _, __ = next(iter(metric_loader))
        self.input_shape = tf.shape(input_batch)
        if self.grid_size is None:
            self.grid_size = self.input_shape[1]

    @tf.function
    def _apply_perturbations(
        self,
        inputs: tf.Tensor,
        targets: tf.Tensor,
        baselines: tf.Tensor,
        explanations: tf.Tensor,
        resize_shape: tf.Tensor,
    ) -> tf.Tensor:
        """
        Apply perturbations to the inputs and calculate the changes in model predictions.

        Args: 
            inputs (tf.Tensor): Input tensor of shape (batch_size, ...).
            targets (tf.Tensor): One-hot encoded class targets (batch_size, nb_classes).
            baselines (tf.Tensor): Tensor with the same shape as inputs used to replace masked regions.
            explanations (tf.Tensor): explanations (tf.Tensor): Explanation tensor for the inputs.
            resize_shape (tf.Tensor): Shape to resize the perturbation masks to.

        Returns:
            Tuple[tf.Tensor, tf.Tensor]:
                - delta_preds : The differences in model predictions after perturbations.
                - attr_scores : The sum of attribution scores in the perturbed regions.
        """
        # Calculate "original" predictions
        base_predictions = tf.reduce_sum(self.model(inputs) * targets, axis=-1)
        # Define a function to apply perturbations for a single perturbation mask
        def perturb_and_predict_for_step(step):
            # Generate a random perturbation mask
            subset_mask = tf.random.uniform((self.grid_size, self.grid_size), 0, 1, tf.float32)
            subset_mask = tf.greater(subset_mask, self.subset_percent)
            subset_mask = tf.cast(subset_mask, tf.float32)
            subset_mask = tf.expand_dims(subset_mask, axis=-1) # (H, W, 1)
            subset_mask = tf.image.resize(subset_mask, resize_shape, method="nearest")
            # Apply the mask to inputs and baselines to get the perturbed matrix
            perturbed = inputs * subset_mask + baselines * (1.0 - subset_mask)
            # Predict using the model for perturbed inputs
            predictions = tf.reduce_sum(self.model(perturbed) * targets, axis=-1) # (bs,)
            # Calculate the difference in predictions (delta)
            delta_preds = base_predictions - predictions # (bs,)
            # Calculate the attribution scores for the regions affected by the mask
            attr_scores = tf.reduce_sum(explanations * (1.0 - tf.squeeze(subset_mask)), axis=[1, 2]) # (bs,)
            return delta_preds, attr_scores
        # Apply perturbations for nb_samples different perturbation masks
        steps = tf.range(self.nb_samples)
        delta_preds, attr_scores = tf.map_fn(
            perturb_and_predict_for_step,
            steps,
            fn_output_signature=(tf.float32, tf.float32)
        )
        return delta_preds, attr_scores

    def _batch_evaluate(self, inputs: tf.Tensor, targets: tf.Tensor, explanations: tf.Tensor) -> tf.Tensor:
        """
        Evaluate the fidelity score for the batch.

        Args:
            inputs (tf.Tensor): Input batch.
            targets (tf.Tensor): One-hot class labels.
            explanations (tf.Tensor): Attribution maps corresponding to inputs.

        Returns:
            tf.Tensor: Spearman correlation scores for each input in the batch.
        """
        # Get the baselines based on the baseline_mode
        if isfunction(self.baseline_mode):
            baselines = self.baseline_mode(inputs)
        else:
            baselines = tf.ones_like(inputs, dtype=tf.float32) * self.baseline_mode
        # Reduce explanations if needed
        if len(explanations.shape) == 4:
            explanations = tf.reduce_mean(explanations, axis=-1)
        # Apply perturbations and get perturbed predictions and attribution scores
        resize_shape = inputs.shape[1:-1]
        resize_shape = tf.cast(resize_shape, tf.int32)
        delta_preds, attr_scores = self._apply_perturbations(inputs, targets, baselines, explanations, resize_shape)
        # shape of delta_preds and attr_scores: (nb_samples, batch_size)
        # Compute the Spearman correlation between the perturbation effects and attribution scores
        delta_preds, attr_scores = tf.transpose(delta_preds), tf.transpose(attr_scores)
        correlations = spearman_correlation(attr_scores, delta_preds)
        return correlations
