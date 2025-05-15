"""
Base module for attribution metrics.

Defines an abstract base class for attribution metric implementations used in
model explanation evaluation pipelines.
"""

from abc import ABC, abstractmethod
from typing import Optional
from tqdm import tqdm

import tensorflow as tf

class BaseAttributionMetric(ABC):
    """
    Abstract base class for attribution metrics.

    This class should be subclassed when implementing new evaluation metrics
    for attribution (explanation) methods.

    Attributes:
        model (tf.keras.Model): The model used for computing explanations.
        batch_size (int): Number of samples to evaluate at once.
    """

    def __init__(
        self,
        model: tf.keras.Model,
        batch_size: Optional[int] = 64,
    ):
        """
        Initialize the BaseAttributionMetric.

        Args:
            model (tf.keras.Model): Model used for computing explanations.
            batch_size (Optional[int]): Number of samples to evaluate at once.
                If None, compute all at once. Defaults to 64.
        """
        self.model = model
        self._batch_size = batch_size

    @property
    def batch_size(self) -> int:
        """int: The batch size used for evaluation."""
        return self._batch_size

    @batch_size.setter
    def batch_size(self, value: int):
        """Set the batch size used for evaluation.

        Args:
            value (int): The new batch size.
        """
        self._batch_size = value

    @abstractmethod
    def _set_params(self, metric_loader: tf.data.Dataset):
        """
        Set parameters for the metric, depending on the dataset.

        Args:
            metric_loader (tf.data.Dataset): Dataset used for evaluation.
        """
        raise NotImplementedError("Subclasses must implement this method.")

    @abstractmethod
    def _batch_evaluate(
        self,
        inputs: tf.Tensor,
        targets: tf.Tensor = None,
        explanations: tf.Tensor = None,
    ) -> tf.Tensor:
        """
        Evaluate the metric for a batch of samples.

        Args:
            inputs (tf.Tensor): Input data.
            targets (tf.Tensor, optional): Target labels for the inputs.
            explanations (tf.Tensor, optional): Explanations for the inputs.

        Returns:
            tf.Tensor: Computed scores for the batch.
        """
        raise NotImplementedError("Subclasses must implement this method.")

    def evaluate(self, metric_loader: tf.data.Dataset) -> tf.Tensor:
        """
        Compute metric scores for the given explanations.

        Args:
            metric_loader (tf.data.Dataset): Dataset providing batches of
                (inputs, targets, explanations).

        Returns:
            tf.Tensor: Concatenated scores for all batches.
        """
        self._set_params(metric_loader)
        batch_scores = []
        for inputs, targets, explanations in tqdm(
            metric_loader,
            desc="Computing Scores for batch",
            unit="batch"
        ):
            batch_scores.append(self._batch_evaluate(inputs, targets, explanations))
        scores = tf.concat(batch_scores, axis=0)
        return scores

    def __call__(self, metric_loader: tf.data.Dataset) -> tf.Tensor:
        """
        Evaluate the metric by calling the instance.

        Args:
            metric_loader (tf.data.Dataset): Dataset for evaluation.

        Returns:
            tf.Tensor: Metric scores.
        """
        return self.evaluate(metric_loader)

    @staticmethod
    def prepare_metric_loader(
        xai_ds: tf.data.Dataset, explanations: tf.data.Dataset
    ) -> tf.data.Dataset:
        """
        Prepares a metric loader dataset yielding
        (inputs, predictions, explanations) triplets.

        Args:
            xai_ds (tf.data.Dataset): Dataset containing tuples of
                (preprocessed inputs, model predictions).
            explanations (tf.data.Dataset): Dataset of explanations
                (same ordering as xai_ds).

        Returns:
            tf.data.Dataset: Zipped dataset yielding
                (inputs, predictions, explanations).
        """
        zip_ds = tf.data.Dataset.zip((xai_ds, explanations))
        metric_ds = zip_ds.map(lambda x, y: (x[0], x[1], y))
        return metric_ds
