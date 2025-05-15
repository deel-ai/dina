from abc import ABC, abstractmethod
from enum import Enum
from functools import partial

import tensorflow as tf
import tensorflow_probability as tfp
from einops import rearrange, repeat

from xplique.attributions.base import BlackBoxExplainer
from xplique.attributions.global_sensitivity_analysis.perturbations import inpainting, blurring, amplitude
from xplique.attributions.global_sensitivity_analysis.kernels import Kernel
from xplique.types import OperatorSignature, Tuple, Union, Optional, Callable
from xplique.commons import Tasks, batch_tensor, repeat_labels


class ReplicatedSampler(ABC):
    """
    Base class for replicated design sampling.
    """

    @staticmethod
    @tf.function
    def build_replicated_design(sampling_a: tf.Tensor, sampling_b: tf.Tensor) -> tf.Tensor:
        """
        Build the replicated design matrix C using A & B via TF and einops.

        Parameters
        ----------
        sampling_a : tf.Tensor
            The masks values for the sampling matrix A, shape (nb_design, d).
        sampling_b : tf.Tensor
            The masks values for the sampling matrix B, shape (nb_design, d).

        Returns
        -------
        replication_c : tf.Tensor
            The replicated design matrix C of shape (nb_design * d, d).
        """
        # Get the number of dimensions (d). (Works in eager or graph mode.)
        d = tf.shape(sampling_a)[-1]
        # Create d copies of sampling_a with shape (d, nb_design, d)
        replication_c = repeat(sampling_a, 'b d -> i b d', i=d)
        # Similarly, replicate sampling_b to the same shape
        replication_b = repeat(sampling_b, 'b d -> i b d', i=d)
        # Create a diagonal mask: shape (d, 1, d) then broadcast to (d, nb_design, d)
        diag_mask = tf.eye(tf.cast(d, tf.int32), dtype=sampling_a.dtype)  # (d, d)
        diag_mask = tf.expand_dims(diag_mask, axis=1)  # (d, 1, d)
        diag_mask = tf.broadcast_to(diag_mask, tf.shape(replication_c))
        # For each "replication" i, replace the i-th column with sampling_b
        replication_c = replication_c * (1 - diag_mask) + replication_b * diag_mask
        # Flatten the first two dimensions so that replication_c has shape (nb_design * d, d)
        replication_c = rearrange(replication_c, 'i b d -> (i b) d')
        return replication_c

    @abstractmethod
    def __call__(self, dimension: int, nb_design: int) -> tf.Tensor:
        raise NotImplementedError()


class TFSobolSequenceRS(ReplicatedSampler):
    """
    Tensorflow Sobol LP tau sequence sampler.
    Ref. I. M. Sobol., The distribution of points in a cube and the accurate evaluation of
    integrals (1967).
    https://www.sciencedirect.com/science/article/abs/pii/0041555367901449
    """
    @tf.function
    def __call__(self, dimension: int, nb_design: int) -> tf.Tensor:
        # Generate 2*dimension numbers per design point.
        sampling_ab = tf.math.sobol_sample(dimension * 2, nb_design, dtype=tf.float32)
        sampling_a = sampling_ab[:, :dimension]  # (nb_design, dimension)
        sampling_b = sampling_ab[:, dimension:]  # (nb_design, dimension)
        replicated_c = ReplicatedSampler.build_replicated_design(sampling_a, sampling_b)
        return tf.concat([sampling_a, sampling_b, replicated_c], axis=0)


class SobolEstimator(ABC):
    """
    Base class for Sobol' total order estimators.
    """

    @staticmethod
    @tf.function
    def masks_dim(masks: tf.Tensor) -> tf.Tensor:
        """
        Deduce the number of dimensions using the sampling masks.

        Parameters
        ----------
        masks : tf.Tensor
            Low resolution masks (before upsampling), one per output.

        Returns
        -------
        nb_dim : tf.Tensor
            The total number of dimensions (e.g. grid_size**2).
        """
        return tf.reduce_prod(tf.shape(masks)[1:])

    @staticmethod
    @tf.function
    def split_abc(outputs: tf.Tensor, nb_design: int, nb_dim: int) -> Tuple[tf.Tensor, tf.Tensor, tf.Tensor]:
        """
        Split the outputs into matrices A, B and C.

        Parameters
        ----------
        outputs : tf.Tensor
            Model outputs for each sample point (concatenated A, B and C).
        nb_design : int
            Number of samples in matrices A and B.
        nb_dim : int
            Number of dimensions (determines the number of C replications).

        Returns
        -------
        sampling_a : tf.Tensor
            Outputs corresponding to matrix A.
        sampling_b : tf.Tensor
            Outputs corresponding to matrix B.
        replication_c : tf.Tensor
            Outputs corresponding to matrix C, reshaped to (nb_dim, nb_design).
        """
        sampling_a = outputs[:nb_design]
        sampling_b = outputs[nb_design:nb_design * 2]
        replication_c = outputs[nb_design * 2:]
        replication_c = rearrange(replication_c, '(i b) -> i b', i=nb_dim)
        return sampling_a, sampling_b, replication_c

    @staticmethod
    @tf.function
    def post_process(stis: tf.Tensor, masks: tf.Tensor) -> tf.Tensor:
        """
        Reshape the Sobol' indices to match the spatial layout of the masks.

        Parameters
        ----------
        stis : tf.Tensor
            Total order Sobol' indices, one per dimension.
        masks : tf.Tensor
            The original low resolution masks.

        Returns
        -------
        stis : tf.Tensor
            Reshaped Sobol' indices.
        """
        # Assuming masks is shaped (n, H, W, 1) and we want a (H, W) map.
        return tf.reshape(stis, tf.shape(masks)[1:-1])

    @abstractmethod
    def __call__(self, masks: tf.Tensor, outputs: tf.Tensor, nb_design: int) -> tf.Tensor:
        raise NotImplementedError()


class JansenEstimator(SobolEstimator):
    """
    Jansen estimator for total order Sobol' indices.
    Ref. Jansen, M., Analysis of variance designs for model output (1999)
    https://www.sciencedirect.com/science/article/abs/pii/S0010465598001544
    """
    @tf.function
    def __call__(self, masks: tf.Tensor, outputs: tf.Tensor, nb_design: int) -> tf.Tensor:
        # Here we assume masks has shape (n, grid_size, grid_size, 1),
        # so that the number of dimensions is grid_size**2.
        grid_size = masks.shape[1]
        nb_dim = grid_size * grid_size
        sampling_a, _, replication_c = self.split_abc(outputs, nb_design, nb_dim)
        mu_a = tf.reduce_mean(sampling_a)
        var = tf.reduce_sum(tf.square(sampling_a - mu_a)) / (tf.cast(tf.size(sampling_a), tf.float32) - 1.0)
        diff = sampling_a[None, :] - replication_c  # shape: (nb_dim, nb_design)
        # Compute the numerator (sum of squared differences) along the design axis.
        numerator = tf.reduce_sum(tf.square(diff), axis=-1)  # shape: (nb_dim,)
        # Compute all Sobol indices at once.
        sti = numerator / (2.0 * tf.cast(nb_design, tf.float32) * var)  # shape: (nb_dim,)
        # Pack the flat vector of indices into a (grid_size, grid_size) grid.
        stis = rearrange(sti, '(h w) -> h w', h=grid_size, w=grid_size)

        return stis


# -------------------------
# SAMPLERS
# -------------------------
class Sampler(ABC):
    """
    Base class for sampling.
    """

    def __init__(self, binary=False):
        self.binary = binary

    @abstractmethod
    def __call__(self, dimension, nb_design) -> tf.Tensor:
        raise NotImplementedError()

    @staticmethod
    def make_binary(masks: tf.Tensor) -> tf.Tensor:
        """
        Transform [0, 1]^d masks into binary {0, 1}^d masks.
        """
        return tf.round(masks)


class TFSobolSequence(Sampler):
    """
    Tensorflow Sobol LP tau sequence sampler.
    Ref. I. M. Sobol., The distribution of points in a cube and the accurate evaluation of integrals (1967)
    https://www.sciencedirect.com/science/article/abs/pii/0041555367901449
    """

    def __call__(self, dimension, nb_design) -> tf.Tensor:
        # Remain in TF (no .numpy() conversion)
        points = tf.math.sobol_sample(dimension, nb_design, dtype=tf.float32)
        if self.binary:
            points = self.make_binary(points)
        return points


# -------------------------
# HSIC ESTIMATORS
# -------------------------
class HsicEstimator(ABC):
    """
    Base class for HSIC estimator.
    """

    def __init__(self, output_kernel="rbf"):
        self.output_kernel = output_kernel
        assert output_kernel in ["rbf"], "Only 'rbf' output kernel is supported for now."
        # Set a high batch size (can be updated via set_batch_size).
        self.batch_size = 100000

    @staticmethod
    def masks_dim(masks: tf.Tensor) -> tf.Tensor:
        """
        Deduce the number of dimensions (d) from the masks.
        """
        return tf.reduce_prod(tf.shape(masks)[1:])

    @staticmethod
    def post_process(score: tf.Tensor, masks: tf.Tensor) -> tf.Tensor:
        """
        Reshape the HSIC scores to the desired spatial layout.
        For example, if masks is (n, H, W, 1) and score is flat of length H*W,
        we reshape and then transpose axes as needed.
        """
        # Reshape to (H, W, 1) and then swap the first two axes.
        reshaped = tf.reshape(score, tf.shape(masks)[1:])
        return tf.transpose(reshaped, perm=[1, 0, 2])

    @abstractmethod
    def input_kernel_func(self, X: tf.Tensor, Y: tf.Tensor) -> tf.Tensor:
        """
        Compute the kernel matrix for the input.
        """
        raise NotImplementedError()

    @abstractmethod
    def output_kernel_func(self, X: tf.Tensor, Y: tf.Tensor) -> tf.Tensor:
        """
        Compute the kernel matrix for the output.
        """
        raise NotImplementedError()

    def set_batch_size(self, batch_size: Optional[int] = None):
        if batch_size is not None:
            self.batch_size = batch_size

    # We do not decorate this with @tf.function to avoid retracing due to dynamic shapes.
    def estimator(self, masks: tf.Tensor, L: tf.Tensor, nb_dim: tf.Tensor, nb_design: int) -> tf.Tensor:
        """
        Compute the HSIC estimates.
        We use einops to streamline reshaping.
        """
        # Use einops to reshape masks:
        # masks shape: (nb_design, H, W, 1)
        # Rearrange to get (d, nb_design) where d = H*W*1.
        X = rearrange(masks, 'n h w c -> (c w h) n')
        # Add singleton dimensions: shape becomes (d, 1, nb_design, 1)
        X1 = rearrange(X, 'd n -> d 1 n 1')
        # Swap last two axes: shape becomes (d, 1, 1, nb_design)
        X2 = rearrange(X1, 'd a n b -> d a b n')

        # Use the minimum of self.batch_size and nb_dim to avoid OOM.
        batch_size = tf.cond(
            nb_dim > self.batch_size,
            lambda: tf.cast(self.batch_size, tf.int64),
            lambda: tf.cast(nb_dim, tf.int64)
        )

        scores = tf.zeros((0,), dtype=tf.float32)
        # Batch over the mask dimensions (using batch_tensor from xplique.commons).
        for x1, x2 in batch_tensor((X1, X2), batch_size):
            K = self.input_kernel_func(x1, x2)
            # Here we reduce over axis=1 (the kernel is computed per mask dimension)
            K = tf.math.reduce_prod(1 + K, axis=1)
            H = tf.eye(nb_design) - tf.ones((nb_design, nb_design), dtype=tf.float32) / tf.cast(nb_design, tf.float32)
            HK = tf.einsum("jk,ikl->ijl", H, K)
            HL = tf.einsum("jk,kl->jl", H, L)
            Kc = tf.einsum("ijk,kl->ijl", HK, H)
            Lc = tf.einsum("jk,kl->jl", HL, H)
            score = tf.math.reduce_sum(Kc * tf.transpose(Lc), axis=[1, 2]) / tf.cast(nb_design, tf.float32)
            scores = tf.concat([scores, score], axis=0)

        return scores

    def __call__(self, masks: tf.Tensor, outputs: tf.Tensor, nb_design: int) -> tf.Tensor:
        nb_dim = self.masks_dim(masks)
        # Cast outputs to float and reshape to (nb_design, 1)
        Y = tf.cast(outputs, tf.float32)
        Y = tf.reshape(Y, (nb_design, 1))
        # Use tfp.stats.percentile to compute the median if needed in output_kernel_func.
        L = self.output_kernel_func(Y, tf.transpose(Y))
        score = self.estimator(masks, L, nb_dim, nb_design)
        return self.post_process(score, masks)


class BinaryEstimator(HsicEstimator):
    """
    HSIC estimator using the binary (Dirac) kernel for the input.
    """

    def input_kernel_func(self, X: tf.Tensor, Y: tf.Tensor) -> tf.Tensor:
        # Use the binary kernel defined via Kernel.from_string.
        return Kernel.from_string("binary")(X, Y)

    def output_kernel_func(self, X: tf.Tensor, Y: tf.Tensor) -> tf.Tensor:
        # Use tfp.stats.percentile to obtain the median.
        width_y = tfp.stats.percentile(Y, 50.0, interpolation='linear')
        width_y = tf.cast(width_y, tf.float32)
        kernel_func = partial(Kernel.from_string(self.output_kernel), width=width_y)
        return kernel_func(X, Y)


class RbfEstimator(HsicEstimator):
    """
    HSIC estimator using the RBF kernel for the input.
    """

    def input_kernel_func(self, X: tf.Tensor, Y: tf.Tensor) -> tf.Tensor:
        width_x = 0.5
        kernel_func = partial(Kernel.from_string("rbf"), width=width_x)
        return kernel_func(X, Y)

    def output_kernel_func(self, X: tf.Tensor, Y: tf.Tensor) -> tf.Tensor:
        width_y = tfp.stats.percentile(Y, 50.0, interpolation='linear')
        width_y = tf.cast(width_y, tf.float32)
        kernel_func = partial(Kernel.from_string(self.output_kernel), width=width_y)
        return kernel_func(X, Y)


class SobolevEstimator(HsicEstimator):
    """
    HSIC estimator using the Sobolev kernel for the input.
    """

    def input_kernel_func(self, X: tf.Tensor, Y: tf.Tensor) -> tf.Tensor:
        return Kernel.from_string("sobolev")(X, Y)

    def output_kernel_func(self, X: tf.Tensor, Y: tf.Tensor) -> tf.Tensor:
        width_y = tfp.stats.percentile(Y, 50.0, interpolation='linear')
        width_y = tf.cast(width_y, tf.float32)
        kernel_func = partial(Kernel.from_string(self.output_kernel), width=width_y)
        return kernel_func(X, Y)


class PerturbationFunction(Enum):
    """
    GSA Perturbation function interface.
    """
    INPAINTING = inpainting
    BLURRING = blurring
    AMPLITUDE = amplitude

    def __call__(self, *args, **kwargs):
        # Allow the enum member to be called directly.
        return self.value(*args, **kwargs)

    @staticmethod
    def from_string(perturbation_function: str) -> "PerturbationFunction":
        """
        Restore a perturbation function from a string.

        Parameters
        ----------
        perturbation_function : str
            Must be one of 'inpainting', 'blurring', or 'amplitude'.

        Returns
        -------
        PerturbationFunction
            The corresponding perturbation function enum member.
        """
        assert perturbation_function in ["inpainting", "blurring", "amplitude"], \
            "Only 'inpainting', 'blurring' and 'amplitude' are supported."
        if perturbation_function == "amplitude":
            return PerturbationFunction.AMPLITUDE
        if perturbation_function == "blurring":
            return PerturbationFunction.BLURRING
        return PerturbationFunction.INPAINTING


class GSABaseAttributionMethod(BlackBoxExplainer):
    """
    GSA base Attribution Method.
    Base explainer for attribution methods based on Global Sensitivity Analysis.

    Parameters
    ----------
    model : tf.keras.Model
        Model used for computing explanations.
    grid_size : int
        The image is split into a (grid_size x grid_size) grid.
    nb_design : int
        Must be a power of two. The number of designs; the number of forward passes
        will be nb_design * (grid_size**2 + 2).
    sampler : Callable
        Function used to generate the masks.
    estimator : Callable
        The estimator used to compute the attribution scores (e.g., Sobol or HSIC estimator).
    perturbation_function : Optional[Union[Callable, str]]
        Perturbation function to apply (or one of 'inpainting', 'blurring', or 'amplitude').
    batch_size : int
        Batch size to use during inference.
    operator : Optional[Union[Tasks, str, OperatorSignature]]
        Function g to explain. If None, use the standard operator g(f, x, y) = f(x)[y].
    """

    def __init__(
        self,
        model: tf.keras.Model,
        sampler: Callable,
        estimator: Callable,
        grid_size: int = 7,
        nb_design: int = 32,
        perturbation_function: Optional[Union[Callable, str]] = "inpainting",
        batch_size: int = 256,
        operator: Optional[Union[Tasks, str, OperatorSignature]] = None,
    ):
        super().__init__(model, batch_size, operator)

        self.grid_size = grid_size
        self.nb_design = nb_design

        if isinstance(perturbation_function, str):
            self.perturbation_function = PerturbationFunction.from_string(perturbation_function)
        else:
            self.perturbation_function = perturbation_function

        self.sampler = sampler
        self.estimator = estimator

        masks = self.sampler(grid_size**2, nb_design)
        self.masks = tf.reshape(masks, (-1, grid_size, grid_size, 1))

    def explain(
        self,
        inputs: Union[tf.data.Dataset, tf.Tensor],
        targets: Optional[Union[tf.Tensor]] = None,
    ) -> tf.Tensor:
        """
        Compute the Sobol' indices for the given inputs.

        Parameters
        ----------
        inputs : Union[tf.data.Dataset, tf.Tensor]
            The images to be explained.
        targets : Optional[Union[tf.Tensor]]
            The corresponding targets.

        Returns
        -------
        attributions_maps : tf.Tensor
            The computed attribution maps.
        """
        input_shape = (tf.shape(inputs)[1], tf.shape(inputs)[2])
        heatmaps = None

        for i, (inp, target) in enumerate(zip(inputs, targets)):
            perturbator = self.perturbation_function(inp)
            outputs = None

            for batch_masks in batch_tensor(self.masks, self.batch_size):
                batch_x, batch_y = self._batch_perturbations(batch_masks, perturbator, target, input_shape)
                batch_outputs = self.inference_function(self.model, batch_x, batch_y)
                outputs = batch_outputs if outputs is None else tf.concat([outputs, batch_outputs], axis=0)

            heatmap = self.estimator(self.masks, outputs, self.nb_design)
            if tf.rank(heatmap) == 2:
                heatmap = heatmap[..., tf.newaxis]
            heatmap = tf.image.resize(heatmap, input_shape, method=tf.image.ResizeMethod.BICUBIC)
            heatmap = tf.expand_dims(heatmap, axis=0)
            heatmaps = heatmap if heatmaps is None else tf.concat([heatmaps, heatmap], axis=0)

        return heatmaps

    @staticmethod
    def _batch_perturbations(
        masks: tf.Tensor,
        perturbator: Callable,
        target: tf.Tensor,
        input_shape: Tuple[int, int],
    ) -> Union[tf.Tensor, tf.Tensor]:
        """
        Prepare perturbated input and replicated targets before a batch inference.

        Parameters
        ----------
        masks
            Perturbation masks in lower dimensions (grid_size, grid_size).
        perturbator
            Perturbation function to be called with the upsampled masks.
        target
            Label of a single prediction
        input_shape
            Shape of a single input

        Returns
        -------
        perturbated_inputs
            One inputs perturbated for each masks, according to the pertubation function
            modulated by the masks values.
        repeated_targets
            Replicated labels, one for each masks.
        """
        repeated_targets = repeat_labels(target[None, :], len(masks))

        upsampled_masks = tf.image.resize(masks, input_shape, method="nearest")
        perturbated_inputs = perturbator(upsampled_masks)

        return perturbated_inputs, repeated_targets
