"""
Main script to generate Randomization scores that have been computed for DINA

Notes:
    In contrast to the other compute metrics scripts, this one allow to run only
    one metric at a time. This is because the randomization process do not allow to
    run multiple metrics at the same time. Indeed, the randomization is done at the
    model level and so there is not advantage to run multiple metrics at the same time.
    You can run multiple metrics in parallel using the command line.

    Furthermore, the scores are not computed on the 50 000 samples of ImageNet but only
    on the first 1024 samples.
"""
import os
import argparse
import json
from typing import Optional, List

import tensorflow as tf

from deel.dina.metrics.randomization import ModelRandomizationMetric, RandomLogitMetric, ProgressiveLayerRandomization

from deel.dina.utils import get_model, get_explainer, get_explainer_args
from deel.dina.utils import compute_dataset_mean
from deel.dina.utils import generate_or_load_preprocess_ds, create_explanations_dataset_from_tfrecord

DINA_RANDOMIZATION_METRICS_ARGS = {
    "ModelRandomizationMetric01": {"stop_layer": 0.015},
    "ModelRandomizationMetric05": {"stop_layer": 0.05},
    "RandomLogitMetric": {}
}
DINA_RANDOMIZATION_SUBSAMPLE_SIZE = 1024

# Create a lookup for available metric names
RANDOMIZATION_METRICS = {
    "ModelRandomizationMetric01": ModelRandomizationMetric,
    "ModelRandomizationMetric05": ModelRandomizationMetric,
    "RandomLogitMetric": RandomLogitMetric,
}

def generate_randomization(
    model_name: str,
    explainer_name: str,
    metric_name: str,
    output_dir: str,
    batch_size: int,
    imagenet_json: Optional[str] = None
):
    """
    Generate randomization scores for a given model, explainer and the associated explanations.

    Args:
        model_name (str): The model to be explained.
        explainer_name (str): The name of the explainer used.
        metric_name (str): The metric name of the randomization metrics to compute.
        output_dir (str): Directory to save the benchmark results.
        batch_size (int): Batch size for computing randomization scores.
        imagenet_json (Optional[str]): Path to the ImageNet JSON file. If a precomputed dataset is not found,
                                       this will be used to generate the dataset. If not provided, a precomputed
                                       dataset must be available at output dir.
    """
    # Load the explanations
    explanations_dir = os.path.join(output_dir, model_name, explainer_name)
    print(f"> Loading explanations from: {explanations_dir}")
    explanations_tfrecord_path = f"{explanations_dir}/explanations.tfrecord"
    assert os.path.exists(explanations_tfrecord_path), f"Explanations not found at {explanations_dir}"
    explanations_ds = create_explanations_dataset_from_tfrecord(explanations_tfrecord_path)

    # Get the model
    print(f"> Loading model: {model_name}")
    model = get_model(model_name)
    # Get the preprocess dataset (that is preprocessed_inputs, labels)
    print(f"> Loading dataset")
    model_dir = f"{output_dir}/{model_name}"
    # Get the preprocess dataset (that is preprocessed_inputs, labels)
    print(f"> Loading dataset")
    preprocess_dataset = generate_or_load_preprocess_ds(
        model=model,
        output_dir=model_dir,
        batch_size=batch_size,
        imagenet_json_path=imagenet_json
    )
    # Get the metric loader
    print(f"> Building metrics data loader")
    metric_loader = tf.data.Dataset.zip((preprocess_dataset, explanations_ds))
    metric_loader = metric_loader.map(lambda x, y: (x[0], x[1], y)).take(DINA_RANDOMIZATION_SUBSAMPLE_SIZE)
    metric_loader = metric_loader.batch(batch_size=batch_size)
    # Get the explainer
    print(f"> Loading explainer: {explainer_name}")
    explainer_args = get_explainer_args(explainer_name)

    if "mean" in explainer_args.values():
        mean_value = compute_dataset_mean(preprocess_dataset)
        for k, v in explainer_args.items():
            if v == "mean":
                explainer_args[k] = mean_value

    if model.name == "dinov2_vit_small14":
        @tf.function
        def dino_operator(model, inputs, targets):
            scores = tf.reduce_sum(model(inputs) * targets, axis=-1)
            return scores
        explainer_args["operator"] = dino_operator

    explainer = get_explainer(
        explainer_name=explainer_name,
        model=model,
        batch_size=batch_size,
        explainer_args=explainer_args
    )
    # Get the metric
    print(f"> Building metric")
    metric_args = DINA_RANDOMIZATION_METRICS_ARGS[metric_name]
    if metric_name.startswith("ModelRandomizationMetric"):
        # Get the progressive layer randomization strategy
        randomization_strategy = ProgressiveLayerRandomization(stop_layer=metric_args["stop_layer"])
        metric_args["randomization_strategy"] = randomization_strategy
        del metric_args["stop_layer"]
    metric = RANDOMIZATION_METRICS[metric_name](model=model, batch_size=batch_size, explainer=explainer, **metric_args)
    print(f"> Generating {metric_name} scores")
    scores = metric.evaluate(metric_loader)
    print(f">> {metric_name} scores generated")
    print(f">> Scores shape: {scores.shape}")
    print(f">> Mean score: {tf.reduce_mean(scores)}")

    # Save the score
    scores = scores.numpy()
    with open(f"{explanations_dir}/{metric_name}_scores.json", "w") as f:
        json.dump(scores.tolist(), f)
    print(f">> Scores saved to {explanations_dir}/{metric_name}_scores.json")
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compute randomization scores for a given model, explainer and the associated explanations.")
    parser.add_argument("--model_name", type=str, required=True, help="Name of the model.")
    parser.add_argument("--explainer_name", type=str, required=True, help="Name of the explainer.")
    parser.add_argument("--metric_name", type=str, required=True, help="The randomization metric name to compute. Options: ModelRandomizationMetric01, ModelRandomizationMetric05, RandomLogitMetric.")
    parser.add_argument("--output_dir", type=str, required=True, help="Directory to save the benchmark results.")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size for computing randomization scores.")
    parser.add_argument("--imagenet_json", type=str, help="Path to the ImageNet JSON file. If a precomputed dataset is not found, this will be used to generate the dataset. If not provided, a precomputed dataset must be available at output dir.")
    args = parser.parse_args()

    # Check if the specified metrics are valid
    if args.metric_name not in RANDOMIZATION_METRICS.keys():
        raise ValueError(f"Invalid metric requested: {args.metric}. Must be one of {RANDOMIZATION_METRICS.keys()}")

    generate_randomization(
            model_name=args.model_name,
            explainer_name=args.explainer_name,
            metric_name=args.metric_name,
            output_dir=args.output_dir,
            batch_size=args.batch_size,
            imagenet_json=args.imagenet_json
    )
