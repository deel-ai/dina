"""
Main script to generate Fidelity scores that have been computed for DINA
"""
import os
import argparse
import json
from typing import Optional, List

import tensorflow as tf

from deel.dina.metrics.fidelity import Insertion, Deletion, MuFidelity

from deel.dina.utils import get_model, get_explainer, get_explainer_args
from deel.dina.utils import compute_dataset_mean
from deel.dina.utils import generate_or_load_preprocess_ds, create_explanations_dataset_from_tfrecord

DINA_FIDELITY_METRICS_ARGS = {
    "Insertion": {"nb_steps": 200},
    "Deletion": {"nb_steps": 200},
    "MuFidelity": {"nb_samples": 200}
}

# Create a lookup for available metric names
FIDELITY_METRICS = {
    "Insertion": Insertion,
    "Deletion": Deletion,
    "MuFidelity": MuFidelity,
}

def generate_fidelity(
        model_name: str,
        explainer_name: str,
        metrics: List[str],
        output_dir: str,
        batch_size: int,
        imagenet_json: Optional[str] = None
):
    """
    Generate fidelity scores for a given model, explainer and the associated explanations.

    Args:
        model_name (str): The model to be explained.
        explainer_name (str): The name of the explainer used.
        metrics (List[str]): The list of fidelity metrics to compute.
        output_dir (str): Directory to save the benchmark results.
        batch_size (int): Batch size for computing fidelity scores.
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
    preprocess_dataset = generate_or_load_preprocess_ds(
        model=model,
        output_dir=model_dir,
        batch_size=batch_size,
        imagenet_json_path=imagenet_json
    )
    # Get the metric loader
    print(f"> Building metrics data loader")
    metric_loader = tf.data.Dataset.zip((preprocess_dataset, explanations_ds))
    metric_loader = metric_loader.map(lambda x, y: (x[0], x[1], y))
    metric_loader = metric_loader.batch(batch_size=batch_size)

    # Loop over the metrics
    for metric_name in metrics:
        print(f"> Generating {metric_name} scores")
        metric_args = DINA_FIDELITY_METRICS_ARGS[metric_name]
        metric = FIDELITY_METRICS[metric_name](model=model, batch_size=batch_size, **metric_args)
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
    parser = argparse.ArgumentParser(description="Compute fidelity scores for a given model, explainer and the associated explanations.")
    parser.add_argument("--model_name", type=str, required=True, help="Name of the model.")
    parser.add_argument("--explainer_name", type=str, required=True, help="Name of the explainer.")
    parser.add_argument("--metrics", type=str, nargs='+', required=True, help="List of fidelity metrics to compute. Options: Insertion, Deletion, MuFidelity.")
    parser.add_argument("--output_dir", type=str, required=True, help="Directory to save the benchmark results.")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size for computing fidelity scores.")
    parser.add_argument("--imagenet_json", type=str, help="Path to the ImageNet JSON file. If a precomputed dataset is not found, this will be used to generate the dataset. If not provided, a precomputed dataset must be available at output dir.")
    args = parser.parse_args()

    # Check if the specified metrics are valid
    invalid = [m for m in args.metrics if m not in FIDELITY_METRICS]
    if invalid:
        raise ValueError(f"Invalid metrics requested: {invalid}. Must be one of {FIDELITY_METRICS.keys()}")

    generate_fidelity(
            model_name=args.model_name,
            explainer_name=args.explainer_name,
            metrics=args.metrics,
            output_dir=args.output_dir,
            batch_size=args.batch_size,
            imagenet_json=args.imagenet_json
    )
