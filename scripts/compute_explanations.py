"""
Main script to compute explanations
"""
import os
import argparse

import tensorflow as tf

from deel.dina.utils import get_model, get_explainer, get_explainer_args
from deel.dina.utils import compute_dataset_mean
from deel.dina.utils import generate_or_load_preprocess_ds, generate_explanations_to_tfrecord

def generate_explanations(
    model_name: str,
    explainer_name: str,
    batch_size: int,
    imagenet_json: str,
    output_dir: str
):
    """
    Generate explanations for the specified model and explainer.

    Args:
        model_name (str): The name of the model being explained.
        explainer_name (str): The name of the explainer to use.
        batch_size (int): Batch size for computing explanations.
        imagenet_json (str): Path to the ImageNet JSON file.
        output_dir (str): Directory to save the explanations as a tfrecord.
    """
    # Get the model
    print(f"> Loading model: {model_name}")
    model = get_model(model_name)
    # Get the preprocess dataset (that is preprocessed_inputs, labels)
    print(f"> Loading dataset")
    model_dir = f"{output_dir}/{model_name}"
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    preprocess_dataset = generate_or_load_preprocess_ds(
        model=model,
        output_dir=model_dir,
        batch_size=batch_size,
        imagenet_json_path=imagenet_json
    ).batch(batch_size)
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
    # Generate explanations
    print(f"> Generating explanations")
    tfrecord_dir = f"{model_dir}/{explainer_name}"
    if not os.path.exists(tfrecord_dir):
        os.makedirs(tfrecord_dir)
    generate_explanations_to_tfrecord(loader=preprocess_dataset, explainer=explainer, tfrecord_file=f"{tfrecord_dir}/explanations.tfrecord")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compute explanations for a given model and explainer.")
    parser.add_argument("--model_name", type=str, required=True, help="The name of the model being explained.")
    parser.add_argument("--explainer_name", type=str, required=True, help="The name of the explainer to use.")
    parser.add_argument("--batch_size", type=int, default=64, help="Batch size for computing explanations.")
    parser.add_argument("--imagenet_json", type=str, help="Path to the ImageNet JSON file. If a precomputed dataset is not found, this will be used to generate the dataset. If not provided, a precomputed dataset must be available at output dir.")
    parser.add_argument("--output_dir", type=str, required=True, help="Directory to save the benchmark results.")
    args = parser.parse_args()
    # Check the directory exists or create it
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    # Generate explanations
    generate_explanations(
        model_name=args.model_name,
        explainer_name=args.explainer_name,
        batch_size=args.batch_size,
        imagenet_json=args.imagenet_json,
        output_dir=args.output_dir
    )
