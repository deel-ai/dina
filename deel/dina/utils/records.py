"""
Utility functions for handling TFRecords and datasets in TensorFlow.
"""
import os
import tensorflow as tf
from tqdm import tqdm
from typing import Optional

from keras_cv_attention_models.imagenet import data

# Methods for computing XAI datasets as TFRecord files
def serialize_example(preprocess_input, one_hot_prediction):
    """
    Creates a tf.train.Example message ready to be written to a file.
    Both input_image and prediction are TensorFlow tensors.
    """
    preprocess_input_raw = tf.io.serialize_tensor(preprocess_input).numpy()  # Convert to raw byte string
    one_hot_prediction_raw = tf.io.serialize_tensor(one_hot_prediction).numpy()  # Convert to raw byte string
    # Create a dictionary with features
    feature = {
        'preprocess_input': tf.train.Feature(bytes_list=tf.train.BytesList(value=[preprocess_input_raw])),
        'one_hot_prediction': tf.train.Feature(bytes_list=tf.train.BytesList(value=[one_hot_prediction_raw])),
    }
    # Create an Example message from the dictionary
    example_proto = tf.train.Example(features=tf.train.Features(feature=feature))
    return example_proto.SerializeToString()

# Define methods to load the precomputed xai datasets in TFRecord format
def parse_tfrecord_fn(example_proto):
    """
    Parses a single TFRecord example into input_image and prediction.
    """
    feature_description = {
        'preprocess_input': tf.io.FixedLenFeature([], tf.string),
        'one_hot_prediction': tf.io.FixedLenFeature([], tf.string),
    }
    # Parse the input tf.train.Example
    parsed_features = tf.io.parse_single_example(example_proto, feature_description)
    # Deserialize the tensors from byte strings
    preprocess_input = tf.io.parse_tensor(parsed_features['preprocess_input'], out_type=tf.float32)
    one_hot_prediction = tf.io.parse_tensor(parsed_features['one_hot_prediction'], out_type=tf.float32)
    return preprocess_input, one_hot_prediction

def create_tf_dataset_from_tfrecord(tfrecord_file):
    """
    Creates a tf.data.Dataset from a TFRecord file.
    """
    raw_dataset = tf.data.TFRecordDataset(tfrecord_file)
    parsed_dataset = raw_dataset.map(parse_tfrecord_fn)
    return parsed_dataset

def serialize_explanations(explanations):
    """
    Serialize the explanations tensor to a raw byte string.
    """
    explanations = tf.io.serialize_tensor(explanations).numpy()
    feature = {'explanations': tf.train.Feature(bytes_list=tf.train.BytesList(value=[explanations]))}
    example_proto = tf.train.Example(features=tf.train.Features(feature=feature))
    return example_proto.SerializeToString()

def generate_explanations_to_tfrecord(loader, explainer, tfrecord_file):
    """
    Generates explanations for the ImageNet validation dataset and saves all batches to a single TFRecord file.
    """
    with tf.io.TFRecordWriter(tfrecord_file) as writer:
        for preprocessed_batch, batch_targets in tqdm(loader, desc="Generating Explanations", unit="preprocessed_batch"):
        # batch_processed = 0
        # for preprocessed_batch, batch_targets in loader:
            explanations = explainer(preprocessed_batch, batch_targets)  # Generate explanations
            # print(f">> Batch {batch_processed} explained")
            # Write each instance in the batch to the TFRecord file
            for i in range(explanations.shape[0]):  # Iterate over each sample in the batch
                example = serialize_explanations(explanations[i])
                writer.write(example)
            # batch_processed += 1
        print(f"Explanations written to {tfrecord_file}")

def parse_explanations_record_fn(example_proto):
    """
    Parses a single TFRecord example into input_image and prediction.
    """
    feature_description = {
        'explanations': tf.io.FixedLenFeature([], tf.string)
    }
    # Parse the input tf.train.Example
    parsed_features = tf.io.parse_single_example(example_proto, feature_description)
    # Deserialize the tensors from byte strings
    explanations = tf.io.parse_tensor(parsed_features['explanations'], out_type=tf.float32)
    return explanations

def create_explanations_dataset_from_tfrecord(tfrecord_file):
    """
    Creates a tf.data.Dataset from a TFRecord file.
    """
    raw_dataset = tf.data.TFRecordDataset(tfrecord_file)
    parsed_dataset = raw_dataset.map(parse_explanations_record_fn)
    return parsed_dataset

@tf.function
def run_inference(model, img_batch):
    return model(img_batch)

def build_dataset_to_tfrecord(loader, model, tfrecord_file):
    """
    Preprocesses the ImageNet validation dataset and saves all batches to a single TFRecord file.
    """
    with tf.io.TFRecordWriter(tfrecord_file) as writer:
        for img_batch, _ in tqdm(loader, "Building TFRecord", total=len(loader)):
            predictions = run_inference(model, img_batch)
            num_classes = predictions.shape[-1]
            predicted_classes = tf.argmax(predictions, axis=-1)
            one_hot_predictions = tf.one_hot(predicted_classes, depth=num_classes)
            # Write each instance in the batch to the TFRecord file
            for i in range(img_batch.shape[0]):  # Iterate over each sample in the batch
                example = serialize_example(img_batch[i], one_hot_predictions[i])
                writer.write(example)
        print(f">> Dataset written to {tfrecord_file}")

def generate_or_load_preprocess_ds(
        model: tf.keras.Model,
        output_dir: str,
        batch_size: int,
        imagenet_json_path: Optional[str] = None) -> tf.data.Dataset:
    """
    Generates or loads the preprocess dataset (TFRecord). The preprocess dataset is the dataset ready to be used for
    computing explanations of a given model. Indeed, as each model has its own preprocessing, the preprocess dataset
    is the dataset where inputs (images) has been preprocessed with the model preprocessing. 
    
    If the dataset is already precomputed, it will be loaded from the TFRecord file. If not, it is generated and
    saved as a TFRecord file. The TFRecord file is saved in the output_dir as preprocess.tfrecord.

    Note:
        The generated dataset will be around 30GB for the ImageNet dataset.

    Args:
        model (tf.keras.Model): The model to be explained.
        image_loader (Optional[tf.data.Dataset]): The dataset used to generate the XAI dataset if needed.
    
    Returns:
        tf.data.Dataset: The TensorFlow dataset (either loaded or generated).
    
    Raises:
        ValueError: If no precomputed dataset is found and no image_loader is provided.
    """
    tfrecord_file = f"{output_dir}/preprocess_dataset.tfrecord"
    if os.path.exists(tfrecord_file):
        # Load precomputed dataset from TFRecord
        print(f">> Loading precomputed dataset from {tfrecord_file}")
        return create_tf_dataset_from_tfrecord(tfrecord_file)
    elif imagenet_json_path is not None:
        print(f">> Building loader")
        print(f">> Data JSON path: {imagenet_json_path}")
        # Load the dataset from the provided JSON file
        assert os.path.exists(imagenet_json_path), f"Data JSON file not found at {imagenet_json_path}"
        input_shape = model.input_shape[1:]
        rescale_mode = getattr(model, "rescale_mode", "torch")
        batch_size = batch_size
        image_loader = data.init_dataset(
            imagenet_json_path, input_shape=input_shape, batch_size=batch_size,
            eval_central_crop=0.95,
            resize_method="bicubic",
            resize_antialias=True,
            rescale_mode=rescale_mode)[1]
        print(f">> Image loader built")
        # Generate the dataset from image_loader if TFRecord doesn't exist
        print(f">> Generating dataset and saving to {tfrecord_file}")
        build_dataset_to_tfrecord(image_loader, model, tfrecord_file)
        return create_tf_dataset_from_tfrecord(tfrecord_file)
    else:
        raise ValueError("No precomputed dataset found, and no image_loader provided.")