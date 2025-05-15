"""
Helpers to manage the imagenet dataset.
"""
import os
import sys
import argparse

def structure_imagenet_val_dataset(raw_dir: str):
    """
    This function structures the validation dataset of imagenet. The validation
    dataset is not structured in the same way as the training dataset. The
    validation dataset is structured in a way that the images are not in folders
    corresponding to their labels. 

    This function will create the folders and move the images to the correct
    folders if the validation dataset is not already structured.

    Args:
        raw_dir (str): the path pointing to the raw data. This must point to the
            `ILSVRC/Data/CLS-LOC` folder. The train/val split are made by appending
            `train` and `val` to this path.
    """
    path_test = os.path.join(raw_dir, "val")        
    # build structure for val dataset
    is_unformatted = len([name for name in os.listdir(path_test)]) > 5000
    if is_unformatted:
        path_test_labels = os.path.join(
            raw_dir, "imagenet_2012_validation_synset_labels.txt"
        )
        # Read in the 50000 synsets associated with the validation data set.
        labels = [l.strip() for l in open(path_test_labels).readlines()]
        unique_labels = set(labels)

        # Make all sub-directories in the validation data dir.
        for label in unique_labels:
            labeled_data_dir = os.path.join(path_test, label)
            os.makedirs(labeled_data_dir)

        # Move all of the image to the appropriate sub-directory.
        for i in range(len(labels)):
            basename = "ILSVRC2012_val_000%.5d.JPEG" % (i + 1)
            original_filename = os.path.join(path_test, basename)
            if not os.path.exists(original_filename):
                print("Failed to find: ", original_filename)
                sys.exit(-1)
            new_filename = os.path.join(path_test, labels[i], basename)
            os.rename(original_filename, new_filename)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--raw_dir",
        type=str,
        required=True,
        help="This should point to the `ILSVRC/Data/CLS-LOC` folder",
    )
    args = parser.parse_args()
    structure_imagenet_val_dataset(args.raw_dir)
