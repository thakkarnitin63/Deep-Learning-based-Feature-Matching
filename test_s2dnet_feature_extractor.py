import matplotlib.pyplot as plt
import numpy as np
import sys
import os
import h5py

from features.feature_map_extractor import (
    FeatureMapExtractor,
)  # Import the FeatureExtractor class


def get_project_root():
    """Returns the absolute path of the project root."""
    # __file__ gives the path of the current script; os.path.dirname() gives the directory containing the script.
    current_dir = os.path.dirname(os.path.abspath(__file__))
    # Adjust the path as necessary based on your project's structure
    # For example, if this script is in a subfolder of the root, you might need to use os.path.dirname() multiple times
    project_root = os.path.dirname(
        current_dir
    )  # Modify if your script is deeper in the directory structure
    return project_root, current_dir


def get_dataset_path(path):
    """Constructs the absolute path to the dataset."""
    root_dir, curr_dir = get_project_root()
    dataset_path = os.path.join(root_dir, curr_dir, path)
    return dataset_path


if __name__ == "__main__":
    # Initialize the feature extractor
    extractor = FeatureMapExtractor()
    dataset_path = get_dataset_path("dataset/Compressed_buddha")

    extractor.process_directory(directory=dataset_path, batch_size=1)
    output_dir = extractor.output_dir
    assert os.path.exists(output_dir), "Output directory was not created."

    files = [f for f in os.listdir(output_dir) if f.endswith(".hdf5")]
    assert len(files) > 0, "No HDF5 files were created."

    sample_file = os.path.join(output_dir, files[0])
    with h5py.File(sample_file, "r") as f:
        assert len(f.keys()) > 0, "No feature maps saved in HDF5 file."
        first_map = list(f.keys())[0]
        feature_map = f[first_map][:]
        assert (
            feature_map.ndim == 3
        ), f"Feature map dimensions are incorrect. {feature_map.ndim}"
        print("Feature map shape:", feature_map.shape)

    print("All tests passed successfully!")
