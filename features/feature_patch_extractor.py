import h5py
import numpy as np
import cv2
import os


class FeaturePatchExtractor:
    def __init__(self, feature_map_directory, patch_size=16):
        """
        Initializes the patch extractor.

        :param feature_map_directory: Path to the directory containing feature map files (.h5).
        :param patch_size: Size of the square patch (default is 16x16 pixels).
        """
        self.feature_map_directory = feature_map_directory
        self.patch_size = patch_size
        self.half_patch = patch_size // 2

    def load_feature_map(self, image_id):
        """
        Load a feature map from a HDF5 file.

        :param image_id: Identifier of the image to load the feature map for.
        :return: Feature map as a numpy array.
        """
        file_path = os.path.join(self.feature_map_directory, f"{image_id}.hdf5")
        with h5py.File(file_path, "r") as file:
            feature_map = file["feature_map_0"][
                :
            ]  # Adjust if stored under a different name
        return feature_map

    def extract_patch(self, feature_map, keypoint):
        """
        Extract a patch centered around the keypoint from the feature map.

        :param feature_map: The feature map array.
        :param keypoint: Keypoint (cv2.KeyPoint object).
        :return: Extracted patch as a numpy array.
        """
        x, y = int(keypoint.pt[0]), int(keypoint.pt[1])
        start_x = max(x - self.half_patch, 0)
        start_y = max(y - self.half_patch, 0)
        end_x = min(
            start_x + self.patch_size, feature_map.shape[1]
        )  # Ensure within bounds
        end_y = min(
            start_y + self.patch_size, feature_map.shape[2]
        )  # Ensure within bounds

        return feature_map[:, start_x:end_x, start_y:end_y]

    def get_patches_for_image(self, image_id, keypoints):
        """
        Retrieve all patches for the given keypoints in the specified image.

        :param image_id: Identifier of the image.
        :param keypoints: List of cv2.KeyPoint objects indicating keypoint positions.
        :return: List of patches centered on the keypoints.
        """
        feature_map = self.load_feature_map(image_id)
        print("Feature map shape:", feature_map.shape)  # Print the shape for debugging
        patches = [self.extract_patch(feature_map, kp) for kp in keypoints]
        del feature_map
        return patches
