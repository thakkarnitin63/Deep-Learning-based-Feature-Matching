# sift_extractor.py
import cv2
import numpy as np
from pathlib import Path
from PIL import Image


class SIFTFeatureExtractor:
    def __init__(self, nfeatures=500):
        self.nfeatures = nfeatures
        self.sift = cv2.SIFT_create(nfeatures=self.nfeatures)

    def extract_features(self, image_path):
        """Extract SIFT features and keypoints from an image."""
        image = Image.open(image_path)
        image = np.array(image, dtype=np.uint8)
        if len(image.shape) == 3 and image.shape[2] == 4:
            # Convert RGBA to RGB
            image = cv2.cvtColor(image, cv2.COLOR_RGBA2RGB)
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        keypoints, descriptors = self.sift.detectAndCompute(gray, None)
        return keypoints, descriptors

    def extract_patches(self, image, keypoints, patch_size=32):
        """Extract patches centered around each keypoint."""
        patches = []
        img_array = np.array(image)
        for kp in keypoints:
            x, y = int(kp.pt[0]), int(kp.pt[1])
            if (
                x >= patch_size // 2
                and x < img_array.shape[1] - patch_size // 2
                and y >= patch_size // 2
                and y < img_array.shape[0] - patch_size // 2
            ):
                patch = img_array[
                    y - patch_size // 2 : y + patch_size // 2,
                    x - patch_size // 2 : x + patch_size // 2,
                ]
                patches.append(patch)
        return np.array(patches)
