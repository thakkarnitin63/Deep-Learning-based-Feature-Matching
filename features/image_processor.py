import cv2
import numpy as np
import os
import glob
from tqdm import tqdm
from matplotlib import pyplot as plt
import mayavi.mlab as mlab


class FeatureDescriptors:
    ORB = "ORB"
    SIFT = "SIFT"
    AKAZE = "AKAZE"


class Matcher:
    FLANN = "FLANN"
    BRUTE_FORCE = "BRUTE_FORCE"


class ImageProcessor:
    def __init__(
        self,
        folder_path,
        camera_matrix=np.eye(3),
        descriptor_type=FeatureDescriptors.AKAZE,
        matcher_type=Matcher.BRUTE_FORCE,
        max_features=5000,
        use_clahe=False,
    ):
        self.folder_path = folder_path
        self.camera_matrix = camera_matrix
        self.descriptor_type = descriptor_type
        self.matcher_type = matcher_type
        self.max_features = max_features
        self.use_clahe = use_clahe
        self.keypoints = {}
        self.descriptors = {}
        self.matches = {}
        self.images = {}
        self.init_feature_detector()
        self.init_matcher()

    def init_feature_detector(self):
        if self.descriptor_type == FeatureDescriptors.SIFT:
            self.detector = cv2.SIFT_create(nfeatures=self.max_features)
        elif self.descriptor_type == FeatureDescriptors.AKAZE:
            self.detector = cv2.AKAZE_create()
        elif self.descriptor_type == FeatureDescriptors.ORB:
            self.detector = cv2.ORB_create(nfeatures=self.max_features)

    def init_matcher(self):
        if self.matcher_type == Matcher.FLANN:
            if self.descriptor_type == FeatureDescriptors.SIFT:
                index_params = dict(algorithm=1, trees=5)
                search_params = dict(checks=50)
                self.matcher = cv2.FlannBasedMatcher(index_params, search_params)
            else:
                raise ValueError("FLANN matching currently supports only SIFT.")
        elif self.matcher_type == Matcher.BRUTE_FORCE:
            if self.descriptor_type in [
                FeatureDescriptors.SIFT,
                FeatureDescriptors.AKAZE,
            ]:
                self.matcher = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)
            elif self.descriptor_type == FeatureDescriptors.ORB:
                self.matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

    def load_images(self):
        for filepath in sorted(
            glob.glob(os.path.join(self.folder_path, "*.jpg"))
            + glob.glob(os.path.join(self.folder_path, "*.png"))
        ):
            img_id = os.path.splitext(os.path.basename(filepath))[
                0
            ]  # Removes the extension from the filename
            img = cv2.imread(filepath, cv2.IMREAD_GRAYSCALE)
            if self.use_clahe:
                clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
                img = clahe.apply(img)
            self.images[img_id] = img
            (
                self.keypoints[img_id],
                self.descriptors[img_id],
            ) = self.detector.detectAndCompute(img, None)

    def match_features(self, img_id1, img_id2):
        kp1, des1 = self.keypoints[img_id1], self.descriptors[img_id1]
        kp2, des2 = self.keypoints[img_id2], self.descriptors[img_id2]
        matches = self.matcher.match(des1, des2)
        # Sort them in the order of their distance.
        matches = sorted(matches, key=lambda x: x.distance)
        good_matches = [m for m in matches if m.distance < 50 * matches[0].distance]
        self.matches[(img_id1, img_id2)] = good_matches[:200]  # Store top 200 matches

    def visualize_matches(self, img_id1, img_id2):
        img1, img2 = self.images[img_id1], self.images[img_id2]
        matches = self.matches[(img_id1, img_id2)]
        kp1, kp2 = self.keypoints[img_id1], self.keypoints[img_id2]
        img_matches = cv2.drawMatches(
            img1,
            kp1,
            img2,
            kp2,
            matches,
            None,
            flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS,
        )
        plt.imshow(img_matches)
        plt.show()
        cv2.imwrite("plot.png", img_matches)

    def compute_essential_matrix(self, img_id1, img_id2):
        kp1, kp2 = self.keypoints[img_id1], self.keypoints[img_id2]
        matches = self.matches[(img_id1, img_id2)]
        pts1 = np.int32([kp1[m.queryIdx].pt for m in matches])
        pts2 = np.int32([kp2[m.trainIdx].pt for m in matches])
        essential_matrix, mask = cv2.findEssentialMat(
            pts1, pts2, self.camera_matrix, method=cv2.RANSAC, prob=0.999, threshold=1.0
        )
        return essential_matrix, mask

    def process_all_images(self, visualize=False):
        self.load_images()
        # Implement logic to match features between all pairs of images
        for img_id1 in self.images:
            for img_id2 in self.images:
                if img_id1 < img_id2:
                    self.match_features(img_id1, img_id2)
                    if visualize:
                        self.visualize_matches(img_id1, img_id2)


if __name__ == "__main__":
    folder_path = "path_to_images"
    camera_matrix = np.array(
        [[1000, 0, 320], [0, 1000, 240], [0, 0, 1]]
    )  # Example camera matrix
    processor = ImageProcessor(folder_path, camera_matrix)
    processor.process_all_images()
