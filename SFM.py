import os
import cv2
import numpy as np
from PIL import Image
import plotly.graph_objs as go
import gtsam


# Class to fetch and load images from a specified directory
class ImageFetcher:
    def __init__(self, path):
        self.images = []
        self.load_images(path)

    def load_images(self, path):
        # Sort files numerically and load each image, converting to RGB
        file_list = sorted(
            os.listdir(path),
            key=lambda x: int(os.path.splitext(x)[0]) if x.isdigit() else x,
        )
        for filename in file_list:
            img_path = os.path.join(path, filename)
            img = cv2.imread(img_path)
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            self.images.append(img_rgb)


# Class to handle feature detection and matching using SIFT
class FeatureMatcher:
    def __init__(self):
        # Initialize SIFT detector
        self.sift = cv2.SIFT_create(contrastThreshold=0.04, edgeThreshold=10)

    def detect_and_compute(self, images):
        # Detect and compute keypoints and descriptors for a list of images
        keypoints, descriptors = [], []
        for image in images:
            kp, des = self.sift.detectAndCompute(image, None)
            keypoints.append(kp)
            descriptors.append(des)
        return keypoints, descriptors


# Class to estimate poses using feature matches
class PoseEstimator:
    def __init__(self, images):
        # Initialize camera matrix based on the first image dimensions
        self.K = np.matrix(
            [
                [1690, 0, images[0].shape[1] / 2],
                [0, 1690, images[0].shape[0] / 2],
                [0, 0, 1],
            ]
        )
        self.projections = []
        self.trajectories = []
        self.correspondences = {}

    def find_essential_matrix(self, des1, des2, kp1, kp2):
        # Match descriptors and compute the essential matrix
        bf = cv2.BFMatcher(cv2.NORM_L2)
        matches = bf.knnMatch(
            np.array(des1, dtype=np.float32), np.array(des2, dtype=np.float32), k=2
        )
        good_matches = [m for m, n in matches if m.distance < 0.7 * n.distance]
        pts1 = np.float32([kp1[m.queryIdx].pt for m in good_matches])
        pts2 = np.float32([kp2[m.trainIdx].pt for m in good_matches])
        E, _ = cv2.findEssentialMat(pts1, pts2, self.K, cv2.RANSAC, 0.999, 1.0)
        return E, pts1, pts2

    def recover_pose_and_triangulate(self, E, pts1, pts2):
        # Decompose essential matrix to obtain possible poses and triangulate
        U, _, Vt = np.linalg.svd(E)
        W = np.matrix("0 -1 0; 1 0 0; 0 0 1")
        possible_rotations = [U @ W @ Vt, U @ W.T @ Vt]
        possible_translations = [U[:, 2], -U[:, 2]]

        best_count = 0
        best_projection = None
        best_points = None
        for R in possible_rotations:
            for T in possible_translations:
                R = np.asarray(R)
                T = np.asarray(T).reshape(-1, 1)
                P = self.K @ np.hstack((R, T))
                pts3d = cv2.triangulatePoints(
                    self.K @ np.hstack((np.eye(3), np.zeros((3, 1)))), P, pts1.T, pts2.T
                )
                pts3d /= pts3d[3, :]
                positive_depth = np.sum(pts3d[2, :] > 0)

                if positive_depth > best_count:
                    best_count = positive_depth
                    best_projection = P
                    best_points = pts3d[:3, :]

        self.projections.append(best_projection)
        self.trajectories.append(best_points)
        if len(self.projections) == 1:
            self.correspondences[1] = {
                tuple(pts2[i]): tuple(best_points[:, i]) for i in range(pts2.shape[0])
            }

        return best_projection, best_points

    def incremental_pose_estimation(self, descriptors, keypoints):
        # Use previously estimated poses to find new poses incrementally
        for i in range(1, len(descriptors) - 1):
            pts1, pts2 = self.find_correspondences(
                keypoints[i], keypoints[i + 1], self.correspondences[i]
            )
            _, rvec, tvec, _ = cv2.solvePnPRansac(
                np.array(pts1), np.array(pts2), self.K, None
            )
            R, _ = cv2.Rodrigues(rvec)
            T = tvec.reshape(-1, 1)
            P = self.K @ np.hstack((R, T))
            self.projections.append(P)
            pts3d = cv2.triangulatePoints(self.projections[i - 1], P, pts1.T, pts2.T)
            pts3d /= pts3d[3, :]
            self.trajectories.append(pts3d[:3, :])
            self.correspondences[i + 1] = {
                tuple(pts2[j]): tuple(pts3d[:3, j]) for j in range(pts2.shape[0])
            }

    def find_correspondences(self, kp_next, kp_actual, pre_correspondences):
        # Identify matching points between subsequent frames
        next_pts = []
        actual_pts = []
        for pt in kp_next:
            if tuple(pt) in pre_correspondences:
                next_pts.append(pt)
                actual_pts.append(pre_correspondences[tuple(pt)])
        return actual_pts, next_pts


class StructureFromMotion:
    def __init__(self, image_folder, use_bundle_adjustment=False):
        # Initialize with the path to images and whether to use bundle adjustment
        self.image_folder = image_folder
        self.use_bundle_adjustment = use_bundle_adjustment
        self.fetcher = ImageFetcher(image_folder)
        self.matcher = FeatureMatcher()
        self.estimator = PoseEstimator(self.fetcher.images)

    def run(self):
        # Main execution flow of the Structure from Motion process
        keypoints, descriptors = self.matcher.detect_and_compute(self.fetcher.images)
        E, pts1, pts2 = self.estimator.find_essential_matrix(
            descriptors[0], descriptors[1], keypoints[0], keypoints[1]
        )
        _, _ = self.estimator.recover_pose_and_triangulate(E, pts1, pts2)
        self.estimator.incremental_pose_estimation(descriptors, keypoints)
        if self.use_bundle_adjustment:
            self.perform_bundle_adjustment()
        self.visualize_results()

    def perform_bundle_adjustment(self):
        # Detailed implementation of the bundle adjustment process using GTSAM
        K_values = self.estimator.K
        Kmat = gtsam.Cal3_S2(
            K_values[0, 0], K_values[1, 1], 0, K_values[0, 2], K_values[1, 2]
        )
        graph = gtsam.NonlinearFactorGraph()
        measurement_noise = gtsam.noiseModel.Isotropic.Sigma(2, 1.0)
        pose_noise = gtsam.noiseModel.Diagonal.Sigmas(
            np.array([0.3, 0.3, 0.3, 0.1, 0.1, 0.1])
        )
        initial_pose = gtsam.Pose3(
            gtsam.Rot3(self.estimator.projections[0][:3, :3]),
            self.estimator.projections[0][:3, 3],
        )
        graph.add(gtsam.PriorFactorPose3(X(0), initial_pose, pose_noise))
        countL = 0
        for i, projection in enumerate(self.estimator.projections):
            for pt2d, pt3d in self.estimator.correspondences[i + 1].items():
                point3d = gtsam.Point3(pt3d)
                measurement = gtsam.Point2(pt2d[0], pt2d[1])
                graph.add(
                    gtsam.GenericProjectionFactorCal3_S2(
                        measurement, measurement_noise, X(i), L(countL), Kmat
                    )
                )
                countL += 1
        initial_estimate = gtsam.Values()
        for i, projection in enumerate(self.estimator.projections):
            pose = gtsam.Pose3(gtsam.Rot3(projection[:3, :3]), projection[:3, 3])
            initial_estimate.insert(X(i), pose)
            for pt2d, pt3d in self.estimator.correspondences[i + 1].items():
                initial_estimate.insert(L(countL), gtsam.Point3(pt3d))
        optimizer = gtsam.LevenbergMarquardtOptimizer(graph, initial_estimate)
        result = optimizer.optimize()
        optimized_poses = []
        optimized_landmarks = []
        for i in range(len(self.estimator.projections)):
            optimized_pose = result.atPose3(X(i))
            optimized_poses.append(optimized_pose)
        for i in range(countL):
            optimized_landmark = result.atPoint3(L(i))
            optimized_landmarks.append(optimized_landmark)
        print("Initial error =", graph.error(initial_estimate))
        print("Final error =", graph.error(result))
        print("Optimized poses:", len(optimized_poses))
        print("Optimized landmarks:", len(optimized_landmarks))

    def visualize_results(self):
        # Visualize the results of the SfM process and optionally the bundle adjustment
        def plot_point_cloud(points, title, color="blue"):
            trace = go.Scatter3d(
                x=points[0, :],  # X coordinates of points
                y=points[1, :],  # Y coordinates of points
                z=points[2, :],  # Z coordinates of points
                mode="markers",
                marker=dict(size=1, opacity=0.8, color=color),
                name="Point Cloud",
            )
            layout = go.Layout(
                title=title,
                scene=dict(
                    xaxis=dict(title="X Axis", range=[-160, 160]),
                    yaxis=dict(title="Y Axis", range=[-200, 200]),
                    zaxis=dict(title="Z Axis", range=[-20, 500]),
                ),
            )
            fig = go.Figure(data=[trace], layout=layout)
            fig.show()

        def plot_camera_poses(poses, title):
            traces = []
            for i, pose in enumerate(poses):
                position = pose[:3, 3]
                x_axis = pose[:3, 0] * 5 + position
                y_axis = pose[:3, 1] * 5 + position
                z_axis = pose[:3, 2] * 5 + position
                traces.extend(
                    [
                        go.Scatter3d(
                            x=[position[0], x_axis[0]],
                            y=[position[1], x_axis[1]],
                            z=[position[2], x_axis[2]],
                            mode="lines",
                            line=dict(color="red", width=2),
                            name=f"X Axis {i + 1}",
                        ),
                        go.Scatter3d(
                            x=[position[0], y_axis[0]],
                            y=[position[1], y_axis[1]],
                            z=[position[2], y_axis[2]],
                            mode="lines",
                            line=dict(color="green", width=2),
                            name=f"Y Axis {i + 1}",
                        ),
                        go.Scatter3d(
                            x=[position[0], z_axis[0]],
                            y=[position[1], z_axis[1]],
                            z=[position[2], z_axis[2]],
                            mode="lines",
                            line=dict(color="blue", width=2),
                            name=f"Z Axis {i + 1}",
                        ),
                    ]
                )
            layout = go.Layout(
                title=title,
                scene=dict(
                    xaxis=dict(title="X Axis"),
                    yaxis=dict(title="Y Axis"),
                    zaxis=dict(title="Z Axis"),
                ),
            )
            fig = go.Figure(data=traces, layout=layout)
            fig.show()

        if self.estimator.trajectories:
            plot_point_cloud(
                np.hstack(self.estimator.trajectories), "Initial 3D Reconstruction"
            )
        if self.estimator.projections:
            plot_camera_poses(self.estimator.projections, "Initial Camera Trajectories")

        if self.use_bundle_adjustment:
            if hasattr(self, "optimized_poses") and hasattr(
                self, "optimized_landmarks"
            ):
                plot_point_cloud(
                    np.hstack(self.optimized_landmarks),
                    "Optimized 3D Reconstruction",
                    color="red",
                )
                plot_camera_poses(self.optimized_poses, "Optimized Camera Trajectories")


def main():
    # Set the path to the directory containing your images
    image_directory = "/path/to/your/images"

    # Initialize the StructureFromMotion system
    sfm_system = StructureFromMotion(image_directory, use_bundle_adjustment=True)

    # Run the Structure from Motion process
    sfm_system.run()


if __name__ == "__main__":
    main()
