from scipy.optimize import minimize
import numpy as np
import cv2
from features.feature_patch_extractor import FeaturePatchExtractor
from features.track_builder import TrackBuilder

class TrackOptimizer:
    def __init__(self, patch_extractor: FeaturePatchExtractor, tracks_file_path):
        self.patch_extractor = patch_extractor
        self.tracks = TrackBuilder.load_tracks_from_hdf5(tracks_file_path)

    def optimize_track(self, track, fixed_kp_index):
        """
        Optimizes the keypoint locations within a track using L2 norm and bilinear interpolation.

        Args:
            track: A list of keypoint dictionaries (including image_id, kp_index, position).
            fixed_kp_index: Index of the keypoint with the highest connectivity to be fixed in place.

        Returns:
            A list of optimized keypoint dictionaries.
        """

        def objective_function(params):
            """
            Objective function to minimize the L2 norm of differences between feature vectors.

            Args:
                params: A flattened array containing all keypoint location updates (dx, dy for each keypoint).

            Returns:
                The L2 norm of the difference between feature vectors after applying updates.
            """
            total_diff = 0
            for i, keypoint in enumerate(track):
                if i == fixed_kp_index:
                    continue  # Skip the fixed keypoint

                # Extract original keypoint location and feature patch
                x, y = keypoint["position"]
                patch = self.patch_extractor.extract_patch(
                    self.patch_extractor.load_feature_map(keypoint["image_id"]),
                    cv2.KeyPoint(x, y, 1.0),
                )

                # Apply parameter updates (dx, dy) for this keypoint
                dx, dy = params[2 * i : 2 * (i + 1)]
                new_x = x + dx
                new_y = y + dy

                # Use bilinear interpolation to get the feature vector at the new location
                interpolated_descriptor = self.bilinear_interpolation(
                    patch, (new_x, new_y)
                )

                # Compare with the original feature vector in the next image
                next_image_id = track[i + 1]["image_id"]
                next_kp_index = track[i + 1]["kp_index"]
                next_feature_map = self.patch_extractor.load_feature_map(next_image_id)
                next_descriptor = next_feature_map[
                    track[i + 1]["position"][1], track[i + 1]["position"][0]
                ]

                # L2 norm of the difference
                diff = np.linalg.norm(interpolated_descriptor - next_descriptor)
                total_diff += diff

            return total_diff
        
        # Initial guess: no movement for all keypoints except the first one (fixed)
        num_keypoints = len(track)
        initial_params = np.zeros(2 * num_keypoints)
        initial_params[1 : 2 * fixed_kp_index] = (
            1  # Set initial movement for all except the fixed one
        )

        # Define optimization constraints (limit movement to 8 pixels in each direction)
        bounds = [(-8, 8)] * num_keypoints

        # Perform optimization using L-BFGS-B algorithm
        res = minimize(
            objective_function,
            initial_params,
            method="L-BFGS-B",
            bounds=bounds,
        )

        # Update keypoint positions with optimized parameters
        optimized_track = []
        for i, keypoint in enumerate(track):
            if i == fixed_kp_index:
                optimized_track.append(keypoint)
                continue

            dx, dy = res.x[2 * i : 2 * (i + 1)]
            optimized_track.append(
                {
                    "image_id": keypoint["image_id"],
                    "kp_index": keypoint["kp_index"],
                    "position": (
                        keypoint["position"][0] + dx,
                        keypoint["position"][1] + dy,
                    ),
                }
            )

        return optimized_track

    def optimize_all_tracks(self):
        """
        Optimizes all tracks in the list using the optimize_track function.

        Returns:
            A list of optimized tracks.
        """
        optimized_tracks = []
        for track in self.tracks:
            # Find the keypoint with the highest connectivity (most edges in the graph)
            max_connections = 0
            fixed_kp_index = None
            for i, keypoint in enumerate(track):
                connections = len(self.patch_extractor.graph.edges(keypoint))
                if connections > max_connections:
                    max_connections = connections
                    fixed_kp_index = i

            optimized_track = self.optimize_track(track.copy(), fixed_kp_index)
            optimized_tracks.append(optimized_track)

        return optimized_tracks
    
    def bilinear_interpolation(self, feature_map, subpixel_loc):
        """
        Performs bilinear interpolation on a 4-pixel feature vector given a subpixel location.

        Args:
            feature_map: A numpy array of shape (4, feature_size) containing the feature vectors of the 4 surrounding pixels.
            subpixel_loc: A numpy array of shape (2,) containing the subpixel location (x, y) within the patch.

        Returns:
            A numpy array of shape (feature_size,) representing the interpolated feature vector at the subpixel location.
        """

        # Extract integer coordinates of the surrounding pixels
        x_floor = int(np.floor(subpixel_loc[0]))
        y_floor = int(np.floor(subpixel_loc[1]))

        # Handle edge cases (subpixel location at the border)
        x_floor = min(max(x_floor, 0), 1)  # Clamp x between 0 and 1 (inclusive)
        y_floor = min(max(y_floor, 0), 1)  # Clamp y between 0 and 1 (inclusive)

        # Calculate distance weights for bilinear interpolation
        dx = subpixel_loc[0] - x_floor
        dy = subpixel_loc[1] - y_floor

        # Perform bilinear interpolation
        top = (1 - dx) * feature_map[y_floor] + dx * feature_map[y_floor + 1]
        bottom = (1 - dx) * feature_map[y_floor + 2] + dx * feature_map[y_floor + 3]
        interpolated_feature = (1 - dy) * top + dy * bottom

        return interpolated_feature


