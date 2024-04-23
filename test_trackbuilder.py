import cv2
import numpy as np
import os
import glob
from tqdm import tqdm
from matplotlib import pyplot as plt
import networkx as nx
from features.image_processor import ImageProcessor, FeatureDescriptors, Matcher
from features.track_builder import TrackBuilder


def main(folder_path, visualize_tracks=True, max_images=None, verbose=True):
    # Initialize the ImageProcessor
    camera_matrix = np.array(
        [[1000, 0, 320], [0, 1000, 240], [0, 0, 1]]
    )  # Example camera matrix
    processor = ImageProcessor(
        folder_path,
        camera_matrix,
        descriptor_type=FeatureDescriptors.SIFT,
        matcher_type=Matcher.BRUTE_FORCE,
    )

    if verbose:
        print(f"Running Feature Detection and Matches")
    processor.process_all_images(
        visualize=False
    )  # Set visualize to True to see matches

    # Initialize the TrackBuilder
    if verbose:
        print(f"Initialising TrackBuilder")
    track_builder = TrackBuilder()

    # Add keypoints and descriptors to the TrackBuilder
    if verbose:
        print(f"Adding keypoints and descriptors to TrackBuilder")
    for image_id, descriptors in processor.descriptors.items():
        keypoints = processor.keypoints[image_id]
        track_builder.add_keypoints_and_descriptors(image_id, keypoints, descriptors)

    # Iterate over all pairs of images to find matches and build tracks
    if verbose:
        print(f"Building tracks for all images")
    image_ids = list(processor.images.keys())
    if max_images is not None:
        image_ids = image_ids[:max_images]

    for i in range(len(image_ids)):
        for j in range(i + 1, len(image_ids)):
            img_id1 = image_ids[i]
            img_id2 = image_ids[j]
            if (img_id1, img_id2) in processor.matches:
                matches = processor.matches[(img_id1, img_id2)]
                kp_pairs = [(m.queryIdx, m.trainIdx) for m in matches]
                descriptors1 = processor.descriptors[img_id1]
                descriptors2 = processor.descriptors[img_id2]
                track_builder.add_matches(
                    img_id1, img_id2, kp_pairs, descriptors1, descriptors2
                )

    # Extract and save tracks
    if verbose:
        print(f"Save tracks to file")
    tracks = track_builder.extract_tracks()
    if verbose:
        print(f"Extraced tracks list :: {len(tracks)}")
    # track_builder.save_tracks_to_hdf5(tracks, filename="features/data/tracks_compressed_buddha.hdf5")

    if verbose:
        print(f"Visualising tracks")
    # Visualize a few tracks if requested
    if visualize_tracks:
        # Choose a few tracks to visualize
        for track in tracks[:5]:  # Visualize first 5 tracks
            track_builder.visualize_track(track)


if __name__ == "__main__":
    path_to_images = "dataset/Compressed_buddha"
    main(path_to_images, visualize_tracks=True)
