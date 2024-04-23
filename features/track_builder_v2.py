import h5py
import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
import json
import pickle


class TrackBuilder:
    def __init__(self, descriptor_type="SIFT"):
        self.graph = nx.Graph()
        self.descriptor_type = descriptor_type

    def add_keypoints_and_descriptors(
        self, image_id, keypoints, descriptors, is_colmap_keypoints=False
    ):
        """
        Adds keypoints and their descriptors to the graph.
        """
        for idx, (kp, desc) in enumerate(zip(keypoints, descriptors)):
            kp_index = idx
            if is_colmap_keypoints:
                # Extract the COLMAP keypoint position (x, y) and index
                kp_position = (kp[0], kp[1])
            else:
                # Use pt attribute for OpenCV keypoints
                kp_position = kp.pt

            node_id = (image_id, kp_index)
            self.graph.add_node(
                node_id,
                image=image_id,
                kp_index=kp_index,
                position=kp_position,
                descriptor=desc,
            )

    def add_matches(self, image_id1, image_id2, kp_pairs, descriptors1, descriptors2):
        """
        Adds matches as edges between keypoints from two images with weights based on descriptor similarity.
        """
        for index1, index2 in kp_pairs:
            node1 = (image_id1, index1)
            node2 = (image_id2, index2)
            if self.graph.has_node(node1) and self.graph.has_node(node2):
                if self.descriptor_type == "SIFT":
                    # Cosine similarity for SIFT descriptors
                    desc1 = descriptors1[index1]
                    desc2 = descriptors2[index2]
                    similarity = np.dot(desc1, desc2) / (
                        np.linalg.norm(desc1) * np.linalg.norm(desc2)
                    )
                elif self.descriptor_type == "AKAZE":
                    # Hamming distance for AKAZE descriptors
                    desc1 = np.unpackbits(descriptors1[index1]).astype(np.int32)
                    desc2 = np.unpackbits(descriptors2[index2]).astype(np.int32)
                    hamming_distance = np.sum(desc1 != desc2)
                    similarity = 1 / (1 + hamming_distance)  # Convert to similarity
                self.graph.add_edge(node1, node2, weight=similarity)

    def extract_tracks(self):
        """
        Extracts tracks using a greedy merging strategy followed by an optional recursive graph cut.

        Returns:
            List of tracks, each track containing node identifiers.
        """

        # Initialize each node to its own track
        node_to_track = {node: {node[0]: node} for node in self.graph.nodes()}

        # Sort edges by descending weight (similarity)
        sorted_edges = sorted(
            self.graph.edges(data=True), key=lambda x: -x[2]["weight"]
        )

        # Greedy merging based on highest similarity edges
        for u, v, data in sorted_edges:
            track_u = node_to_track[u]
            track_v = node_to_track[v]
            if track_u != track_v and not any(k in track_v for k in track_u):
                # Merge tracks if they don't share images and avoid duplicate image IDs
                for node in track_v:
                    track_u[node] = track_v[node]
                # Update node_to_track pointers for all nodes in track v
                for node in track_v:
                    node_to_track[node] = track_u

        # Extract unique tracks from node_to_track mapping
        tracks = []
        seen_tracks = set()
        for track in node_to_track.values():
            # Convert dictionary to list of nodes and ensure unique tracks
            track_id = tuple(sorted(track.values()))
            if track_id not in seen_tracks:
                seen_tracks.add(track_id)
                tracks.append(list(track.values()))

        return tracks

    def visualize_tracks(self, tracks, image_data=None):
        """
        Visualizes tracks on top of images (if provided).

        Args:
            tracks: List of tracks, each track containing a list of node identifiers.
            image_data: Optional dictionary mapping image IDs to image data (e.g., NumPy arrays).
        """

        plt.figure(figsize=(10, 6))

        if image_data is not None:
            for image_id, image in image_data.items():
                plt.imshow(image, cmap="gray")
                break  # Only show the first image (optional)

        # Extract node positions from the graph
        node_positions = {
            node: self.graph.nodes[node]["position"] for node in self.graph.nodes()
        }

        for track in tracks:
            track_nodes = [node_positions[node] for node in track]
            plt.plot(
                [p[0] for p in track_nodes],
                [p[1] for p in track_nodes],
                color="red",
                linewidth=2,
            )

        plt.title("Visualized Tracks")
        plt.xlabel("X")
        plt.ylabel("Y")
        plt.show()

    @staticmethod
    def save(tracks, filename, format="pickle"):
        """
        Saves tracks to a file in the specified format.

        Args:
            tracks: List of tracks, each track containing a list of node identifiers.
            filename: Path to the output file.
            format: Output format (e.g., "json", "pickle"). Defaults to "json".
        """

        if format == "json":
            with open(filename, "w") as f:
                json.dump(tracks, f, indent=4)
        elif format == "pickle":
            with open(filename, "wb") as f:
                pickle.dump(tracks, f)
        else:
            raise ValueError(f"Unsupported format: {format}")

        print(f"Tracks saved to {filename} in {format.upper()} format.")

    @staticmethod
    def load(filename, format="pickle"):
        """
        Loads tracks from a file in the specified format.

        Args:
            filename: Path to the input file.
            format: Input format (e.g., "json", "pickle"). Defaults to "json".

        Returns:
            List of tracks, each track containing a list of node identifiers.
        """

        if format == "json":
            with open(filename, "r") as f:
                return json.load(f)
        elif format == "pickle":
            with open(filename, "rb") as f:
                return pickle.load(f)
        else:
            raise ValueError(f"Unsupported format: {format}")
