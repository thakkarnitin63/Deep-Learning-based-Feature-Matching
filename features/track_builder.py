import h5py
import networkx as nx
import numpy as np
import matplotlib.pyplot as plt


class TrackBuilder:
    def __init__(self, descriptor_type="AKAZE"):
        self.graph = nx.Graph()
        self.descriptor_type = descriptor_type

    # def add_keypoints_and_descriptors(self, image_id, keypoints, descriptors):
    #     """
    #     Adds keypoints and their descriptors to the graph.
    #     """
    #     for idx, (kp, desc) in enumerate(zip(keypoints, descriptors)):
    #         node_id = (image_id, idx)
    #         # Adding kp_index explicitly
    #         self.graph.add_node(
    #             node_id, image=image_id, kp_index=idx, position=kp.pt, descriptor=desc
    #         )

    def add_keypoints_and_descriptors(
        self, image_id, keypoints, descriptors, is_colmap_keypoints=False
    ):
        """
        Adds keypoints and their descriptors to the graph.
        """
        for idx, (kp, desc) in enumerate(zip(keypoints, descriptors)):
            kp_index = idx
            if is_colmap_keypoints:
                # Extract the COLMAP keypoint index
                kp_position = (kp[0], kp[1])  # Extract the COLMAP keypoint position
            else:
                kp_position = kp.pt  # Use the pt attribute for OpenCV keypoints

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

    # def extract_tracks(self):
    #     """
    #     Extracts tracks as connected components, ensuring no track has multiple keypoints from the same image.
    #     This method merges tracks based on the highest similarity of connecting edges, ensuring each image is represented only once per track.
    #     """
    #     # Initialize each node to its own track
    #     node_to_track = {node: {node[0]: node} for node in self.graph.nodes()}

    #     # Sort edges by descending weight (similarity) and attempt to merge tracks
    #     for u, v, data in sorted(
    #         self.graph.edges(data=True), key=lambda x: -x[2]["weight"]
    #     ):
    #         track_u = node_to_track[u]
    #         track_v = node_to_track[v]
    #         if track_u != track_v:
    #             # Check if merging the tracks results in any image being represented more than once
    #             if not any(k in track_v for k in track_u):
    #                 # Merge the tracks
    #                 for node in track_v:
    #                     track_u[node[0]] = node
    #                 # Update node_to_track pointers for all nodes in v's track
    #                 for node in track_v:
    #                     node_to_track[node] = track_u

    #     # Extract unique tracks from node_to_track mapping
    #     # Extract unique tracks from node_to_track mapping
    #     unique_tracks = set(tuple(sorted(track.items())) for track in node_to_track.values())
    #     tracks = [list(dict(track).values()) for track in unique_tracks]
    #     return tracks

    def extract_tracks(self):
        """
        Extracts tracks as connected components, ensuring no track has multiple keypoints from the same image.

        Returns:
            List of tracks, each track containing node identifiers.
        """
        # Initialize each node to its own track
        node_to_track = {node: {node[0]: node} for node in self.graph.nodes()}

        # Merge tracks based on the highest similarity of connecting edges
        for u, v, data in sorted(
            self.graph.edges(data=True), key=lambda x: -x[2]["weight"]
        ):
            if node_to_track[u] != node_to_track[v]:
                if not (set(node_to_track[u].keys()) & set(node_to_track[v].keys())):
                    # Combine tracks
                    for node in node_to_track[v]:
                        node_to_track[u][node] = node_to_track[v][node]
                    # Update all nodes in track v to point to track u
                    for node in node_to_track[v]:
                        node_to_track[node] = node_to_track[u]

        # Gather tracks from node_to_track mapping, ensuring unique tracks
        seen_tracks = set()
        tracks = []
        for track in node_to_track.values():
            # Convert dictionary to list of nodes
            track_id = tuple(sorted(track.values()))
            if track_id not in seen_tracks:
                seen_tracks.add(track_id)
                tracks.append(list(track.values()))

        return tracks

    def save_tracks_to_hdf5(self, tracks, filename="tracks.hdf5"):
        """
        Saves tracks to an HDF5 file using a compound data type.
        :param tracks: List of tracks.
        :param filename: Name of the HDF5 file to save the tracks.
        """
        # Define the data type for HDF5 storage
        dt = np.dtype(
            [
                ("image_id", h5py.string_dtype(encoding="utf-8")),
                ("kp_index", np.int32),
                ("position_x", np.float32),
                ("position_y", np.float32),
                (
                    "descriptor",
                    h5py.special_dtype(vlen=np.float32),
                ),  # Handling variable-length float data
            ]
        )

        with h5py.File(filename, "w") as f:
            for i, track in enumerate(tracks):
                grp = f.create_group(f"track_{i}")
                data = []
                for node_id in track:
                    node_data = self.graph.nodes.get(node_id)
                    if (
                        node_data
                        and "position" in node_data
                        and "descriptor" in node_data
                    ):
                        data.append(
                            (
                                node_data["image"],
                                node_data["kp_index"],
                                node_data["position"][0],  # position_x
                                node_data["position"][1],  # position_y
                                np.array(
                                    node_data["descriptor"], dtype=np.float32
                                ),  # Ensure correct dtype
                            )
                        )
                if data:
                    ds = grp.create_dataset("keypoints", data=np.array(data, dtype=dt))

        print(f"Tracks saved to {filename}")

    @classmethod
    def load_tracks_from_hdf5(self, filename):
        tracks = []
        with h5py.File(filename, "r") as f:
            # Iterate over each track group in the HDF5 file
            for track_key in f.keys():
                track_group = f[track_key]
                # Load the dataset within this group
                keypoints_dataset = track_group["keypoints"]
                # Convert the dataset to a numpy array
                keypoints = keypoints_dataset[:]
                # Extract track information
                track_info = [
                    {
                        "image_id": kp["image_id"].decode(
                            "utf-8"
                        ),  # Decode from bytes to string
                        "kp_index": int(kp["kp_index"]),
                        "position": (float(kp["position_x"]), float(kp["position_y"])),
                        "descriptor": np.array(
                            kp["descriptor"]
                        ),  # Ensure descriptor is a numpy array
                    }
                    for kp in keypoints
                ]
                tracks.append(track_info)
        return tracks

    def visualize_track(self, track):
        """
        Visualizes a single track for debugging and inspection.
        """
        g = nx.Graph()
        for i in range(len(track) - 1):
            u, v = track[i], track[i + 1]
            if self.graph.has_edge(u, v):
                weight = self.graph[u][v]["weight"]
                g.add_edge(u, v, weight=weight)

        pos = nx.spring_layout(g)
        nx.draw(g, pos, with_labels=True, node_size=50, font_size=8)
        edge_labels = nx.get_edge_attributes(g, "weight")
        nx.draw_networkx_edge_labels(g, pos, edge_labels=edge_labels)
        plt.show()
