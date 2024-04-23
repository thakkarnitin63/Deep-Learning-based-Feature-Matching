from .colmap_database import COLMAPDatabase, blob_to_array, array_to_blob, pair_id_to_image_ids, image_ids_to_pair_id
import numpy as np

class COLMAPExtractor:
    def __init__(self, database_path):
        self.db = COLMAPDatabase.connect(database_path)

    def get_cameras(self):
        cameras = {}
        query = "SELECT * FROM cameras"
        for camera_id, model, width, height, params, prior in self.db.execute(query):
            cameras[camera_id] = {
                "model": model,
                "width": width,
                "height": height,
                "params": blob_to_array(params, np.float64),
                "prior_focal_length": prior,
            }
        return cameras

    def get_images(self):
        images = {}
        query = "SELECT * FROM images"
        for image_id, name, camera_id, prior_qw, prior_qx, prior_qy, prior_qz, prior_tx, prior_ty, prior_tz in self.db.execute(query):
            images[image_id] = {
                "name": name.split(".")[0],
                "camera_id": camera_id,
                "prior_q": np.array([prior_qw, prior_qx, prior_qy, prior_qz]),
                "prior_t": np.array([prior_tx, prior_ty, prior_tz]),
            }
        return images

    def get_keypoints(self):
        keypoints = {}
        query = "SELECT image_id, data FROM keypoints"
        for image_id, data in self.db.execute(query):
            keypoints[image_id] = blob_to_array(data, np.float32, (-1, 2))
        return keypoints

    def get_descriptors(self):
        descriptors = {}
        query = "SELECT image_id, data FROM descriptors"
        for image_id, data in self.db.execute(query):
            descriptors[image_id] = blob_to_array(data, np.uint8)
        return descriptors

    def get_matches(self):
        matches = {}
        query = "SELECT pair_id, data FROM matches"
        for pair_id, data in self.db.execute(query):
            if data is not None:  # Check for NoneType
                image_id1, image_id2 = pair_id_to_image_ids(pair_id)
                pair = (image_id1, image_id2)
                matches[pair] = blob_to_array(data, np.uint32, (-1, 2))
            # else:
            #     # print("Warning: Data is None for pair_id:", pair)
                
        return matches



    def get_two_view_geometries(self):
        two_view_geometries = {}
        query = "SELECT pair_id, F, E, H, qvec, tvec FROM two_view_geometries"
        for pair_id, F, E, H, qvec, tvec in self.db.execute(query):
            if F is not None and E is not None and H is not None and qvec is not None and tvec is not None:  # Check for NoneType
                image_id1, image_id2 = pair_id_to_image_ids(pair_id)
                two_view_geometries[(image_id1, image_id2)] = {
                    "F": blob_to_array(F, np.float64),
                    "E": blob_to_array(E, np.float64),
                    "H": blob_to_array(H, np.float64),
                    "qvec": blob_to_array(qvec, np.float64),
                    "tvec": blob_to_array(tvec, np.float64),
                }
            # else:
            #     print("Warning: Data is None for pair_id:", pair_id)
        return two_view_geometries

    def close(self):
        self.db.close()



