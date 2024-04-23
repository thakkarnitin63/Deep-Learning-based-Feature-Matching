import torch
import h5py
from PIL import Image
import torchvision.transforms.functional as tvf
from models.s2dnet import S2DNet
import os


class FeatureMapExtractor:
    def __init__(
        self, model_path="models/s2dnet_weights.pth", device=None, output_dir="output"
    ):
        self.device = (
            device
            if device
            else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        )
        self.model = S2DNet(device=self.device, checkpoint_path=model_path)
        self.model.to(self.device)
        self.model.eval()  # Set the model to inference mode
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)

    @torch.no_grad()
    def _preprocess(self, image: Image.Image) -> torch.Tensor:
        """Converts a PIL Image to a normalized tensor."""
        tens = tvf.pil_to_tensor(image)
        return tens.float().div(255) if isinstance(tens, torch.ByteTensor) else tens

    def load_and_preprocess_image(self, image_path):
        try:
            image = Image.open(image_path).convert("RGB")
            image_tensor = self._preprocess(image)
            return image_tensor.unsqueeze(0).to(self.device)
        except Exception as e:
            print(f"Failed to load image {image_path}: {e}")
            return None

    def extract_features(self, image_path, save_to_hdf5=True):
        image_tensor = self.load_and_preprocess_image(image_path)
        with torch.no_grad():
            feature_maps = self.model(image_tensor)
        if save_to_hdf5:
            self.save_features_to_hdf5(image_path, feature_maps)
        return feature_maps

    def save_features_to_hdf5(self, image_path, feature_maps):
        file_name = os.path.basename(image_path).split(".")[0] + ".hdf5"
        hdf5_path = os.path.join(self.output_dir, file_name)
        with h5py.File(hdf5_path, "w") as f:
            # Convert tensor to numpy array and store in HDF5
            for i, fmap in enumerate(feature_maps):
                f.create_dataset(f"feature_map_{i}", data=fmap.cpu().numpy())

    def process_directory(
        self,
        directory,
        extensions=[".jpg", ".jpeg", ".png", ".bmp", ".tiff"],
        batch_size=4,
    ):
        image_paths = self.list_image_paths(directory, extensions)
        for i in range(0, len(image_paths), batch_size):
            batch_paths = image_paths[i : i + batch_size]
            batch_tensors = [
                self.load_and_preprocess_image(path)
                for path in batch_paths
                if path is not None
            ]
            if batch_tensors:
                # batch_tensors = torch.stack(
                #     batch_tensors
                # )  # Stack all image tensors to form a batch
                # print(f"Size of batch_tensors :: {batch_tensors.shape}")

                batch_tensors = torch.cat(
                    batch_tensors, dim=0
                )  # Concatenate along the batch dimension
                print(f"Size of batch_tensors :: {batch_tensors.shape}")

                with torch.no_grad():
                    batch_features = self.model(batch_tensors)
                # Assuming saving or further processing per batch
                for path, features in zip(batch_paths, batch_features):
                    self.save_features_to_hdf5(path, features)

    @staticmethod
    def list_image_paths(directory, extensions):
        image_paths = []
        for root, _, files in os.walk(directory):
            for file in files:
                if any(file.lower().endswith(ext) for ext in extensions):
                    image_paths.append(os.path.join(root, file))
        image_paths.sort()
        return image_paths
