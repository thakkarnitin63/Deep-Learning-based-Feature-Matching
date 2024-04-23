from PIL import Image
import torchvision.transforms as transforms
import torch
from models.s2dnet import S2DNet
import os
import matplotlib.pyplot as plt


def load_image(image_path):
    transform = transforms.Compose(
        [
            transforms.ToTensor(),  # Converts to [0, 1] range
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],  # Standard ImageNet means
                std=[0.229, 0.224, 0.225],
            ),  # Standard ImageNet stds
        ]
    )
    image = Image.open(image_path).convert("RGB")
    return transform(image).unsqueeze(0)  # Add batch dimension


def extract_features(
    image_path, device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
):
    image_tensor = load_image(image_path)
    model = S2DNet(device=device, checkpoint_path="models/s2dnet_weights.pth")
    model.to(device)
    model.eval()  # Set the model to inference mode

    with torch.no_grad():
        feature_maps = model(image_tensor.to(device))
    return feature_maps


def list_image_paths(directory, extensions=[".jpg", ".jpeg", ".png", ".bmp", ".tiff"]):
    image_paths = []
    for root, dirs, files in os.walk(directory):
        for file in files:
            if any(file.lower().endswith(ext) for ext in extensions):
                image_paths.append(os.path.join(root, file))
    return image_paths


def visualize_feature_maps(feature_maps):
    for i, fmap in enumerate(feature_maps):
        plt.figure(figsize=(20, 10))
        for j in range(min(fmap.size(1), 10)):
            ax = plt.subplot(2, 5, j + 1)
            plt.imshow(fmap[0, j].cpu().detach().numpy(), cmap="gray")
            plt.title(f"Feature Map {j + 1}")
            plt.axis("off")
        plt.suptitle(f"Feature Maps from Layer {i + 1}")
        plt.show()


if __name__ == "__main__":
    image_paths = list_image_paths(
        "dataset/buddha"
    )  # Adjusted path to the Buddha dataset
    # print(f"Paths :: {image_paths}")
    feature_maps = extract_features(image_paths[0], device="cpu")

    print("Extracted feature maps have shapes:")
    for fm in feature_maps:
        print(fm.shape)

    visualize_feature_maps(feature_maps)
