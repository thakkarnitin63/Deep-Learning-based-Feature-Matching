import cv2
import matplotlib.pyplot as plt
from features.feature_patch_extractor import FeaturePatchExtractor

# Assuming you have some image ID and keypoints
image_id = "buddha_001-min"
keypoints = [cv2.KeyPoint(x=50, y=50, size=10)]  # Example keypoint

# Initialize the FeaturePatchExtractor
feature_map_directory = "output"
patch_extractor = FeaturePatchExtractor(feature_map_directory)

# Get patches for the image and keypoints
patches = patch_extractor.get_patches_for_image(image_id, keypoints)
print(f"SHape of patches: {patches[0].shape}")


# Plotting the first 8 channels of the first patch
num_channels_to_plot = 8
fig, axs = plt.subplots(2, 4, figsize=(24, 24))  # Fixed figure size

for i in range(2):
    for j in range(4):
        axs[i, j].imshow(patches[0][i * 4 + j], cmap="gray")
        axs[i, j].set_title(f"Channel {i * 4 + j + 1}")
plt.show()
