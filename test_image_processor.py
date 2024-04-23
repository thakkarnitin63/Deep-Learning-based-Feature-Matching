from features.image_processor import ImageProcessor
import numpy as np


def main():
    folder_path = "dataset/Compressed_buddha"
    camera_matrix = np.array(
        [[1000, 0, 320], [0, 1000, 240], [0, 0, 1]]
    )  # Example camera matrix
    processor = ImageProcessor(folder_path, camera_matrix)
    processor.process_all_images(visualize=False)
    print(f"Image ids :: {len(processor.keypoints['buddha_001-min'])}")


if __name__ == "__main__":
    main()
