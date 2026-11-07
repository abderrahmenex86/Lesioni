import torch
import numpy as np
from pathlib import Path
from PIL import Image
from tqdm import tqdm


def check_pixel_distribution(dataset_path="dataset/plantseg/annotations/train", num_classes=116):
    mask_paths = list(Path(dataset_path).glob("*.png"))
    pixel_counts = torch.zeros(num_classes, dtype=torch.long)

    for path in tqdm(mask_paths, desc="Scanning Dataset"):
        mask = torch.from_numpy(np.array(Image.open(path).convert("L")))
        unique, counts = torch.unique(mask, return_counts=True)
        for u, c in zip(unique, counts):
            if u < num_classes:
                pixel_counts[u] += c

    total_pixels = pixel_counts.sum()
    percentages = (pixel_counts / total_pixels) * 100

    print("\nClass Pixel Distribution:")
    for i, p in enumerate(percentages):
        if p > 0:
            print(f"Class {i}: {p:.4f}%")


if __name__ == "__main__":
    check_pixel_distribution()
