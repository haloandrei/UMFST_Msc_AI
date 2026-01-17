import os
import random
from PIL import Image
import numpy as np
from torch.utils.data import Dataset, DataLoader

from utils import normalize_image, to_tensor


IMG_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp"}


def is_image_file(name):
    _, ext = os.path.splitext(name.lower())
    return ext in IMG_EXTENSIONS


def discover_samples(root):
    class_names = [d for d in sorted(os.listdir(root)) if os.path.isdir(os.path.join(root, d))]
    class_to_idx = {name: idx for idx, name in enumerate(class_names)}
    samples = []

    for class_name in class_names:
        class_dir = os.path.join(root, class_name)
        for fname in sorted(os.listdir(class_dir)):
            if is_image_file(fname):
                path = os.path.join(class_dir, fname)
                samples.append((path, class_to_idx[class_name]))

    return samples, class_names


class FruitDataset(Dataset):
    def __init__(
        self,
        root,
        image_size,
        max_samples,
        seed,
    ):
        self.samples, self.class_names = discover_samples(root)

        if max_samples and max_samples < len(self.samples):
            rng = random.Random(seed)
            self.samples = rng.sample(self.samples, max_samples)

        self.image_size = image_size

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        image = Image.open(path).convert("RGB")
        image = image.resize((self.image_size, self.image_size))
        arr = np.asarray(image, dtype=np.float32) / 255.0
        arr = normalize_image(arr)
        tensor = to_tensor(arr)
        return tensor, label


def build_loader(
    data_root,
    image_size,
    batch_size,
    num_workers,
    max_samples,
    seed,
    shuffle,
):
    dataset = FruitDataset(
        root=data_root,
        image_size=image_size,
        max_samples=max_samples,
        seed=seed,
    )
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True,
    )
