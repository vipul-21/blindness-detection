import torch
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
import pandas as pd
from PIL import Image
import os
import numpy as np

from config import *


class BD_Dataset(Dataset):
    def __init__(self, csv_file=TRAIN_CSV, data_dir=TRAIN_DIR, transform=None):
        self.data_dir = data_dir
        self.transform = transform
        self.class_labels = ["No DR", "Mild", "Moderate", "Severe", "Proliferative DR"]
        self.data = pd.read_csv(TRAIN_CSV)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        id, label = self.data.iloc[idx]
        im = Image.open(os.path.join(self.data_dir, id + ".png"))
        sample = {"image": im, "label": label}
        if self.transform:
            sample["image"] = self.transform(sample["image"])
        return sample


class Rescale(object):
    def __init__(self, output_size):
        assert isinstance(output_size, (int, tuple))
        if isinstance(output_size, int):
            self.output_size = (output_size, output_size)
        else:
            assert len(output_size) == 2
            self.output_size = output_size

    def __call__(self, sample):
        image, label = sample["image"], sample["label"]

        new_h, new_w = self.output_size
        new_h, new_w = int(new_h), int(new_w)
        img = image.resize((new_w, new_h))
        return {"image": img, "label": label}


class RandomCrop(object):
    def __init__(self, output_size):
        assert isinstance(output_size, (int, tuple))
        if isinstance(output_size, int):
            self.output_size = (output_size, output_size)
        else:
            assert len(output_size) == 2
            self.output_size = output_size

    def __call__(self, sample):
        image, label = sample["image"], sample["label"]

        h, w = image.size[:2]
        new_h, new_w = self.output_size

        top = np.random.randint(0, h - new_h)
        left = np.random.randint(0, w - new_w)

        image = image.crop((left, top, left + new_w, top + new_h))

        return {"image": image, "label": label}


class ToTensor(object):
    def __call__(self, sample):
        image, label = sample["image"], sample["label"]

        # swap color axis because
        # numpy image: H x W x C
        # torch image: C X H X W
        image = np.array(image).transpose((2, 0, 1))
        return {"image": torch.from_numpy(image), "label": torch.tensor(label)}


if __name__ == "__main__":
    # visualize data
    import matplotlib.pyplot as plt

    data_transform = transforms.Compose(
        [
            transforms.Resize(512),
            transforms.RandomResizedCrop(256),
            transforms.RandomHorizontalFlip(),
        ]
    )
    dataset = BD_Dataset(transform=data_transform)
    plt.figure()
    for i in range(4):
        sample = dataset[i]
        ax = plt.subplot(2, 2, i + 1)
        ax.imshow(sample["image"])
    plt.tight_layout()
    plt.show()

