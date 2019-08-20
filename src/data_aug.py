import torch
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
import pandas as pd
from PIL import Image
import os
import numpy as np

from config import *


class BD_Dataset(Dataset):
    def __init__(self, csv_file=TRAIN_CSV, data_dir=TRAIN_DIR, transform=None, is_test=False):
        self.data_dir = data_dir
        self.transform = transform
        self.is_test = is_test
        self.class_labels = ["No DR", "Mild",
                             "Moderate", "Severe", "Proliferative DR"]
        self.data = pd.read_csv(csv_file)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        if not self.is_test:
            id, label = self.data.iloc[idx]
        else:
            id,  = self.data.iloc[idx]
            label = ""

        im = Image.open(os.path.join(self.data_dir, id + ".png"))
        sample = (im, label)
        if self.transform:
            sample = (self.transform(sample[0]), sample[1])
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
        return (img, label)


class RandomCrop(object):
    def __init__(self, output_size):
        assert isinstance(output_size, (int, tuple))
        if isinstance(output_size, int):
            self.output_size = (output_size, output_size)
        else:
            assert len(output_size) == 2
            self.output_size = output_size

    def __call__(self, sample):
        image, label = sample

        h, w = image.size[:2]
        new_h, new_w = self.output_size

        top = np.random.randint(0, h - new_h)
        left = np.random.randint(0, w - new_w)

        image = image.crop((left, top, left + new_w, top + new_h))

        return (image, label)


class ToTensor(object):
    def __call__(self, sample):
        image, label = sample

        # swap color axis because
        # numpy image: H x W x C
        # torch image: C X H X W
        image = np.array(image).transpose((2, 0, 1))
        return (torch.from_numpy(image), torch.tensor(label))


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
