import torch
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
import pandas as pd
from PIL import Image
import os
import numpy as np
import cv2

from config import *


class BD_Dataset(Dataset):
    def __init__(self, csv_file=TRAIN_CSV, data_dir=TRAIN_DIR, transform=None, is_test=False, regression_data=False):
        self.data_dir = data_dir
        self.transform = transform
        self.is_test = is_test
        self.regression_data = regression_data
        self.class_labels = ["No DR", "Mild",
                             "Moderate", "Severe", "Proliferative DR"]
        self.data = pd.read_csv(csv_file)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        if not self.is_test:
            id, label = self.data.iloc[idx]
            if self.regression_data:
                label = torch.tensor(label)
        else:
            id,  = self.data.iloc[idx]
            label = ""

        im = Image.open(os.path.join(self.data_dir, id + ".png"))
        sample = (im, label)
        if self.transform:
            sample = (self.transform(sample[0]), sample[1])
        sample = (np.array(sample[0]), sample[1])
        return sample


def crop_image_from_gray(img, tol=7):
    if img.ndim == 2:
        mask = img > tol
        img = img[np.ix_(mask.any(1), mask.any(0))]
    elif img.ndim == 3:
        gray_img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        mask = gray_img > tol

        check_shape = img[:, :, 0][np.ix_(mask.any(1), mask.any(0))].shape[0]
        if check_shape != 0:
            img1 = img[:, :, 0][np.ix_(mask.any(1), mask.any(0))]
            img2 = img[:, :, 1][np.ix_(mask.any(1), mask.any(0))]
            img3 = img[:, :, 2][np.ix_(mask.any(1), mask.any(0))]
            img = np.stack([img1, img2, img3], axis=-1)
    return img


def load_ben_color(image, img_shape, sigmaX=10):
    # image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    # image = crop_image_from_gray(image)
    image = cv2.resize(image, img_shape)
    image = cv2.addWeighted(image, 4, cv2.GaussianBlur(
        image, (0, 0), sigmaX), -4, 128)

    return image


class CircleCrop(object):
    def __init__(self, img_shape, sigmaX=20):
        self.sigmaX = sigmaX
        self.img_shape = img_shape

    def __call__(self, img):
        img = np.asarray(img)  # cv2.cvtColor(np.array(img), cv2.COLOR_BGR2RGB)
        img = crop_image_from_gray(img)
        height, width, depth = img.shape
        largest_side = np.max((height, width))
        img = cv2.resize(img, (largest_side, largest_side))

        height, width, depth = img.shape

        x = int(width / 2)
        y = int(height / 2)
        r = np.amin((x, y))

        circle_img = np.zeros((height, width), np.uint8)
        cv2.circle(circle_img, (x, y), int(r), 1, thickness=-1)
        img = cv2.bitwise_and(img, img, mask=circle_img)
        img = load_ben_color(img, self.img_shape, self.sigmaX)

        return Image.fromarray(img)


if __name__ == "__main__":
    # visualize data
    import matplotlib.pyplot as plt

    circleCrop = CircleCrop((512, 512))
    data_transform = transforms.Compose(
        [
            circleCrop,
            transforms.RandomHorizontalFlip(),
        ]
    )
    dataset = BD_Dataset(transform=data_transform)
    plt.figure()
    for i in range(25):
        sample = dataset[i]
        ax = plt.subplot(5, 5, i + 1)
        ax.imshow(sample[0])
    plt.tight_layout()
    plt.show()
