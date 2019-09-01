import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, ConcatDataset, Subset
import torchvision
from torchvision import transforms, models
import matplotlib.pyplot as plt
from sklearn.metrics import cohen_kappa_score
from efficientnet_pytorch import EfficientNet
import time
import copy

from data_aug import *
from config import *
from utils import *

device = getDevice()


if __name__ == "__main__":
    img_shape = (256, 256)
    transform = transforms.Compose([
        # CircleCrop(img_shape),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation((0, 360)),
        transforms.ToTensor(),
        # transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])

    dataset1 = BD_Dataset(OLD_TRAIN_CSV, OLD_PREPROCESSED_DIR,
                          transform, False, True, False, False)
    dataset2 = BD_Dataset(OLD_TEST_CSV, OLD_PREPROCESSED_DIR,
                          transform, False, True, False, False)
    dataset2.data = dataset2.data.drop(["Usage"], axis=1)
    train_dataset = ConcatDataset([dataset1, dataset2])
    # train_dataset = Subset(train_dataset, [x for x in range(100)])

    validation_dataset = BD_Dataset(
        TRAIN_CSV, PREPROCESSED_DIR, transform, False, True, True, False)
    # validation_dataset = Subset(validation_dataset, [x for x in range(50)])

    dataset_sizes = {"train": len(train_dataset),
                     "val": len(validation_dataset)}
    print(dataset_sizes)

    bs = 20
    dataloaders = {
        "train": DataLoader(
            train_dataset, batch_size=bs, shuffle=True, num_workers=16),
        "val": DataLoader(
            validation_dataset, batch_size=bs, shuffle=True, num_workers=16)}

    # load pre trained model
    # Note that we are training the entire model first.
    model_ft = torch.load("finetuned_efficienet_b2_20e_old_new_data.pt")
    model_ft = EfficientNet.from_pretrained("efficientnet-b2")
    num_ftrs = model_ft._fc.in_features
    model_ft._fc = nn.Linear(num_ftrs, 1)
    model_ft = model_ft.to(device)

    optimizer_ft = optim.Adam(model_ft.parameters(),
                              lr=0.001, weight_decay=1e-5)
    criterion = nn.MSELoss()
    exp_lr_scheduler = optim.lr_scheduler.StepLR(
        optimizer_ft, step_size=5, gamma=0.1)

    # model_ft = train_model(model_ft, dataloaders, dataset_sizes, criterion,
    #                        optimizer_ft, exp_lr_scheduler, 1, "temp.pt")
    model_ft = train_model(model_ft, dataloaders, dataset_sizes, criterion,
                           optimizer_ft, exp_lr_scheduler, 20, "efficientnet_b2_25e_old_data.pt")

    # It's time to fine tune now to our validation dataset
    print("-------------Finetuning on 2019 dataset now-----------")
    tDataset = validation_dataset
    vDataset = Subset(validation_dataset, [x for x in range(30)])
    dataset_sizes = {"train": len(tDataset),
                     "val": len(vDataset)}
    print(dataset_sizes)

    dataloaders = {"train": DataLoader(tDataset, batch_size=bs, shuffle=True, num_workers=4), "val": DataLoader(
        vDataset, batch_size=bs, shuffle=True, num_workers=4)}

    for parameters in model_ft.parameters():
        parameters.requires_grad = False

    for parameters in model_ft._fc.parameters():
        parameters.requires_grad = True

    model_ft = train_model(model_ft, dataloaders, dataset_sizes, criterion, optimizer_ft,
                           exp_lr_scheduler, 50, "finetuned_efficienet_b2_50e_old_new_data.pt")
    # model_ft = train_model(model_ft, dataloaders, dataset_sizes, criterion,
    #                        optimizer_ft, exp_lr_scheduler, 1, "temp1.pt")
