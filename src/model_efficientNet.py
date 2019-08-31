import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
import torchvision
from torchvision import transforms, models
import matplotlib.pyplot as plt
from sklearn.metrics import cohen_kappa_score
from efficientnet_pytorch import EfficientNet
import time
import copy

from data_aug import *
from config import *
import utils

device = utils.getDevice()

accs = []
kappas = []
losses = []


def get_actual_predictions(preds, coeff=[0.5, 1.5, 2.5, 3.5]):
    actual_preds = torch.zeros(preds.shape, device=device)
    for i, p in enumerate(preds):
        if p < coeff[0]:
            ap = 0
        elif p < coeff[1]:
            ap = 1
        elif p < coeff[2]:
            ap = 2
        elif p < coeff[3]:
            ap = 3
        else:
            ap = 4
        actual_preds[i] = torch.tensor(ap, device=device, dtype=torch.float)
    return actual_preds


def train_model(model, dataloaders, dataset_sizes, criterion, optimizer, scheduler, num_epochs=25):
    since = time.time()
    best_model_wts = copy.deepcopy(model.state_dict())
    best_kappa_score = -float("inf")
    for epoch in range(num_epochs):
        print("Epoch{}/{}".format(epoch, num_epochs-1))
        print("-"*10)
        for phase in ["train", "val"]:
            if phase == "train":
                scheduler.step()
                model.train()
            else:
                model.eval()

            running_loss = 0.
            running_corrects = 0.

            predictions_all = torch.empty(size=(0,), device=device)
            labels_all = torch.empty(size=(0,), device=device)
            curr_batch = -1
            for inputs, labels in dataloaders[phase]:
                curr_batch += 1
                if curr_batch % 10 == 0:
                    print("\tBatch: ", curr_batch)

                inputs = inputs.to(device, dtype=torch.float)
                labels = labels.to(device, dtype=torch.float)
                labels = labels.view(-1, 1)
                optimizer.zero_grad()
                with torch.set_grad_enabled(phase == "train"):
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)
                    if phase == "train":
                        loss.backward()
                        optimizer.step()

                    actual_preds = get_actual_predictions(outputs[:, 0])
                    predictions_all = torch.cat(
                        (predictions_all, actual_preds), 0)
                    labels_all = torch.cat((labels_all, labels[:, 0]), 0)

                batch_loss = loss.item() * inputs.size(0)
                running_loss += batch_loss
                batch_corrects = torch.sum(actual_preds == labels[:, 0])
                running_corrects += batch_corrects
                if curr_batch % 10 == 0:
                    print("\t\tBatch loss: ", batch_loss)
                    print("\t\tBatch corrects: ", batch_corrects)
            epoch_loss = running_loss/dataset_sizes[phase]
            epoch_acc = running_corrects.double()*100 / dataset_sizes[phase]
            epoch_kappa_score = cohen_kappa_score(
                predictions_all.tolist(), labels_all.tolist(), weights="quadratic")
            print("{} Loss: {:.4f} Acc: {:.4f} Kappa: {:.4f}".format(
                phase, epoch_loss, epoch_acc, epoch_kappa_score))

            losses.append(epoch_loss.item())
            accs.append(epoch_acc.item())
            kappas.append(epoch_kappa_score)

            if phase == "val" and epoch_kappa_score > best_kappa_score:
                best_kappa_score = epoch_kappa_score
                best_model_wts = copy.deepcopy(model.state_dict())
                # torch.save(model_ft, "temp.pt")
                torch.save(model, "finetuned_efficientNet_b2_30e_regression.pt")
        print()

    time_elapsed = time.time() - since
    print("Training copmlete in {:.0f}m{:.0f}s".format(
        time_elapsed//60, time_elapsed % 60))
    print("Best val Acc: {:4f}".format(best_acc))

    model.load_state_dict(best_model_wts)
    return model


if __name__ == "__main__":
    img_shape = (256, 256)
    transform = transforms.Compose([
        CircleCrop(img_shape),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])

    dataset = BD_Dataset(transform=transform, regression_data=True)
    # dataset = Subset(dataset, [i for i in range(200)])

    # Make Train - Val Split
    train_size = int((1-VALIDATION_FRACTION)*len(dataset))
    validation_size = len(dataset) - train_size
    dataset_sizes = {"train": train_size, "val": validation_size}
    train_dataset, validation_dataset = torch.utils.data.random_split(
        dataset, [train_size, validation_size])
    print(dataset_sizes)

    dataloaders = {
        "train": DataLoader(
            train_dataset, batch_size=20, shuffle=True, num_workers=4),
        "val": DataLoader(
            validation_dataset, batch_size=20, shuffle=True, num_workers=4)}

    # load pre trained model
    model_ft = EfficientNet.from_pretrained("efficientnet-b2")
    num_ftrs = model_ft._fc.in_features
    model_ft._fc = nn.Linear(num_ftrs, 1)
    model_ft = model_ft.to(device)

    optimizer_ft = optim.Adam(model_ft.parameters(), lr=0.001, weight_decay=1e-5)
    criterion = nn.MSELoss()
    exp_lr_scheduler = optim.lr_scheduler.StepLR(
        optimizer_ft, step_size=5, gamma=0.1)

    model_ft = train_model(model_ft, dataloaders, dataset_sizes, criterion,
                           optimizer_ft, exp_lr_scheduler, 30)

    # torch.save(model_ft, "temp.pt")
    torch.save(model_ft, "finetuned_efficientNet_b2_30e_regression.pt")
