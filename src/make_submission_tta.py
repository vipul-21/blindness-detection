import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm
import torch
from torch.utils.data import DataLoader, Subset

from data_aug import *
from utils import *
from config import *

device = getDevice()
print("Using ", device)

img_shape = (256, 256)
data_transform = transforms.Compose([
    CircleCrop(img_shape),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation((0, 360)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])


# test
dataset = BD_Dataset(TRAIN_CSV, TRAIN_DIR, data_transform, False, True)
# dataset = Subset(dataset, [i for i in range(500)])
dataloader = DataLoader(dataset, batch_size=50, shuffle=False, num_workers=4)

model = torch.load("finetuned_resnet50_30e_regression.pt")
model.eval()

optR = OptimizedRounder()
coeff = [1.12913722, 1.39349474, 2.03201554, 4.59752889]

tta = 2
with torch.no_grad():
    outputs_all = torch.empty(size=(0, len(dataset)), device=device)
    labels_all = []
    for i in range(tta):
        print("tta: ", i)
        outputs_curr = torch.empty(size=(0,), device=device)
        batch = 0
        for inputs, labels in dataloader:
            if i == 0:
                labels_all += labels.tolist()
            batch += 1
            print("batch: ", batch)
            inputs = inputs.to(device)
            outputs = model(inputs)
            outputs_curr = torch.cat((outputs_curr, outputs[:, 0]), 0)
        outputs_all = torch.cat((outputs_all, outputs_curr.view(1, -1)), 0)
    raw_predictions = torch.mean(outputs_all, 0)
    predictions = optR.predict(raw_predictions.tolist(), coeff)
    
# labels = []
# for i in range(len(dataset)):
#     labels.append(dataset[i][1])

print("Predictions ", len(predictions), " ", predictions[:5])
print("Kappa score: ", cohen_kappa_score(predictions, labels_all, weights="quadratic"))

# # create submission
# test_data = pd.read_csv(TEST_CSV)
# with open("submission.csv", "w") as f:
#     f.write("id_code,diagnosis\n")
#     for i in range(len(predictions)):
#         f.write(test_data.id_code[i] + "," +
#                 str(int(predictions[i])) + "\n")