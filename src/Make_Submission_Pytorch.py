import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm
import torch
from torch.utils.data import DataLoader

from data_aug import *
from utils import getDevice

DATA_DIR = "../input/aptos2019-blindness-detection/" 
TEST_DIR = DATA_DIR + "test_images/"
TEST_CSV = DATA_DIR + "test.csv"

device = getDevice()
print("Using ", device)

data_transform = transforms.Compose([
    transforms.RandomResizedCrop(224),
    # transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])


# test
dataset = BD_Dataset(TEST_CSV, TEST_DIR, data_transform, True)
dataloader = DataLoader(dataset, batch_size=10, shuffle=False, num_workers=4)
# Recreate the exact same model, including weights and optimizer.
model = torch.load("finetuned_resnet50.pt")
model.eval()

with torch.no_grad():
    predictions = []
    for inputs, _ in dataloader:
        inputs = inputs.to(device)
        outputs = model(inputs)
        _, preds = torch.max(outputs, 1)
        predictions += preds.tolist()

# print("Predictions ", len(predictions), " ", predictions[:5])

# create submission
test_data = pd.read_csv(TEST_CSV)
with open("submissions.csv", "w") as f:
    f.write("id_code,diagnosis\n")
    for i in range(len(predictions)):
        f.write(test_data.id_code[i] + "," +
                str(predictions[i]) + "\n")
