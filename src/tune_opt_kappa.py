import torch

from config import *
from utils import *
from data_aug import *

device = getDevice()

model = torch.load("resnet18_2e_regression.pt")
model.eval()

img_shape = (256, 256)
transform = transforms.Compose([
    CircleCrop(img_shape),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])

dataset = BD_Dataset(transform=transform, regression_data=True)

dataloader = DataLoader(dataset, batch_size=64,
                        shuffle=False, num_workers=4)


predictions_all = []
labels_all = []
with torch.set_grad_enabled(False):
    for inputs, labels in dataloader:
        inputs = inputs.to(device, dtype=torch.float)
        preds = model(inputs)
        predictions_all.append(preds[:, 0].tolist())
        labels_all.append(labels.tolist())

print(len(predictions_all), len(labels_all))
optR = OptimizedRounder()
optR.fit(predictions_all, labels_all)
print(optR.coefficients())