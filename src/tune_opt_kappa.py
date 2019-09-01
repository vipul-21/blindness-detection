import torch

from config import *
from utils import *
from data_aug import *

device = getDevice()

model = torch.load("finetuned_efficienet_b2_50e_old_new_data.pt")
model.eval()

# img_shape = (256, 256)
transform = transforms.Compose([
    # CircleCrop(img_shape),
    # transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])

dataset = BD_Dataset(TRAIN_CSV, PREPROCESSED_DIR, transform, False, True, True, False)

dataloader = DataLoader(dataset, batch_size=20, shuffle=False, num_workers=4)


predictions_all = []
labels_all = []
with torch.set_grad_enabled(False):
    batch = 0
    for inputs, labels in dataloader:
        batch += 1
        print("Batch: ", batch)
        inputs = inputs.to(device, dtype=torch.float)
        preds = model(inputs)
        predictions_all += preds[:, 0].tolist()
        labels_all += labels.tolist()

print(len(predictions_all), len(labels_all))
optR = OptimizedRounder()

orig_preds = optR.predict(predictions_all, [0.5, 1.5, 2.5, 3.5])
print("Original Kappa Score: ", cohen_kappa_score(orig_preds, labels_all))

optR.fit(predictions_all, labels_all)
print(optR.coefficients())

new_preds = optR.predict(predictions_all, optR.coefficients())
print("New Kappa Score: ", cohen_kappa_score(new_preds, labels_all))