import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision
from torchvision import transforms
import matplotlib.pyplot as plt
from sklearn.metrics import cohen_kappa_score

from data_aug import *
from config import *


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16*61*61, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16*61*61)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


def imshow(inp, title=None):
    """Imshow for Tensor."""
    inp = inp.numpy().transpose((1, 2, 0))
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    inp = std * inp + mean
    inp = np.clip(inp, 0, 1)
    plt.imshow(inp)
    if title is not None:
        plt.title(title)
    plt.pause(0.001)  # pause a bit so that plots are updated

def train(net, dataset_loader, optimizer, criterion, epochs):
    for epoch in range(epochs):
        print("Epoch {}/{}".format(epoch, epochs))
        running_loss = 0.0
        for i, data in enumerate(dataset_loader, 0):
            inputs, labels = data['image'].to(device), data['label'].to(device)
            optimizer.zero_grad()
            outputs = net(inputs)
            loss = criterion(outputs, labels) 
            loss.backward()
            optimizer.step()
 
            # print statistics
            running_loss += loss.item()
            n = 10
            if i%n == n-1:
                print('[%d, %5d] loss: %.3f' % (epoch+1, i+1, running_loss/n)) 
                running_loss = 0.0

def validate(net, validation_loader):
    correct = 0.
    total = 0.
    predictions = []
    true_labels = []
    with torch.no_grad():
        for data in validation_loader:
            inputs, labels = data['image'].to(device), data['label'].to(device)
            outputs = net(inputs)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            predictions += predicted.tolist()
            true_labels += labels.tolist()

        print("Accuracy: ", 100*correct/total)
        print("Kappa: ", cohen_kappa_score(predictions, true_labels))

if __name__ == "__main__":
    data_transform = transforms.Compose(
        [
            transforms.Resize(512),
            transforms.RandomResizedCrop(256),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )
    dataset = BD_Dataset(transform=data_transform)

    train_size = int((1-VALIDATION_FRACTION)*len(dataset))
    validation_size = len(dataset) - train_size
    train_dataset, validation_dataset = torch.utils.data.random_split(dataset, [train_size, validation_size])

    train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True, num_workers=4)
    validation_loader = DataLoader(validation_dataset, batch_size=4, shuffle=True, num_workers=4)

    # # Visualize data
    # plt.figure()
    # batch = next(iter(dataset_loader))
    # inputs, classes = batch["image"], batch["label"]
    # out = torchvision.utils.make_grid(inputs)
    # imshow(out)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print ("Using ", device)

    # Define net, loss function and learning algorithm
    net = Net()
    net.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)
    epochs = 5 

    print("Training({}):".format(train_size))
    train(net, train_loader, optimizer, criterion, epochs)
    torch.save(net, "basic_pytorch.pt")
    
if PREPARE_SUBMIT:
    print("Training on validation set as well({}):".format(validation_size))
    train(net, validation_loader, optimizer, criterion, epochs)
    torch.save(net, "basic_pytorch.pt")
    print("Finished Training")

print("Validating({}):".format(validation_size))
validate(net, validation_loader)