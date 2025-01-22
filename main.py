# Import standard libraries for deep learning (torch), image processing (torchvision), and medical datasets (medmnist).
from tqdm import tqdm
import numpy as np
import torch
import torchvision
import matplotlib.pyplot as plt    
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
import torchvision.transforms as transforms
from torchvision.models import resnet18
import math

import medmnist
# INFO: Metadata about the MedMNIST dataset, e.g., task type and number of classes.
# Evaluator: Calculates metrics like AUC and accuracy.
from medmnist import INFO, Evaluator

# First work on 2D dataset
# data_flag: Selects the BloodMNIST dataset.
data_flag = 'bloodmnist'
download = True

NUM_EPOCHS = 5
BATCH_SIZE = 64
lr = 0.01

# info: Retrieves dataset-specific information, such as the type of task (multi-class) and the number of channels (n_channels = 1 for grayscale images).
info = INFO[data_flag]
task = info['task']
n_channels = info['n_channels']
n_classes = len(info['label'])

DataClass = getattr(medmnist, info['python_class'])

# First, we read the MedMNIST data, preprocess them and encapsulate them into dataloader form.
# preprocessing
# Converts images to PyTorch tensors (ToTensor) and normalises pixel values to the range [0, 1] (Normalize).
data_transform = transforms.Compose([
    transforms.RandomHorizontalFlip(p=0.5),  # Flip images horizontally with 50% probability
    transforms.RandomVerticalFlip(p=0.2),    # Flip images vertically with 20% probability
    transforms.RandomRotation(degrees=20),   # Rotate images randomly within a range of ±20 degrees
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5], std=[0.5]),  # Normalize to [-1, 1] assuming grayscale
])

# load the data
train_dataset = DataClass(split='train', transform=data_transform, download=download)
val_dataset = DataClass(split='val', transform=data_transform, download=download)
test_dataset = DataClass(split='test', transform=data_transform, download=download)

# encapsulate data into dataloader form
train_loader = data.DataLoader(dataset=train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader = data.DataLoader(dataset=val_dataset, batch_size=2*BATCH_SIZE, shuffle=False)
test_loader = data.DataLoader(dataset=test_dataset, batch_size=2*BATCH_SIZE, shuffle=False)

# define a simple CNN model
class EnhancedNet(nn.Module):
    def __init__(self, in_channels, num_classes):
        super(EnhancedNet, self).__init__()
        
        self.features = nn.Sequential(
            nn.Conv2d(in_channels, 16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Dropout(0.2),
            
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Dropout(0.25),
            
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Dropout(0.3),
        )

        self.fc = nn.Sequential(
            nn.Linear(64*3*3, 128),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(128, num_classes)
        )
    
    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x

model = EnhancedNet(in_channels=n_channels, num_classes=n_classes)
    
# define loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9)

# Training and evaluation loop
for epoch in range(NUM_EPOCHS):
    model.train()
    train_loss = 0.0
    correct = 0
    total = 0

    for inputs, targets in tqdm(train_loader):
        optimizer.zero_grad()
        outputs = model(inputs)
        targets = targets.squeeze().long()
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

    train_acc = 100. * correct / total
    print(f"Epoch {epoch+1}/{NUM_EPOCHS}, Loss: {train_loss:.4f}, Accuracy: {train_acc:.2f}%")

# Evaluation function
def evaluate(loader):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, targets in loader:
            outputs = model(inputs)
            targets = targets.squeeze().long()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
    return 100. * correct / total

# Evaluate on validation and test sets
val_acc = evaluate(val_loader)
test_acc = evaluate(test_loader)
print(f"Validation Accuracy: {val_acc:.2f}%")
print(f"Test Accuracy: {test_acc:.2f}%")

# Display dataset distribution
labels = np.array(train_dataset.labels)
unique_classes, counts = np.unique(labels, return_counts=True)
for cls, count in zip(unique_classes, counts):
    print(f"Class {cls}: {count} samples")
class_ratios = counts / counts.sum()
print(f"Class ratios: {class_ratios}")
