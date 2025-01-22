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
import os

import medmnist
# INFO: Metadata about the MedMNIST dataset, e.g., task type and number of classes.
# Evaluator: Calculates metrics like AUC and accuracy.
from medmnist import INFO, Evaluator

# First work on 2D dataset
# data_flag: Selects the BloodMNIST dataset.
data_flag = 'bloodmnist'
download = True

NUM_EPOCHS = 20
BATCH_SIZE = 16
lr = 0.006386990932880774

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
            nn.Dropout(0.1719734197141942),
            
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Dropout(0.2656156743608801),
            
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Dropout(0.15540328363230912),
        )

        self.fc = nn.Sequential(
            nn.Linear(64*3*3, 128),
            nn.ReLU(),
            nn.Dropout(0.37852373003007855),
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
training_loss = []
validation_loss = []
epochs = list(range(1, NUM_EPOCHS + 1))

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
    training_loss.append(train_loss / len(train_loader))
    
    # Evaluate on validation set
    model.eval()
    val_loss = 0.0
    with torch.no_grad():
        for inputs, targets in val_loader:
            outputs = model(inputs)
            targets = targets.squeeze().long()
            loss = criterion(outputs, targets)
            val_loss += loss.item()
    validation_loss.append(val_loss / len(val_loader))

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

# Save metrics and graphs
output_dir = r"C:\\Users\\dariu\\Documents\\1. UCL\\4th Year\\Applied Machine Learning Systems I\\AMLS_24-25_SN21026121\\B"
os.makedirs(output_dir, exist_ok=True)

# Save metrics to a text file
metrics_path = os.path.join(output_dir, "metrics.txt")
with open(metrics_path, "w") as f:
    f.write(f"Validation Accuracy: {val_acc:.2f}%\n")
    f.write(f"Test Accuracy: {test_acc:.2f}%\n")
    f.write("Class Distribution in Training Dataset:\n")
    for cls, count in zip(unique_classes, counts):
        f.write(f"Class {cls}: {count} samples\n")
    f.write("Class Ratios:\n")
    for cls, ratio in zip(unique_classes, class_ratios):
        f.write(f"Class {cls}: {ratio:.4f}\n")

print(f"Metrics saved to {metrics_path}")

# Save the loss curve graph
loss_curve_path = os.path.join(output_dir, "loss_curve.png")
plt.figure(figsize=(8, 6))
plt.plot(epochs, training_loss, label="Training Loss", marker="o")
plt.plot(epochs, validation_loss, label="Validation Loss", marker="s")
plt.title("Training vs Validation Loss Curve", fontsize=14)
plt.xlabel("Epochs", fontsize=12)
plt.ylabel("Loss", fontsize=12)
plt.legend(fontsize=12)
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(loss_curve_path)
plt.close()
print(f"Loss curve saved to {loss_curve_path}")

# Save the class distribution bar chart
class_distribution_path = os.path.join(output_dir, "class_distribution.png")
plt.figure(figsize=(8, 6))
plt.bar(unique_classes, counts, tick_label=[f"Class {cls}" for cls in unique_classes])
plt.title("Class Distribution in Training Dataset", fontsize=14)
plt.xlabel("Class", fontsize=12)
plt.ylabel("Count", fontsize=12)
plt.tight_layout()
plt.savefig(class_distribution_path)
plt.close()
print(f"Class distribution chart saved to {class_distribution_path}")
