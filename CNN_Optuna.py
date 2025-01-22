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
import optuna
from medmnist import INFO, Evaluator

# Dataset and preprocessing
from medmnist import INFO
import medmnist

# Define dataset parameters
data_flag = 'bloodmnist'
download = True
NUM_EPOCHS = 5
info = INFO[data_flag]
task = info['task']
n_channels = info['n_channels']
n_classes = len(info['label'])
DataClass = getattr(medmnist, info['python_class'])

data_transform = transforms.Compose([
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomVerticalFlip(p=0.2),
    transforms.RandomRotation(degrees=20),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5], std=[0.5]),
])

train_dataset = DataClass(split='train', transform=data_transform, download=download)
val_dataset = DataClass(split='val', transform=data_transform, download=download)
test_dataset = DataClass(split='test', transform=data_transform, download=download)

# Define evaluation function
def evaluate(loader, model, device):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, targets in loader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            targets = targets.squeeze().long()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
    return 100. * correct / total

# Define the Optuna objective function
def objective(trial):
    # Hyperparameter search space
    lr = trial.suggest_loguniform('lr', 1e-5, 1e-1)
    batch_size = trial.suggest_categorical('batch_size', [16, 32, 64])
    dropout1 = trial.suggest_uniform('dropout1', 0.1, 0.5)
    dropout2 = trial.suggest_uniform('dropout2', 0.1, 0.5)
    dropout3 = trial.suggest_uniform('dropout3', 0.1, 0.5)
    dropout_fc = trial.suggest_uniform('dropout_fc', 0.1, 0.5)

    class TunedNet(nn.Module):
        def __init__(self, in_channels, num_classes):
            super(TunedNet, self).__init__()
            self.features = nn.Sequential(
                nn.Conv2d(in_channels, 16, kernel_size=3, padding=1),
                nn.BatchNorm2d(16),
                nn.ReLU(),
                nn.MaxPool2d(2, 2),
                nn.Dropout(dropout1),

                nn.Conv2d(16, 32, kernel_size=3, padding=1),
                nn.BatchNorm2d(32),
                nn.ReLU(),
                nn.MaxPool2d(2, 2),
                nn.Dropout(dropout2),

                nn.Conv2d(32, 64, kernel_size=3, padding=1),
                nn.BatchNorm2d(64),
                nn.ReLU(),
                nn.MaxPool2d(2, 2),
                nn.Dropout(dropout3),
            )

            self.fc = nn.Sequential(
                nn.Linear(64*3*3, 128),
                nn.ReLU(),
                nn.Dropout(dropout_fc),
                nn.Linear(128, num_classes)
            )

        def forward(self, x):
            x = self.features(x)
            x = x.view(x.size(0), -1)
            x = self.fc(x)
            return x

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Initialize model and optimizer
    model = TunedNet(in_channels=n_channels, num_classes=n_classes).to(device)
    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9)
    criterion = nn.CrossEntropyLoss()

    # Data loaders
    train_loader = data.DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = data.DataLoader(dataset=val_dataset, batch_size=batch_size, shuffle=False)

    # Training loop
    for epoch in range(NUM_EPOCHS):
        model.train()
        for inputs, targets in train_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            targets = targets.squeeze().long()
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

    # Evaluate on validation set
    val_acc = evaluate(val_loader, model, device)
    return val_acc

# Optuna study
study = optuna.create_study(direction='maximize')
study.optimize(objective, n_trials=50)

# Print best hyperparameters
print("Best hyperparameters:", study.best_params)

# Train final model with best hyperparameters
best_params = study.best_params

# Define device globally
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# Define global data loaders
val_loader = data.DataLoader(dataset=val_dataset, batch_size=best_params['batch_size'], shuffle=False)
test_loader = data.DataLoader(dataset=test_dataset, batch_size=best_params['batch_size'], shuffle=False)

# Final model setup
final_model = nn.Sequential(
    nn.Conv2d(n_channels, 16, kernel_size=3, padding=1),
    nn.BatchNorm2d(16),
    nn.ReLU(),
    nn.MaxPool2d(2, 2),
    nn.Dropout(best_params['dropout1']),

    nn.Conv2d(16, 32, kernel_size=3, padding=1),
    nn.BatchNorm2d(32),
    nn.ReLU(),
    nn.MaxPool2d(2, 2),
    nn.Dropout(best_params['dropout2']),

    nn.Conv2d(32, 64, kernel_size=3, padding=1),
    nn.BatchNorm2d(64),
    nn.ReLU(),
    nn.MaxPool2d(2, 2),
    nn.Dropout(best_params['dropout3']),

    nn.Flatten(),
    nn.Linear(64*3*3, 128),
    nn.ReLU(),
    nn.Dropout(best_params['dropout_fc']),
    nn.Linear(128, n_classes)
).to(device)

optimizer = optim.SGD(final_model.parameters(), lr=best_params['lr'], momentum=0.9)
criterion = nn.CrossEntropyLoss()
train_loader = data.DataLoader(dataset=train_dataset, batch_size=best_params['batch_size'], shuffle=True)
test_loader = data.DataLoader(dataset=test_dataset, batch_size=best_params['batch_size'], shuffle=False)

# Final training loop
for epoch in range(NUM_EPOCHS):
    final_model.train()
    for inputs, targets in train_loader:
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs = final_model(inputs)
        targets = targets.squeeze().long()
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

# Final evaluation
val_acc = evaluate(val_loader, final_model, device)
test_acc = evaluate(test_loader, final_model, device)
print(f"Validation Accuracy: {val_acc:.2f}%")
print(f"Test Accuracy: {test_acc:.2f}%")
