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


import medmnist
# INFO: Metadata about the MedMNIST dataset, e.g., task type and number of classes.
# Evaluator: Calculates metrics like AUC and accuracy.
from medmnist import INFO, Evaluator

# First work on 2D dataset
# data_flag: Selects the BreastMNIST dataset.
data_flag = 'breastmnist'
download = True

NUM_EPOCHS = 3
BATCH_SIZE = 128
lr = 0.001

# info: Retrieves dataset-specific information, such as the type of task (binary-class) and the number of channels (n_channels = 1 for grayscale images).
info = INFO[data_flag]
task = info['task']
n_channels = info['n_channels']
n_classes = len(info['label'])

DataClass = getattr(medmnist, info['python_class'])

# First, we read the MedMNIST data, preprocess them and encapsulate them into dataloader form.
# preprocessing
# Converts images to PyTorch tensors (ToTensor) and normalises pixel values to the range [-1, 1] (Normalize).
data_transform = transforms.Compose([
    transforms.RandomHorizontalFlip(p=0.5),  # Flip images horizontally with 50% probability
    transforms.RandomVerticalFlip(p=0.2),    # Flip images vertically with 20% probability
    transforms.RandomRotation(degrees=20),   # Rotate images randomly within a range of ±20 degrees
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),  # Adjust brightness, contrast, etc.
    transforms.RandomResizedCrop(size=28, scale=(0.8, 1.0), ratio=(0.9, 1.1)),  # Randomly crop and resize
    transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),  # Random translation
    transforms.RandomGrayscale(p=0.1),        # Convert to grayscale with a small probability
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5], std=[0.5])  # Normalize pixel values to [-1, 1]
])

# load the data
train_dataset = DataClass(split='train', transform=data_transform, download=download)
test_dataset = DataClass(split='test', transform=data_transform, download=download)

pil_dataset = DataClass(split='train', download=download)

# encapsulate data into dataloader form
# Wraps the datasets into DataLoader objects for batching and shuffling.
train_loader = data.DataLoader(dataset=train_dataset, batch_size=BATCH_SIZE, shuffle=True)
train_loader_at_eval = data.DataLoader(dataset=train_dataset, batch_size=2*BATCH_SIZE, shuffle=False)
test_loader = data.DataLoader(dataset=test_dataset, batch_size=2*BATCH_SIZE, shuffle=False)


# define a simple CNN model
#   Conv2d: Applies convolutional filters to extract spatial features.
#   BatchNorm2d: Normalises layer outputs to stabilise training.
#   ReLU: Introduces non-linearity.
class EnhancedNet(nn.Module):
    def __init__(self, in_channels, num_classes):
        super(EnhancedNet, self).__init__()
        
        self.features = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2, 2)
        )
        
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(128, num_classes)
        )
    
    def forward(self, x):
        x = self.features(x)
        x = self.gap(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x

model = EnhancedNet(in_channels=n_channels, num_classes=n_classes)
    
# define loss function and optimizer
#   BCEWithLogitsLoss: For multi-label binary classification.
#   CrossEntropyLoss: For multi-class classification.
if task == "multi-label, binary-class":
    criterion = nn.BCEWithLogitsLoss()
else:
    criterion = nn.CrossEntropyLoss()
    
# Stochastic Gradient Descent (SGD) with a learning rate (lr) and momentum.
optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9)

# train

#   Backward pass (loss.backward()).
#   Update weights (optimizer.step()).
for epoch in range(NUM_EPOCHS):
    train_correct = 0
    train_total = 0
    test_correct = 0
    test_total = 0
    
    model.train()
    for inputs, targets in tqdm(train_loader):
        # forward + backward + optimize
        # Zero gradients
        optimizer.zero_grad()
        # Forward pass
        outputs = model(inputs)

        # Compute loss
        if task == 'multi-label, binary-class':
            targets = targets.to(torch.float32)
            loss = criterion(outputs, targets)
        else:
            targets = targets.squeeze().long()
            loss = criterion(outputs, targets)
        
        # Backwards pass
        loss.backward()
        # Update weights
        optimizer.step()

# evaluation
def test(split):
    model.eval()
    y_true = torch.tensor([])
    y_score = torch.tensor([])
    
    data_loader = train_loader_at_eval if split == 'train' else test_loader

    with torch.no_grad():
        for inputs, targets in data_loader:
            outputs = model(inputs)

            # Computes prediction using softmax
            if task == 'multi-label, binary-class':
                targets = targets.to(torch.float32)
                outputs = outputs.softmax(dim=-1)
            else:
                targets = targets.squeeze().long()
                outputs = outputs.softmax(dim=-1)
                targets = targets.float().resize_(len(targets), 1)

            # Appends true labels and predictions to y_true and y_score
            y_true = torch.cat((y_true, targets), 0)
            y_score = torch.cat((y_score, outputs), 0)

        y_true = y_true.numpy()
        y_score = y_score.detach().numpy()
        
        # Evaluator calculates AUC and accuracy for the given split (train/test).
        evaluator = Evaluator(data_flag, split)
        metrics = evaluator.evaluate(y_score)
    
        print('%s  auc: %.3f  acc:%.3f' % (split, *metrics))


# Run Evaluation: The results show metrics for the training and test datasets.
print('==> Evaluating ...')
test('train')
test('test')

# Determine balance of dataset
# Access the labels in the training dataset
labels = np.array(train_dataset.labels)
# Count occurrences of each class
unique_classes, counts = np.unique(labels, return_counts=True)
# Display the class distribution
for cls, count in zip(unique_classes, counts):
    print(f"Class {cls}: {count} samples")
# Check the ratio of class distribution
class_ratios = counts / counts.sum()
print(f"Class ratios: {class_ratios}")