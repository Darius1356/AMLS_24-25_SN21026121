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
    transforms.RandomHorizontalFlip(),  # Flip images horizontally
    transforms.RandomRotation(15),     # Rotate images randomly
    transforms.RandomResizedCrop(28),  # Crop images randomly
    transforms.ToTensor(),
    transforms.Normalize(mean=[.5], std=[.5])
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

# visualisation
"""
print(train_dataset)
print("===================")
print(test_dataset)

image = train_dataset.montage(length=20)  # Generate the montage
plt.imshow(image)                        # Display the image
plt.axis('off')                          # Optional: Turn off axes
plt.show()                               # Display the image
print(n_channels)
"""

# Load the ResNet-18 model
model = resnet18(pretrained=False)  # Use pretrained=True for transfer learning

# Adjust the input for grayscale images
model.conv1 = nn.Conv2d(1, 64, kernel_size=3, stride=1, padding=1, bias=False)
model.maxpool = nn.Identity()  # Remove the max pooling for smaller images

# Adjust the final fully connected layer to match the number of classes
num_classes = len(train_dataset.labels)  # Number of classes in the dataset
model.fc = nn.Linear(model.fc.in_features, num_classes)
    
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

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

NUM_EPOCHS = 10  # Number of epochs

for epoch in range(NUM_EPOCHS):
    model.train()
    running_loss = 0.0

    for inputs, targets in train_loader:
        inputs, targets = inputs.to(device), targets.to(device)

        # Zero gradients
        optimizer.zero_grad()

        # Forward pass
        outputs = model(inputs)
        loss = criterion(outputs, targets)

        # Backward pass and optimisation
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    print(f"Epoch {epoch + 1}/{NUM_EPOCHS}, Loss: {running_loss / len(train_loader):.4f}")

# evaluation
def test(split):
    model.eval()
    y_true = []
    y_score = []
    
    # Select the appropriate data loader based on the split
    data_loader = train_loader if split == 'train' else test_loader

    with torch.no_grad():
        for inputs, targets in data_loader:
            # Move data to the appropriate device
            inputs, targets = inputs.to(device), targets.to(device)

            # Forward pass to get predictions
            outputs = model(inputs)

            # Process predictions and labels based on the task
            if task == 'multi-label, binary-class':
                targets = targets.float()  # Ensure targets are in float32
                outputs = torch.sigmoid(outputs)  # Sigmoid for multi-label classification
            else:
                targets = targets.long()  # Ensure targets are in long (for classification)
                outputs = torch.softmax(outputs, dim=-1)  # Softmax for multi-class classification

            # Append true labels and predictions
            y_true.append(targets.cpu())
            y_score.append(outputs.cpu())

    # Concatenate all batches
    y_true = torch.cat(y_true, dim=0).numpy()
    y_score = torch.cat(y_score, dim=0).numpy()
    
    # Evaluate metrics
    evaluator = Evaluator(data_flag, split)  # Ensure `data_flag` is defined elsewhere in your script
    metrics = evaluator.evaluate(y_score, y_true)  # Ensure `evaluator.evaluate` is compatible with y_true and y_score

    # Print results
    print('%s  auc: %.3f  acc: %.3f' % (split, *metrics))



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