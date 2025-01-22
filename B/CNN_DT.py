from sklearn.metrics import f1_score, classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import os
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
from torchvision.transforms import Compose, ToTensor, Normalize, RandomHorizontalFlip, RandomVerticalFlip, RandomRotation
import medmnist
from medmnist import INFO

# Define the CNN model
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

# Load the data
info = INFO['bloodmnist']
n_channels = info['n_channels']
n_classes = len(info['label'])
data_transform = Compose([
    RandomHorizontalFlip(p=0.5),
    RandomVerticalFlip(p=0.2),
    RandomRotation(degrees=20),
    ToTensor(),
    Normalize(mean=[0.5], std=[0.5]),
])

train_dataset = medmnist.BloodMNIST(split='train', transform=data_transform, download=True)
val_dataset = medmnist.BloodMNIST(split='val', transform=data_transform, download=True)
test_dataset = medmnist.BloodMNIST(split='test', transform=data_transform, download=True)

train_loader = data.DataLoader(dataset=train_dataset, batch_size=16, shuffle=True)
val_loader = data.DataLoader(dataset=val_dataset, batch_size=32, shuffle=False)
test_loader = data.DataLoader(dataset=test_dataset, batch_size=32, shuffle=False)

# Initialize the model
model = EnhancedNet(in_channels=n_channels, num_classes=n_classes)
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.006386990932880774, momentum=0.9)

# Training loop
NUM_EPOCHS = 20
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

# Feature extraction for decision tree
def extract_features(model, loader):
    model.eval()
    features = []
    labels = []

    with torch.no_grad():
        for inputs, targets in tqdm(loader):
            x = model.features(inputs)
            x = x.view(x.size(0), -1)
            features.append(x.numpy())
            labels.append(targets.numpy())

    features = np.concatenate(features, axis=0)
    labels = np.concatenate(labels, axis=0).ravel()
    return features, labels

train_features, train_labels = extract_features(model, train_loader)
val_features, val_labels = extract_features(model, val_loader)
test_features, test_labels = extract_features(model, test_loader)

scaler = StandardScaler()
train_features = scaler.fit_transform(train_features)
val_features = scaler.transform(val_features)
test_features = scaler.transform(test_features)

# Train and evaluate decision tree
decision_tree = DecisionTreeClassifier(random_state=42)
decision_tree.fit(train_features, train_labels)

val_predictions = decision_tree.predict(val_features)
test_predictions = decision_tree.predict(test_features)

val_accuracy = accuracy_score(val_labels, val_predictions)
test_accuracy = accuracy_score(test_labels, test_predictions)

print(f"Validation Accuracy (Decision Tree): {val_accuracy:.4f}")
print(f"Test Accuracy (Decision Tree): {test_accuracy:.4f}")

# Optional AUC for binary tasks
if len(np.unique(train_labels)) == 2:
    val_probs = decision_tree.predict_proba(val_features)[:, 1]
    test_probs = decision_tree.predict_proba(test_features)[:, 1]

    val_auc = roc_auc_score(val_labels, val_probs)
    test_auc = roc_auc_score(test_labels, test_probs)

    print(f"Validation AUC (Decision Tree): {val_auc:.4f}")
    print(f"Test AUC (Decision Tree): {test_auc:.4f}")

# ---------------------------
# Save Metrics, Confusion Matrix, and PCA Plot
# ---------------------------
output_dir = r'C:\Users\dariu\Documents\1. UCL\4th Year\Applied Machine Learning Systems I\AMLS_24-25_SN21026121\B'
os.makedirs(output_dir, exist_ok=True)

val_f1 = f1_score(val_labels, val_predictions, average='weighted')
test_f1 = f1_score(test_labels, test_predictions, average='weighted')

metrics_file = os.path.join(output_dir, 'decision_tree_metrics.txt')
with open(metrics_file, 'w') as file:
    file.write(f"Decision Tree Validation Accuracy: {val_accuracy:.4f}\n")
    file.write(f"Decision Tree Test Accuracy: {test_accuracy:.4f}\n")
    file.write(f"Decision Tree Validation F1 Score: {val_f1:.4f}\n")
    file.write(f"Decision Tree Test F1 Score: {test_f1:.4f}\n")
    file.write("\nClassification Report (Test):\n")
    file.write(classification_report(test_labels, test_predictions))

print(f"Metrics saved to {metrics_file}")

# Confusion Matrix
conf_matrix = confusion_matrix(test_labels, test_predictions)
conf_matrix_file = os.path.join(output_dir, 'confusion_matrix.png')
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=np.unique(test_labels), yticklabels=np.unique(test_labels))
plt.title('Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.savefig(conf_matrix_file)
plt.close()
print(f"Confusion matrix saved as '{conf_matrix_file}'")

# PCA Plot
pca = PCA(n_components=2)
pca_features = pca.fit_transform(test_features)

pca_plot_file = os.path.join(output_dir, 'pca_plot.png')
plt.figure(figsize=(10, 8))
for label in np.unique(test_labels):
    plt.scatter(
        pca_features[test_labels == label, 0],
        pca_features[test_labels == label, 1],
        label=f"Class {label}", alpha=0.7
    )
plt.title('PCA Visualization')
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.legend()
plt.grid()
plt.savefig(pca_plot_file)
plt.close()
print(f"PCA plot saved as '{pca_plot_file}'")
