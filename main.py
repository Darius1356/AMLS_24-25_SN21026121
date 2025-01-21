import os
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV, learning_curve
from sklearn.metrics import (confusion_matrix, ConfusionMatrixDisplay, roc_curve,
                             precision_recall_curve, accuracy_score, roc_auc_score)
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.decomposition import PCA
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline

from sklearn.tree import DecisionTreeClassifier, plot_tree
from tensorflow.keras import backend as K


# Suppress TensorFlow logs
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
tf.random.set_seed(42)

# ---------------------------
# Load and Preprocess Dataset
# ---------------------------
data_dir = r'C:\Users\dariu\Documents\1. UCL\4th Year\Applied Machine Learning Systems I\AMLS_24-25_SN21026121\Datasets\breastmnist.npz'
data = np.load(data_dir)

x_train, y_train = data['train_images'], data['train_labels']
x_val, y_val = data['val_images'], data['val_labels']
x_test, y_test = data['test_images'], data['test_labels']

# Normalize and add channel dimension
x_train, x_val, x_test = [np.expand_dims(arr.astype('float32') / 255.0, axis=-1)
                          for arr in (x_train, x_val, x_test)]

y_train, y_val, y_test = [arr.ravel() for arr in (y_train, y_val, y_test)]

# --------------------------
# Data Augmentation Pipeline
# --------------------------
data_augmentation = keras.Sequential([
    layers.RandomFlip("horizontal"),
    layers.RandomRotation(0.1),
    layers.RandomZoom(0.1),
    layers.RandomContrast(0.1)
], name="data_augmentation")

# ---------------------------
# Build CNN Model
# ---------------------------
def build_cnn():
    inputs = keras.Input(shape=(28, 28, 1))
    x = data_augmentation(inputs)

    # Layer 1
    x = layers.Conv2D(32, 3, activation='relu')(x)
    x = layers.MaxPooling2D()(x)
    x = layers.Dropout(0.1)(x)

    # Layer 2
    x = layers.Conv2D(64, 3, activation='relu')(x)
    x = layers.MaxPooling2D()(x)
    x = layers.Dropout(0.3)(x)

    # Fully Convolutional Layer
    x = layers.Flatten()(x)
    x = layers.Dense(128, activation='relu')(x)
    x = layers.Dropout(0.4)(x)
    outputs = layers.Dense(1, activation='sigmoid')(x)

    model = keras.Model(inputs, outputs)
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model

cnn_model = build_cnn()

# ---------------------------
# Train CNN Model with Early Stopping
# ---------------------------
early_stopping = keras.callbacks.EarlyStopping(
    monitor='val_loss', patience=5, restore_best_weights=True
)

history = cnn_model.fit(
    x_train, y_train,
    validation_data=(x_val, y_val),
    epochs=50,
    batch_size=128,
    callbacks=[early_stopping]
)

# ------------------------------------
# Feature Extraction
# ------------------------------------
def extract_features(cnn_model, datasets):
    feature_extractor = keras.Model(inputs=cnn_model.input, outputs=cnn_model.layers[-2].output)
    return [feature_extractor.predict(dataset) for dataset in datasets]

train_features, val_features, test_features = extract_features(cnn_model, (x_train, x_val, x_test))

scaler = StandardScaler()
train_features = scaler.fit_transform(train_features)
val_features = scaler.transform(val_features)
test_features = scaler.transform(test_features)

# ------------------------------------
# Model Training and Evaluation
# ------------------------------------
def train_and_evaluate_model(model, param_grid, X_train, y_train, X_val, y_val, X_test, y_test):
    grid_search = GridSearchCV(model, param_grid, scoring='accuracy', cv=3, n_jobs=-1, verbose=1)
    grid_search.fit(X_train, y_train)
    best_model = grid_search.best_estimator_
    val_accuracy = best_model.score(X_val, y_val)
    test_accuracy = best_model.score(X_test, y_test)
    return best_model, val_accuracy, test_accuracy, grid_search

# Decision Tree
dt_params = {
    'max_depth': [None, 5, 10, 20],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
}

dt_model, val_acc_dt, test_acc_dt, dt_grid_search = train_and_evaluate_model(
    DecisionTreeClassifier(random_state=42), dt_params, train_features, y_train, val_features, y_val, test_features, y_test
)

# Random Forest
rf_params = {
    'n_estimators': [50, 100, 200],
    'max_depth': [None, 5, 10, 20],
    'min_samples_split': [2, 5, 10]
}

rf_model, val_acc_rf, test_acc_rf, rf_grid_search = train_and_evaluate_model(
    RandomForestClassifier(random_state=42), rf_params, train_features, y_train, val_features, y_val, test_features, y_test
)

# Display Results
print("Model Comparison")
print(f"Decision Tree  - Val Acc: {val_acc_dt:.4f}, Test Acc: {test_acc_dt:.4f}")
print(f"Random Forest  - Val Acc: {val_acc_rf:.4f}, Test Acc: {test_acc_rf:.4f}")

# Best Random Forest Hyperparameters
print("\nBest Random Forest Hyperparameters:")
for param, value in rf_grid_search.best_params_.items():
    print(f"{param}: {value}")

# ------------------------------------
# Learning Curve Plot
# ------------------------------------
def plot_learning_curves(estimator, X, y, title):
    train_sizes, train_scores, val_scores = learning_curve(
        estimator, X, y, cv=3, n_jobs=-1, train_sizes=np.linspace(0.1, 1.0, 5), scoring='accuracy'
    )

    plt.figure(figsize=(8, 6))
    plt.plot(train_sizes, train_scores.mean(axis=1), 'o-', label='Training score')
    plt.plot(train_sizes, val_scores.mean(axis=1), 'o-', label='Validation score')
    plt.fill_between(train_sizes, train_scores.mean(axis=1) - train_scores.std(axis=1),
                     train_scores.mean(axis=1) + train_scores.std(axis=1), alpha=0.1)
    plt.fill_between(train_sizes, val_scores.mean(axis=1) - val_scores.std(axis=1),
                     val_scores.mean(axis=1) + val_scores.std(axis=1), alpha=0.1)
    plt.title(title)
    plt.xlabel("Training Size")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.grid()
    plt.show()

plot_learning_curves(dt_model, train_features, y_train, title="Decision Tree Learning Curve")
plot_learning_curves(rf_model, train_features, y_train, title="Random Forest Learning Curve")

# ------------------------------------
# PCA and Visualisation
# ------------------------------------
pca = PCA(n_components=2)
train_features_2d = pca.fit_transform(train_features)
val_features_2d = pca.transform(val_features)

rf_2d = RandomForestClassifier(n_estimators=100, random_state=42)
rf_2d.fit(train_features_2d, y_train)

x_min, x_max = train_features_2d[:, 0].min() - 1, train_features_2d[:, 0].max() + 1
y_min, y_max = train_features_2d[:, 1].min() - 1, train_features_2d[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.02), np.arange(y_min, y_max, 0.02))

Z = rf_2d.predict(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)

plt.figure(figsize=(8, 6))
plt.contourf(xx, yy, Z, alpha=0.4)
plt.scatter(train_features_2d[:, 0], train_features_2d[:, 1], c=y_train, edgecolors='k')
plt.title("Random Forest Decision Boundary (2D PCA)")
plt.xlabel("PCA Component 1")
plt.ylabel("PCA Component 2")
plt.show()

# Visualise a Single Decision Tree
plt.figure(figsize=(20, 10))
plot_tree(rf_model.estimators_[0], filled=True, feature_names=[f"feature_{i}" for i in range(train_features.shape[1])])
plt.show()
