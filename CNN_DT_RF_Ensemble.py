import os
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras import layers, Sequential
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV, learning_curve
from sklearn.metrics import (
    confusion_matrix, roc_curve, precision_recall_curve, accuracy_score, roc_auc_score
)
from sklearn.ensemble import RandomForestClassifier
from sklearn.decomposition import PCA
from sklearn.tree import DecisionTreeClassifier, plot_tree
from imblearn.over_sampling import SMOTE
import matplotlib.pyplot as plt

# Suppress TensorFlow logs
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
tf.random.set_seed(42)

# ---------------------------
# Dataset Loading and Preprocessing
# ---------------------------
DATA_PATH = r'C:\Users\dariu\Documents\1. UCL\4th Year\Applied Machine Learning Systems I\AMLS_24-25_SN21026121\Datasets\breastmnist.npz'
data = np.load(DATA_PATH)

x_train, y_train = data['train_images'], data['train_labels']
x_val, y_val = data['val_images'], data['val_labels']
x_test, y_test = data['test_images'], data['test_labels']

# Normalize data and reshape
x_train, x_val, x_test = [
    np.expand_dims(arr.astype('float32') / 255.0, axis=-1)
    for arr in (x_train, x_val, x_test)
]

y_train, y_val, y_test = [arr.ravel() for arr in (y_train, y_val, y_test)]

# ---------------------------
# Data Augmentation
# ---------------------------
data_augmentation = Sequential([
    layers.RandomFlip("horizontal"),
    layers.RandomRotation(0.1),
    layers.RandomZoom(0.1),
    layers.RandomContrast(0.1),
], name="data_augmentation")

# ---------------------------
# CNN Model Definition
# ---------------------------
def build_cnn_model():
    inputs = tf.keras.Input(shape=(28, 28, 1))
    x = data_augmentation(inputs)

    x = layers.Conv2D(32, 3, activation='relu')(x)
    x = layers.MaxPooling2D()(x)
    x = layers.Dropout(0.1)(x)

    x = layers.Conv2D(64, 3, activation='relu')(x)
    x = layers.MaxPooling2D()(x)
    x = layers.Dropout(0.3)(x)

    x = layers.Flatten()(x)
    x = layers.Dense(128, activation='relu')(x)
    x = layers.Dropout(0.4)(x)

    outputs = layers.Dense(1, activation='sigmoid')(x)
    
    model = tf.keras.Model(inputs, outputs)
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model

cnn_model = build_cnn_model()

# ---------------------------
# Train CNN Model
# ---------------------------
early_stopping_cb = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

cnn_model.fit(
    x_train, y_train,
    validation_data=(x_val, y_val),
    epochs=50,
    batch_size=128,
    callbacks=[early_stopping_cb]
)

# ---------------------------
# Feature Extraction
# ---------------------------
def extract_features(model, datasets):
    extractor = tf.keras.Model(inputs=model.input, outputs=model.layers[-2].output)
    return [extractor.predict(ds) for ds in datasets]

train_features, val_features, test_features = extract_features(cnn_model, (x_train, x_val, x_test))

scaler = StandardScaler()
train_features = scaler.fit_transform(train_features)
val_features = scaler.transform(val_features)
test_features = scaler.transform(test_features)

# ---------------------------
# Model Training and Evaluation
# ---------------------------
def evaluate_model(model, params, X_train, y_train, X_val, y_val, X_test, y_test):
    grid_search = GridSearchCV(model, params, cv=3, scoring='accuracy', n_jobs=-1, verbose=1)
    grid_search.fit(X_train, y_train)
    
    best_model = grid_search.best_estimator_
    val_probs = best_model.predict_proba(X_val)[:, 1]
    test_probs = best_model.predict_proba(X_test)[:, 1]

    val_auc = roc_auc_score(y_val, val_probs)
    test_auc = roc_auc_score(y_test, test_probs)

    val_acc = accuracy_score(y_val, best_model.predict(X_val))
    test_acc = accuracy_score(y_test, best_model.predict(X_test))
    
    return best_model, val_auc, test_auc, val_acc, test_acc, grid_search

# Decision Tree
dt_params = {
    'max_depth': [None, 5, 10, 20],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
}
dt_model, val_auc_dt, test_auc_dt, val_acc_dt, test_acc_dt, dt_grid = evaluate_model(
    DecisionTreeClassifier(random_state=42), dt_params, train_features, y_train, val_features, y_val, test_features, y_test
)

# Random Forest
rf_params = {
    'n_estimators': [50, 100, 200],
    'max_depth': [None, 5, 10, 20],
    'min_samples_split': [2, 5, 10]
}
rf_model, val_auc_rf, test_auc_rf, val_acc_rf, test_acc_rf, rf_grid = evaluate_model(
    RandomForestClassifier(random_state=42), rf_params, train_features, y_train, val_features, y_val, test_features, y_test
)

print("Model Comparison:")
print(f"Decision Tree  - Validation AUC: {val_auc_dt:.4f}, Test AUC: {test_auc_dt:.4f}, Validation ACC: {val_acc_dt:.4f}, Test ACC: {test_acc_dt:.4f}")
print(f"Random Forest  - Validation AUC: {val_auc_rf:.4f}, Test AUC: {test_auc_rf:.4f}, Validation ACC: {val_acc_rf:.4f}, Test ACC: {test_acc_rf:.4f}")

# ---------------------------
# Visualisations
# ---------------------------
def plot_learning_curve(estimator, X, y, title):
    train_sizes, train_scores, val_scores = learning_curve(
        estimator, X, y, cv=3, n_jobs=-1, train_sizes=np.linspace(0.1, 1.0, 5), scoring='accuracy'
    )

    plt.plot(train_sizes, train_scores.mean(axis=1), label='Train')
    plt.plot(train_sizes, val_scores.mean(axis=1), label='Validation')
    plt.fill_between(train_sizes, train_scores.mean(axis=1) - train_scores.std(axis=1),
                     train_scores.mean(axis=1) + train_scores.std(axis=1), alpha=0.2)
    plt.fill_between(train_sizes, val_scores.mean(axis=1) - val_scores.std(axis=1),
                     val_scores.mean(axis=1) + val_scores.std(axis=1), alpha=0.2)
    plt.title(title)
    plt.xlabel("Training Size")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.grid()
    plt.show()

plot_learning_curve(dt_model, train_features, y_train, "Decision Tree Learning Curve")
plot_learning_curve(rf_model, train_features, y_train, "Random Forest Learning Curve")
