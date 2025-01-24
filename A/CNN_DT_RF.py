import os
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras import layers, Sequential
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV, learning_curve
from sklearn.metrics import (
    confusion_matrix, roc_curve, precision_recall_curve, accuracy_score, roc_auc_score, f1_score,classification_report, precision_score, recall_score
)
from sklearn.ensemble import RandomForestClassifier
from sklearn.decomposition import PCA
from sklearn.tree import DecisionTreeClassifier, plot_tree
from imblearn.over_sampling import SMOTE
import matplotlib.pyplot as plt
import seaborn as sns

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

# ---------------------------
# Random Forest Classifier and Evaluation
# ---------------------------
save_dir = r'C:\Users\dariu\Documents\1. UCL\4th Year\Applied Machine Learning Systems I\AMLS_24-25_SN21026121\A'
os.makedirs(save_dir, exist_ok=True)

# Train Random Forest Classifier
rf_model = RandomForestClassifier(random_state=42)
rf_model.fit(x_train.reshape(x_train.shape[0], -1), y_train)

# Predictions and probabilities
x_test_flat = x_test.reshape(x_test.shape[0], -1)
y_pred = rf_model.predict(x_test_flat)
y_pred_prob = rf_model.predict_proba(x_test_flat)[:, 1]

# Metrics
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
roc_auc = roc_auc_score(y_test, y_pred_prob)

# Save metrics to a text file
metrics_file_path = os.path.join(save_dir, "random_forest_metrics.txt")
with open(metrics_file_path, "w") as f:
    f.write(f"Accuracy: {accuracy:.4f}\n")
    f.write(f"Precision: {precision:.4f}\n")
    f.write(f"Recall: {recall:.4f}\n")
    f.write(f"F1 Score: {f1:.4f}\n")
    f.write(f"ROC AUC: {roc_auc:.4f}\n")
print(f"Metrics saved to {metrics_file_path}")

# Plot and save ROC curve
fpr, tpr, _ = roc_curve(y_test, y_pred_prob)
plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, label=f"ROC Curve (AUC = {roc_auc:.4f})")
plt.plot([0, 1], [0, 1], 'k--', label="Random Guess")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve for Random Forest Classifier")
plt.legend()
roc_curve_path = os.path.join(save_dir, "random_forest_roc_curve.png")
plt.savefig(roc_curve_path)
plt.close()
print(f"ROC curve saved to {roc_curve_path}")
# ---------------------------
# Accuracy and F1 Score Evaluation
# ---------------------------
dt_predictions = dt_model.predict(test_features)
dt_accuracy = accuracy_score(y_test, dt_predictions)
dt_f1 = f1_score(y_test, dt_predictions, average='binary')

# Save metrics to a text file
metrics_file = r'C:\Users\dariu\Documents\1. UCL\4th Year\Applied Machine Learning Systems I\AMLS_24-25_SN21026121\A\decision_tree_metrics.txt'
with open(metrics_file, 'w') as file:
    file.write(f"Decision Tree Accuracy: {dt_accuracy:.4f}\n")
    file.write(f"Decision Tree F1 Score: {dt_f1:.4f}\n")
    file.write("\nClassification Report:\n")
    file.write(classification_report(y_test, dt_predictions))

print(f"Metrics saved to {metrics_file}")

# ---------------------------
# Confusion Matrix
# ---------------------------
conf_matrix = confusion_matrix(y_test, dt_predictions)
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=['Benign', 'Malignant'], yticklabels=['Benign', 'Malignant'])
plt.title('Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('Actual')
conf_matrix_file = r'C:\Users\dariu\Documents\1. UCL\4th Year\Applied Machine Learning Systems I\AMLS_24-25_SN21026121\A\decision_tree_confusion_matrix.png'
plt.savefig(conf_matrix_file)
plt.close()
print(f"Confusion matrix saved as '{conf_matrix_file}'")

# ---------------------------
# PCA Plot
# ---------------------------
pca = PCA(n_components=2)
pca_features = pca.fit_transform(test_features)

plt.figure(figsize=(10, 8))
for label in np.unique(y_test):
    plt.scatter(
        pca_features[y_test == label, 0],
        pca_features[y_test == label, 1],
        label=f"Class {label}", alpha=0.7
    )
plt.title('PCA Visualization')
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.legend()
plt.grid()
pca_plot_file = r'C:\Users\dariu\Documents\1. UCL\4th Year\Applied Machine Learning Systems I\AMLS_24-25_SN21026121\A\decision_tree_pca_plot.png'
plt.savefig(pca_plot_file)
plt.close()
print(f"PCA plot saved as '{pca_plot_file}'")

# ---------------------------
# Learning Curve Plot and Save
# ---------------------------
def plot_and_save_learning_curve(estimator, title, X, y, save_path, cv=None, n_jobs=None, train_sizes=np.linspace(0.1, 1.0, 5)):
    plt.figure(figsize=(10, 6))
    plt.title(title)
    plt.xlabel("Training examples")
    plt.ylabel("Score")

    train_sizes, train_scores, test_scores = learning_curve(
        estimator, X, y, cv=cv, n_jobs=n_jobs, train_sizes=train_sizes, scoring="accuracy"
    )
    
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)

    plt.grid()

    # Plot the learning curve
    plt.fill_between(train_sizes, train_scores_mean - train_scores_std,
                     train_scores_mean + train_scores_std, alpha=0.1, color="r")
    plt.fill_between(train_sizes, test_scores_mean - test_scores_std,
                     test_scores_mean + test_scores_std, alpha=0.1, color="g")
    plt.plot(train_sizes, train_scores_mean, 'o-', color="r", label="Training score")
    plt.plot(train_sizes, test_scores_mean, 'o-', color="g", label="Cross-validation score")

    plt.legend(loc="best")
    plt.tight_layout()

    # Save the plot
    plt.savefig(save_path)
    plt.close()
    print(f"Learning curve saved as '{save_path}'")

# File path to save the learning curve
learning_curve_path = r'C:\Users\dariu\Documents\1. UCL\4th Year\Applied Machine Learning Systems I\AMLS_24-25_SN21026121\A\decision_tree_learning_curve.png'

# Use DecisionTreeClassifier
plot_and_save_learning_curve(
    dt_model,
    "Learning Curve: Decision Tree Classifier",
    train_features, y_train,
    save_path=learning_curve_path,
    cv=5, n_jobs=-1
)
