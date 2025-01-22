import os
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV, learning_curve
from sklearn.metrics import (confusion_matrix, ConfusionMatrixDisplay, roc_curve,
                             precision_recall_curve, accuracy_score, roc_auc_score, f1_score)
import matplotlib.pyplot as plt

from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline


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
    x = layers.Dropout(0.09041545937192769)(x)

    # Layer 2
    x = layers.Conv2D(64, 3, activation='relu')(x)
    x = layers.MaxPooling2D()(x)
    x = layers.Dropout(0.30541401607988583)(x)

    # Fully Convolutional Layer
    x = layers.Flatten()(x)
    x = layers.Dense(128, activation='relu')(x)
    x = layers.Dropout(0.23425114121921314)(x)
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

# ---------------------------
# Feature Extraction for SVM
# ---------------------------
feature_extractor = keras.Model(inputs=cnn_model.input, outputs=cnn_model.layers[-2].output)
train_features, val_features, test_features = [
    feature_extractor.predict(dataset)
    for dataset in (x_train, x_val, x_test)
]

scaler = StandardScaler()
train_features = scaler.fit_transform(train_features)
val_features = scaler.transform(val_features)
test_features = scaler.transform(test_features)

# Save directory for metrics and graphs
save_dir = r"C:\Users\dariu\Documents\1. UCL\4th Year\Applied Machine Learning Systems I\AMLS_24-25_SN21026121\A"

# ---------------------------
# Evaluate SVM and Save Graphs
# ---------------------------
# Function to save evaluation metrics to a text file
def save_metrics_to_text(file_path, accuracy, auc, f1):
    with open(file_path, 'w') as f:
        f.write("CNN-SVM Evaluation Metrics\n")
        f.write("===========================\n")
        f.write(f"Accuracy: {accuracy:.4f}\n")
        f.write(f"Area Under Curve (AUC): {auc:.4f}\n")
        f.write(f"F1 Score: {f1:.2f}\n")  # Added F1 score to output


# Evaluate SVM and save graphs
def evaluate_svm_and_save(svm_model, X_test, y_test, save_dir):
    os.makedirs(save_dir, exist_ok=True)
    y_pred = svm_model.predict(X_test)
    y_prob = svm_model.predict_proba(X_test)[:, 1]

    # Confusion Matrix
    cm_display = ConfusionMatrixDisplay.from_predictions(y_test, y_pred, display_labels=["Benign", "Malignant"], cmap="Blues")
    cm_display.ax_.set_title("Confusion Matrix")
    cm_path = os.path.join(save_dir, "CNN_SVM_Confusion_Matrix.png")
    plt.savefig(cm_path)
    plt.close()

    # ROC Curve
    fpr, tpr, _ = roc_curve(y_test, y_prob)
    plt.figure()
    plt.plot(fpr, tpr, label=f"AUC = {roc_auc_score(y_test, y_prob):.2f}")
    plt.plot([0, 1], [0, 1], 'k--')
    plt.title("ROC Curve")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.legend()
    plt.grid()
    roc_path = os.path.join(save_dir, "CNN_SVM_ROC_Curve.png")
    plt.savefig(roc_path)
    plt.close()

    # Precision-Recall Curve
    precision, recall, _ = precision_recall_curve(y_test, y_prob)
    plt.figure()
    plt.plot(recall, precision)
    plt.title("Precision-Recall Curve")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.grid()
    pr_path = os.path.join(save_dir, "CNN_SVM_Precision_Recall_Curve.png")
    plt.savefig(pr_path)
    plt.close()

    # Calculate metrics
    accuracy = accuracy_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_prob)
    f1 = f1_score(y_test, y_pred)  # Added F1 score calculation

    metrics_path = os.path.join(save_dir, "CNN_SVM_Evaluation_Metrics.txt")
    save_metrics_to_text(metrics_path, accuracy, auc, f1)

    return {
        "Confusion Matrix": cm_path,
        "ROC Curve": roc_path,
        "Precision-Recall Curve": pr_path,
        "Metrics File": metrics_path
    }

param_grid = {
    'svm__C': [0.1, 1, 10],
    'svm__kernel': ['linear', 'rbf'],
    'svm__gamma': ['scale', 'auto'],
    'svm__class_weight': [None, 'balanced']  
}

pipeline = Pipeline([
    ('smote', SMOTE(random_state=42)),  # SMOTE for oversampling
    ('svm', SVC(probability=True))      # SVM as the classifier
])

svm = SVC(probability=True)
grid_search = GridSearchCV(
    pipeline,
    param_grid,
    cv=3,
    scoring='accuracy',
    verbose=2,
    n_jobs=-1
)

grid_search.fit(train_features, y_train)

best_svm = grid_search.best_estimator_
results = evaluate_svm_and_save(grid_search.best_estimator_, test_features, y_test, save_dir)
results