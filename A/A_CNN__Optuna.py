import optuna
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow import keras
import numpy as np

# Suppress TensorFlow logs
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

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

# Objective function for Optuna
def objective(trial):
    # Suggest values for dropout rates
    dropout1 = trial.suggest_float("dropout1", 0.0, 0.5)
    dropout2 = trial.suggest_float("dropout2", 0.0, 0.5)
    dropout3 = trial.suggest_float("dropout3", 0.0, 0.5)

    # Build CNN Model
    inputs = keras.Input(shape=(28, 28, 1))
    x = data_augmentation(inputs)

    # Layer 1
    x = layers.Conv2D(32, 3, activation='relu')(x)
    x = layers.MaxPooling2D()(x)
    x = layers.Dropout(dropout1)(x)

    # Layer 2
    x = layers.Conv2D(64, 3, activation='relu')(x)
    x = layers.MaxPooling2D()(x)
    x = layers.Dropout(dropout2)(x)

    # Fully Convolutional Layer
    x = layers.Flatten()(x)
    x = layers.Dense(128, activation='relu')(x)
    x = layers.Dropout(dropout3)(x)
    outputs = layers.Dense(1, activation='sigmoid')(x)

    model = keras.Model(inputs, outputs)
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

    # Train the model
    model.fit(
        x_train, y_train,
        validation_data=(x_val, y_val),
        epochs=5,  # Keep epochs low for faster optimisation
        batch_size=64,
        verbose=0
    )

    # Evaluate the model
    val_loss, val_accuracy = model.evaluate(x_val, y_val, verbose=0)
    return val_loss

# Run Optuna optimisation
study = optuna.create_study(direction="minimize")
study.optimize(objective, n_trials=50)

# Best dropout values
print("Best trial:", study.best_trial.params)
