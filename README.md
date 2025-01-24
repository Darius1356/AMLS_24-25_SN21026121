# AMLS_24-25_SN21026121
# Project Title: Advanced Machine Learning for Medical Imaging

## Project Overview

This repository contains an advanced machine learning project focusing on medical imaging classification tasks using the **BloodMNIST** and **BreastMNIST** datasets. The project explores a variety of machine learning algorithms, including **Decision Trees**, **Random Forests**, **Support Vector Machines (SVMs)**, and **Convolutional Neural Networks (CNNs)**, and evaluates their performance on these datasets. 

The primary goal of this project is to apply state-of-the-art machine learning techniques for medical diagnosis and demonstrate how various models can be tuned and optimised for different evaluation metrics.

## Repository Structure

A/
Scripts and resources for the BloodMNIST dataset.
B/
Scripts and resources for the BreastMNIST dataset.
Datasets/
Preprocessed BloodMNIST and BreastMNIST datasets.
main.py
Entry point to run experiments.
README.md
This file.
requirements.txt
Python dependencies required to run the project.
.gitignore
Git ignore file for excluding unnecessary files.

### Key Files and Folders

- **A**: Contains machine learning and CNN-related scripts for the BloodMNIST dataset.

- **B**: Contains similar scripts as folder A but tailored for the BreastMNIST dataset.

- **Datasets**: Folder for '.npz' format datasets to be pasted into.

- **main.py**: Orchestrates the execution of various experiments across datasets and models.

## Installation Instructions

### 1. Clone the Repository
git clone <repository-url>
cd <repository-folder>

### 2. Set Up a Virtual Environment
python -m venv venv

### 3. Activate the Virtual Environment
## On Windows:
venv\Scripts\activate

## On Mac/Linux:
source venv/bin/activate

### 4. Install Dependencies
pip install -r requirements.txt
### Running the Project
## Once the dependencies are installed, you can run the main script using the following command:
venv\Scripts\python main.py

## This will execute the pre-configured experiments for BloodMNIST and BreastMNIST datasets and output the results for each model.

### Key Features
## Multi-Model Comparison:
# Evaluate and compare the performance of Decision Trees, Random Forests, SVMs, and CNNs.
## Model Tuning:
# Includes hyperparameter tuning for optimising the models.
## Visualisation:
# Generates performance plots such as confusion matrices, precision-recall curves, and ROC curves.
## Results and Metrics
# Evaluation metrics for each model are stored in the project as .txt files (e.g., CNN_metrics.txt) or visualised in .png files (e.g., random_forest_roc_curve.png).