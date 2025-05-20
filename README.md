# Toxic Comment Classification

## Overview

This document provides comprehensive documentation for the `toxicCommentClassification.py` script, which implements a BERT-based deep learning model to classify toxic comments across multiple categories. The model is designed to handle the Jigsaw Toxic Comment Classification Challenge dataset, identifying various forms of toxicity in text data.

## Table of Contents

1. [Dependencies](#dependencies)
2. [Configuration](#configuration)
3. [Dataset](#dataset)
4. [Code Structure](#code-structure)
5. [Key Functions](#key-functions)
6. [Model Architecture](#model-architecture)
7. [Training Process](#training-process)
8. [Evaluation Metrics](#evaluation-metrics)
9. [Visualization](#visualization)
10. [Prediction and Submission](#prediction-and-submission)
11. [Usage Guide](#usage-guide)
12. [Optimization Tips](#optimization-tips)
13. [Troubleshooting](#troubleshooting)

## Dependencies

The script relies on the following libraries:

```python
# Data manipulation
import numpy as np
import pandas as pd

# Visualization
import matplotlib.pyplot as plt
import seaborn as sns

# Text processing
import re

# Deep Learning - PyTorch
import torch
from torch.utils.data import Dataset, DataLoader
from torch import nn
from torch.optim import AdamW

# Hugging Face Transformers
from transformers import BertTokenizer, BertForSequenceClassification, get_linear_schedule_with_warmup

# Machine Learning
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, hamming_loss, accuracy_score
```

## Configuration

The script uses the following configuration parameters:

| Parameter | Value | Description |
|-----------|-------|-------------|
| `MODEL_NAME` | 'bert-base-uncased' | Pretrained BERT model to use |
| `MAX_LENGTH` | 128 | Maximum sequence length for tokenization |
| `BATCH_SIZE` | 16 | Number of samples per batch |
| `EPOCHS` | 3 | Number of training epochs |
| `LEARNING_RATE` | 2e-5 | Learning rate for optimizer |
| `SEED` | 42 | Random seed for reproducibility |

These parameters can be adjusted to optimize performance based on hardware resources and specific requirements.

## Dataset

The script is designed to work with the Jigsaw Toxic Comment Classification Challenge dataset, which includes:

- `train.csv`: Contains comments with toxicity labels
- `test.csv`: Contains comments for prediction (no labels)
- `sample_submission.csv`: Template for submission format

The dataset includes the following toxicity labels:
- toxic
- severe_toxic
- obscene
- threat
- insult
- identity_hate

If the data files are not found, the script creates a small dummy dataset to demonstrate functionality.

## Code Structure

The script follows a modular structure:

1. **Setup and Configuration**: Imports, constants, and seed initialization
2. **Data Loading and Exploration**: Functions to load and analyze data
3. **Data Preprocessing**: Text cleaning and preparation
4. **Dataset Classes**: Custom PyTorch Dataset classes for training and testing
5. **Training Functions**: Train and evaluation loops
6. **Prediction Function**: Model inference on test data
7. **Visualization Functions**: Training history and data exploration plots
8. **Main Execution Function**: Coordinates the entire workflow

## Key Functions

### `load_data()`

Loads the training and test datasets from CSV files. Creates a dummy dataset if files are not found.

```python
def load_data():
    try:
        train_df = pd.read_csv('/jigsaw toxic comment classification/train.csv')
        test_df = pd.read_csv('/jigsaw toxic comment classification/test.csv')
        sample_submission_df = pd.read_csv('/jigsaw toxic comment classification/sample_submission.csv')
        print("Data loaded successfully.")
    except FileNotFoundError:
        print("Error: Dataset files not found. Using dummy data for script structure.")
        # Creates dummy data if files not found
        # ...
    
    return train_df, test_df, sample_submission_df
```

### `explore_data(train_df, test_df, label_cols)`

Performs exploratory data analysis, visualizing:
- Label distribution
- Correlation between toxicity types
- Comment length distribution
- Statistics on clean vs. toxic comments

```python
def explore_data(train_df, test_df, label_cols):
    # Prints dataset information
    # Creates visualizations for label distribution
    # Analyzes comment lengths
    # Generates correlation matrix for labels
    # ...
    return train_df, test_df
```

### `clean_text(text)`

Preprocesses comment text by removing excessive whitespace.

```python
def clean_text(text):
    text = str(text)
    # Remove excessive whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    return text
```

### Dataset Classes

Two custom PyTorch Dataset classes handle data loading and tokenization:

1. `ToxicCommentDataset`: For training and validation data (includes labels)
2. `TestCommentDataset`: For test data (no labels)

These classes convert text to BERT input format with tokenization, padding, and attention masks.

### Training Functions

#### `train_epoch(model, data_loader, loss_fn, optimizer, device, scheduler)`

Performs one training epoch:
- Processes batches
- Computes loss
- Updates model parameters
- Reports progress

#### `eval_model(model, data_loader, loss_fn, device, label_cols)`

Evaluates model on validation data:
- Computes loss
- Calculates metrics (ROC AUC, Hamming loss)
- Returns evaluation metrics

### `predict(model, data_loader, device)`

Generates predictions on test data:
- Processes test batches
- Applies sigmoid to convert logits to probabilities
- Returns probability array for each toxicity category

### `plot_training_history(history, epochs)`

Visualizes training metrics over epochs:
- Training and validation loss
- Validation ROC AUC

## Model Architecture

The script uses a fine-tuned BERT model (`bert-base-uncased`) for multi-label text classification:

- **Base Model**: Pre-trained BERT with 12 layers, 768 hidden dimensions
- **Classification Head**: Linear layer mapping BERT output to 6 classes (one for each toxicity type)
- **Activation**: Sigmoid (via BCEWithLogitsLoss) for multi-label classification

The model is loaded and configured as follows:

```python
model = BertForSequenceClassification.from_pretrained(
    MODEL_NAME,
    num_labels=len(label_cols),
    output_attentions=False,
    output_hidden_states=False,
)
```

## Training Process

The training workflow follows these steps:

1. **Data Preparation**:
   - Load and clean data
   - Split into training/validation sets
   - Create PyTorch Dataset and DataLoader objects

2. **Model Setup**:
   - Load pre-trained BERT model
   - Configure for multi-label classification
   - Initialize optimizer and learning rate scheduler

3. **Training Loop**:
   - Train for specified number of epochs
   - Monitor training and validation metrics
   - Save best model based on validation ROC AUC

4. **Optimization Techniques**:
   - Learning rate scheduler with warmup
   - Gradient clipping to prevent exploding gradients
   - Early stopping (implicitly via best model saving)

## Evaluation Metrics

The model performance is evaluated using:

1. **ROC AUC Score**: Area under the ROC curve, calculated individually for each toxicity label and averaged. Higher is better.
2. **Hamming Loss**: Fraction of labels incorrectly predicted. Lower is better.
3. **Training and Validation Loss**: Binary cross-entropy loss.

These metrics provide a comprehensive view of model performance for multi-label classification.

## Visualization

The script includes several visualizations:

1. **Label Distribution**: Bar chart showing counts for each toxicity type
2. **Label Correlation**: Heatmap showing relationships between different toxicity types
3. **Comment Length**: Histogram of comment lengths (word count)
4. **Training History**: Line charts of loss and ROC AUC over epochs

## Prediction and Submission

After training, the model:
1. Loads the best-performing model state
2. Generates predictions on test data
3. Creates a submission file in the required format

```python
# Generate predictions
test_predictions = predict(model, test_dataloader, device)

# Create submission DataFrame
submission_df = pd.DataFrame(test_predictions, columns=label_cols)
submission_df['id'] = test_df['id']
submission_df = submission_df[['id'] + label_cols]

# Save to CSV
submission_df.to_csv('submission.csv', index=False)
```

## Usage Guide

To use this script:

1. **Prepare the Dataset**:
   - Download the Jigsaw Toxic Comment Classification Challenge dataset
   - Place CSV files in the `/jigsaw toxic comment classification/` directory

2. **Install Dependencies**:
   ```bash
   pip install numpy pandas matplotlib seaborn torch transformers scikit-learn
   ```

3. **Run the Script**:
   ```bash
   python toxicCommentClassification.py
   ```

4. **Review Output**:
   - The script will print training progress and evaluation metrics
   - Visualizations will be displayed during execution
   - The final submission file will be saved as `submission.csv`
   - The best model will be saved as `best_model_state.bin`


## Troubleshooting

Common issues and solutions:

1. **CUDA Out of Memory**:
   - Reduce `BATCH_SIZE`
   - Decrease `MAX_LENGTH`
   - Use gradient accumulation

2. **Slow Training**:
   - Enable mixed precision training
   - Use a smaller BERT variant
   - Reduce dataset size for prototyping

3. **Multiprocessing Issues**:
   - The script includes `torch.multiprocessing.freeze_support()` to prevent problems
   - Set `num_workers=0` in DataLoader as implemented

4. **Poor Performance**:
   - Check class imbalance in training data
   - Try different evaluation thresholds (default is 0.5)
   - Implement class weights in loss function

---
