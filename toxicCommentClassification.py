import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import re  # Regular expressions for text cleaning

# Deep Learning Framework - Using PyTorch
import torch
from torch.utils.data import Dataset, DataLoader
from torch import nn
from torch.optim import AdamW

# Hugging Face Transformers
from transformers import BertTokenizer, BertForSequenceClassification, get_linear_schedule_with_warmup

# Scikit-learn
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, hamming_loss, accuracy_score

# Set random seeds for reproducibility
SEED = 42
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

# Configuration
MODEL_NAME = 'bert-base-uncased'
MAX_LENGTH = 128  # Max sequence length BERT can handle
BATCH_SIZE = 16   # Adjust based on GPU memory
EPOCHS = 3        # Number of training epochs
LEARNING_RATE = 2e-5  # Common learning rate for BERT fine-tuning


def load_data():
    try:
        train_df = pd.read_csv('/jigsaw toxic comment classification/train.csv')
        test_df = pd.read_csv('/jigsaw toxic comment classification/test.csv')
        sample_submission_df = pd.read_csv('/jigsaw toxic comment classification/sample_submission.csv')
        print("Data loaded successfully.")
    except FileNotFoundError:
        print("Error: Dataset files not found. Using dummy data for script structure.")
        # Dummy data for script structure if files are missing
        train_df = pd.DataFrame({
            'id': ['1', '2', '3', '4', '5'],
            'comment_text': ['This is fine.', 'This is bad and obscene!', 'You are an idiot.', 
                            'Explanation why the edits made under my username Hardcore Metallica Fan were reverted?', 'Go away!'],
            'toxic': [0, 1, 1, 0, 0], 'severe_toxic': [0, 0, 0, 0, 0], 'obscene': [0, 1, 0, 0, 0],
            'threat': [0, 0, 0, 0, 0], 'insult': [0, 0, 1, 0, 0], 'identity_hate': [0, 0, 0, 0, 0]
        })
        test_df = pd.DataFrame({
            'id': ['10', '11'],
            'comment_text': ['Testing one two.', 'Another comment here.']
        })
        sample_submission_df = pd.DataFrame({
            'id': ['10', '11'], 'toxic': [0.5]*2, 'severe_toxic': [0.5]*2, 'obscene': [0.5]*2,
            'threat': [0.5]*2, 'insult': [0.5]*2, 'identity_hate': [0.5]*2
        })
    
    return train_df, test_df, sample_submission_df


def explore_data(train_df, test_df, label_cols):
    print("Train Data Shape:", train_df.shape)
    print("Test Data Shape:", test_df.shape)
    
    print("\nTrain Data Head:")
    print(train_df.head())
    print("\nTest Data Head:")
    print(test_df.head())
    
    # Calculate label counts and percentages
    label_counts = train_df[label_cols].sum()
    label_percentages = (label_counts / len(train_df)) * 100
    
    # Create a DataFrame for plotting
    label_stats_df = pd.DataFrame({'count': label_counts, 'percentage': label_percentages})
    print("Label Distribution:")
    print(label_stats_df)
    
    # Plotting label distribution
    plt.figure(figsize=(12, 6))
    sns.barplot(x=label_stats_df.index, y=label_stats_df['count'])
    plt.title('Distribution of Toxicity Labels')
    plt.ylabel('Number of Comments')
    plt.xlabel('Toxicity Type')
    plt.xticks(rotation=45)
    plt.show()
    
    # Check for comments with no labels (clean comments)
    no_label_count = len(train_df[train_df[label_cols].sum(axis=1) == 0])
    print(f"\nNumber of comments with NO toxicity labels: {no_label_count} ({no_label_count / len(train_df) * 100:.2f}%)")
    
    # Multi-label Correlation
    plt.figure(figsize=(8, 6))
    sns.heatmap(train_df[label_cols].corr(), annot=True, cmap='coolwarm', fmt=".2f")
    plt.title('Correlation Matrix of Toxicity Labels')
    plt.show()
    
    # Comment length analysis
    train_df['comment_length'] = train_df['comment_text'].apply(lambda x: len(str(x).split()))
    test_df['comment_length'] = test_df['comment_text'].apply(lambda x: len(str(x).split()))
    
    plt.figure(figsize=(12, 6))
    sns.histplot(train_df['comment_length'], bins=100, kde=True)
    plt.title('Distribution of Comment Length (Words)')
    plt.xlabel('Number of Words')
    plt.ylabel('Frequency')
    plt.xlim(0, 500)  # Limit x-axis for better visualization
    plt.show()
    
    print("\nComment Length Statistics (Words):")
    print(train_df['comment_length'].describe())
    
    return train_df, test_df


def clean_text(text):
    text = str(text)
    # Remove excessive whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    return text


class ToxicCommentDataset(Dataset):
    def __init__(self, comments, labels, tokenizer, max_len):
        self.comments = comments
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len
    
    def __len__(self):
        return len(self.comments)
    
    def __getitem__(self, item):
        comment = str(self.comments[item])
        target = self.labels[item]
        
        encoding = self.tokenizer.encode_plus(
            comment,
            add_special_tokens=True,
            max_length=self.max_len,
            return_token_type_ids=False,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt',
        )
        
        return {
            'comment_text': comment,
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(target, dtype=torch.float)
        }


class TestCommentDataset(Dataset):
    def __init__(self, comments, tokenizer, max_len):
        self.comments = comments
        self.tokenizer = tokenizer
        self.max_len = max_len
    
    def __len__(self):
        return len(self.comments)
    
    def __getitem__(self, item):
        comment = str(self.comments[item])
        
        encoding = self.tokenizer.encode_plus(
            comment,
            add_special_tokens=True,
            max_length=self.max_len,
            return_token_type_ids=False,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt',
        )
        
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
        }


def train_epoch(model, data_loader, loss_fn, optimizer, device, scheduler):
    model.train()
    total_loss = 0
    num_batches = len(data_loader)
    
    for i, batch in enumerate(data_loader):
        # Move batch to device
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)
        
        # Clear previous gradients
        optimizer.zero_grad()
        
        # Forward pass
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        logits = outputs.logits
        
        # Calculate loss
        loss = loss_fn(logits, labels)
        total_loss += loss.item()
        
        # Backward pass
        loss.backward()
        
        # Clip gradients to prevent exploding gradients
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        # Update parameters
        optimizer.step()
        scheduler.step()
        
        # Print progress
        if (i + 1) % 100 == 0:
            print(f'  Batch {i + 1}/{num_batches} | Loss: {loss.item():.4f}')
    
    avg_train_loss = total_loss / num_batches
    print(f"\n  Average Training Loss: {avg_train_loss:.4f}")
    return avg_train_loss


def eval_model(model, data_loader, loss_fn, device, label_cols):
    model.eval()
    total_loss = 0
    all_preds = []
    all_labels = []
    num_batches = len(data_loader)
    
    with torch.no_grad():
        for batch in data_loader:
            # Move batch to device
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)
            
            # Forward pass
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask
            )
            logits = outputs.logits
            
            # Calculate loss
            loss = loss_fn(logits, labels)
            total_loss += loss.item()
            
            # Store predictions and true labels
            probs = torch.sigmoid(logits)
            all_preds.append(probs.cpu().numpy())
            all_labels.append(labels.cpu().numpy())
    
    avg_val_loss = total_loss / num_batches
    print(f"  Average Validation Loss: {avg_val_loss:.4f}")
    
    # Concatenate results from all batches
    all_preds = np.concatenate(all_preds, axis=0)
    all_labels = np.concatenate(all_labels, axis=0)
    
    # Calculate metrics
    roc_auc_scores = {}
    mean_roc_auc = 0
    try:
        # Calculate AUC for each label individually
        for i, label_name in enumerate(label_cols):
            # Check if both classes are present for the current label
            if len(np.unique(all_labels[:, i])) > 1:
                roc_auc_scores[label_name] = roc_auc_score(all_labels[:, i], all_preds[:, i])
            else:
                roc_auc_scores[label_name] = np.nan
        
        # Calculate mean AUC, ignoring NaNs
        mean_roc_auc = np.nanmean(list(roc_auc_scores.values()))
        print(f"  Mean ROC AUC: {mean_roc_auc:.4f}")
        print("  Individual ROC AUC Scores:")
        for name, score in roc_auc_scores.items():
            print(f"    {name}: {score:.4f}")
    except Exception as e:
        print(f"  Could not calculate ROC AUC: {e}")
    
    # Calculate Hamming Loss
    threshold = 0.5
    binary_preds = (all_preds > threshold).astype(int)
    hamming = hamming_loss(all_labels, binary_preds)
    print(f"  Hamming Loss: {hamming:.4f}")
    
    return avg_val_loss, mean_roc_auc, hamming


def predict(model, data_loader, device):
    model.eval()
    predictions = []
    print("\nGenerating predictions on test data...")
    
    with torch.no_grad():
        for batch in data_loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask
            )
            logits = outputs.logits
            probs = torch.sigmoid(logits)
            predictions.append(probs.cpu().numpy())
    
    return np.concatenate(predictions, axis=0)


def plot_training_history(history, epochs):
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.plot(range(1, epochs + 1), history['train_loss'], label='Train Loss')
    plt.plot(range(1, epochs + 1), history['val_loss'], label='Validation Loss')
    plt.title('Loss Over Epochs')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.plot(range(1, epochs + 1), history['val_roc_auc'], label='Validation Mean ROC AUC')
    plt.title('Mean ROC AUC Over Epochs')
    plt.xlabel('Epochs')
    plt.ylabel('Mean ROC AUC')
    plt.legend()
    
    plt.tight_layout()
    plt.show()


def main():
    # Setup device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Load data
    train_df, test_df, sample_submission_df = load_data()
    
    # Define target labels
    label_cols = ['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']
    
    # Explore data
    train_df, test_df = explore_data(train_df, test_df, label_cols)
    
    # Clean text
    train_df['comment_text_cleaned'] = train_df['comment_text'].apply(clean_text)
    test_df['comment_text_cleaned'] = test_df['comment_text'].apply(clean_text)
    
    # Initialize tokenizer
    tokenizer = BertTokenizer.from_pretrained(MODEL_NAME)
    
    # Example tokenization
    sample_text = "This is a sample comment for tokenization."
    tokens = tokenizer.encode_plus(
        sample_text,
        max_length=32,
        padding='max_length',
        truncation=True,
        return_tensors='pt'
    )
    
    print("\nSample Tokenization:")
    print(f"Text: {sample_text}")
    print(f"Tokens: {tokenizer.convert_ids_to_tokens(tokens['input_ids'][0])}")
    print(f"Input IDs: {tokens['input_ids']}")
    print(f"Attention Mask: {tokens['attention_mask']}")
    
    # Prepare data for Dataset class
    X = train_df['comment_text_cleaned'].values
    y = train_df[label_cols].values
    
    # Split data into training and validation sets
    X_train, X_val, y_train, y_val = train_test_split(
        X, y,
        test_size=0.1,
        random_state=SEED,
    )
    
    print(f"\nTrain size: {len(X_train)}, Validation size: {len(X_val)}")
    
    # Create Datasets
    train_dataset = ToxicCommentDataset(X_train, y_train, tokenizer, MAX_LENGTH)
    val_dataset = ToxicCommentDataset(X_val, y_val, tokenizer, MAX_LENGTH)
    
    # Create DataLoaders - use num_workers=0 to avoid multiprocessing issues
    train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    val_dataloader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
    
    # Check a sample batch
    data = next(iter(train_dataloader))
    print("\nSample batch shapes:")
    print("Input IDs:", data['input_ids'].shape)
    print("Attention Mask:", data['attention_mask'].shape)
    print("Labels:", data['labels'].shape)
    
    # Load model
    model = BertForSequenceClassification.from_pretrained(
        MODEL_NAME,
        num_labels=len(label_cols),
        output_attentions=False,
        output_hidden_states=False,
    )
    model.to(device)
    print("\nModel loaded successfully.")
    
    # Optimizer
    optimizer = AdamW(model.parameters(), lr=LEARNING_RATE, eps=1e-8)
    
    # Total number of training steps
    total_steps = len(train_dataloader) * EPOCHS
    
    # Learning rate scheduler
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=0,
        num_training_steps=total_steps
    )
    
    # Loss function
    loss_fn = nn.BCEWithLogitsLoss().to(device)
    
    # Training
    history = {'train_loss': [], 'val_loss': [], 'val_roc_auc': [], 'val_hamming': []}
    best_roc_auc = -1
    best_model_state = None
    
    print("\nStarting Training...")
    for epoch in range(EPOCHS):
        print(f'\n--- Epoch {epoch + 1}/{EPOCHS} ---')
        
        train_loss = train_epoch(
            model,
            train_dataloader,
            loss_fn,
            optimizer,
            device,
            scheduler
        )
        history['train_loss'].append(train_loss)
        
        print(f"\n--- Validation Epoch {epoch + 1} ---")
        val_loss, val_roc_auc, val_hamming = eval_model(
            model,
            val_dataloader,
            loss_fn,
            device,
            label_cols
        )
        history['val_loss'].append(val_loss)
        history['val_roc_auc'].append(val_roc_auc)
        history['val_hamming'].append(val_hamming)
        
        # Save the best model
        if val_roc_auc > best_roc_auc:
            best_roc_auc = val_roc_auc
            best_model_state = model.state_dict()
            torch.save(best_model_state, 'best_model_state.bin')
            print(f"  ** New best model saved with ROC AUC: {best_roc_auc:.4f} **")
    
    print("\nTraining Finished.")
    print(f"Best Validation ROC AUC: {best_roc_auc:.4f}")
    
    # Load the best model state for prediction
    if best_model_state:
        model.load_state_dict(best_model_state)
        print("Loaded best model state for prediction.")
    
    # Plot training history
    plot_training_history(history, EPOCHS)
    
    # Prepare test data
    test_texts = test_df['comment_text_cleaned'].values
    test_dataset = TestCommentDataset(test_texts, tokenizer, MAX_LENGTH)
    test_dataloader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
    
    # Generate predictions
    test_predictions = predict(model, test_dataloader, device)
    print("Predictions generated successfully.")
    print("Shape of predictions:", test_predictions.shape)
    
    # Create submission DataFrame
    submission_df = pd.DataFrame(test_predictions, columns=label_cols)
    submission_df['id'] = test_df['id']
    
    # Reorder columns to match sample_submission.csv format
    submission_df = submission_df[['id'] + label_cols]
    
    print("\nSubmission DataFrame Head:")
    print(submission_df.head())
    
    # Save to CSV
    submission_df.to_csv('submission.csv', index=False)
    print("\nSubmission file 'submission.csv' created successfully.")


if __name__ == '__main__':
    # This is the critical line that solves the multiprocessing issue
    torch.multiprocessing.freeze_support()
    main()