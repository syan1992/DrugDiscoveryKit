import pandas as pd
import numpy as np
import torch
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from torch.utils.data import TensorDataset, DataLoader
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from tqdm import tqdm
import torch.nn as nn

tqdm.pandas()

# Tokenization function
def tokenize(string):
    encodings = tokenizer.encode_plus(
        string,
        add_special_tokens=True,
        truncation=True,
        padding="max_length",
        max_length=max_length,
        return_attention_mask=True
    )
    input_ids = encodings["input_ids"]
    attention_mask = encodings["attention_mask"]
    return input_ids, attention_mask

# Function to save test predictions to a CSV file
def save_test_predictions(true_labels, predictions, output_path):
    df = pd.DataFrame({"true_labels": true_labels, "predictions": predictions})
    df.to_csv(output_path, index=False)

# Function to convert data to PyTorch tensors
def get_tensor_data(data):
    input_ids_tensor = torch.tensor(data["input_ids"].tolist(), dtype=torch.int32)
    attention_mask_tensor = torch.tensor(data["attention_mask"].tolist(), dtype=torch.int32)
    #labels_tensor = torch.tensor(data["p_np"].tolist(), dtype=torch.float32)
    labels_tensor = torch.tensor(data["Class"].tolist(), dtype=torch.float32)
    return TensorDataset(input_ids_tensor, attention_mask_tensor, labels_tensor)

model_name = "DeepChem/ChemBERTa-77M-MLM"
model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=1)
tokenizer = AutoTokenizer.from_pretrained(model_name)
max_length = tokenizer.model_max_length

batch_size = 32
epochs = 50
torch.manual_seed(12345)

# Dataset parameters
dataset = 'bace'
for i in [1, 2, 3]:
    train_df = pd.read_csv(f'/home/UWO/ysun2443/code/trimol_dataset/{dataset}_scaffold_{i}/raw/train_{dataset}_{i}.csv')
    test_df = pd.read_csv(f'/home/UWO/ysun2443/code/trimol_dataset/{dataset}_scaffold_{i}/raw/test_{dataset}_{i}.csv')
    val_df = pd.read_csv(f'/home/UWO/ysun2443/code/trimol_dataset/{dataset}_scaffold_{i}/raw/valid_{dataset}_{i}.csv')

    # Tokenization
    train_df[["input_ids", "attention_mask"]] = train_df["mol"].progress_apply(lambda x: pd.Series(tokenize(x)))
    test_df[["input_ids", "attention_mask"]] = test_df["mol"].progress_apply(lambda x: pd.Series(tokenize(x)))
    val_df[["input_ids", "attention_mask"]] = val_df["mol"].progress_apply(lambda x: pd.Series(tokenize(x)))

    train_dataset = get_tensor_data(train_df)
    val_dataset = get_tensor_data(val_df)
    test_dataset = get_tensor_data(test_df)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)
    test_loader = DataLoader(test_dataset, batch_size=batch_size)

    criterion = torch.nn.BCEWithLogitsLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    best_val_auroc = 0  # Initialize best AUROC
    best_model_state = None

    for epoch in tqdm(range(epochs)):
        # Training loop
        model.train()
        total_train_loss = 0
        for batch in train_loader:
            optimizer.zero_grad(set_to_none=True)
            input_ids, attention_mask, labels = batch
            input_ids, attention_mask, labels = input_ids.to(device), attention_mask.to(device), labels.to(device)
            output_dict = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            predictions = output_dict.logits.squeeze(dim=1)
            loss = criterion(predictions, labels)
            loss.backward()
            optimizer.step()
            total_train_loss += loss.item()
        avg_train_loss = total_train_loss / len(train_loader)

        # Validation loop
        model.eval()
        val_predictions = []
        val_labels = []
        with torch.no_grad():
            for batch in val_loader:
                input_ids, attention_mask, labels = batch
                input_ids, attention_mask, labels = input_ids.to(device), attention_mask.to(device), labels.to(device)
                output_dict = model(input_ids, attention_mask=attention_mask, labels=labels)
                predictions = torch.sigmoid(output_dict.logits.squeeze(dim=1))
                val_predictions.extend(predictions.cpu().numpy())
                val_labels.extend(labels.cpu().numpy())
        val_auroc = roc_auc_score(val_labels, val_predictions)

        print(f"epoch {epoch + 1}: Train Loss {avg_train_loss:.4f}, Val AUROC {val_auroc:.4f}")

        # Track the best validation AUROC and save model state
        if val_auroc > best_val_auroc:
            best_val_auroc = val_auroc
            best_model_state = model.state_dict()

    # Load the best model state for testing
    model.load_state_dict(best_model_state)

    # Testing loop
    model.eval()
    test_predictions = []
    test_labels = []
    with torch.no_grad():
        for batch in test_loader:
            input_ids, attention_mask, labels = batch
            input_ids, attention_mask, labels = input_ids.to(device), attention_mask.to(device), labels.to(device)
            output_dict = model(input_ids, attention_mask=attention_mask, labels=labels)
            predictions = torch.sigmoid(output_dict.logits.squeeze(dim=1))
            test_predictions.extend(predictions.cpu().numpy())
            test_labels.extend(labels.cpu().numpy())
    test_auroc = roc_auc_score(test_labels, test_predictions)
    print(f"Test AUROC: {test_auroc:.4f}")

    # Save test predictions to CSV
    output_path = f"test_predictions_{dataset}_{i}.csv"
    save_test_predictions(test_labels, test_predictions, output_path)
    print(f"Test predictions saved to {output_path}")
