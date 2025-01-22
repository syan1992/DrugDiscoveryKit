import pandas as pd
import numpy as np
import torch
from sklearn.metrics import roc_auc_score
from torch.utils.data import TensorDataset, DataLoader
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from tqdm import tqdm
import torch.nn as nn
import pdb
tqdm.pandas()

'''
sider
column_names = [
    "Hepatobiliary disorders",
    "Metabolism and nutrition disorders",
    "Product issues",
    "Eye disorders",
    "Investigations",
    "Musculoskeletal and connective tissue disorders",
    "Gastrointestinal disorders",
    "Social circumstances",
    "Immune system disorders",
    "Reproductive system and breast disorders",
    "Neoplasms benign, malignant and unspecified (incl cysts and polyps)",
    "General disorders and administration site conditions",
    "Endocrine disorders",
    "Surgical and medical procedures",
    "Vascular disorders",
    "Blood and lymphatic system disorders",
    "Skin and subcutaneous tissue disorders",
    "Congenital, familial and genetic disorders",
    "Infections and infestations",
    "Respiratory, thoracic and mediastinal disorders",
    "Psychiatric disorders",
    "Renal and urinary disorders",
    "Pregnancy, puerperium and perinatal conditions",
    "Ear and labyrinth disorders",
    "Cardiac disorders",
    "Nervous system disorders",
    "Injury, poisoning and procedural complications"
]
'''
'''
column_names = [
        "FDA_APPROVED",
        "CT_TOX"
]
'''

column_names = [
            "NR-AR",
    "NR-AR-LBD",
    "NR-AhR",
    "NR-Aromatase",
    "NR-ER",
    "NR-ER-LBD",
    "NR-PPAR-gamma",
    "SR-ARE",
    "SR-ATAD5",
    "SR-HSE",
    "SR-MMP",
    "SR-p53"
]

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

# Function to convert data to PyTorch tensors
def get_tensor_data(data):
    input_ids_tensor = torch.tensor(data["input_ids"].tolist(), dtype=torch.int32)
    attention_mask_tensor = torch.tensor(data["attention_mask"].tolist(), dtype=torch.int32)
    labels_tensor = torch.tensor(data[column_names].values, dtype=torch.float32)
    return TensorDataset(input_ids_tensor, attention_mask_tensor, labels_tensor)

# Initialize model and tokenizer
model_name = "DeepChem/ChemBERTa-77M-MLM"
model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=len(column_names))
tokenizer = AutoTokenizer.from_pretrained(model_name)
max_length = tokenizer.model_max_length

batch_size = 32
epochs = 50
torch.manual_seed(12345)

# Dataset parameters
dataset = 'tox21'
for i in [1, 2, 3]:
    train_df = pd.read_csv(f'/home/UWO/ysun2443/code/trimol_dataset/{dataset}_{i}/raw/train_{dataset}_{i}.csv')
    test_df = pd.read_csv(f'/home/UWO/ysun2443/code/trimol_dataset/{dataset}_{i}/raw/test_{dataset}_{i}.csv')
    val_df = pd.read_csv(f'/home/UWO/ysun2443/code/trimol_dataset/{dataset}_{i}/raw/valid_{dataset}_{i}.csv')

    # Tokenization
    train_df[["input_ids", "attention_mask"]] = train_df["smiles"].progress_apply(
        lambda x: pd.Series(tokenize(x))
    )
    test_df[["input_ids", "attention_mask"]] = test_df["smiles"].progress_apply(
        lambda x: pd.Series(tokenize(x))
    )
    val_df[["input_ids", "attention_mask"]] = val_df["smiles"].progress_apply(
        lambda x: pd.Series(tokenize(x))
    )

    train_dataset = get_tensor_data(train_df)
    val_dataset = get_tensor_data(val_df)
    test_dataset = get_tensor_data(test_df)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)
    test_loader = DataLoader(test_dataset, batch_size=batch_size)

    # Loss and optimizer
    criterion = nn.BCEWithLogitsLoss()  # For multi-task classification
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    best_val_auroc = float('-inf')  # Initialize best AUROC as the minimum
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
            predictions = output_dict.logits
            loss = criterion(predictions, labels)
            loss.backward()
            optimizer.step()
            total_train_loss += loss.item()
        avg_train_loss = total_train_loss / len(train_loader)

        # Validation loop
        model.eval()
        val_predictions = []
        val_labels = []
        total_val_loss = 0
        with torch.no_grad():
            for batch in val_loader:
                input_ids, attention_mask, labels = batch
                input_ids, attention_mask, labels = input_ids.to(device), attention_mask.to(device), labels.to(device)
                output_dict = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
                predictions = torch.sigmoid(output_dict.logits)  # Apply sigmoid for probabilities
                loss = criterion(predictions, labels)
                total_val_loss += loss.item()
                val_predictions.extend(predictions.cpu().numpy())
                val_labels.extend(labels.cpu().numpy())
        avg_val_loss = total_val_loss / len(val_loader)

        # Calculate AUROC for the validation set

        val_predictions = np.array(val_predictions)
        val_labels = np.array(val_labels)

        val_auroc = []

        for task in range(len(column_names)):
            if not np.isnan(val_labels[:, task]).all():
                try:
                    auroc = roc_auc_score(val_labels[:, task], val_predictions[:, task])
                    val_auroc.append(auroc)
                except ValueError as e:
                    print(f"Task {column_names[task]} has invalid data for AUROC calculation.")
        val_auroc = np.mean(val_auroc) if val_auroc else 0
        print(f"epoch {epoch + 1}: Train Loss {avg_train_loss:.4f}, Val Loss {avg_val_loss:.4f}, Val AUROC {val_auroc:.4f}")

        # Track the best validation AUROC and save model state
        if val_auroc > best_val_auroc:
            best_val_auroc = val_auroc
            best_model_state = model.state_dict()

    # Load the best model state for testing
    model.load_state_dict(best_model_state)

    # Testing loop
    model.eval()
    total_test_loss = 0
    test_labels = []
    test_predictions = []
    with torch.no_grad():
        for batch in test_loader:
            input_ids, attention_mask, labels = batch
            input_ids, attention_mask, labels = input_ids.to(device), attention_mask.to(device), labels.to(device)
            output_dict = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            predictions = torch.sigmoid(output_dict.logits)  # Apply sigmoid for probabilities
            loss = criterion(predictions, labels)
            total_test_loss += loss.item()
            test_labels.extend(labels.cpu().numpy())
            test_predictions.extend(predictions.cpu().numpy())
    avg_test_loss = total_test_loss / len(test_loader)
    print(f"Test Loss {avg_test_loss:.4f}")

    # Save predictions and true labels to separate CSVs
    pred_output_path = f"test_predictions_{dataset}_{i}.csv"
    true_output_path = f"test_true_labels_{dataset}_{i}.csv"

    pd.DataFrame(test_predictions, columns=column_names).to_csv(pred_output_path, index=False)
    pd.DataFrame(test_labels, columns=column_names).to_csv(true_output_path, index=False)

    print(f"Test predictions saved to {pred_output_path}")
    print(f"True labels saved to {true_output_path}")
