import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification, AdamW
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import argparse
import os

# Define Dataset Class
class NotesDataset(Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item["labels"] = torch.tensor(self.labels[idx])
        return item

# Parse command-line arguments
parser = argparse.ArgumentParser(description="Train ClinicalBERT for ICD Code Prediction.")
parser.add_argument("--data_path", type=str, default="data/preprocessed_data.csv", help="Path to preprocessed data.")
parser.add_argument("--labels_path", type=str, default="data/encoded_labels.csv", help="Path to encoded labels.")
parser.add_argument("--model_name", type=str, default="emilyalsentzer/Bio_ClinicalBERT", help="Name of the pre-trained model.")
parser.add_argument("--output_dir", type=str, default="models", help="Directory to save the trained model.")
parser.add_argument("--batch_size", type=int, default=8, help="Batch size for training.")
parser.add_argument("--epochs", type=int, default=3, help="Number of training epochs.")
parser.add_argument("--learning_rate", type=float, default=5e-5, help="Learning rate for the optimizer.")
args = parser.parse_args()

# Load preprocessed data
data = pd.read_csv(args.data_path)
labels = pd.read_csv(args.labels_path).values

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(
    data["clean_diagnosis"], labels, test_size=0.2, random_state=42
)

# Tokenize data
tokenizer = AutoTokenizer.from_pretrained(args.model_name)

def tokenize_texts(texts):
    return tokenizer(
        list(texts),
        max_length=128,
        padding="max_length",
        truncation=True,
        return_tensors="pt"
    )

train_encodings = tokenize_texts(X_train)
test_encodings = tokenize_texts(X_test)

# Create Datasets and DataLoaders
train_dataset = NotesDataset(train_encodings, y_train)
test_dataset = NotesDataset(test_encodings, y_test)

train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=args.batch_size)

# Load Pre-trained Model
model = AutoModelForSequenceClassification.from_pretrained(
    args.model_name,
    num_labels=y_train.shape[1]
)

# Move model to GPU if available
device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)

# Define Optimizer
optimizer = AdamW(model.parameters(), lr=args.learning_rate)

# Training Loop
for epoch in range(args.epochs):
    model.train()
    loop = tqdm(train_loader, leave=True)
    for batch in loop:
        optimizer.zero_grad()
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)

        outputs = model(input_ids, attention_mask=attention_mask, labels=labels.float())
        loss = outputs.loss
        loss.backward()
        optimizer.step()

        # Update progress bar
        loop.set_description(f"Epoch {epoch}")
        loop.set_postfix(loss=loss.item())

# Save the trained model
os.makedirs(args.output_dir, exist_ok=True)
model.save_pretrained(args.output_dir)
tokenizer.save_pretrained(args.output_dir)

print(f"Model saved to {args.output_dir}")
