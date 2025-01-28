import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from sklearn.metrics import classification_report
import argparse
import os
import pickle


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
parser = argparse.ArgumentParser(description="Evaluate ClinicalBERT for ICD Code Prediction.")
parser.add_argument("--data_path", type=str, default="data/preprocessed_data.csv", help="Path to preprocessed data.")
parser.add_argument("--labels_path", type=str, default="data/encoded_labels.csv", help="Path to encoded labels.")
parser.add_argument("--model_dir", type=str, default="models", help="Directory of the trained model.")
parser.add_argument("--batch_size", type=int, default=8, help="Batch size for evaluation.")
args = parser.parse_args()

# Load preprocessed test data
data = pd.read_csv(args.data_path)
labels = pd.read_csv(args.labels_path).values

# Split data into test set (assuming 20% used for testing during training)
test_data = data.sample(frac=0.2, random_state=42)
test_labels = labels[test_data.index]

# Tokenize test data
tokenizer = AutoTokenizer.from_pretrained(args.model_dir)

def tokenize_texts(texts):
    return tokenizer(
        list(texts),
        max_length=128,
        padding="max_length",
        truncation=True,
        return_tensors="pt"
    )

test_encodings = tokenize_texts(test_data["clean_diagnosis"])

# Create DataLoader for the test set
test_dataset = NotesDataset(test_encodings, test_labels)
test_loader = DataLoader(test_dataset, batch_size=args.batch_size)

# Load trained model
model = AutoModelForSequenceClassification.from_pretrained(args.model_dir)
device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)

with open("data/mlb.pkl", "rb") as f:
    mlb = pickle.load(f)

# Evaluate the model
model.eval()
y_true = []
y_pred = []

with torch.no_grad():
    for batch in test_loader:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)

        outputs = model(input_ids, attention_mask=attention_mask)
        logits = outputs.logits
        predictions = (torch.sigmoid(logits) > 0.5).cpu().numpy()

        y_pred.extend(predictions)
        y_true.extend(labels.cpu().numpy())

# Calculate evaluation metrics
y_true = torch.tensor(y_true)
y_pred = torch.tensor(y_pred)

# Generate classification report
print("Classification Report:")
#print(classification_report(y_true, y_pred, target_names=tokenizer.convert_ids_to_tokens(range(labels.shape[1]))))
print(classification_report(y_true, y_pred, target_names=mlb.classes_))

