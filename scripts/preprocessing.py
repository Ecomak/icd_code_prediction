import sqlite3
import pandas as pd
import re
import pickle
from sklearn.preprocessing import MultiLabelBinarizer
import argparse

# Parse command-line arguments
parser = argparse.ArgumentParser(description="Preprocess MIMIC-III data.")
parser.add_argument("--db_path", type=str, default="data/MIMIC3_demo.db", help="Path to SQLite database.")
parser.add_argument("--output_dir", type=str, default="data", help="Directory to save preprocessed files.")
parser.add_argument("--top_n", type=int, default=10, help="Number of top ICD codes to include.")
args = parser.parse_args()

# Connect to SQLite database
conn = sqlite3.connect(args.db_path)

# Load ADMISSIONS and DIAGNOSES_ICD tables
admissions_df = pd.read_sql_query("SELECT HADM_ID, DIAGNOSIS FROM admissions", conn)
diagnoses_df = pd.read_sql_query("SELECT HADM_ID, ICD9_CODE FROM diagnoses_icd", conn)

conn.close()

merged_df = admissions_df.merge(diagnoses_df, on="hadm_id", how="inner")

# Preview the merged data
#print(merged_df.head())


def clean_text(text):
    # Convert to lowercase
    text = text.lower()
    # Remove special characters and digits
    text = re.sub(r"[^a-z\s]", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

# Apply cleaning to DIAGNOSIS column
merged_df["clean_diagnosis"] = merged_df["diagnosis"].apply(clean_text)

# Drop rows with missing or empty text
merged_df = merged_df.dropna(subset=["clean_diagnosis"])
merged_df = merged_df[merged_df["clean_diagnosis"] != ""]

# Find the top 10 most frequent ICD codes
top_icd_codes = merged_df["icd9_code"].value_counts().head(args.top_n).index

# Filter the dataset to include only these ICD codes
filtered_df = merged_df[merged_df["icd9_code"].isin(top_icd_codes)]


# Convert ICD9_CODE into lists (one code per admission in this case)
filtered_df["icd_list"] = filtered_df["icd9_code"].apply(lambda x: [x])

# Use MultiLabelBinarizer to encode ICD codes
mlb = MultiLabelBinarizer()
y = mlb.fit_transform(filtered_df["icd_list"])

# Save the MultiLabelBinarizer
with open("data/mlb.pkl", "wb") as f:
    pickle.dump(mlb, f)


# Preview encoded ICD codes
#print("Classes:", mlb.classes_)
#print("Encoded Labels Example:", y[:5])

# Save preprocessed data
output_dir = args.output_dir
filtered_df.to_csv(f"{output_dir}/preprocessed_data.csv", index=False)
pd.DataFrame(y, columns=mlb.classes_).to_csv(f"{output_dir}/encoded_labels.csv", index=False)
