import os
import pandas as pd
from sqlalchemy import create_engine

# Set up the database
db_name = "MIMIC3_demo.db"  
engine = create_engine(f"sqlite:///{db_name}")

# Directory containing the CSV files
data_directory = "physionet.org/files/mimiciii-demo/1.4/"

# List of all files in the directory
files = [f for f in os.listdir(data_directory) if f.endswith(".csv")]

for file in files:
    table_name = os.path.splitext(file)[0].lower()  # Use the file name (without extension) as the table name
    file_path = os.path.join(data_directory, file)
    
    # Read the CSV file
    print(f"Importing {file} into table '{table_name}'...")
    try:
        df = pd.read_csv(file_path, dtype=str)  # Read as strings to avoid parsing issues
        df.to_sql(table_name, engine, if_exists="replace", index=False)
        print(f"Table '{table_name}' created successfully.")
    except Exception as e:
        print(f"Error importing {file}: {e}")

print(f"All CSV files have been imported into the SQLite database '{db_name}'.")