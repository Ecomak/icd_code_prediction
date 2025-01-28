# ICD Code Prediction Using Clinical Notes

This project fine-tunes `Bio_ClinicalBERT` to predict ICD codes from clinical notes using the MIMIC-III dataset.


## Setup

1. **Clone the repository**:
   ```bash
   git clone https://github.com/yourusername/ICD_Code_Prediction.git
   cd ICD_Code_Prediction
2. **Install dependencies**:
    pip install -r requirements.txt
3. **Run preprocessing**:
    python scripts/preprocessing.py
4. **Train the model**:
    python scripts/train.py
5. **Evaluate the mdoel**:
    python scripts/evaluate.py
