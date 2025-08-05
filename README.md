# OHCA Classifier

A BERT-based classifier for detecting Out-of-Hospital Cardiac Arrest (OHCA) cases in medical text. This package provides a complete solution for training and using OHCA classification models on your own datasets.

## What is this?

This classifier identifies OHCA (Out-of-Hospital Cardiac Arrest) cases from medical discharge summaries and clinical notes. It uses a fine-tuned BERT model (specifically PubMedBERT) to achieve high accuracy in medical text classification.

**Key Features:**
- Built on PubMedBERT for medical domain expertise
- Handles class imbalance with smart oversampling
- Special handling for transfer patients
- Easy to adapt to your own datasets
- Comprehensive evaluation tools

## Installation

### Option 1: Install from source
```bash
git clone https://github.com/monajm36/OHCA-classifier.git
cd OHCA-classifier
pip install -e .
```

### Option 2: Download and use directly
1. Download `ohca_classifier.py` 
2. Install requirements: `pip install -r requirements.txt`
3. Import in your code: `from ohca_classifier import OHCAClassifier`

## Quick Start

### Train on Your Data

```python
import pandas as pd
from ohca_classifier import OHCAClassifier

# 1. Load your data
df = pd.read_csv("your_medical_records.csv")

# 2. Initialize classifier  
classifier = OHCAClassifier()

# 3. Prepare data (update column names for your dataset)
prepared_df = classifier.prepare_data(
    df, 
    text_column='discharge_summary',  # <- Your text column name
    label_column='ohca_status'        # <- Your label column name
)

# 4. Train
classifier.train(prepared_df, output_dir="./my_ohca_model")
```

### Make Predictions

```python
# Load trained model
classifier = OHCAClassifier()
classifier.load_model("./my_ohca_model")

# Predict on new texts
texts = ["Patient had cardiac arrest in ED", "Routine discharge"]
predictions = classifier.predict(texts, return_probabilities=True)

for text, pred in zip(texts, predictions):
    print(f"Text: {text}")
    print(f"OHCA: {'YES' if pred['is_ohca'] else 'NO'} (confidence: {pred['confidence']:.3f})")
```

## Data Format Requirements

### Training Data
Your dataset should have these columns:

| Column | Description | Values |
|--------|-------------|--------|
| Text column | Medical text (discharge summaries, notes) | Free text |
| Label column | OHCA classification | 0=non-OHCA, 1=OHCA, 2=transfer, 3=other |

**Example:**
```csv
discharge_summary,ohca_status
"Patient presented with chest pain...",0
"Cardiac arrest in emergency department...",1
"Transferred from outside hospital...",2
```

### Label Meanings
- **0**: Non-OHCA cases (normal patients)
- **1**: OHCA cases (your target class) 
- **2**: Transfer patients (handled specially)
- **3**: Other/miscellaneous cases

## Advanced Usage

### Batch Processing Large Datasets

```python
# Process large CSV files efficiently
df = pd.read_csv("large_dataset.csv")
texts = df['medical_text'].tolist()

# Predict in batches
predictions = classifier.predict(texts, return_probabilities=True)

# Add results to dataframe
df['predicted_ohca'] = [p['prediction'] for p in predictions]
df['ohca_probability'] = [p['ohca_probability'] for p in predictions]
df.to_csv("results.csv", index=False)
```

### Model Evaluation

```python
# Evaluate on test set
test_df = pd.read_csv("test_data.csv")
results = classifier.evaluate(
    test_df, 
    text_column='medical_text',
    label_column='true_ohca'
)

print(f"F1 Score: {results['f1']:.3f}")
print(f"Precision: {results['precision']:.3f}")  
print(f"Recall: {results['recall']:.3f}")
```

## Adapting to Your Data

### Different Label Format

If your labels are different (e.g., text labels), map them first:

```python
# Your labels: 'normal', 'cardiac_arrest', 'transfer', 'other'
label_mapping = {
    'normal': 0,
    'cardiac_arrest': 1,  # This is your OHCA class
    'transfer': 2,
    'other': 3
}

df['mapped_labels'] = df['original_labels'].map(label_mapping)
prepared_df = classifier.prepare_data(df, label_column='mapped_labels')
```

### Binary Labels Only

If you only have OHCA vs non-OHCA:

```python
# Convert binary labels to expected format
df['label'] = df['is_ohca'].astype(int)  # True/False -> 1/0
df['text'] = df['medical_text']
df['is_transfer'] = 0  # No transfer info

# Skip complex preparation
prepared_df = df[['text', 'label', 'is_transfer']]
classifier.train(prepared_df, output_dir="./binary_model")
```

## Performance Tips

### Handling Imbalanced Data

The classifier automatically handles imbalanced datasets:

```python
prepared_df = classifier.prepare_data(
    df,
    text_column='your_text',
    label_column='your_labels',
    oversample_rare_classes=True,     # Boost rare negatives
    balance_final=True                # Balance OHCA vs non-OHCA
)
```

### GPU Acceleration

The classifier automatically uses GPU if available:

```python  
import torch
print(f"Using device: {torch.device('cuda' if torch.cuda.is_available() else 'cpu')}")
```

## Example Usage

See `example_usage.py` for a complete working example.

## FAQ

**Q: My data has different column names. What should I do?**
A: Just update the `text_column` and `label_column` parameters when calling `prepare_data()`.

**Q: Can I use this for other medical classification tasks?**
A: Yes! The architecture works for any binary medical text classification. Just change your labels and retrain.

**Q: What if I get out-of-memory errors?**
A: Reduce `batch_size` in training, use a smaller `max_length`, or switch to CPU training.

## Contributing

Contributions welcome! Please open an issue or submit a pull request.

## License

MIT License - see LICENSE file for details.

## Contact

- GitHub: [@monajm36](https://github.com/monajm36)
- Issues: [GitHub Issues](https://github.com/monajm36/OHCA-classifier/issues)

---

**Star this repo if it helps your research!**
