"""
OHCA Classifier Package
A reusable package for training and using OHCA classification models
"""

import pandas as pd
import numpy as np
import torch
import torch.nn.functional as F
from sklearn.model_selection import train_test_split
from sklearn.utils import compute_class_weight, resample
from sklearn.metrics import confusion_matrix, classification_report, precision_recall_fscore_support, accuracy_score
from transformers import (
    AutoTokenizer, AutoModelForSequenceClassification,
    Trainer, TrainingArguments, DataCollatorWithPadding, EarlyStoppingCallback
)
from datasets import Dataset
from tqdm import tqdm
import os
from typing import Optional, List, Dict, Tuple

class OHCAClassifier:
    """
    Out-of-Hospital Cardiac Arrest (OHCA) Classifier
    
    A BERT-based binary classifier for identifying OHCA cases in medical text.
    """
    
    def __init__(self, 
                 model_name: str = "microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract",
                 threshold: float = 0.3,
                 max_length: int = 512):
        """
        Initialize the OHCA Classifier
        
        Args:
            model_name: HuggingFace model name or path to trained model
            threshold: Classification threshold for OHCA prediction
            max_length: Maximum sequence length for tokenization
        """
        self.model_name = model_name
        self.threshold = threshold
        self.max_length = max_length
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Initialize tokenizer and model
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = None
        
    def prepare_data(self, 
                     df: pd.DataFrame,
                     text_column: str = 'clean_text',
                     label_column: str = 'predicted_ohca',
                     oversample_rare_classes: bool = True,
                     balance_final: bool = True) -> pd.DataFrame:
        """
        Prepare and balance dataset for training
        
        Args:
            df: Input dataframe
            text_column: Name of text column
            label_column: Name of label column (should contain 0,1,2,3 where 1=OHCA)
            oversample_rare_classes: Whether to oversample classes 2 and 3
            balance_final: Whether to balance OHCA vs non-OHCA classes
            
        Returns:
            Prepared dataframe with 'text', 'label', and 'is_transfer' columns
        """
        # Clean data
        df_clean = df[[text_column, label_column]].dropna().copy()
        
        # Create binary labels (1 = OHCA, 0 = everything else)
        df_clean['label'] = df_clean[label_column].apply(lambda x: 1 if x == 1 else 0)
        df_clean['is_transfer'] = df_clean[label_column].apply(lambda x: 1 if x == 2 else 0)
        df_clean['text'] = df_clean[text_column]
        
        # Add transfer patient prefix
        df_clean['text'] = df_clean.apply(
            lambda row: f"TRANSFERRED_PATIENT {row['text']}" 
            if row['label'] == 0 and row['is_transfer'] == 1 
            else row['text'],
            axis=1
        )
        
        if not oversample_rare_classes and not balance_final:
            return df_clean[['text', 'label', 'is_transfer']]
        
        # Split into subgroups
        non_ohca_0 = df_clean[df_clean[label_column] == 0]
        non_ohca_2 = df_clean[df_clean[label_column] == 2]  
        non_ohca_3 = df_clean[df_clean[label_column] == 3]  
        ohca_1 = df_clean[df_clean[label_column] == 1]
        
        print(f"Original class distribution:")
        print(f"Class 0 (non-OHCA): {len(non_ohca_0)}")
        print(f"Class 1 (OHCA): {len(ohca_1)}")
        print(f"Class 2 (transfer): {len(non_ohca_2)}")
        print(f"Class 3 (other): {len(non_ohca_3)}")
        
        # Oversample rare negatives
        if oversample_rare_classes and len(non_ohca_2) > 0:
            non_ohca_2 = resample(non_ohca_2, replace=True, n_samples=min(100, len(non_ohca_0)), random_state=42)
        if oversample_rare_classes and len(non_ohca_3) > 0:
            non_ohca_3 = resample(non_ohca_3, replace=True, n_samples=min(100, len(non_ohca_0)), random_state=42)
        
        # Combine all negatives
        df_balanced = pd.concat([non_ohca_0, non_ohca_2, non_ohca_3, ohca_1])
        
        # Balance OHCA vs non-OHCA
        if balance_final:
            minority = df_balanced[df_balanced['label'] == 1]
            majority = df_balanced[df_balanced['label'] == 0]
            
            if len(minority) < len(majority):
                minority_upsampled = resample(minority, replace=True, n_samples=len(majority), random_state=42)
                df_final = pd.concat([majority, minority_upsampled])
            else:
                df_final = df_balanced
        else:
            df_final = df_balanced
        
        print(f"Final class distribution:")
        print(f"Non-OHCA: {len(df_final[df_final['label'] == 0])}")
        print(f"OHCA: {len(df_final[df_final['label'] == 1])}")
        
        return df_final[['text', 'label', 'is_transfer']].reset_index(drop=True)
    
    def train(self,
              df: pd.DataFrame,
              output_dir: str,
              test_size: float = 0.3,
              learning_rate: float = 2e-5,
              batch_size: int = 8,
              num_epochs: int = 5,
              weight_decay: float = 0.01,
              early_stopping_patience: int = 2) -> Dict:
        """
        Train the OHCA classifier
        
        Args:
            df: Prepared dataframe with 'text' and 'label' columns
            output_dir: Directory to save the trained model
            test_size: Fraction of data to use for validation
            learning_rate: Learning rate for training
            batch_size: Training batch size
            num_epochs: Number of training epochs
            weight_decay: Weight decay for regularization
            early_stopping_patience: Patience for early stopping
            
        Returns:
            Dictionary with training metrics
        """
        print("Starting training...")
        
        # Split data
        train_df, val_df = train_test_split(
            df, test_size=test_size, stratify=df['label'], random_state=42
        )
        
        print(f"Training samples: {len(train_df)}")
        print(f"Validation samples: {len(val_df)}")
        
        # Compute class weights
        class_weights = compute_class_weight(
            class_weight='balanced', 
            classes=np.unique(train_df['label']), 
            y=train_df['label']
        )
        weights_tensor = torch.tensor(class_weights, dtype=torch.float)
        
        # Create datasets
        train_dataset = Dataset.from_pandas(train_df)
        val_dataset = Dataset.from_pandas(val_df)
        
        # Tokenize
        def tokenize(batch):
            encoding = self.tokenizer(
                batch['text'], 
                truncation=True, 
                padding="max_length", 
                max_length=self.max_length
            )
            encoding["label"] = batch["label"]
            return encoding
        
        train_dataset = train_dataset.map(tokenize, batched=True)
        val_dataset = val_dataset.map(tokenize, batched=True)
        
        # Load model
        self.model = AutoModelForSequenceClassification.from_pretrained(
            self.model_name, num_labels=2
        )
        
        # Custom loss with class weights
        def compute_loss(model, inputs, return_outputs=False):
            labels = inputs.pop("labels")
            outputs = model(**inputs)
            logits = outputs.logits
            loss_fn = torch.nn.CrossEntropyLoss(weight=weights_tensor.to(logits.device))
            loss = loss_fn(logits, labels)
            return (loss, outputs) if return_outputs else loss
        
        self.model.compute_loss = compute_loss
        
        # Metrics
        def compute_metrics(pred):
            labels = pred.label_ids
            preds = np.argmax(pred.predictions, axis=1)
            precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average='binary')
            acc = accuracy_score(labels, preds)
            return {
                'accuracy': acc,
                'precision': precision,
                'recall': recall,
                'f1': f1   
            }
        
        # Training arguments
        training_args = TrainingArguments(
            output_dir=output_dir,
            eval_strategy="epoch",
            save_strategy="epoch",
            load_best_model_at_end=True,
            metric_for_best_model="f1",
            greater_is_better=True,
            learning_rate=learning_rate,
            per_device_train_batch_size=batch_size,
            per_device_eval_batch_size=batch_size,
            num_train_epochs=num_epochs,
            weight_decay=weight_decay,
            logging_dir=f"{output_dir}/logs",
            seed=42,
        )
        
        # Trainer
        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            tokenizer=self.tokenizer,
            data_collator=DataCollatorWithPadding(tokenizer=self.tokenizer),
            callbacks=[EarlyStoppingCallback(early_stopping_patience=early_stopping_patience)],
            compute_metrics=compute_metrics
        )
        
        # Train
        train_result = trainer.train()
        
        # Save model
        self.model.save_pretrained(output_dir)
        self.tokenizer.save_pretrained(output_dir)
        
        print(f"✅ Model saved to {output_dir}")
        
        return {
            "train_loss": train_result.training_loss,
            "train_samples": len(train_df),
            "val_samples": len(val_df)
        }
    
    def load_model(self, model_path: str):
        """Load a trained model"""
        print(f"Loading model from {model_path}")
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_path)
        self.model.to(self.device)
        self.model.eval()
        print("✅ Model loaded successfully")
    
    def predict(self, texts: List[str], return_probabilities: bool = False) -> List[Dict]:
        """
        Predict OHCA for a list of texts
        
        Args:
            texts: List of medical texts to classify
            return_probabilities: Whether to return probability scores
            
        Returns:
            List of prediction dictionaries
        """
        if self.model is None:
            raise ValueError("Model not loaded. Call load_model() first.")
        
        results = []
        
        with torch.no_grad():
            for text in tqdm(texts, desc="Predicting"):
                # Tokenize
                inputs = self.tokenizer(
                    text, 
                    return_tensors="pt", 
                    truncation=True, 
                    padding="max_length", 
                    max_length=self.max_length
                )
                inputs = {k: v.to(self.device) for k, v in inputs.items()}
                
                # Predict
                outputs = self.model(**inputs)
                probs = F.softmax(outputs.logits, dim=1)
                
                ohca_prob = probs[0][1].item()
                prediction = 1 if ohca_prob >= self.threshold else 0
                
                result = {
                    "prediction": prediction,
                    "is_ohca": bool(prediction)
                }
                
                if return_probabilities:
                    result.update({
                        "ohca_probability": ohca_prob,
                        "non_ohca_probability": probs[0][0].item(),
                        "confidence": max(ohca_prob, probs[0][0].item())
                    })
                
                results.append(result)
        
        return results
    
    def evaluate(self, df: pd.DataFrame, text_column: str = 'text', label_column: str = 'label') -> Dict:
        """
        Evaluate model performance on a dataset
        
        Args:
            df: Dataframe with text and labels
            text_column: Name of text column
            label_column: Name of label column
            
        Returns:
            Dictionary with evaluation metrics
        """
        if self.model is None:
            raise ValueError("Model not loaded. Call load_model() first.")
        
        texts = df[text_column].tolist()
        true_labels = df[label_column].tolist()
        
        predictions = self.predict(texts)
        pred_labels = [p["prediction"] for p in predictions]
        
        # Calculate metrics
        precision, recall, f1, _ = precision_recall_fscore_support(true_labels, pred_labels, average='binary')
        accuracy = accuracy_score(true_labels, pred_labels)
        
        # Classification report
        report = classification_report(true_labels, pred_labels, target_names=["non-OHCA", "OHCA"])
        cm = confusion_matrix(true_labels, pred_labels)
        
        return {
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "classification_report": report,
            "confusion_matrix": cm.tolist()
        }

def quick_start_example():
    """Example of how to use the OHCA Classifier"""
    
    # Load your data
    # df = pd.read_excel("your_data.xlsx")
    
    # Initialize classifier
    classifier = OHCAClassifier()
    
    # Prepare data (adapt column names to your dataset)
    # prepared_df = classifier.prepare_data(df, text_column='your_text_column', label_column='your_label_column')
    
    # Train model
    # training_results = classifier.train(prepared_df, output_dir="./ohca_model")
    
    # Load trained model
    # classifier.load_model("./ohca_model")
    
    # Make predictions
    # sample_texts = ["Patient had cardiac arrest", "Routine discharge"]
    # predictions = classifier.predict(sample_texts, return_probabilities=True)
    
    # Evaluate on test set
    # eval_results = classifier.evaluate(test_df)
    
    print("See the README for complete usage examples!")

if __name__ == "__main__":
    quick_start_example()
