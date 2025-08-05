"""
Simple example of how to use the OHCA Classifier
"""

import pandas as pd
from ohca_classifier import OHCAClassifier

def main():
    # Example: Train on your own data
    print("=== OHCA Classifier Example ===")
    
    # 1. Load your data (replace with your file path)
    # df = pd.read_csv("your_medical_data.csv")
    
    # 2. Initialize classifier
    classifier = OHCAClassifier()
    
    # 3. Prepare data (update column names for your dataset)
    # prepared_df = classifier.prepare_data(
    #     df, 
    #     text_column='your_text_column',    # <- Change this
    #     label_column='your_label_column'   # <- Change this
    # )
    
    # 4. Train model
    # classifier.train(prepared_df, output_dir="./my_ohca_model")
    
    # 5. Load trained model and make predictions
    # classifier.load_model("./my_ohca_model")
    # predictions = classifier.predict(["Patient had cardiac arrest"])
    
    print("✅ See the comments above for usage instructions!")
    print("📚 Check the README.md for complete documentation")

if __name__ == "__main__":
    main()
