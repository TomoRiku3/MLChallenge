"""
Test script for the Random Forest prediction model.
This script tests the prediction code on the cleaned_data_combined.csv file.
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from collections import Counter
import time
import random

# Import the prediction code
from pred import predict_all, process

# RANDOM SEED
SEED = 42

def test_model():
    """
    Test the Random Forest prediction model on a small subset of the data.
    """
    # Load the full dataset
    df = pd.read_csv("cleaned_data_combined.csv")
    
    
    # Split the data into train and test sets
    X_train, X_test = train_test_split(df, test_size=0.2, random_state=SEED)
    
    # Save the test set to a temporary CSV file
    test_file = "temp_test_data.csv"
    X_test.to_csv(test_file, index=False)
    
    # Measure prediction time
    predictions = predict_all(test_file)
    
    
    # Count the number of predictions for each class
    pred_counts = Counter(predictions)
    print("\nPrediction distribution:")
    for label, count in pred_counts.items():
        print(f"  {label}: {count} predictions ({count/len(predictions):.1%})")
    
    # If the test data has labels, calculate accuracy
    if 'Label' in X_test.columns:
        correct = sum(pred == true for pred, true in zip(predictions, X_test['Label']))
        accuracy = correct / len(X_test)
        print(f"\nAccuracy: {accuracy:.2%} ({correct}/{len(X_test)})")
     
    return predictions, accuracy

if __name__ == "__main__":

    print("Testing Random Forest prediction model...")
    results = []
    for i in range(20):
        # Set a new random SEED value
        SEED = random.randint(1, 10000)
        result = test_model()
    
        # Run the test
        result, accuracy = test_model()
    
        # Store the result
        results.append({
            "seed": SEED,
            "result": accuracy
        })



    # Output summary
    print("All tests completed.")
    for r in results:
        print(f"SEED: {r['seed']}, Result: {r['result']}")
        print("Testing complete.")