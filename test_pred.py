"""
Test script for the Random Forest prediction model.
This script tests the prediction code on the cleaned_data_combined.csv file.
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from collections import Counter
import time

# Import the prediction code
from example_pred import predict_all, process

def test_model():
    """
    Test the Random Forest prediction model on a small subset of the data.
    """
    print("Loading data...")
    # Load the full dataset
    df = pd.read_csv("cleaned_data_combined.csv")
    
    # Print basic info about the dataset
    print(f"Dataset shape: {df.shape}")
    print(f"Dataset columns: {df.columns.tolist()}")
    
    # Check the distribution of labels in the dataset
    if 'Label' in df.columns:
        print("\nLabel distribution:")
        label_counts = Counter(df['Label'])
        for label, count in label_counts.items():
            print(f"  {label}: {count} samples ({count/len(df):.1%})")
    
    # Split the data into train and test sets
    print("\nSplitting data into train/test sets...")
    X_train, X_test = train_test_split(df, test_size=0.2, random_state=42)
    
    # Save the test set to a temporary CSV file
    test_file = "temp_test_data.csv"
    X_test.to_csv(test_file, index=False)
    
    # Measure prediction time
    print(f"\nMaking predictions on {len(X_test)} samples...")
    start_time = time.time()
    predictions = predict_all(test_file)
    end_time = time.time()
    
    # Print prediction statistics
    print(f"Prediction completed in {end_time - start_time:.2f} seconds")
    print(f"Average time per sample: {(end_time - start_time) / len(X_test):.4f} seconds")
    
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
        
        # Create a confusion matrix
        labels = sorted(set(X_test['Label']))
        conf_matrix = {}
        for true_label in labels:
            conf_matrix[true_label] = {}
            for pred_label in labels:
                conf_matrix[true_label][pred_label] = 0
        
        for pred, true in zip(predictions, X_test['Label']):
            conf_matrix[true][pred] += 1
        
        # Print confusion matrix
        print("\nConfusion Matrix:")
        print("True \\ Pred", end="")
        for label in labels:
            print(f"\t{label}", end="")
        print()
        
        for true_label in labels:
            print(f"{true_label}", end="")
            for pred_label in labels:
                print(f"\t{conf_matrix[true_label][pred_label]}", end="")
            print()
    
    # Test on a few individual samples
    print("\nSample predictions:")
    for i in range(min(5, len(X_test))):
        sample = X_test.iloc[i]
        pred = predictions[i]
        print(f"Sample {i+1}:")
        
        # Use column position rather than names to avoid key errors
        print(f"  Complexity (Q1): {sample.iloc[1]}")  # Q1 is usually at position 1
        print(f"  Ingredients (Q2): {sample.iloc[2]}")  # Q2 is usually at position 2
        print(f"  Setting (Q3): {sample.iloc[3]}")  # Q3 is usually at position 3
        
        print(f"  Prediction: {pred}")
        if 'Label' in sample:
            print(f"  True label: {sample['Label']}")
        print()
    
    return predictions

if __name__ == "__main__":
    print("Testing Random Forest prediction model...")
    test_model()
    print("Testing complete.")