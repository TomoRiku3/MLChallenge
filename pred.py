"""
This Python file is example of how your `pred.py` script should
look. Your file should contain a function `predict_all` that takes
in the name of a CSV file, and returns a list of predictions.

Your `pred.py` script can use different methods to process the input
data, but the format of the input it takes and the output your script produces should be the same.

Here's an example of how your script may be used in our test file:

    from example_pred import predict_all
    predict_all("example_test_set.csv")
"""

# basic python imports are permitted
import sys
import csv
import json
import os

# numpy and pandas are also permitted
import numpy as np
import pandas as pd

# ========================================== GLOBAL VARIABLES ============================================

# Original question names - these are the expected full column names
FULL_Q1 = "Q1: From a scale 1 to 5, how complex is it to make this food? (Where 1 is the most simple, and 5 is the most complex)"
FULL_Q2 = "Q2: How many ingredients would you expect this food item to contain?"
FULL_Q3 = "Q3: In what setting would you expect this food to be served? Please check all that apply"
FULL_Q4 = "Q4: How much would you expect to pay for one serving of this food item?"
FULL_Q5 = "Q5: What movie do you think of when thinking of this food item?"
FULL_Q6 = "Q6: What drink would you pair with this food item?"
FULL_Q7 = "Q7: When you think about this food item, who does it remind you of?"
FULL_Q8 = "Q8: How much hot sauce would you add to this food item?"
LABEL = "Label"

# Shortened question names for internal use
Q1 = "Q1"
Q2 = "Q2"
Q3 = "Q3"
Q4 = "Q4"
Q5 = "Q5"
Q6 = "Q6"
Q7 = "Q7"
Q8 = "Q8"

# ======================== HELPER FUNCTIONS =======================================

def find_digits(s):
    """
    Returns a list of all contiguous digit sequences in the input string.
    """
    results = []  # list to store found digit sequences
    current = ""  # temporary string to accumulate digits

    for char in s:
        if char.isdigit():
            current += char  # accumulate digits
        else:
            if current:
                results.append(current)  # save the digit sequence when a non-digit is encountered
                current = ""  # reset for the next sequence

    # If the string ends with a digit, append the last sequence
    if current:
        results.append(current)

    return results


def find_numbers_no_regex(s):
    """
    Extracts contiguous digit sequences that may contain a single decimal point.
    """
    results = []        # List to store found number sequences
    current = ""        # Current number sequence being built
    dot_in_current = False  # Flag to track if the current sequence already has a decimal point

    for char in s:
        if char.isdigit():
            # Append digit to the current sequence
            current += char
        elif char == '.' and current and not dot_in_current:
            # Append a decimal point if one hasn't been added yet and the sequence already has a digit
            current += char
            dot_in_current = True
        else:
            # On encountering any non-valid character, store the current sequence (if any) and reset
            if current:
                results.append(current)
                current = ""
                dot_in_current = False

    # Append any remaining sequence after the loop ends
    if current:
        results.append(current)

    return results


class CustomMultiLabelBinarizer:
    """
    A custom implementation of MultiLabelBinarizer that transforms collections of
    labels to binary matrices and back.
    """
    
    def __init__(self, classes=None):
        """
        Initialize the binarizer.
        
        Parameters:
        - classes: Optional array of class labels known by the binarizer.
        """
        self.classes_ = classes
    
    def fit(self, y):
        """
        Fit the binarizer by determining all classes present in the input data.
        
        Parameters:
        - y: List of lists of labels to be binarized.
        
        Returns:
        - self
        """
        # Get unique classes across all lists
        self.classes_ = sorted(set().union(*[set(labels) for labels in y]))
        return self
    
    def fit_transform(self, y):
        """
        Fit the binarizer and transform the input data.
        
        Parameters:
        - y: List of lists of labels to be binarized.
        
        Returns:
        - Array of binary vectors for each sample.
        """
        self.fit(y)
        return self.transform(y)
    
    def transform(self, y):
        """
        Transform the input into binary indicators for each class.
        
        Parameters:
        - y: List of lists of labels to be binarized.
        
        Returns:
        - Binary matrix where each column represents a class.
        """
        # Initialize output array
        result = np.zeros((len(y), len(self.classes_)), dtype=int)
        
        # For each sample and its labels
        for i, labels in enumerate(y):
            # For each label in the sample
            for label in labels:
                if label in self.classes_:
                    # Set the corresponding cell to 1
                    j = self.classes_.index(label)
                    result[i, j] = 1
        
        return result


# ======================== DATA PROCESSING FUNCTIONS =======================================

def standardize_column_names(df):
    """
    Standardize column names in the DataFrame to use the short version (Q1, Q2, etc.)
    
    Parameters:
    - df: DataFrame with original column names
    
    Returns:
    - DataFrame with standardized column names
    """
    # Create a mapping of full column names to shortened versions
    column_map = {
        FULL_Q1: Q1,
        FULL_Q2: Q2,
        FULL_Q3: Q3,
        FULL_Q4: Q4,
        FULL_Q5: Q5,
        FULL_Q6: Q6,
        FULL_Q7: Q7,
        FULL_Q8: Q8
    }
    
    # Create a new mapping for the columns that are in the DataFrame
    actual_map = {}
    for full_col, short_col in column_map.items():
        # Try exact match first
        if full_col in df.columns:
            actual_map[full_col] = short_col
            continue
            
        # Try to find columns that start with "Qn:" 
        for col in df.columns:
            # Check if column matches the pattern "Qn:" where n is a digit
            if col.startswith(f"Q{short_col[1]}:") or col.startswith(f"Q{short_col[1]}: "):
                actual_map[col] = short_col
                break
    
    # Rename the columns that were found
    df = df.rename(columns=actual_map)
    
    return df


def process_q1(x):
    """
    Process the complexity question.
    """
    try:
        # Handle missing values
        if pd.isna(x):
            return 0
        
        x_str = str(x)
        # Extract numbers from the string
        numbers = find_digits(x_str)
        
        if numbers:
            return float(numbers[0])
        else:
            return 0
    except:
        return 0


def process_q2(x):
    """
    Process the number of ingredients question.
    """
    try:
        # Handle missing values
        if pd.isna(x):
            return 0  # Return default value for NaN input

        x_str = str(x)  # Convert input to string for processing

        # Case 1: Input contains commas, treat as a list
        if ',' in x_str:
            # Split by commas, remove extra whitespace, and ignore empty strings
            tokens = [t.strip() for t in x_str.split(',') if t.strip()]
            return float(len(tokens))  # Return count of valid items as float for consistency

        else:
            # Case 2: Extract numbers from the string using the helper function
            numbers = find_digits(x_str)

            if numbers:
                # If there's a hyphen, assume a range and compute the average
                if '-' in x_str:
                    nums = [float(num) for num in numbers]
                    return sum(nums) / len(nums)
                # Otherwise, return the first found number as a float
                return float(numbers[0])
            else:
                return 0.0  # No numbers found, return default value

    except Exception as e:
        # Return default value in case of error
        return 0.0


def process_q4(x):
    """
    Process the price question.
    """
    try:
        # Check if input is missing; return default value if so
        if pd.isna(x):
            return 0.0

        x_str = str(x)  # Convert input to a string for processing

        # Use the helper function to extract numbers (as strings) from the input string
        numbers_str = find_numbers_no_regex(x_str)

        if numbers_str:
            # Convert each extracted number to a float
            numbers = [float(num) for num in numbers_str]

            # If the input string indicates a range (using '-' or 'to'), return the average of all found numbers
            if '-' in x_str or 'to' in x_str:
                return sum(numbers) / len(numbers)

            # Otherwise, return the first number found
            return numbers[0]
        else:
            return 0.0  # Return default value if no numbers were found
    except Exception as e:
        # Return default value in case of any error
        return 0.0


def process_q3_q7(x):
    """
    Process the setting question.
    """
    try:
        # Return empty list if input is NaN or exactly zero
        if pd.isna(x) or x == 0:
            return []

        # Split the string by commas, strip whitespace, and ignore empty entries
        return [item.strip() for item in x.split(',') if item.strip()]
    except Exception as e:
        # Return empty list if any unexpected error occurs
        return []


def process_q8(x):
    """
    Process the hot sauce question.
    """
    try:
        # Handle explicit zero value
        if x == 0:
            return 0.0

        # Map specific text responses to numeric values
        elif x == "A little (mild)":
            return 1.0
        elif x == "A moderate amount (medium)":
            return 2.0
        elif x == "A lot (hot)":
            return 3.0
        elif x == "I will have some of this food item with my hot sauce":
            return 0.5

        # Default case for unknown input
        else:
            return 0.0
    except Exception as e:
        # Return default value in case of error
        return 0.0


def process_label(x):
    """
    Process the food label.
    """
    try:
        # Check for each specific food label and return corresponding code
        if x == "Pizza":
            return 1
        elif x == "Shawarma":  # Note: Fixed spelling from "Shwarama" to "Shawarma"
            return 2
        elif x == "Sushi":
            return 3
        else:
            return 0  # Return 0 for unknown labels
    except Exception as e:
        return 0  # Return 0 in case of any error


def process(A):
    """
    Process the data for model inference.
    """
    # First standardize column names
    A = standardize_column_names(A)
    
    # Q1: Complexity scale
    if Q1 in A.columns:
        A[Q1] = A[Q1].apply(process_q1).astype(float)
    else:
        print(f"Warning: {Q1} column not found in data")
        A[Q1] = 0.0  # Default value
    
    # Q2: Number of ingredients
    if Q2 in A.columns:
        A[Q2] = A[Q2].apply(process_q2).astype(float)
    else:
        print(f"Warning: {Q2} column not found in data")
        A[Q2] = 0.0  # Default value

    # Q3: Expected setting → list
    if Q3 in A.columns:
        A["Q3_list"] = A[Q3].apply(process_q3_q7)
    else:
        print(f"Warning: {Q3} column not found in data")
        A["Q3_list"] = [[]]  # Empty list for each row

    # Q4: Expected price
    if Q4 in A.columns:
        A[Q4] = A[Q4].apply(process_q4).astype(float)
    else:
        print(f"Warning: {Q4} column not found in data")
        A[Q4] = 0.0  # Default value

    # Q7: Reminds you of who → list
    if Q7 in A.columns:
        A["Q7_list"] = A[Q7].apply(process_q3_q7)
    else:
        print(f"Warning: {Q7} column not found in data")
        A["Q7_list"] = [[]]  # Empty list for each row

    # Q8: Hot sauce level
    if Q8 in A.columns:
        A[Q8] = A[Q8].apply(process_q8).astype(float)
    else:
        print(f"Warning: {Q8} column not found in data")
        A[Q8] = 0.0  # Default value

    # label: Food type - Only process if LABEL column exists (it won't in test data)
    if LABEL in A.columns:
        A[LABEL] = A[LABEL].apply(process_label)

    # Drop Q5 and Q6
    A = A.drop(columns=[Q5, Q6, 'id'], errors="ignore")

    # Multi-label binarize Q3 list using custom implementation
    mlb_q3 = CustomMultiLabelBinarizer()
    q3_encoded = mlb_q3.fit_transform(A["Q3_list"])
    df_q3 = pd.DataFrame(q3_encoded, columns=[f"Q3_{col}" for col in mlb_q3.classes_], index=A.index)
    A = pd.concat([A, df_q3], axis=1)
    A.drop(columns=["Q3_list"], inplace=True)

    # Multi-label binarize Q7 list using custom implementation
    mlb_q7 = CustomMultiLabelBinarizer()
    q7_encoded = mlb_q7.fit_transform(A["Q7_list"])
    df_q7 = pd.DataFrame(q7_encoded, columns=[f"Q7_{col}" for col in mlb_q7.classes_], index=A.index)
    A = pd.concat([A, df_q7], axis=1)
    A.drop(columns=["Q7_list"], inplace=True)

    # Ensure all numerical columns are float type
    for col in A.columns:
        if col.startswith('Q') and len(col) == 2:  # Basic question columns (Q1-Q8)
            A[col] = pd.to_numeric(A[col], errors='coerce').fillna(0.0)
        elif col.startswith('Q3_') or col.startswith('Q7_'):  # Binarized columns
            A[col] = A[col].astype(float)
    
    # Print column types for debugging
    # print("Column types after processing:")
    # print(A.dtypes.head())
    
    return A


# ======================== RANDOM FOREST FUNCTIONS =======================================

def load_forest_structure(json_path):
    """
    Load the Random Forest structure from a JSON file.
    """
    try:
        with open(json_path, 'r') as f:
            forest_structure = json.load(f)
        return forest_structure
    except Exception as e:
        # If there's an error, print it and return None
        print(f"Error loading forest structure: {e}")
        return None


def manual_predict_single_tree(x, tree):
    """
    Make a prediction using a single tree from the Random Forest.
    """
    # Start at the root node (node 0)
    node_id = 0
    
    # Convert pandas Series to numpy array if needed
    if isinstance(x, pd.Series):
        x = x.to_numpy()
    
    # Traverse the tree until reaching a leaf node
    try:
        while tree[str(node_id)]['type'] != 'leaf':
            # Get the current node
            node = tree[str(node_id)]
            
            # Get the feature index and threshold
            feature_idx = node['feature_index']
            threshold = node['threshold']
            
            # Ensure the feature value is numeric
            feature_value = float(x[feature_idx])
            
            # Compare the feature value with the threshold
            if feature_value <= threshold:
                node_id = node['left_child']
            else:
                node_id = node['right_child']
        
        # Get the predicted class from the leaf node
        leaf_node = tree[str(node_id)]
        return leaf_node['predicted_class']
    except Exception as e:
        # If there's an error in traversing the tree, print a detailed message
        print(f"Error in tree traversal: {e}")
        print(f"Node ID: {node_id}, Feature Index: {feature_idx if 'feature_idx' in locals() else 'N/A'}")
        print(f"Feature value: {x[feature_idx] if 'feature_idx' in locals() else 'N/A'}")
        print(f"Feature value type: {type(x[feature_idx]) if 'feature_idx' in locals() else 'N/A'}")
        print(f"Threshold: {threshold if 'threshold' in locals() else 'N/A'}")
        return 0  # Default prediction in case of error


def manual_predict(x, forest_structure):
    """
    Make a prediction using the entire Random Forest.
    """
    # Get predictions from each tree
    tree_predictions = []
    
    for i, tree in enumerate(forest_structure['trees']):
        try:
            prediction = manual_predict_single_tree(x, tree)
            tree_predictions.append(prediction)
        except Exception as e:
            print(f"Error in tree {i}: {e}")
            tree_predictions.append(0)  # Default prediction in case of error
    
    # Use majority voting to get the final prediction
    if not tree_predictions:
        return "Pizza"  # Default prediction if all trees failed
    
    unique_classes, counts = np.unique(tree_predictions, return_counts=True)
    final_prediction = unique_classes[np.argmax(counts)]
    
    # Map the numerical prediction back to the original class
    predicted_class = forest_structure['classes'][final_prediction]
    
    # Convert numerical prediction to label
    if predicted_class == 1:
        return "Pizza"
    elif predicted_class == 2:
        return "Shawarma"
    elif predicted_class == 3:
        return "Sushi"
    else:
        # Default case (should not happen)
        return "Pizza"


# ======================== MAIN CODE =======================================

# Global variable to store the loaded forest structure
FOREST_STRUCTURE = None

def predict(x, processed_features):
    """
    Make a prediction for a single sample using the Random Forest model.
    """
    global FOREST_STRUCTURE
    
    # If forest structure is not loaded yet, load it
    if FOREST_STRUCTURE is None:
        forest_path = "random_forest_structure.json"
        FOREST_STRUCTURE = load_forest_structure(forest_path)
        
        # If loading failed, return a default prediction
        if FOREST_STRUCTURE is None:
            return "Pizza"  # Default prediction
    
    # Make prediction using the Random Forest
    return manual_predict(processed_features, FOREST_STRUCTURE)


def predict_all(filename):
    """
    Make predictions for all samples in the given file.
    """
    # Print debug information
    # print(f"Reading file: {filename}")
    
    # Read the CSV file into a pandas DataFrame
    try:
        df = pd.read_csv(filename)
        # print(f"File read successfully. Shape: {df.shape}")
        # print(f"Columns: {df.columns.tolist()}")
    except Exception as e:
        print(f"Error reading file {filename}: {e}")
        return []
    
    # Process the data using the provided processing function
    try:
        processed_df = process(df)
        print(f"Data processed successfully. Shape: {processed_df.shape}")
    except Exception as e:
        print(f"Error processing data: {e}")
        import traceback
        traceback.print_exc()
        return []
    
    # Drop the label column if it exists (it should not exist in test data)
    if LABEL in processed_df.columns:
        processed_df = processed_df.drop(columns=[LABEL])
    
    # Initialize list to store predictions
    predictions = []
    
    # Make prediction for each sample
    for idx, row in df.iterrows():
        try:
            # Get processed features for this sample
            processed_features = processed_df.iloc[idx]
            
            # Make prediction
            pred = predict(row, processed_features)
            
            # Add prediction to list
            predictions.append(pred)
        except Exception as e:
            print(f"Error predicting for sample {idx}: {e}")
            predictions.append("Pizza")  # Default prediction in case of error
    
    print(f"Made {len(predictions)} predictions")
    return predictions
