import pandas as pd
import os
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier
from sklearn.pipeline import Pipeline
from sklearn.metrics import hamming_loss, accuracy_score, jaccard_score, classification_report
import joblib # Added for saving the model

# Get the directory of the current script (backend/model/)
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
# Path to the processed GoEmotions data (backend/data/goemotions/)
DATA_FILE = os.path.join(SCRIPT_DIR, "..", "data", "goemotions", "go_emotions_6categories.csv")

EMOTION_CATEGORIES = [
    'Positive_Affect_Joy',
    'Sadness_Low_Arousal_Negative',
    'Anger_High_Arousal_Negative',
    'Fear_Anxiety',
    'Surprise_Epistemic',
    'Neutral_Cat'
]

def load_and_split_data(file_path, label_columns, test_size=0.3, val_size_of_temp=0.5, random_state=42):
    """Loads data, checks for necessary columns, and splits it into train, validation, and test sets."""
    try:
        df = pd.read_csv(file_path)
    except FileNotFoundError:
        print(f"Error: Data file {file_path} not found.")
        return None, None, None, None, None, None
    except Exception as e:
        print(f"Error loading CSV: {e}")
        return None, None, None, None, None, None

    if 'text' not in df.columns:
        print("Error: 'text' column not found in the dataset.")
        return None, None, None, None, None, None
    
    for col in label_columns:
        if col not in df.columns:
            print(f"Error: Label column '{col}' not found in the dataset.")
            return None, None, None, None, None, None
            
    df.dropna(subset=['text'], inplace=True) # Ensure no NaN texts
    df[label_columns] = df[label_columns].fillna(0).astype(int) # Ensure labels are int and NaNs are 0

    X = df['text']
    y = df[label_columns]

    # Split into training (70%) and temp (30%)
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )
    # Split temp into validation (15% of total) and test (15% of total)
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=val_size_of_temp, random_state=random_state 
    )
    
    print(f"Training samples: {len(X_train)}")
    print(f"Validation samples: {len(X_val)}")
    print(f"Test samples: {len(X_test)}")

    return X_train, y_train, X_val, y_val, X_test, y_test

def train_and_evaluate_multilabel_classifier(X_train, y_train, X_val, y_val):
    """Trains a multi-label classifier using GridSearchCV and evaluates it on the validation set."""

    pipeline = Pipeline([
        ('tfidf', TfidfVectorizer(min_df=5, max_df=0.8)), # Initial min_df, max_df
        # Added class_weight='balanced' to LogisticRegression
        ('clf', OneVsRestClassifier(LogisticRegression(solver='liblinear', random_state=42, class_weight='balanced'), n_jobs=-1))
    ])

    param_grid = {
        'tfidf__ngram_range': [(1, 1), (1, 2)],
        'tfidf__max_features': [None, 10000, 20000], # Test with limited and full vocab
        'clf__estimator__C': [0.1, 1, 10] # Regularization for Logistic Regression
    }

    # Using f1_weighted for scoring as it handles label imbalance well for averaging.
    grid_search = GridSearchCV(pipeline, param_grid, cv=3, 
                               scoring='f1_weighted', 
                               verbose=2, n_jobs=-1) # Increased verbosity

    print("\nStarting GridSearchCV for multi-label classifier...")
    grid_search.fit(X_train, y_train)
    print("GridSearchCV training complete.")

    print("\nBest parameters found:")
    print(grid_search.best_params_)
    
    best_model = grid_search.best_estimator_

    print("\nEvaluating best model on validation set...")
    predictions = best_model.predict(X_val)

    subset_acc = accuracy_score(y_val, predictions)
    hamming = hamming_loss(y_val, predictions)
    jaccard_samp_avg = jaccard_score(y_val, predictions, average='samples')

    print(f"Subset Accuracy (Exact Match Ratio): {subset_acc:.4f}")
    print(f"Hamming Loss: {hamming:.4f}")
    print(f"Jaccard Score (Sample Average): {jaccard_samp_avg:.4f}")

    print("\nClassification Report (per label):")
    report = classification_report(y_val, predictions, target_names=EMOTION_CATEGORIES, zero_division=0)
    print(report)

    return best_model

if __name__ == "__main__":
    print("Loading and splitting GoEmotions data with 6 categories...")
    X_train, y_train, X_val, y_val, X_test, y_test = load_and_split_data(DATA_FILE, EMOTION_CATEGORIES)

    if X_train is not None:
        best_model_on_val = train_and_evaluate_multilabel_classifier(X_train, y_train, X_val, y_val)
        
        if best_model_on_val and X_test is not None and y_test is not None:
            print("\n\n=======================================================")
            print("PERFORMANCE ON FINAL TEST SET")
            print("=======================================================")
            print("\nEvaluating best model (from validation tuning) on the test set...")
            test_predictions = best_model_on_val.predict(X_test)
            
            test_subset_acc = accuracy_score(y_test, test_predictions)
            test_hamming = hamming_loss(y_test, test_predictions)
            test_jaccard_samp_avg = jaccard_score(y_test, test_predictions, average='samples', zero_division=0)

            print(f"Test Set Subset Accuracy (Exact Match Ratio): {test_subset_acc:.4f}")
            print(f"Test Set Hamming Loss: {test_hamming:.4f}")
            print(f"Test Set Jaccard Score (Sample Average): {test_jaccard_samp_avg:.4f}")

            print("\nTest Set Classification Report (per label):")
            test_report = classification_report(y_test, test_predictions, target_names=EMOTION_CATEGORIES, zero_division=0)
            print(test_report)
            
            # Save the model that was tuned on val and now tested.
            # Define the target models directory at the workspace root level
            workspace_root = os.path.abspath(os.path.join(SCRIPT_DIR, "..", ".."))
            models_dir = os.path.join(workspace_root, "models")
            os.makedirs(models_dir, exist_ok=True) # Ensure the directory exists
            model_save_path = os.path.join(models_dir, "goemotions_multilabel_classifier.joblib")
            joblib.dump(best_model_on_val, model_save_path)
            print(f"Model saved to {model_save_path}")
    else:
        print("Could not proceed due to data loading/splitting errors.") 