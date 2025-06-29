import pandas as pd
import os
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVR
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_absolute_error, mean_squared_error
from scipy.stats import pearsonr
import numpy as np
import joblib # For saving the model

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
BACKEND_DIR = os.path.dirname(SCRIPT_DIR)
DATA_DIR = os.path.join(BACKEND_DIR, "data", "semeval_intensity")

TRAIN_FILE = os.path.join(DATA_DIR, "train_cleaned.csv")
DEV_FILE = os.path.join(DATA_DIR, "dev_cleaned.csv")
TEST_FILE = os.path.join(DATA_DIR, "test_cleaned.csv")
MODEL_SAVE_PATH = os.path.join(SCRIPT_DIR, "final_svr_model.joblib") # Save in the same dir as script

def load_data(file_path):
    """Loads data from a CSV file."""
    try:
        df = pd.read_csv(file_path)
        df.dropna(subset=['Cleaned_Tweet'], inplace=True)
        return df
    except FileNotFoundError:
        print(f"Error: {file_path} not found.")
        return None
    except Exception as e:
        print(f"An error occurred while loading {file_path}: {e}")
        return None

def finalize_and_evaluate_svr(train_df, dev_df, test_df):
    """Trains the final SVR model on combined train+dev data and evaluates on the test set."""
    if train_df is None or dev_df is None or test_df is None:
        print("Training, development, or test data is missing. Aborting.")
        return

    # Combine training and development data for final model training
    combined_train_df = pd.concat([train_df, dev_df], ignore_index=True)
    print(f"\nCombined training data size: {len(combined_train_df)} samples")

    X_train_final = combined_train_df['Cleaned_Tweet']
    y_train_final = combined_train_df['Intensity_Score']
    
    X_test = test_df['Cleaned_Tweet']
    y_test = test_df['Intensity_Score']

    # Using the best parameters found by GridSearchCV earlier
    best_tfidf_params = {
        'max_df': 0.7,
        'min_df': 3,
        'ngram_range': (1, 2)
    }
    best_svr_params = {
        'C': 0.1,
        'epsilon': 0.01,
        'kernel': 'linear'
    }

    pipeline = Pipeline([
        ('tfidf', TfidfVectorizer(**best_tfidf_params)),
        ('svr', SVR(**best_svr_params))
    ])

    print("\nTraining final SVR model on combined train+dev data...")
    pipeline.fit(X_train_final, y_train_final)
    print("Final model training complete.")

    print("\nEvaluating final model on the test set...")
    predictions = pipeline.predict(X_test)

    mae = mean_absolute_error(y_test, predictions)
    mse = mean_squared_error(y_test, predictions)
    rmse = np.sqrt(mse)
    pearson_corr, _ = pearsonr(y_test, predictions)

    print(f"Test Set MAE: {mae:.4f}")
    print(f"Test Set RMSE: {rmse:.4f}")
    print(f"Test Set Pearson Correlation: {pearson_corr:.4f}")

    joblib.dump(pipeline, MODEL_SAVE_PATH)
    print(f"\nFinal model saved to {MODEL_SAVE_PATH}")

if __name__ == "__main__":
    print("Loading training data...")
    train_df = load_data(TRAIN_FILE)
    print("Loading development data...")
    dev_df = load_data(DEV_FILE)
    print("Loading test data...")
    test_df = load_data(TEST_FILE)

    if train_df is not None and dev_df is not None and test_df is not None:
        print(f"Training data loaded: {len(train_df)} samples")
        print(f"Development data loaded: {len(dev_df)} samples")
        print(f"Test data loaded: {len(test_df)} samples")
        finalize_and_evaluate_svr(train_df, dev_df, test_df)
    else:
        print("Could not proceed due to data loading errors.") 