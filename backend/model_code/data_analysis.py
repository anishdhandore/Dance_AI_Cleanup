import pandas as pd
import os

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
BACKEND_DIR = os.path.dirname(SCRIPT_DIR)
DATA_DIR = os.path.join(BACKEND_DIR, "data", "semeval_intensity")

def analyze_dataframe(df_name, df_path):
    print(f"Analysis for {df_name} ({df_path}):")
    try:
        df = pd.read_csv(df_path)

        print("\nIntensity_Score Distribution:")
        print(df['Intensity_Score'].describe())

        print("\nMissing Values per Column:")
        print(df.isnull().sum())

        print("\n" + "="*50 + "\n")

    except FileNotFoundError:
        print(f"Error: {df_path} not found.")
    except Exception as e:
        print(f"An error occurred while analyzing {df_path}: {e}")

if __name__ == "__main__":
    datasets = {
        "Train": os.path.join(DATA_DIR, "train_cleaned.csv"),
        "Dev": os.path.join(DATA_DIR, "dev_cleaned.csv"),
        "Test": os.path.join(DATA_DIR, "test_cleaned.csv")
    }

    for name, path in datasets.items():
        analyze_dataframe(name, path) 