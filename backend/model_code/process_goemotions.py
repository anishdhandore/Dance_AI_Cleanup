import pandas as pd
import os

# Define the mapping from original emotions to new categories
EMOTION_MAPPING = {
    'Positive_Affect_Joy': [
        'joy', 'amusement', 'excitement', 'love', 'gratitude', 
        'optimism', 'pride', 'relief', 'admiration', 'approval', 
        'caring', 'desire'
    ],
    'Sadness_Low_Arousal_Negative': [
        'sadness', 'grief', 'disappointment', 'remorse', 'embarrassment'
    ],
    'Anger_High_Arousal_Negative': [
        'anger', 'annoyance', 'disapproval', 'disgust'
    ],
    'Fear_Anxiety': [
        'fear', 'nervousness'
    ],
    'Surprise_Epistemic': [
        'surprise', 'curiosity', 'confusion', 'realization'
    ],
    'Neutral_Cat': [
        'neutral'
    ]
}

# Get the directory of the current script (which is backend/model/)
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
# Navigate to the data directory (backend/data/goemotions/)
DATA_DIR = os.path.join(SCRIPT_DIR, "..", "data", "goemotions") 
INPUT_FILE = os.path.join(DATA_DIR, "go_emotions_dataset.csv")
OUTPUT_FILE = os.path.join(DATA_DIR, "go_emotions_6categories.csv")

def process_emotions(input_path, output_path, mapping):
    print(f"Loading dataset from {input_path}...")
    try:
        df = pd.read_csv(input_path)
    except FileNotFoundError:
        print(f"Error: Input file {input_path} not found.")
        return
    except Exception as e:
        print(f"Error loading CSV: {e}")
        return

    print("Processing emotions and mapping to new categories...")
    all_original_emotions = [emotion for sublist in mapping.values() for emotion in sublist]
    missing_cols = [col for col in all_original_emotions if col not in df.columns]
    if missing_cols:
        print(f"Error: The following emotion columns are missing in the input CSV: {missing_cols}")
        if 'emotion' in df.columns and all(label in df['emotion'].unique() for label in missing_cols):
            print("Found 'emotion' column. The dataset might be in a different format than expected (multi-label binary columns).")
        return

    # Create new category columns
    for new_category, original_emotions in mapping.items():
        # Ensure all original_emotions for this category exist before trying to sum
        existing_original_emotions = [emo for emo in original_emotions if emo in df.columns]
        if not existing_original_emotions:
            print(f"Warning: No columns found for new category '{new_category}'. Skipping.")
            df[new_category] = 0 # Initialize with 0 if no source columns found
            continue
        
        df[new_category] = df[existing_original_emotions].any(axis=1).astype(int)

    # Select relevant columns for the output: id, text, and the new categories
    output_columns = ['id', 'text'] + list(mapping.keys())
    
    # Check if 'id' and 'text' columns exist, if not, adapt gracefully or warn
    if 'id' not in df.columns:
        print("Warning: 'id' column not found in input. It will not be in the output.")
        output_columns.remove('id')
    if 'text' not in df.columns:
        print("Warning: 'text' column not found in input. It will not be in the output.")
        output_columns.remove('text')
        print("Error: 'text' column is essential for further processing. Aborting output generation.")
        return
        
    # Ensure all new category columns are actually in df before selecting
    final_output_columns = [col for col in output_columns if col in df.columns]

    try:
        df_output = df[final_output_columns]
        print(f"\nFirst 5 rows of the processed data with new categories:")
        print(df_output.head())

        print(f"\nSaving processed data to {output_path}...")
        df_output.to_csv(output_path, index=False)
        print("Processing complete. New CSV saved.")
    except KeyError as e:
        print(f"Error selecting columns for output: {e}. Available columns: {df.columns.tolist()}")
    except Exception as e:
        print(f"Error saving CSV: {e}")

if __name__ == "__main__":
    process_emotions(INPUT_FILE, OUTPUT_FILE, EMOTION_MAPPING) 