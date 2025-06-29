import pandas as pd
import os
from text_cleaner import clean_tweet
from sklearn.feature_extraction.text import TfidfVectorizer
import joblib

data_path = "backend/data/semeval_intensity"
columns = ["ID", "Tweet", "Affect_Dimension", "Intensity_Score"]

splits = ["train", "dev", "test"]

for split in splits:
    txt_path = os.path.join(data_path, f"{split}.txt")
    df = pd.read_csv(txt_path, sep="\t", names=columns, skiprows=1)

    df["Cleaned_Tweet"] = df["Tweet"].apply(clean_tweet)
    
    # Save cleaned CSV
    csv_path = os.path.join(data_path, f"{split}_cleaned.csv")
    df.to_csv(csv_path, index=False)
    print(f"✅ Saved: {csv_path}")
