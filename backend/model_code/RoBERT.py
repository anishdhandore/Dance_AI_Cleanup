import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, hamming_loss, jaccard_score
import torch
from torch.utils.data import Dataset
from transformers import (
    RobertaTokenizer,
    RobertaForSequenceClassification,
    Trainer,
    TrainingArguments,
    EarlyStoppingCallback
)

# Config
MODEL_NAME = 'roberta-base'
MAX_LEN = 128
BATCH_SIZE = 16
EPOCHS = 4
LEARNING_RATE = 2e-5

# Paths
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_FILE = os.path.join(SCRIPT_DIR, "..", "data", "goemotions", "go_emotions_6categories.csv")
MODEL_SAVE_PATH = os.path.join(SCRIPT_DIR, "..", "models", "roberta_goemotions_6cat")

os.makedirs(MODEL_SAVE_PATH, exist_ok=True)

class GoEmotionsDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_len=128):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = str(self.texts[idx])
        label = self.labels[idx]
        
        encoding = self.tokenizer(
            text,
            truncation=True,
            padding='max_length',
            max_length=self.max_len,
            return_tensors='pt'
        )
        
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(label, dtype=torch.float)
        }

def load_data():
    print("Loading data...")
    df = pd.read_csv(DATA_FILE)
    df.columns = df.columns.str.strip()  # Remove spaces
    print("CSV Columns:", df.columns.tolist())

    # Adjust to your CSV's actual columns if different
    label_columns = ['Positive_Affect_Joy', 'Sadness_Low_Arousal_Negative', 'Anger_High_Arousal_Negative', 'Fear_Anxiety', 'Surprise_Epistemic', 'Neutral_Cat']
    text_column = 'text'

    if not all(col in df.columns for col in label_columns):
        raise ValueError("❌ One or more label columns missing from CSV.")

    texts = df[text_column].values
    labels = df[label_columns].values

    train_texts, temp_texts, train_labels, temp_labels = train_test_split(
        texts, labels, test_size=0.3, random_state=42
    )
    val_texts, test_texts, val_labels, test_labels = train_test_split(
        temp_texts, temp_labels, test_size=0.5, random_state=42
    )

    print(f"✅ Data Split - Train: {len(train_texts)}, Val: {len(val_texts)}, Test: {len(test_texts)}")
    return train_texts, val_texts, test_texts, train_labels, val_labels, test_labels

def compute_metrics(pred):
    predictions, labels = pred
    predictions = (predictions > 0).astype(int)

    accuracy = accuracy_score(labels.flatten(), predictions.flatten())
    hamming = hamming_loss(labels.flatten(), predictions.flatten())
    jaccard = jaccard_score(labels, predictions, average='samples')

    return {
        'accuracy': accuracy,
        'hamming_loss': hamming,
        'jaccard_score': jaccard
    }

def main():
    train_texts, val_texts, test_texts, train_labels, val_labels, test_labels = load_data()

    tokenizer = RobertaTokenizer.from_pretrained(MODEL_NAME)
    model = RobertaForSequenceClassification.from_pretrained(
        MODEL_NAME,
        num_labels=6,
        problem_type="multi_label_classification"
    )

    train_dataset = GoEmotionsDataset(train_texts, train_labels, tokenizer, MAX_LEN)
    val_dataset = GoEmotionsDataset(val_texts, val_labels, tokenizer, MAX_LEN)
    test_dataset = GoEmotionsDataset(test_texts, test_labels, tokenizer, MAX_LEN)

    training_args = TrainingArguments(
        output_dir=MODEL_SAVE_PATH,
        num_train_epochs=EPOCHS,
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE,
        eval_strategy="epoch",
        save_strategy="epoch",
        learning_rate=LEARNING_RATE,
        logging_dir=os.path.join(MODEL_SAVE_PATH, "logs"),
        logging_steps=50,
        save_total_limit=2,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        report_to=[],
        fp16=torch.cuda.is_available()
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=2)]
    )

    print("🚀 Starting training...")
    trainer.train()

    print("💾 Saving model...")
    trainer.save_model(MODEL_SAVE_PATH)
    tokenizer.save_pretrained(MODEL_SAVE_PATH)

    print("📊 Evaluating on test set...")
    test_results = trainer.evaluate(test_dataset)
    print(f"Test Results: {test_results}")

    predictions = trainer.predict(test_dataset)
    pred_labels = (predictions.predictions > 0).astype(int)

    print("\n📝 Classification Report:")
    print(classification_report(test_labels, pred_labels,
                                target_names=['anger', 'fear', 'happy', 'sad', 'surprise', 'disgust']))

    print(f"\n✅ Model saved to: {MODEL_SAVE_PATH}")

if __name__ == "__main__":
    main()
