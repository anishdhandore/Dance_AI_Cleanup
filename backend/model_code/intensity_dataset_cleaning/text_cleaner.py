import re
import html
import string
import joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
import os

BASIC_STOPWORDS = {
    "the", "is", "a", "an", "and", "of", "to", "in", "on", "for", "with",
    "this", "that", "it", "as", "at", "by", "be", "are", "was", "were"
}

def clean_tweet(text, remove_stopwords=False):
    # Decode HTML entities
    text = html.unescape(text)

    # Lowercase
    text = text.lower()

    # Remove URLs
    text = re.sub(r"http\S+|www\S+|https\S+", "", text)

    # Remove @mentions and #hashtags
    text = re.sub(r"@\w+", "", text)
    text = re.sub(r"#\w+", "", text)

    # Remove punctuation and digits (keep emojis)
    text = re.sub(r"[{}0-9]".format(re.escape(string.punctuation)), "", text)

    # Remove extra whitespace
    words = text.strip().split()

    if remove_stopwords:
        words = [w for w in words if w not in BASIC_STOPWORDS]

    return " ".join(words)



