"""
Spam Email Detection - Full Training Pipeline
Merges data/train/spam.csv + data/test/emails.csv, trains, and saves model.
"""

import pandas as pd
import numpy as np
import re
import string
import pickle
import os

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, confusion_matrix, classification_report
)

import nltk
from nltk.corpus import stopwords

# ---------------------------------------------------------------------------
# Setup: download NLTK stopwords if missing
# ---------------------------------------------------------------------------
try:
    stop_words = set(stopwords.words('english'))
except LookupError:
    nltk.download('stopwords')
    stop_words = set(stopwords.words('english'))


# ---------------------------------------------------------------------------
# STEP 1 & 2: Load and standardize both datasets
# ---------------------------------------------------------------------------
def load_and_standardize(path):
    """
    Loads a CSV and standardizes it to two columns: 'label', 'text'.
    Handles common spam-dataset formats automatically.
    """
    # try a couple encodings — SMS spam collection is often latin-1
    try:
        df = pd.read_csv(path, encoding='utf-8')
    except UnicodeDecodeError:
        df = pd.read_csv(path, encoding='latin-1')

    df.columns = [c.strip().lower() for c in df.columns]

    # Map of possible column name variants -> standard name
    label_candidates = ['label', 'category', 'v1', 'class', 'spam']
    text_candidates = ['text', 'message', 'v2', 'email', 'body', 'content']

    label_col = next((c for c in label_candidates if c in df.columns), None)
    text_col = next((c for c in text_candidates if c in df.columns), None)

    if label_col is None or text_col is None:
        raise ValueError(
            f"Could not auto-detect label/text columns in {path}. "
            f"Found columns: {list(df.columns)}. "
            f"Please rename manually to 'label' and 'text'."
        )

    df = df[[label_col, text_col]].copy()
    df.columns = ['label', 'text']

    # Standardize label values -> 'spam' / 'ham'
    df['label'] = df['label'].astype(str).str.strip().str.lower()
    df['label'] = df['label'].replace({
        '1': 'spam', '0': 'ham',
        'true': 'spam', 'false': 'ham',
        'yes': 'spam', 'no': 'ham'
    })

    return df


DATA_DIR = 'data'
TRAIN_PATH = os.path.join(DATA_DIR, 'train', 'spam.csv')
TEST_PATH = os.path.join(DATA_DIR, 'test', 'emails.csv')

print("Loading datasets...")
df1 = load_and_standardize(TRAIN_PATH)
df2 = load_and_standardize(TEST_PATH)

print(f"  spam.csv:   {df1.shape[0]} rows")
print(f"  emails.csv: {df2.shape[0]} rows")

# ---------------------------------------------------------------------------
# Merge the two datasets into one
# ---------------------------------------------------------------------------
df = pd.concat([df1, df2], ignore_index=True)
print(f"Merged dataset: {df.shape[0]} rows")


# ---------------------------------------------------------------------------
# STEP 3: Explore the Data (EDA)
# ---------------------------------------------------------------------------
print("\n--- EDA ---")
print("Label distribution:\n", df['label'].value_counts())
print("Missing values:\n", df.isnull().sum())

df = df.dropna(subset=['text', 'label'])
df = df[df['label'].isin(['spam', 'ham'])]  # drop any unrecognized labels

before = len(df)
df = df.drop_duplicates(subset=['text'])
print(f"Removed {before - len(df)} duplicate rows")

print("Final label distribution:\n", df['label'].value_counts())


# ---------------------------------------------------------------------------
# STEP 4: Preprocess the Text
# ---------------------------------------------------------------------------
def clean_text(text):
    text = str(text).lower()
    text = re.sub(r'http\S+|www\S+', '', text)          # remove URLs
    text = re.sub(r'\d+', '', text)                      # remove numbers
    text = text.translate(str.maketrans('', '', string.punctuation))  # punctuation
    tokens = text.split()
    tokens = [w for w in tokens if w not in stop_words]  # stopwords
    return ' '.join(tokens)


print("\nCleaning text...")
df['clean_text'] = df['text'].apply(clean_text)
df = df[df['clean_text'].str.strip() != '']  # drop empty after cleaning


# ---------------------------------------------------------------------------
# STEP 5: Convert Text into Numerical Features (TF-IDF)
# ---------------------------------------------------------------------------
X = df['clean_text']
y = df['label'].map({'ham': 0, 'spam': 1})

vectorizer = TfidfVectorizer(max_features=5000, ngram_range=(1, 2))
X_vec = vectorizer.fit_transform(X)


# ---------------------------------------------------------------------------
# STEP 6: Split the Dataset
# ---------------------------------------------------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X_vec, y, test_size=0.2, random_state=42, stratify=y
)


# ---------------------------------------------------------------------------
# STEP 7: Train the Model
# ---------------------------------------------------------------------------
print("\nTraining model...")
model = MultinomialNB()
model.fit(X_train, y_train)

# Alternatives you can swap in:
# model = LogisticRegression(max_iter=1000)
# model = LinearSVC()


# ---------------------------------------------------------------------------
# STEP 8: Evaluate the Model
# ---------------------------------------------------------------------------
y_pred = model.predict(X_test)

print("\n--- Evaluation ---")
print("Accuracy: ", accuracy_score(y_test, y_pred))
print("Precision:", precision_score(y_test, y_pred))
print("Recall:   ", recall_score(y_test, y_pred))
print("F1-Score: ", f1_score(y_test, y_pred))
print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))


# ---------------------------------------------------------------------------
# STEP 9: Predict New Messages (example)
# ---------------------------------------------------------------------------
def predict_message(msg):
    cleaned = clean_text(msg)
    vec = vectorizer.transform([cleaned])
    pred = model.predict(vec)[0]
    return 'spam' if pred == 1 else 'ham'


sample = "Congratulations! You have won a free prize. Click here to claim now!"
print(f"\nSample prediction: '{sample}' -> {predict_message(sample)}")


# ---------------------------------------------------------------------------
# STEP 10: Save the Model
# ---------------------------------------------------------------------------
os.makedirs('models', exist_ok=True)

with open('models/spam_model.pkl', 'wb') as f:
    pickle.dump(model, f)

with open('models/vectorizer.pkl', 'wb') as f:
    pickle.dump(vectorizer, f)

print("\nModel and vectorizer saved to models/")