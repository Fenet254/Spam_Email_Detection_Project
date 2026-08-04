import re
import string
import joblib
import pandas as pd

from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, confusion_matrix, classification_report
)

DATA_PATH = "data/spam.csv"

STOPWORDS = set(stopwords.words("english"))
LEMMATIZER = WordNetLemmatizer()


def load_dataset(path=DATA_PATH):
    df = pd.read_csv(path, encoding="latin-1")
    df = df[["v1", "v2"]].rename(columns={"v1": "label", "v2": "message"})
    print(f"Loaded {len(df)} rows from {path}")
    return df


def explore_data(df):
    print(df["label"].value_counts())
    print(df.isnull().sum())
    df = df.drop_duplicates().reset_index(drop=True)
    print(f"Shape after dedup: {df.shape}")
    return df


def clean_text(text):
    text = text.lower()
    text = re.sub(f"[{re.escape(string.punctuation)}]", " ", text)
    text = re.sub(r"\d+", " ", text)
    text = re.sub(r"\s+", " ", text).strip()

    tokens = text.split()
    tokens = [LEMMATIZER.lemmatize(t) for t in tokens if t not in STOPWORDS and len(t) > 1]
    return " ".join(tokens)


def preprocess_dataframe(df):
    df["clean_message"] = df["message"].apply(clean_text)
    return df


def build_features(df):
    vectorizer = TfidfVectorizer(max_features=3000, ngram_range=(1, 2))
    X = vectorizer.fit_transform(df["clean_message"])
    y = df["label"].map({"ham": 0, "spam": 1})
    return X, y, vectorizer


def split_data(X, y):
    return train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)


def train_models(X_train, y_train):
    models = {
        "Multinomial Naive Bayes": MultinomialNB(),
        "Logistic Regression": LogisticRegression(max_iter=1000),
        "Linear SVM": LinearSVC(),
    }
    for model in models.values():
        model.fit(X_train, y_train)
    return models


def evaluate_models(models, X_test, y_test):
    results = {}
    for name, model in models.items():
        y_pred = model.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred)
        rec = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        cm = confusion_matrix(y_test, y_pred)

        results[name] = {"accuracy": acc, "precision": prec, "recall": rec, "f1": f1, "model": model}

        print(f"\n{name}")
        print(f"Accuracy: {acc:.4f}  Precision: {prec:.4f}  Recall: {rec:.4f}  F1: {f1:.4f}")
        print(cm)
        print(classification_report(y_test, y_pred, target_names=["ham", "spam"]))

    return results


def predict_message(model, vectorizer, message):
    cleaned = clean_text(message)
    vec = vectorizer.transform([cleaned])
    pred = model.predict(vec)[0]
    return "Spam" if pred == 1 else "Ham"


def main():
    df = load_dataset()
    df = explore_data(df)
    df = preprocess_dataframe(df)

    X, y, vectorizer = build_features(df)
    X_train, X_test, y_train, y_test = split_data(X, y)

    models = train_models(X_train, y_train)
    results = evaluate_models(models, X_test, y_test)

    best_name = max(results, key=lambda n: results[n]["f1"])
    best_model = results[best_name]["model"]
    print(f"\nBest model: {best_name} (F1={results[best_name]['f1']:.4f})")

    joblib.dump(best_model, "models/spam_model.pkl")
    joblib.dump(vectorizer, "models/vectorizer.pkl")
    print("Saved model and vectorizer to models/")


if __name__ == "__main__":
    main()