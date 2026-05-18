"""
classifier/train.py
Trains a TF-IDF + Logistic Regression classifier on IT support tickets.
"""

import argparse
import json
import joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.pipeline import Pipeline


def train(data_path: str, model_path: str = "classifier/model.pkl"):
    df = pd.read_csv(data_path)
    assert "text" in df.columns and "category" in df.columns, (
        "CSV must have 'text' and 'category' columns"
    )

    X_train, X_test, y_train, y_test = train_test_split(
        df["text"], df["category"], test_size=0.2, random_state=42, stratify=df["category"]
    )

    pipeline = Pipeline([
        ("tfidf", TfidfVectorizer(
            ngram_range=(1, 2),
            max_features=10_000,
            sublinear_tf=True,
        )),
        ("clf", LogisticRegression(
            max_iter=1000,
            C=5.0,
            class_weight="balanced",
        )),
    ])

    pipeline.fit(X_train, y_train)

    y_pred = pipeline.predict(X_test)
    report = classification_report(y_test, y_pred, output_dict=True)
    print(classification_report(y_test, y_pred))
    print(f"Macro F1: {report['macro avg']['f1-score']:.3f}")

    joblib.dump(pipeline, model_path)
    print(f"Model saved to {model_path}")
    return pipeline


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="data/tickets.csv")
    parser.add_argument("--model-out", default="classifier/model.pkl")
    args = parser.parse_args()
    train(args.data, args.model_out)
