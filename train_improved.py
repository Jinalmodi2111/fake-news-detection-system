import pandas as pd
import numpy as np
import pickle
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.metrics import classification_report
import re

CSV_PATH = "data/train.csv"
OUTPUT_MODEL = "model_artifact.pkl"

def clean_text(text):
    text = re.sub(r"http\S+", "", text)
    text = re.sub(r"[^a-zA-Z ]", " ", text)
    text = text.lower()
    text = re.sub(r"\s+", " ", text).strip()
    return text

def load_data():
    df = pd.read_csv(CSV_PATH)

    # FIX: your file format is label,text
    df = df.dropna(subset=['text', 'label'])
    
    # clean text before vectorizing
    df['text'] = df['text'].astype(str).apply(clean_text)

    X = df['text'].tolist()
    y = df['label'].tolist()

    return X, y

def build_model():
    word_vectorizer = TfidfVectorizer(
        ngram_range=(1,2),
        analyzer='word',
        max_features=40000
    )

    char_vectorizer = TfidfVectorizer(
        ngram_range=(3,5),
        analyzer='char',
        max_features=25000
    )

    vect = FeatureUnion([
        ("word", word_vectorizer),
        ("char", char_vectorizer)
    ])

    model = LogisticRegression(
        solver='saga',
        max_iter=5000,
        class_weight='balanced'
    )

    pipe = Pipeline([
        ('vect', vect),
        ('clf', model)
    ])

    return pipe

def train():
    X, y = load_data()

    X_train, X_val, y_train, y_val = train_test_split(
        X, y,
        test_size=0.15,
        stratify=y,
        random_state=42
    )

    pipe = build_model()

    params = {
        'clf__C': [0.5, 1.0, 2.0],
    }

    cv = StratifiedKFold(n_splits=4, shuffle=True, random_state=42)

    gs = GridSearchCV(
        pipe,
        params,
        cv=cv,
        n_jobs=-1,
        scoring="f1_macro",
        verbose=1
    )

    gs.fit(X_train, y_train)
    best_model = gs.best_estimator_

    preds = best_model.predict(X_val)
    print("\nValidation Report:\n")
    print(classification_report(y_val, preds))

    artifact = {
        "pipeline": best_model
    }

    with open(OUTPUT_MODEL, "wb") as f:
        pickle.dump(artifact, f)

    print("\nModel saved as:", OUTPUT_MODEL)

if __name__ == "__main__":
    train()
