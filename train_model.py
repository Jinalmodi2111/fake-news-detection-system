import pandas as pd
import numpy as np
import re
import pickle
import os
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# CLEAN TEXT FUNCTION
def clean_text(text):
    if not isinstance(text, str):
        return ""
    text = re.sub(r"http\S+", "", text)
    text = re.sub(r"[^A-Za-z0-9 ]+", "", text)
    return text.lower()

# LOAD DATASET
df = pd.read_csv("dataset/news.csv")

# FIX LABELS  (FAKE=0, REAL=1)
df["label"] = df["label"].replace({"FAKE": 0, "REAL": 1}).infer_objects(copy=False)

# CLEAN ARTICLES
df["text"] = df["text"].apply(clean_text)

# TRAIN/TEST SPLIT
X_train, X_test, y_train, y_test = train_test_split(
    df["text"], df["label"], test_size=0.2, random_state=42
)

# TF-IDF VECTORIZER
vectorizer = TfidfVectorizer(stop_words="english", max_features=5000)
X_train_tf = vectorizer.fit_transform(X_train)  #Transform text into TF-IDF vectors
X_test_tf = vectorizer.transform(X_test) #transform test text using same vocabulary learned from train set

# MODEL (LOGISTIC REGRESSION)
model = LogisticRegression(max_iter=300)
model.fit(X_train_tf, y_train)

# ACCURACY
pred = model.predict(X_test_tf)
acc = accuracy_score(y_test, pred)
print("Training Completed!")
print("Accuracy:", acc)

# SAVE MODEL & VECTORIZER
os.makedirs("model", exist_ok=True)

pickle.dump(model, open("model/model.pkl", "wb"))
pickle.dump(vectorizer, open("model/vectorizer.pkl", "wb"))

print("Model saved successfully in /model folder!")
