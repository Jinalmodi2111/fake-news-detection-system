import pickle
import re
import os
import colorama
from colorama import Fore, Style

colorama.init(autoreset=True)

MODEL_PATH = "model_artifact.pkl"

# --------------------------
# Load Combined Pipeline
# --------------------------
if not os.path.exists(MODEL_PATH):
    print(Fore.RED + f"\nERROR: {MODEL_PATH} not found!")
    print("Run: python train_improved.py first to generate model_artifact.pkl\n")
    exit()

with open(MODEL_PATH, "rb") as f:
    ART = pickle.load(f)

PIPELINE = ART["pipeline"]

# --------------------------
# Text Cleaning
# --------------------------
def clean_text(text):
    text = re.sub(r"http\S+", "", text)
    text = re.sub(r"[^a-zA-Z ]", " ", text)
    text = text.lower()
    text = re.sub(r"\s+", " ", text).strip()
    return text


# --------------------------
# Main Loop
# --------------------------
print(Fore.CYAN + "---------- Fake News Detector (Offline Mode) ----------")

while True:
    text = input(Fore.YELLOW + "\nEnter news text (or type EXIT to quit):\n")

    if text.lower() == "exit":
        print(Fore.BLUE + "\nExiting... Goodbye! 👋")
        break

    cleaned = clean_text(text)

    # Model Predictions
    pred = PIPELINE.predict([cleaned])[0]
    proba = PIPELINE.predict_proba([cleaned])[0]
    confidence = round(max(proba), 4)

    # Output Formatting
    if pred.upper() == "FAKE":
        print(Fore.RED + f"\n🚨 RESULT: FAKE NEWS")
        print(Fore.RED + f"🔍 Confidence: {confidence}")
    else:
        print(Fore.GREEN + f"\n✅ RESULT: REAL NEWS")
        print(Fore.GREEN + f"🔍 Confidence: {confidence}")

    print(Style.DIM + "\n" + "-"*50)
