import pandas as pd
import re
import nltk
from nltk.corpus import stopwords
from src.config import TRAIN_FILE, TEST_FILE

# Download stopwords 
nltk.download('stopwords')

stop_words = set(stopwords.words('english'))


def clean_text(text):
    # Convert to lowercase
    text = text.lower()

    # Remove special characters and numbers
    text = re.sub(r'[^a-z\s]', '', text)

    # Remove stopwords
    words = text.split()
    words = [word for word in words if word not in stop_words]

    return " ".join(words)


def load_and_preprocess():
    # Load dataset
    train_df = pd.read_csv(TRAIN_FILE)
    test_df = pd.read_csv(TEST_FILE)

    # AG News format: [label, title, description]
    train_df.columns = ["label", "title", "description"]
    test_df.columns = ["label", "title", "description"]

    # Combine title + description
    train_df["text"] = train_df["title"] + " " + train_df["description"]
    test_df["text"] = test_df["title"] + " " + test_df["description"]

    # Clean text
    train_df["text"] = train_df["text"].apply(clean_text)
    test_df["text"] = test_df["text"].apply(clean_text)

    X_train = train_df["text"]
    y_train = train_df["label"]

    X_test = test_df["text"]
    y_test = test_df["label"]

    return X_train, X_test, y_train, y_test
