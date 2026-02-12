import os

# Base directories
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

DATA_RAW_PATH = os.path.join(BASE_DIR, "data", "raw")
DATA_PROCESSED_PATH = os.path.join(BASE_DIR, "data", "processed")

MODEL_PATH = os.path.join(BASE_DIR, "models", "news_classifier.pkl")

RESULTS_PATH = os.path.join(BASE_DIR, "results", "metrics.txt")

# Dataset files
TRAIN_FILE = os.path.join(DATA_RAW_PATH, "train.csv")
TEST_FILE = os.path.join(DATA_RAW_PATH, "test.csv")
