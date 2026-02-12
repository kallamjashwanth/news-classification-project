from sklearn.linear_model import LogisticRegression
import joblib
import os
from src.config import MODEL_PATH


def train_model(X_train, y_train, vectorizer):
    # Create model
    model = LogisticRegression(max_iter=1000)

    # Train model
    model.fit(X_train, y_train)

    # Save model
    joblib.dump(model, MODEL_PATH)

    # Save vectorizer (same folder as model)
    vectorizer_path = os.path.join(os.path.dirname(MODEL_PATH), "tfidf_vectorizer.pkl")
    joblib.dump(vectorizer, vectorizer_path)

    return model

