from src.data_preprocessing import load_and_preprocess
from src.feature_engineering import create_tfidf_features
from src.train import train_model
from src.evaluate import evaluate_model


def main():
    print("Loading and preprocessing data...")
    X_train, X_test, y_train, y_test = load_and_preprocess()

    print("Creating TF-IDF features...")
    X_train_tfidf, X_test_tfidf, vectorizer = create_tfidf_features(X_train, X_test)

    print("Training model...")
    model = train_model(X_train_tfidf, y_train, vectorizer)

    print("Evaluating model...")
    evaluate_model(model, X_test_tfidf, y_test)

    print("Process completed successfully!")


if __name__ == "__main__":
    main()
