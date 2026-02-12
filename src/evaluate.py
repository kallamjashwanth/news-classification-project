from sklearn.metrics import accuracy_score, classification_report
from src.config import RESULTS_PATH


def evaluate_model(model, X_test, y_test):
    # Make predictions
    predictions = model.predict(X_test)

    # Calculate accuracy
    accuracy = accuracy_score(y_test, predictions)

    # Generate detailed report
    report = classification_report(y_test, predictions)

    # Save results to file
    with open(RESULTS_PATH, "w") as f:
        f.write(f"Accuracy: {accuracy}\n\n")
        f.write(report)

    print("Accuracy:", accuracy)
    print("\nClassification Report:\n")
    print(report)
