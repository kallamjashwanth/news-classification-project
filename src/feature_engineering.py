from sklearn.feature_extraction.text import TfidfVectorizer


def create_tfidf_features(X_train, X_test):
    # Create TF-IDF object
    vectorizer = TfidfVectorizer(max_features=5000)

    # Learn vocabulary from training data and transform it
    X_train_tfidf = vectorizer.fit_transform(X_train)

    # Transform test data using same vocabulary
    X_test_tfidf = vectorizer.transform(X_test)

    return X_train_tfidf, X_test_tfidf, vectorizer
