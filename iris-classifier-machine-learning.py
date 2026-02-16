"""
Iris Classifier using Logistic Regression
Author: Yam1let


This script trains a machine learning model to classify iris flowers
based on their physical measurements.
"""

# Try to import scikit-learn, if not installed, install it
try:
    from sklearn.datasets import load_iris
    from sklearn.model_selection import train_test_split
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import accuracy_score, classification_report

except ImportError:
    import sys
    import subprocess
    print("scikit-learn is not installed. Installing now...")
    subprocess.check_call([sys.executable, "-m", "pip", "install", "scikit-learn"])
    print("Installation completed! Continuing...\n")
    from sklearn.datasets import load_iris
    from sklearn.model_selection import train_test_split
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import accuracy_score, classification_report


def main():
    # Load dataset
    iris = load_iris()
    X = iris.data
    y = iris.target

    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Create model
    model = LogisticRegression(max_iter=200)

    # Train model
    model.fit(X_train, y_train)

    # Make predictions
    predictions = model.predict(X_test)

    # Evaluate model
    accuracy = accuracy_score(y_test, predictions)

    print("Model Accuracy:", accuracy)
    print("\nClassification Report:\n")
    print(classification_report(y_test, predictions))


if __name__ == "__main__":
    main()

