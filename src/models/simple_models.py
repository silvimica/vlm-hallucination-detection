import numpy as np
from sklearn.svm import SVC
from sklearn.metrics import classification_report, accuracy_score, f1_score
from sklearn.preprocessing import StandardScaler
from datasets import Dataset
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier

import joblib 
def train_and_evaluate_svm(train_dataset: Dataset, validation_dataset: Dataset):
    # Extract labels
    y_train = train_dataset['label']
    y_val = validation_dataset['label']

    # Convert the features from lists of dictionaries to a matrix format
    X_train = np.vstack(train_dataset['features'])
    X_val = np.vstack(validation_dataset['features'])

    # Standardize the feature matrices
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)  # Fit and transform on training data
    X_val_scaled = scaler.transform(X_val)          # Only transform on validation data

    # Initialize and train the SVM classifier
    svm_classifier = SVC(kernel='sigmoid', random_state=42)  # Adjust kernel as needed
    svm_classifier.fit(X_train_scaled, y_train)

    # Predictions and evaluation
    y_pred = svm_classifier.predict(X_val_scaled)

    # Print evaluation metrics
    accuracy = accuracy_score(y_val, y_pred)
    f1 = f1_score(y_val, y_pred, average='macro')
    report = classification_report(y_val, y_pred)

    print("Accuracy:", accuracy)
    print("F1 Score:", f1)
    print("Classification Report:\n", report)

    return svm_classifier

def train_and_evaluate_logreg(train_dataset: Dataset, validation_dataset: Dataset):
    # Extract labels
    y_train = train_dataset['label']
    y_val = validation_dataset['label']

    # Convert the features from lists of dictionaries to a matrix format
    X_train = np.vstack(train_dataset['features'])
    X_val = np.vstack(validation_dataset['features'])

    # Standardize the feature matrices
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)

    # Initialize and train the Logistic Regression classifier
    logreg_classifier = LogisticRegression(random_state=42, max_iter=1000)  # Adjust max_iter if convergence issues
    logreg_classifier.fit(X_train_scaled, y_train)

    # Predictions and evaluation
    y_pred = logreg_classifier.predict(X_val_scaled)

    # Print evaluation metrics
    accuracy = accuracy_score(y_val, y_pred)
    f1 = f1_score(y_val, y_pred, average='macro')
    report = classification_report(y_val, y_pred)

    print("Logistic Regression - Accuracy:", accuracy)
    print("Logistic Regression - F1 Score:", f1)
    print("Logistic Regression - Classification Report:\n", report)

    return logreg_classifier


def train_and_evaluate_mlp(train_dataset: Dataset, validation_dataset: Dataset):
    # Extract labels
    y_train = train_dataset['label']
    y_val = validation_dataset['label']

    # Convert the features from lists of dictionaries to a matrix format
    X_train = np.vstack(train_dataset['features'])
    X_val = np.vstack(validation_dataset['features'])

    # Standardize the feature matrices
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)

    # Initialize and train the MLP classifier
    mlp_classifier = MLPClassifier(hidden_layer_sizes=(256, 128, 64, 32), max_iter=1000, random_state=42)  # Adjust hidden layers and iterations as needed
    mlp_classifier.fit(X_train_scaled, y_train)
    
    # Predictions and evaluation
    y_pred = mlp_classifier.predict(X_val_scaled)

    # Print evaluation metrics
    accuracy = accuracy_score(y_val, y_pred)
    f1 = f1_score(y_val, y_pred, average='macro')
    report = classification_report(y_val, y_pred)

    print("MLP - Accuracy:", accuracy)
    print("MLP - F1 Score:", f1)
    print("MLP - Classification Report:\n", report)

    return mlp_classifier, scaler 