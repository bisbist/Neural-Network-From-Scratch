from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
import numpy as np

def generate_data():
    """
    Generates a synthetic multi-class classification dataset using scikit-learn's make_classification.

    Returns:
        X_train, X_test, y_train, y_test: Tuple of training and testing features and one-hot encoded labels.
    """
    # Generate synthetic classification data
    X, y = make_classification(
        n_samples=1000,       # Total number of samples
        n_features=20,        # Number of features
        n_classes=3,          # Number of output classes
        n_informative=10,     # Number of informative features
        random_state=42       # Seed for reproducibility
    )

    # Convert class labels to one-hot encoded format
    y_onehot = np.eye(3)[y]

    # Split the dataset into training and testing sets (80% train, 20% test)
    return train_test_split(X, y_onehot, test_size=0.2, random_state=42)


def normalize(train, test):
    """
    Applies z-score normalization to the dataset.

    Args:
        train (np.ndarray): Training feature matrix.
        test (np.ndarray): Testing feature matrix.

    Returns:
        Tuple: Normalized training and testing datasets.
    """
    # Compute mean and standard deviation from the training data
    mean = np.mean(train, axis=0)
    std = np.std(train, axis=0)

    # Normalize both training and test sets using training stats
    return (train - mean) / std, (test - mean) / std
