import pandas as pd
from sklearn.model_selection import train_test_split


def preprocess_data(df, feature, target, test_size=0.2, random_state=42):
    """
    Prepare the dataset for model training.

    Parameters:
        df: Pandas DataFrame
        feature: Independent variable (X)
        target: Dependent variable (y)
        test_size: Percentage of data used for testing
        random_state: Seed for reproducibility
    """

    # Select only the required columns
    data = df[[feature, target]].copy()

    # Remove rows with missing values
    data = data.dropna()

    # Define X and y
    X = data[[feature]]
    y = data[target]

    # Split into training and testing datasets
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state
    )

    return X_train, X_test, y_train, y_test