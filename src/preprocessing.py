# train_test_split is used to divide the dataset into
# training data and testing data.
from sklearn.model_selection import train_test_split


def preprocess_data(df, feature, target, test_size=0.2, random_state=42):
    """
    Prepare the dataset before model training.

    Process:
    1. Select the feature (X) and target (y).
    2. Remove rows with missing values.
    3. Separate X and y.
    4. Split the data into training and testing sets.

    Parameters:
        df: Original dataset.
        feature: Independent variable (X).
        target: Dependent variable (y).
        test_size: Percentage of data used for testing.
        random_state: Controls the random split for reproducibility.

    Returns:
        X_train: Feature data used to train the model.
        X_test: Feature data used to test the model.
        y_train: Target values used for training.
        y_test: Target values used for testing.
    """

    # Keep only the feature and target needed for Linear Regression.
    data = df[[feature, target]].copy()

    # Remove rows containing missing values.
    data = data.dropna()

    # X represents the independent variable used to make predictions.
    X = data[[feature]]

    # y represents the value that the model tries to predict.
    y = data[target]

    # Divide the dataset into training and testing data.
    #
    # random_state makes the split reproducible:
    # running the experiment again with the same value
    # produces the same train/test split.
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state
    )

    return X_train, X_test, y_train, y_test