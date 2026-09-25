from sklearn.linear_model import LinearRegression


def train_model(X_train, y_train):
    """
    Train a Linear Regression model.

    Parameters:
        X_train: Training features
        y_train: Training target values

    Returns:
        Trained Linear Regression model
    """

    # Create the model
    model = LinearRegression()

    # Train the model
    model.fit(X_train, y_train)

    return model