# Scikit-learn Linear Regression is currently used to test
# and validate the MLOps pipeline.
#
# The Session 1 implementation using gradient descent can later
# be integrated into this module. The learning_rate and iterations
# parameters are already defined in experiment_config.yaml.

from sklearn.linear_model import LinearRegression


def train_model(X_train, y_train):
    """
    Train a Linear Regression model using scikit-learn.

    Parameters:
        X_train: Training feature data.
        y_train: Training target values.

    Returns:
        model: Trained Linear Regression model.

    Note:
        This implementation is currently used to validate the
        MLOps architecture. The gradient descent implementation
        from Session 1 can be integrated later.
    """

    # Create the Linear Regression model.
    model = LinearRegression()

    # Train the model using the training data.
    model.fit(X_train, y_train)

    # Return the trained model so it can be evaluated.
    return model