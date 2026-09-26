# File:         model.py
# Assignment:   Linear Regression Architecture / MLOps Workshop
# Course:       CSCN8010 - Foundations of Machine Learning Frameworks
# Date:         September 25th, 2026
# Team 3:       Ricardo Mohammed (7500382), Senay Teweldebrhan (9120588)
#               Zeynep Ozdemir (9045142), Juan Camilo Chirivi (9115141)
# Description:  This file contains the implementation of the Linear Regression model training
#               function.


from sklearn.linear_model import LinearRegression


def train_model(X_train, y_train):
    """
    Train a Linear Regression model using scikit-learn.

    This function fits a scikit-learn LinearRegression model using the
    provided training features and target values.

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