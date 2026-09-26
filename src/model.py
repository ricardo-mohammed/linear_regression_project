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
    Train a Linear Regression model.

    This function fits a scikit-learn LinearRegression model using the
    provided training features and target values.

    Parameters:
        X_train: Training feature matrix.
        y_train: Training target values aligned with X_train.

    Returns:
        Trained Linear Regression model.

    Constraints:
        - X_train and y_train must be aligned row-by-row.
        - Missing values should be handled before calling this function.
        - The input data should be numerical for sklearn compatibility.
    """

    # Create the model
    model = LinearRegression()

    # Train the model
    model.fit(X_train, y_train)

    return model