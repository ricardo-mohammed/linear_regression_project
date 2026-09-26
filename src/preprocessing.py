# File:         preprocessing.py
# Assignment:   Linear Regression Architecture / MLOps Workshop
# Course:       CSCN8010 - Foundations of Machine Learning Frameworks
# Date:         September 25th, 2026
# Team 3:       Ricardo Mohammed (7500382), Senay Teweldebrhan (9120588)
#               Zeynep Ozdemir (9045142), Juan Camilo Chirivi (9115141)
# Description:  This file contains data-cleaning and feature preparation logic for
#               the linear regression workflow.


import pandas as pd
from sklearn.model_selection import train_test_split


def preprocess_data(df, feature, target, test_size=0.2, random_state=42):
    """
    Prepare the dataset before model training.

    Process:
    1. Select the feature (X) and target (y).
    2. Remove rows with missing values.
    3. Separate X and y.
    4. Split the data into training and testing sets.

    This function selects the chosen feature and target columns, removes rows
    with missing values, and splits the dataset into training and testing sets.

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


def prepare_california_regression_data(df, feature_column="avg_bedrooms", target_column="median_house_value", max_feature_value=12):
    """
    Prepare California housing data for univariate regression analysis.

    This function derives the average room and occupancy features used in the
    notebook's regression visuals, removes legacy cumulative columns, and filters
    out extreme values to keep the relationship easier to interpret.

    Parameters:
        df: Pandas DataFrame containing the California housing dataset.
        feature_column: Name of the feature column to use for the regression plot.
        target_column: Name of the target variable to visualize and model.
        max_feature_value: Upper threshold used to exclude extreme outliers.

    Returns:
        Pandas DataFrame filtered and enriched for the regression analysis.

    Constraints:
        - The source DataFrame must include the columns required for the derived
          feature calculations: total_rooms, households, population, and
          median_house_value.
        - The feature_column must be numeric after transformation.
        - The max_feature_value must be positive and should represent the cutoff
          used to remove unrealistic outliers.
    """

    data = df.copy()
    data["avg_bedrooms"] = data["total_rooms"] / data["households"]
    data["avg_occupancy"] = data["population"] / data["households"]
    data = data.drop(columns=["total_rooms", "total_bedrooms", "population"], errors="ignore")

    filtered_data = data[data[feature_column] < max_feature_value].copy()

    return filtered_data